#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "verify_softmax.h"
#include "verify_vector_add.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct BufferSpec {
    std::string type;   // "float" or "uint"
    uint32_t count;     // number of elements
    std::string mode;   // "in", "out", or "const"
    uint32_t constVal;  // value for const mode
};

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <file.metal> <kernel_name> <grid_size> [buffer_specs...]\n"
              << "\n"
              << "  grid_size:    N           (dispatchThreads, auto threadgroup size)\n"
              << "                N,T         (dispatchThreads, N total threads, threadgroup T)\n"
              << "                NxT         (dispatchThreadgroups, N groups of T threads)\n"
              << "\n"
              << "  buffer_spec:  type:count[:mode[:value]]\n"
              << "    type:  float | uint\n"
              << "    count: number of elements\n"
              << "    mode:  in   - fill with sequential data (default)\n"
              << "           out  - zero-initialized, printed after execution\n"
              << "           const - fill with value (for constant uniforms)\n"
              << "    value: integer value for const mode\n"
              << "\n"
              << "Examples:\n"
              << "  " << prog << " vector_add.metal vector_add 1024 "
              << "float:1024:in float:1024:in float:1024:out\n"
              << "  " << prog << " softmax.metal softmax 1x256 "
              << "float:256:in float:256:out uint:1:const:256\n";
}

static BufferSpec parse_buffer_spec(const std::string& spec) {
    BufferSpec bs{};
    bs.mode = "in";
    bs.constVal = 0;

    // Split on ':'
    std::vector<std::string> parts;
    std::istringstream ss(spec);
    std::string part;
    while (std::getline(ss, part, ':')) {
        parts.push_back(part);
    }

    if (parts.size() < 2) {
        std::cerr << "Invalid buffer spec: " << spec << "\n";
        std::exit(1);
    }

    bs.type = parts[0];
    if (bs.type != "float" && bs.type != "uint") {
        std::cerr << "Unknown buffer type: " << bs.type << "\n";
        std::exit(1);
    }

    bs.count = static_cast<uint32_t>(std::stoul(parts[1]));

    if (parts.size() >= 3) {
        bs.mode = parts[2];
        if (bs.mode != "in" && bs.mode != "out" && bs.mode != "const") {
            std::cerr << "Unknown buffer mode: " << bs.mode << "\n";
            std::exit(1);
        }
    }

    if (parts.size() >= 4) {
        bs.constVal = static_cast<uint32_t>(std::stoul(parts[3]));
    }

    return bs;
}

enum class DispatchMode { Threads, Threadgroups };

static void parse_grid_spec(const std::string& spec,
                             uint32_t& gridSize, uint32_t& threadgroupSize,
                             DispatchMode& mode) {
    // "N" → dispatchThreads with N total threads, auto threadgroup size
    // "N,T" → dispatchThreads with N total threads, threadgroup size T
    // "NxT" → dispatchThreadgroups with N threadgroups of T threads each
    auto x = spec.find('x');
    if (x != std::string::npos) {
        mode = DispatchMode::Threadgroups;
        gridSize = static_cast<uint32_t>(std::stoul(spec.substr(0, x)));
        threadgroupSize = static_cast<uint32_t>(std::stoul(spec.substr(x + 1)));
    } else {
        mode = DispatchMode::Threads;
        auto comma = spec.find(',');
        if (comma == std::string::npos) {
            gridSize = static_cast<uint32_t>(std::stoul(spec));
            threadgroupSize = 0; // auto
        } else {
            gridSize = static_cast<uint32_t>(std::stoul(spec.substr(0, comma)));
            threadgroupSize = static_cast<uint32_t>(std::stoul(spec.substr(comma + 1)));
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string metalFile = argv[1];
    std::string kernelName = argv[2];
    std::string gridSpec = argv[3];

    std::vector<BufferSpec> bufferSpecs;
    for (int i = 4; i < argc; i++) {
        bufferSpecs.push_back(parse_buffer_spec(argv[i]));
    }

    uint32_t gridSize, threadgroupSize;
    DispatchMode dispatchMode;
    parse_grid_spec(gridSpec, gridSize, threadgroupSize, dispatchMode);

    // Get the default Metal device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal is not supported on this device." << std::endl;
        return 1;
    }
    std::cout << "Using device: " << device->name()->utf8String() << std::endl;

    // Load the Metal shader source from file
    std::ifstream file(metalFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << metalFile << std::endl;
        return 1;
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    std::string shaderSource = ss.str();

    // Compile the shader library
    NS::Error* error = nullptr;
    NS::String* source = NS::String::string(shaderSource.c_str(),
                                             NS::UTF8StringEncoding);
    MTL::Library* library = device->newLibrary(source, nullptr, &error);
    if (!library) {
        std::cerr << "Failed to compile shader library: "
                  << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    // Create the compute pipeline
    NS::String* funcName = NS::String::string(kernelName.c_str(),
                                               NS::UTF8StringEncoding);
    MTL::Function* function = library->newFunction(funcName);
    if (!function) {
        std::cerr << "Failed to find kernel function '" << kernelName << "'."
                  << std::endl;
        return 1;
    }

    MTL::ComputePipelineState* pipeline =
        device->newComputePipelineState(function, &error);
    if (!pipeline) {
        std::cerr << "Failed to create pipeline state: "
                  << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    // Create Metal buffers
    std::vector<MTL::Buffer*> buffers;
    for (size_t i = 0; i < bufferSpecs.size(); i++) {
        const auto& bs = bufferSpecs[i];
        size_t elemSize = (bs.type == "float") ? sizeof(float) : sizeof(uint32_t);
        size_t bufSize = bs.count * elemSize;

        MTL::Buffer* buf = device->newBuffer(bufSize, MTL::ResourceStorageModeShared);

        if (bs.mode == "in") {
            if (bs.type == "float") {
                auto* ptr = static_cast<float*>(buf->contents());
                for (uint32_t j = 0; j < bs.count; j++) {
                    ptr[j] = static_cast<float>(j);
                }
            } else {
                auto* ptr = static_cast<uint32_t*>(buf->contents());
                for (uint32_t j = 0; j < bs.count; j++) {
                    ptr[j] = j;
                }
            }
        } else if (bs.mode == "const") {
            if (bs.type == "float") {
                auto* ptr = static_cast<float*>(buf->contents());
                for (uint32_t j = 0; j < bs.count; j++) {
                    ptr[j] = static_cast<float>(bs.constVal);
                }
            } else {
                auto* ptr = static_cast<uint32_t*>(buf->contents());
                for (uint32_t j = 0; j < bs.count; j++) {
                    ptr[j] = bs.constVal;
                }
            }
        }
        // "out" mode: buffer is already zero-initialized by Metal

        buffers.push_back(buf);
    }

    // Create command queue and command buffer
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

    // Encode the compute command
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline);

    for (size_t i = 0; i < buffers.size(); i++) {
        encoder->setBuffer(buffers[i], 0, i);
    }

    // Dispatch threads
    if (threadgroupSize == 0) {
        threadgroupSize = std::min(
            static_cast<uint32_t>(pipeline->maxTotalThreadsPerThreadgroup()),
            gridSize);
    }

    MTL::Size tg(threadgroupSize, 1, 1);
    if (dispatchMode == DispatchMode::Threadgroups) {
        MTL::Size grid(gridSize, 1, 1);
        encoder->dispatchThreadgroups(grid, tg);
    } else {
        MTL::Size grid(gridSize, 1, 1);
        encoder->dispatchThreads(grid, tg);
    }
    encoder->endEncoding();

    // Submit and wait
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // Print output buffers
    for (size_t i = 0; i < bufferSpecs.size(); i++) {
        const auto& bs = bufferSpecs[i];
        if (bs.mode != "out") continue;

        std::cout << "Buffer " << i << " (" << bs.type << "[" << bs.count << "]):\n";
        uint32_t printCount = std::min(bs.count, 32u);

        if (bs.type == "float") {
            auto* ptr = static_cast<float*>(buffers[i]->contents());
            for (uint32_t j = 0; j < printCount; j++) {
                std::cout << "  [" << j << "] = " << ptr[j] << "\n";
            }
        } else {
            auto* ptr = static_cast<uint32_t*>(buffers[i]->contents());
            for (uint32_t j = 0; j < printCount; j++) {
                std::cout << "  [" << j << "] = " << ptr[j] << "\n";
            }
        }

        if (bs.count > printCount) {
            std::cout << "  ... (" << (bs.count - printCount) << " more elements)\n";
        }
    }

    // Run verification for known kernels
    bool verified = true;
    if (kernelName == "vector_add" && buffers.size() >= 3 &&
        bufferSpecs[0].type == "float" && bufferSpecs[1].type == "float" &&
        bufferSpecs[2].mode == "out") {
        auto* a = static_cast<const float*>(buffers[0]->contents());
        auto* b = static_cast<const float*>(buffers[1]->contents());
        verified = verify_vector_add(a, b, buffers[2], bufferSpecs[2].count);
    } else if (kernelName == "softmax" && buffers.size() >= 2 &&
               bufferSpecs[0].type == "float" && bufferSpecs[1].mode == "out") {
        auto* input = static_cast<const float*>(buffers[0]->contents());
        verified = verify_softmax(input, buffers[1], bufferSpecs[1].count);
    }

    // Release resources
    for (auto* buf : buffers) {
        buf->release();
    }
    commandQueue->release();
    pipeline->release();
    function->release();
    library->release();
    device->release();

    std::cout << "Done." << std::endl;
    return verified ? 0 : 1;
}
