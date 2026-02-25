#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

int main() {
    const uint32_t count = 1024;

    // Get the default Metal device
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal is not supported on this device." << std::endl;
        return 1;
    }
    std::cout << "Using device: " << device->name()->utf8String() << std::endl;

    // Load the Metal shader source from file
    std::ifstream file("vector_add.metal");
    if (!file.is_open()) {
        std::cerr << "Failed to open vector_add.metal" << std::endl;
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
    NS::String* funcName = NS::String::string("vector_add",
                                               NS::UTF8StringEncoding);
    MTL::Function* function = library->newFunction(funcName);
    if (!function) {
        std::cerr << "Failed to find kernel function 'vector_add'." << std::endl;
        return 1;
    }

    MTL::ComputePipelineState* pipeline =
        device->newComputePipelineState(function, &error);
    if (!pipeline) {
        std::cerr << "Failed to create pipeline state: "
                  << error->localizedDescription()->utf8String() << std::endl;
        return 1;
    }

    // Prepare input data
    std::vector<float> a(count), b(count);
    for (uint32_t i = 0; i < count; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    // Create Metal buffers
    size_t bufferSize = count * sizeof(float);
    MTL::Buffer* bufferA = device->newBuffer(a.data(), bufferSize,
                                              MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferB = device->newBuffer(b.data(), bufferSize,
                                              MTL::ResourceStorageModeShared);
    MTL::Buffer* bufferResult = device->newBuffer(bufferSize,
                                                   MTL::ResourceStorageModeShared);

    // Create command queue and command buffer
    MTL::CommandQueue* commandQueue = device->newCommandQueue();
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();

    // Encode the compute command
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline);
    encoder->setBuffer(bufferA, 0, 0);
    encoder->setBuffer(bufferB, 0, 1);
    encoder->setBuffer(bufferResult, 0, 2);

    // Dispatch threads
    MTL::Size gridSize(count, 1, 1);
    NS::UInteger threadGroupSize = std::min(
        static_cast<NS::UInteger>(pipeline->maxTotalThreadsPerThreadgroup()),
        static_cast<NS::UInteger>(count));
    MTL::Size threadgroupSize(threadGroupSize, 1, 1);
    encoder->dispatchThreads(gridSize, threadgroupSize);
    encoder->endEncoding();

    // Submit and wait
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    // Verify results
    auto* result = static_cast<float*>(bufferResult->contents());
    bool success = true;
    for (uint32_t i = 0; i < count; i++) {
        float expected = a[i] + b[i];
        if (result[i] != expected) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << result[i]
                      << ", expected " << expected << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "SUCCESS: All " << count << " elements matched." << std::endl;
        std::cout << "Sample: " << a[5] << " + " << b[5] << " = " << result[5]
                  << std::endl;
    }

    // Release resources
    bufferResult->release();
    bufferB->release();
    bufferA->release();
    commandQueue->release();
    pipeline->release();
    function->release();
    library->release();
    device->release();

    return success ? 0 : 1;
}
