#include "verify_softmax.h"

#include <cmath>
#include <iostream>

bool verify_softmax(const float* input, MTL::Buffer* resultBuffer, uint32_t count) {
    auto* result = static_cast<float*>(resultBuffer->contents());

    // Compute expected softmax on CPU
    float maxVal = input[0];
    for (uint32_t i = 1; i < count; i++) {
        maxVal = std::max(maxVal, input[i]);
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        sum += std::expf(input[i] - maxVal);
    }

    bool success = true;
    for (uint32_t i = 0; i < count; i++) {
        float expected = std::expf(input[i] - maxVal) / sum;
        float diff = std::fabs(result[i] - expected);
        if (diff > 1e-5f) {
            std::cerr << "Mismatch at index " << i
                      << ": got " << result[i]
                      << ", expected " << expected
                      << " (diff " << diff << ")" << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "SUCCESS: All " << count << " softmax elements matched." << std::endl;
        std::cout << "Sample: softmax[0] = " << result[0]
                  << ", softmax[" << (count - 1) << "] = " << result[count - 1]
                  << std::endl;
    }

    return success;
}
