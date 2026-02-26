#include "verify_vector_add.h"

#include <iostream>

bool verify_vector_add(const float* a, const float* b,
                       MTL::Buffer* resultBuffer, uint32_t count) {
    auto* result = static_cast<float*>(resultBuffer->contents());
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

    return success;
}
