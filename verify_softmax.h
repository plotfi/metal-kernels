#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>

bool verify_softmax(const float* input, MTL::Buffer* resultBuffer, uint32_t count);
