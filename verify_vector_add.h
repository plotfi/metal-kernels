#pragma once

#include <Metal/Metal.hpp>
#include <cstdint>

bool verify_vector_add(const float* a, const float* b,
                       MTL::Buffer* resultBuffer, uint32_t count);
