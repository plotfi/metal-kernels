#include <metal_stdlib>
using namespace metal;

kernel void softmax(device const float* input  [[buffer(0)]],
                    device float* output        [[buffer(1)]],
                    constant uint& length       [[buffer(2)]],
                    uint tid  [[thread_position_in_threadgroup]],
                    uint tgs  [[threads_per_threadgroup]]) {
    threadgroup float shared_data[256];

    // Pass 1: find max for numerical stability
    float local_max = -INFINITY;
    for (uint i = tid; i < length; i += tgs) {
        local_max = max(local_max, input[i]);
    }
    shared_data[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for max
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_max = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 2: compute sum of exp(x - max)
    float local_sum = 0.0;
    for (uint i = tid; i < length; i += tgs) {
        local_sum += exp(input[i] - global_max);
    }
    shared_data[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction for sum
    for (uint stride = tgs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float global_sum = shared_data[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: write normalized output
    for (uint i = tid; i < length; i += tgs) {
        output[i] = exp(input[i] - global_max) / global_sum;
    }
}
