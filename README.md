# Metal Kernels

Just toying around with compute

## Building

```bash
mkdir -p build && cd build
cmake ..
ninja
```

metal-cpp is downloaded automatically on first configure if not already present.

## Usage

```
./metal_kernel <file.metal> <kernel_name> <grid_size> [buffer_specs...]
```

### Grid size

| Format | Dispatch mode | Description |
|--------|--------------|-------------|
| `N` | `dispatchThreads` | N total threads, auto threadgroup size |
| `N,T` | `dispatchThreads` | N total threads, threadgroup size T |
| `NxT` | `dispatchThreadgroups` | N threadgroups of T threads each |

### Buffer specs

Format: `type:count[:mode[:value]]`

- **type**: `float` or `uint`
- **count**: number of elements
- **mode**:
  - `in` (default) — filled with sequential data (0, 1, 2, ...)
  - `out` — zero-initialized, printed after execution
  - `const` — filled with `value` (for constant uniforms)

### Examples

```bash
# vector_add: two input buffers added element-wise into an output buffer
./metal_kernel vector_add.metal vector_add 1024 \
    float:1024:in float:1024:in float:1024:out

# softmax: 1 threadgroup of 256 threads, with a constant length parameter
./metal_kernel softmax.metal softmax 1x256 \
    float:256:in float:256:out uint:1:const:256
```

Built-in verification runs automatically for `vector_add` and `softmax` kernels.
