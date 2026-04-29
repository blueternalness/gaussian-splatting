#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void dequantize_sh_kernel_soa(
    const int8_t* __restrict__ quantized_shs, // [48, N] SoA format
    const float* __restrict__ scales,         // [48]
    const float* __restrict__ zero_points,    // [48]
    int num_gaussians,
    int sh_dim, // 48
    float* __restrict__ out_shs               // [N, 48] AoS format for Rasterizer
) {
    extern __shared__ float shared_mem[];
    float* s_scales = shared_mem;
    float* s_zps = &shared_mem[sh_dim];

    if (threadIdx.x < sh_dim) {
        s_scales[threadIdx.x] = scales[threadIdx.x];
        s_zps[threadIdx.x] = zero_points[threadIdx.x];
    }
    __syncthreads();

    int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussian_idx >= num_gaussians) return;

    for (int i = 0; i < sh_dim; i++) {
        int soa_idx = i * num_gaussians + gaussian_idx; 
        int8_t q_val = quantized_shs[soa_idx];

        float dequantized_val = (static_cast<float>(q_val) - s_zps[i]) * s_scales[i];

        int aos_idx = gaussian_idx * sh_dim + i;
        out_shs[aos_idx] = dequantized_val;
    }
}

// C++ Wrapper
torch::Tensor dequantize_sh_cuda(
    torch::Tensor quantized_shs, 
    torch::Tensor scales, 
    torch::Tensor zero_points) 
{
    int sh_dim = quantized_shs.size(0); // 48
    int num_gaussians = quantized_shs.size(1); // N

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(quantized_shs.device());
    torch::Tensor out_shs = torch::empty({num_gaussians, sh_dim}, options);

    int threads = 256;
    int blocks = (num_gaussians + threads - 1) / threads;
    
    // Shared memory size: 48 scales + 48 zero_points = 96 floats
    size_t shared_mem_size = 2 * sh_dim * sizeof(float);

    dequantize_sh_kernel_soa<<<blocks, threads, shared_mem_size>>>(
        quantized_shs.data_ptr<int8_t>(),
        scales.data_ptr<float>(),
        zero_points.data_ptr<float>(),
        num_gaussians,
        sh_dim,
        out_shs.data_ptr<float>()
    );

    return out_shs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize", &dequantize_sh_cuda, "Parallel SH Dequantization (CUDA)");
}
