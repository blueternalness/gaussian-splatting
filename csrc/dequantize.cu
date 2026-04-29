#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel: Real-time Dequantization
__global__ void dequantize_sh_kernel_soa(
    const int8_t* __restrict__ quantized_shs, // [48, N] SoA format
    const float* __restrict__ scales,         // [48]
    const float* __restrict__ zero_points,    // [48]
    int num_gaussians,
    int sh_dim, // 48
    float* __restrict__ out_shs               // [N, 48] AoS format for Rasterizer
) {
    // 1. Shared Memory Utilization: Scale과 Zero-point를 Shared Memory에 로드
    // Global 메모리 접근(중복)을 최소화
    extern __shared__ float shared_mem[];
    float* s_scales = shared_mem;
    float* s_zps = &shared_mem[sh_dim];

    // Thread Block 내의 스레드들이 협력하여 48개의 scale/zp를 SMEM으로 로드
    if (threadIdx.x < sh_dim) {
        s_scales[threadIdx.x] = scales[threadIdx.x];
        s_zps[threadIdx.x] = zero_points[threadIdx.x];
    }
    __syncthreads(); // 모든 스레드가 로드 완료될 때까지 대기

    // 2. Thread-Level Parallelism: 1 Thread = 1 Gaussian
    int gaussian_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gaussian_idx >= num_gaussians) return;

    // 3. Memory Coalescing: SoA 구조를 읽어서 AoS(원래 렌더러가 요구하는 형태)로 변환
    for (int i = 0; i < sh_dim; i++) {
        // [48, N] SoA Memory Access -> Warp 내 스레드들이 연속된 메모리에 접근하므로 Coalesced Access 달성!
        int soa_idx = i * num_gaussians + gaussian_idx; 
        int8_t q_val = quantized_shs[soa_idx];

        // Dequantize: (q_val - zp) * scale
        float dequantized_val = (static_cast<float>(q_val) - s_zps[i]) * s_scales[i];

        // Write to AoS output: [N, 48]
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

    return out_shs; // 이 결과값을 기존 Rasterizer의 입력으로 던져줍니다.
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dequantize", &dequantize_sh_cuda, "Parallel SH Dequantization (CUDA)");
}
