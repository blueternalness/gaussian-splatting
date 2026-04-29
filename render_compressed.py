import torch
import time
import numpy as np
from scene.gaussian_model import GaussianModel
# 렌더링 파이프라인 (기존 3DGS 코드)
from gaussian_renderer import render 
# 커스텀 CUDA 익스텐션 로드
import dequantize_cuda 

def load_compressed_model(path, device):
    data = np.load(path)
    # [48, N] int8 tensor
    q_shs = torch.tensor(data['shs_q'], dtype=torch.int8, device=device) 
    sh_scale = torch.tensor(data['sh_scale'], dtype=torch.float32, device=device)
    sh_zp = torch.tensor(data['sh_zp'], dtype=torch.float32, device=device)
    return q_shs, sh_scale, sh_zp, data

def benchmark_rendering(q_shs, sh_scale, sh_zp, background, viewpoints, gaussian_model):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    frames = 0
    
    for view in viewpoints:
        # 1. Real-time Decompression (CUDA) - Render Time에 실행됨!
        # [48, N] int8 -> [N, 48] float32 변환
        dequantized_shs = dequantize_cuda.dequantize(q_shs, sh_scale, sh_zp)
        
        # 모델에 임시로 꽂아주거나 렌더 함수에 직접 전달 (Gaussian 렌더러 구조에 맞게)
        # reshape to [N, 16, 3] expected by rasterizer
        gaussian_model.active_shs = dequantized_shs.view(-1, 16, 3) 

        # 2. Rasterization
        render_pkg = render(view, gaussian_model, background, override_shs=gaussian_model.active_shs)
        frames += 1
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    fps = frames / (end_time - start_time)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) # MB 단위
    
    print(f"--- Benchmark Results ---")
    print(f"FPS: {fps:.2f}")
    print(f"Peak VRAM Usage: {peak_vram:.2f} MB")
    
    return fps, peak_vram
