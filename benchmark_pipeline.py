import torch
import time
import os
import numpy as np
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from argparse import ArgumentParser
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import cameraList_from_camInfos
from scene.dataset_readers import sceneLoadTypeCallbacks
import dequantize_cuda

def quantize_and_compress(gaussian_model, save_path):
    print("--- [Offline] Compressing Gaussian SH Parameters ---")
    shs = gaussian_model.get_features.detach().view(gaussian_model.get_xyz.shape[0], -1)
    shs_soa = shs.t().contiguous() # [48, N]
    
    sh_min = shs_soa.min(dim=1, keepdim=True)[0]
    sh_max = shs_soa.max(dim=1, keepdim=True)[0]
    sh_scale = (sh_max - sh_min) / 255.0
    sh_scale[sh_scale == 0] = 1e-5 
    sh_zp = torch.round(-sh_min / sh_scale) - 128.0

    shs_q = torch.round(shs_soa / sh_scale + sh_zp)
    shs_q = torch.clamp(shs_q, -128, 127).to(torch.int8)

    # VRAM/Disk 절약 계산
    orig_mb = shs.element_size() * shs.nelement() / (1024**2)
    comp_mb = shs_q.element_size() * shs_q.nelement() / (1024**2)
    print(f"SH Memory Size: {orig_mb:.2f} MB (FP32) -> {comp_mb:.2f} MB (INT8)")
    print(f"Compression Ratio for SH: {orig_mb/comp_mb:.2f}x")

    np.savez_compressed(save_path, 
                        shs_q=shs_q.cpu().numpy(), 
                        sh_scale=sh_scale.cpu().numpy(), 
                        sh_zp=sh_zp.cpu().numpy())
    return shs_q, sh_scale, sh_zp

def benchmark_rendering(gaussian_model, viewpoints, background, q_shs, sh_scale, sh_zp, mode="baseline"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    frames = len(viewpoints)
    start_time = time.time()
    
    for view in viewpoints:
        if mode == "compressed":
            # 1. Real-time Parallel Decompression
            dequantized_shs = dequantize_cuda.dequantize(q_shs, sh_scale, sh_zp)
            override_shs = dequantized_shs.view(-1, 16, 3)
        else:
            override_shs = None # 사용 fp32 baseline
            
        # 2. Rasterization
        render(view, gaussian_model, background, override_shs=override_shs)
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    fps = frames / (end_time - start_time)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f"[{mode.upper()}] FPS: {fps:.2f} | Peak VRAM: {peak_vram:.2f} MB")
    return fps, peak_vram

if __name__ == "__main__":
    # 임의의 초기화 (테스트용 더미 데이터 생성)
    # 실제로는 argparse로 model_path를 받아 로드해야 합니다.
    print("Setting up dummy 3DGS scene for benchmark (1 Million Gaussians)...")
    num_gaussians = 1_000_000
    device = torch.device("cuda")
    
    class DummyModel:
        def __init__(self):
            self.get_xyz = torch.rand((num_gaussians, 3), device=device)
            self.get_features = torch.rand((num_gaussians, 1, 3), device=device) # DC
            self.get_features_rest = torch.rand((num_gaussians, 15, 3), device=device) # Rest
            self.get_opacity = torch.rand((num_gaussians, 1), device=device)
            self.get_scaling = torch.rand((num_gaussians, 3), device=device)
            self.get_rotation = torch.rand((num_gaussians, 4), device=device)
            self.active_sh_degree = 3
            # Fake the features property
            self.get_features = torch.cat([self.get_features, self.get_features_rest], dim=1)

    model = DummyModel()
    
    # 1. 압축 수행
    q_shs, sh_scale, sh_zp = quantize_and_compress(model, "compressed_scene.npz")
    
    # 2. 벤치마킹 환경 설정 (더미 카메라 100개)
    class DummyCam:
        def __init__(self):
            self.camera_center = torch.zeros(3, device=device)
            self.image_width = 800
            self.image_height = 800
            self.world_view_transform = torch.eye(4, device=device)
            self.full_proj_transform = torch.eye(4, device=device)
            self.tangent_fovx = 1.0
            self.tangent_fovy = 1.0
    
    views = [DummyCam() for _ in range(100)]
    bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    
    # 3. 속도 및 메모리 비교
    print("\n--- Running Baseline Benchmark ---")
    benchmark_rendering(model, views, bg_color, None, None, None, mode="baseline")
    
    print("\n--- Running Compressed Benchmark ---")
    benchmark_rendering(model, views, bg_color, q_shs, sh_scale, sh_zp, mode="compressed")
