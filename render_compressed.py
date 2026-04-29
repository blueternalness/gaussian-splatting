import torch
import time
import numpy as np
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render 
import dequantize_cuda 

def load_compressed_model(path, device):
    data = np.load(path)
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
        dequantized_shs = dequantize_cuda.dequantize(q_shs, sh_scale, sh_zp)
        
        gaussian_model.active_shs = dequantized_shs.view(-1, 16, 3) 

        render_pkg = render(view, gaussian_model, background, override_shs=gaussian_model.active_shs)
        frames += 1
        
    torch.cuda.synchronize()
    end_time = time.time()
    
    fps = frames / (end_time - start_time)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f"--- Benchmark Results ---")
    print(f"FPS: {fps:.2f}")
    print(f"Peak VRAM Usage: {peak_vram:.2f} MB")
    
    return fps, peak_vram
