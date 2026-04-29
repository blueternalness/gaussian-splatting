import torch
import os
import time
import numpy as np
from argparse import ArgumentParser
import torchvision
from utils.general_utils import safe_state
from scene import Scene, GaussianModel
from gaussian_renderer import render
import dequantize_cuda

class DummyPipe:
    compute_cov3D_python = False
    convert_SHs_python = False
    debug = False
    antialiasing = False 

def get_model_vram_mb(gaussians):
    mem = 0
    for name, param in gaussians.__dict__.items():
        if isinstance(param, torch.Tensor) and param.is_cuda:
            mem += param.element_size() * param.nelement()
        elif isinstance(param, torch.nn.Parameter) and param.data.is_cuda:
            mem += param.data.element_size() * param.data.nelement()
    return mem / (1024**2)

def quantize_and_compress(gaussian_model, save_path):
    shs = gaussian_model.get_features.detach().view(gaussian_model.get_xyz.shape[0], -1)
    shs_soa = shs.t().contiguous() 
    
    sh_min = shs_soa.min(dim=1, keepdim=True)[0]
    sh_max = shs_soa.max(dim=1, keepdim=True)[0]
    sh_scale = (sh_max - sh_min) / 255.0
    sh_scale[sh_scale == 0] = 1e-5 
    sh_zp = torch.round(-sh_min / sh_scale) - 128.0

    shs_q = torch.round(shs_soa / sh_scale + sh_zp)
    shs_q = torch.clamp(shs_q, -128, 127).to(torch.int8)
    return shs_q, sh_scale, sh_zp

@torch.no_grad()
def render_baseline(model_path, views, gaussians, background):
    render_path = os.path.join(model_path, "baseline", "renders")
    os.makedirs(render_path, exist_ok=True)
    pipe = DummyPipe()
    
    model_vram = get_model_vram_mb(gaussians)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    start_time = time.time()
    for idx, view in enumerate(views):
        rendering = render(view, gaussians, pipe, background)["render"]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
    torch.cuda.synchronize()
    fps = len(views) / (time.time() - start_time)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f"[BASELINE] Static Model VRAM: {model_vram:.2f} MB")
    print(f"[BASELINE] FPS: {fps:.2f} | Rendering Peak VRAM: {peak_vram:.2f} MB\n")

@torch.no_grad()
def render_compressed(model_path, views, gaussians, background, q_shs, sh_scale, sh_zp):
    render_path = os.path.join(model_path, "compressed", "renders")
    os.makedirs(render_path, exist_ok=True)
    pipe = DummyPipe()
    
    del gaussians._features_dc
    del gaussians._features_rest
    torch.cuda.empty_cache()
    
    model_vram = get_model_vram_mb(gaussians) + (q_shs.element_size() * q_shs.nelement() / (1024**2))
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    start_time = time.time()
    for idx, view in enumerate(views):
        dequantized_shs = dequantize_cuda.dequantize(q_shs, sh_scale, sh_zp).view(-1, 16, 3)
        gaussians._features_dc = dequantized_shs[:, 0:1, :]
        gaussians._features_rest = dequantized_shs[:, 1:16, :]
            
        rendering = render(view, gaussians, pipe, background)["render"]
        
        gaussians._features_dc = None
        gaussians._features_rest = None
        del dequantized_shs
        
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
    torch.cuda.synchronize()
    fps = len(views) / (time.time() - start_time)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
    
    print(f"[COMPRESSED] Static Model VRAM: {model_vram:.2f} MB (Compression Success!)")
    print(f"[COMPRESSED] FPS: {fps:.2f} | Rendering Peak VRAM: {peak_vram:.2f} MB (PyTorch Materialization Overhead)\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True)
    parser.add_argument("-s", "--source_path", type=str, required=True)
    parser.add_argument("-r", "--resolution", type=int, default=2)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--depths", type=str, default="") 
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--data_device", type=str, default="cuda") 
    parser.add_argument("--train_test_exp", action="store_true", default=False)
    parser.add_argument("--sh_degree", type=int, default=3)
    
    args = parser.parse_args()
    safe_state(False)
    
    gaussians = GaussianModel(sh_degree=3)
    scene = Scene(args, gaussians, load_iteration=-1, shuffle=False)
    
    bg_color = [1,1,1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTestCameras() 
    if len(views) == 0: views = scene.getTrainCameras() 

    print("\n--- Running Baseline Rendering ---")
    render_baseline(args.model_path, views, gaussians, background)
    
    q_shs, sh_scale, sh_zp = quantize_and_compress(gaussians, os.path.join(args.model_path, "compressed_shs.npz"))
    
    print("\n--- Running Compressed Rendering ---")
    render_compressed(args.model_path, views, gaussians, background, q_shs, sh_scale, sh_zp)