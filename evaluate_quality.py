import torch
import os
import lpips
from PIL import Image
import torchvision.transforms.functional as TF
from utils.loss_utils import ssim
from argparse import ArgumentParser
from tqdm import tqdm

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def main():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model directory")
    args = parser.parse_args()

    baseline_dir = os.path.join(args.model_path, "baseline", "renders")
    compressed_dir = os.path.join(args.model_path, "compressed", "renders")

    if not os.path.exists(baseline_dir) or not os.path.exists(compressed_dir):
        print("Error: Rendered image directories not found. Run render_and_compare.py first.")
        return

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    
    image_files = sorted(os.listdir(baseline_dir))
    
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    print("--- Evaluating Visual Fidelity (Baseline vs Compressed) ---")
    for img_name in tqdm(image_files):
        if not img_name.endswith(".png"): continue
            
        base_img = TF.to_tensor(Image.open(os.path.join(baseline_dir, img_name))).unsqueeze(0).cuda()
        comp_img = TF.to_tensor(Image.open(os.path.join(compressed_dir, img_name))).unsqueeze(0).cuda()

        psnr_val = calculate_psnr(base_img, comp_img).item()
        psnr_scores.append(psnr_val)

        ssim_val = ssim(base_img, comp_img).item()
        ssim_scores.append(ssim_val)

        base_img_lpips = base_img * 2.0 - 1.0
        comp_img_lpips = comp_img * 2.0 - 1.0
        lpips_val = loss_fn_vgg(base_img_lpips, comp_img_lpips).item()
        lpips_scores.append(lpips_val)

    print("\n========= [ Visual Fidelity ] =========")
    print(f"1. PSNR (higher is better; typically above 30 is considered excellent): {sum(psnr_scores)/len(psnr_scores):.2f} dB")
    print(f"2. SSIM (closer to 1 indicates higher structural similarity):         {sum(ssim_scores)/len(ssim_scores):.4f}")
    print(f"3. LPIPS (closer to 0 means perceptually more similar to the human eye):  {sum(lpips_scores)/len(lpips_scores):.4f}")
    print("===========================================")

if __name__ == "__main__":
    main()

