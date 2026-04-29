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

    # LPIPS 모델 로드 (VGG 네트워크 기반)
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()
    
    image_files = sorted(os.listdir(baseline_dir))
    
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    print("--- Evaluating Visual Fidelity (Baseline vs Compressed) ---")
    for img_name in tqdm(image_files):
        if not img_name.endswith(".png"): continue
            
        # 이미지 로드 및 텐서 변환 (0~1 범위, [1, C, H, W] 형태)
        base_img = TF.to_tensor(Image.open(os.path.join(baseline_dir, img_name))).unsqueeze(0).cuda()
        comp_img = TF.to_tensor(Image.open(os.path.join(compressed_dir, img_name))).unsqueeze(0).cuda()

        # 1. PSNR (클수록 좋음)
        psnr_val = calculate_psnr(base_img, comp_img).item()
        psnr_scores.append(psnr_val)

        # 2. SSIM (1에 가까울수록 좋음)
        ssim_val = ssim(base_img, comp_img).item()
        ssim_scores.append(ssim_val)

        # 3. LPIPS (0에 가까울수록 좋음, lpips 모델은 -1~1 입력을 기대하므로 스케일링)
        base_img_lpips = base_img * 2.0 - 1.0
        comp_img_lpips = comp_img * 2.0 - 1.0
        lpips_val = loss_fn_vgg(base_img_lpips, comp_img_lpips).item()
        lpips_scores.append(lpips_val)

    print("\n========= [ 최종 화질 평가 결과 ] =========")
    print(f"1. PSNR (클수록 좋음, 보통 30 이상이면 우수): {sum(psnr_scores)/len(psnr_scores):.2f} dB")
    print(f"2. SSIM (1에 가까울수록 구조 일치):         {sum(ssim_scores)/len(ssim_scores):.4f}")
    print(f"3. LPIPS (0에 가까울수록 사람 눈에 똑같음):  {sum(lpips_scores)/len(lpips_scores):.4f}")
    print("===========================================")
    print("결론: 위 수치는 양자화(INT8)로 인한 화질 '손실량'을 나타냅니다.")

if __name__ == "__main__":
    main()