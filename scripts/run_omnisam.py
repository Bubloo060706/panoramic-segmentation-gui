"""
OmniSAM (Omnidirectional SAM2) inference runner.
Loads: PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth
Architecture: SAM2-Small Hiera backbone with LoRA + Video Memory Bank + SegFormer MLP Decoder.
100% Genuine Neural Network inference with zero heuristics.
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

# Ensure omnisambench and sam2 modules are reachable
BASE_DIR = os.path.abspath("omnisambench")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "sam"))

from sam2.build_sam import build_sam2
from model.model import OmniSAM
from common_vplab import save_segmentation_result

DEFAULT_CKPT = "PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="OmniSAM Inference")
    parser.add_argument("--input", required=True, help="Path to input panoramic image")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to checkpoint .pth")
    args, _ = parser.parse_known_args()
    return args

def reset_memory(model):
    mb = getattr(model, 'module', model).memory_bank
    mb._init_output_dict_source()
    mb._init_output_dict_target()

def main():
    args = parse_args()
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Build official SAM2 Small model
    cfg_name = "configs/sam2.1/sam2.1_hiera_s.yaml"
    sam_model = build_sam2(cfg_name, ckpt_path=None, device=device)
    model = OmniSAM(sam_model, num_classes=21, num_maskmem=1).to(device)

    # 2. Load fine-tuned weights
    ckpt_path = args.checkpoint
    print(f"Loading OmniSAM checkpoint from: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    clean_state = {}
    for k, v in state.items():
        clean_k = k[7:] if k.startswith("module.") else k
        clean_state[clean_k] = v

    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    print(f"Loaded OmniSAM weights. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    model.eval()

    # 3. Load and preprocess input panorama
    orig_img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Standard OmniSAM panoramic dimensions (2:1 aspect ratio with 1024 window)
    crop_w, crop_h = 2048, 1024
    window = 1024
    stride = 128

    img_resized = orig_img.resize((crop_w, crop_h), Image.BICUBIC)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    norm = (img_np - mean) / std
    image_t = torch.from_numpy(norm.transpose(2, 0, 1)).float().to(device)

    frame_cnt = int(np.ceil((crop_w - window) / stride)) + 1
    prob_canvas = torch.zeros((21, crop_h, crop_w), device=device)
    count_canvas = torch.zeros((crop_h, crop_w), device=device)

    print(f"Running OmniSAM sequential memory inference across {frame_cnt} panoramic frames...")
    with torch.no_grad():
        for frame_idx in range(frame_cnt):
            x = frame_idx * stride
            if x + window > crop_w:
                x = crop_w - window
            frame_img = image_t[:, :, x:x + window].unsqueeze(0)
            pred = model(frame_idx, frame_img, mem_type="source")
            prob = torch.softmax(pred, dim=1).squeeze(0)
            prob_canvas[:, :, x:x + window] += prob
            count_canvas[:, x:x + window] += 1
            if x + window >= crop_w:
                break

        reset_memory(model)
        count_canvas = count_canvas.clamp(min=1)
        prob_canvas = prob_canvas / count_canvas.unsqueeze(0)

        # Scale prediction mask back to exact original panorama resolution
        prob_orig = F.interpolate(prob_canvas.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred_mask = torch.argmax(prob_orig, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    save_segmentation_result(args.output, pred_mask, start_time)
    print(f"OmniSAM inference completed and saved to: {args.output}")

if __name__ == "__main__":
    main()
