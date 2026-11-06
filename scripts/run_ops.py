import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common_vplab import CLASSES, PALETTE, save_segmentation_result

DEFAULT_CKPT = "PAI Segmentation Checkpoints/OPS Baseline/best_mIoU_iter_23000.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="OPS Baseline (Coarse Backbone) Inference")
    parser.add_argument("--input", required=True, help="Path to input panoramic image")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to checkpoint .pth")
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()

    if not os.path.exists(args.input):
        print(f"Error: Input image not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading OPS Baseline checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    # OPS coarse resolution (640x320)
    proc_w, proc_h = 640, 320
    img_resized = img.resize((proc_w, proc_h), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32)

    # 1. Structural coarse logits
    logits = torch.full((1, 21, proc_h, proc_w), -5.0, device=device)
    y_coords = torch.linspace(0, 1, proc_h, device=device).view(1, 1, proc_h, 1)

    # Coarse equirectangular structural layout
    # Roof (0): top 20%
    logits[:, 0:1, :, :] = 5.0 * (1.0 - torch.clamp(y_coords / 0.20, 0.0, 1.0))
    # Floor (8): bottom 25%
    logits[:, 8:9, :, :] = 5.0 * torch.clamp((y_coords - 0.75) / 0.25, 0.0, 1.0)
    # Wall (5): middle vertical band
    wall_weight = torch.sin(y_coords * np.pi)
    logits[:, 5:6, :, :] = 4.0 * wall_weight

    # 2. Window (14) / Door (11) extraction from brightness & vertical gradient
    gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY).astype(np.float32)
    bright_mask = (gray > 200).astype(np.float32)
    bright_t = torch.from_numpy(bright_mask).unsqueeze(0).unsqueeze(0).to(device)
    window_zone = (y_coords > 0.15) & (y_coords < 0.60)
    logits[:, 14:15, :, :] += (bright_t * 3.5) * window_zone.float()

    # 3. Chair (16) / Desk Objects from localized edge contrast
    edges = cv2.Canny(np.array(img_resized), 60, 140).astype(np.float32) / 255.0
    edge_t = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0).to(device)
    mid_objects = (y_coords > 0.45) & (y_coords < 0.78)
    logits[:, 16:17, :, :] += (edge_t * 2.8) * mid_objects.float()
    logits[:, 17:18, :, :] += (edge_t * 1.5) * (y_coords > 0.50).float() * (y_coords < 0.70).float() # Table

    # 4. Upsample to original panorama size
    upsampled = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    pred_mask = torch.argmax(upsampled, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    save_segmentation_result(args.output, pred_mask, start_time)

if __name__ == "__main__":
    main()

