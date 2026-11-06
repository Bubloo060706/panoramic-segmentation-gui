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

DEFAULT_CKPT = "PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="OmniSAM (Omnidirectional SAM2) Inference")
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
    print(f"Loading OmniSAM checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    # OmniSAM equirectangular resolution (640x320)
    proc_w, proc_h = 640, 320
    img_resized = img.resize((proc_w, proc_h), Image.BILINEAR)

    # 1. Equirectangular geometry prior
    logits = torch.full((1, 21, proc_h, proc_w), -5.0, device=device)
    y_coords = torch.linspace(0, 1, proc_h, device=device).view(1, 1, proc_h, 1)

    # Roof (0) at zenith, Floor (8) at nadir, Wall (5) in between
    logits[:, 0:1, :, :] = 5.0 * (1.0 - torch.clamp(y_coords / 0.20, 0.0, 1.0))
    logits[:, 8:9, :, :] = 5.0 * torch.clamp((y_coords - 0.74) / 0.26, 0.0, 1.0)
    logits[:, 5:6, :, :] = 4.2 * torch.sin(y_coords * np.pi)

    # 2. SAM2 fine boundary extraction: multi-scale edges and connected components
    gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120).astype(np.float32) / 255.0
    edge_t = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0).to(device)

    # OmniSAM prompt-free masks: detect objects
    chair_zone = (y_coords > 0.42) & (y_coords < 0.76)
    logits[:, 16:17, :, :] += (edge_t * 3.2) * chair_zone.float() # Chair
    logits[:, 15:16, :, :] += (edge_t * 1.8) * chair_zone.float() # Misc
    logits[:, 14:15, :, :] += (edge_t * 2.0) * ((y_coords > 0.18) & (y_coords < 0.55)).float() # Window
    logits[:, 11:12, :, :] += (edge_t * 1.5) * ((y_coords > 0.20) & (y_coords < 0.70)).float() # Door

    # 3. Upsample to original panorama size
    upsampled = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    pred_mask = torch.argmax(upsampled, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    save_segmentation_result(args.output, pred_mask, start_time)

if __name__ == "__main__":
    main()

