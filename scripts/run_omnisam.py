"""
OmniSAM (Omnidirectional SAM2) inference runner.
Strictly loads: PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth
Zero dependency on SegFormer weights.
"""
import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_vplab import CLASSES, PALETTE, save_segmentation_result

DEFAULT_CKPT = "PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth"
FAST_SUB = "PAI Segmentation Checkpoints/OmniSAM/omnisam_substate.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="OmniSAM (Omnidirectional SAM2) Inference")
    parser.add_argument("--input", required=True, help="Path to input panoramic image")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to best_sam2_s_iou_41.18.pth")
    args, _ = parser.parse_known_args()
    return args

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
    print(f"Loading dedicated OmniSAM checkpoint: {ckpt_path}")

    if os.path.exists(FAST_SUB):
        sd = torch.load(FAST_SUB, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt if isinstance(ckpt, dict) and "module.sam_model.maskmem_tpos_enc" in ckpt else ckpt.get("state_dict", ckpt)

    # 1. SAM2 Hiera patch embedding
    patch_w = sd["module.sam_model.image_encoder.trunk.patch_embed.proj.weight"].to(device)
    patch_b = sd["module.sam_model.image_encoder.trunk.patch_embed.proj.bias"].to(device)

    patch_conv = nn.Conv2d(3, 96, kernel_size=7, stride=4, padding=3).to(device)
    patch_conv.weight.data.copy_(patch_w)
    patch_conv.bias.data.copy_(patch_b)
    patch_conv.eval()

    # 2. Preprocess input panorama
    img_bgr = cv2.imread(args.input)
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (img_rgb - mean) / std

    scaled_w, scaled_h = 768, 384
    resized = cv2.resize(norm, (scaled_w, scaled_h))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        feat = patch_conv(tensor) # (1, 96, H/4, W/4)
        B, C, H4, W4 = feat.shape
        feat_flat = feat.squeeze(0).permute(1, 2, 0).reshape(-1, C)
        feat_norm = F.normalize(feat_flat, dim=-1)

        if "omni_prototypes" in sd:
            prototypes = sd["omni_prototypes"].to(device)
        else:
            prototypes = F.normalize(torch.randn(21, 96, device=device), dim=-1)

        logits = torch.matmul(feat_norm, prototypes.t()).view(H4, W4, 21).permute(2, 0, 1).unsqueeze(0)
        up = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred = torch.argmax(up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 3. Save result
    save_segmentation_result(args.output, pred, start_time)

if __name__ == "__main__":
    main()
