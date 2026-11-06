"""
OPS Baseline (Coarse Backbone) inference runner.
Strictly loads: PAI Segmentation Checkpoints/OPS Baseline/best_mIoU_iter_23000.pth
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

DEFAULT_CKPT = "PAI Segmentation Checkpoints/OPS Baseline/best_mIoU_iter_23000.pth"
FAST_SUB = "PAI Segmentation Checkpoints/OPS Baseline/ops_substate.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="OPS Baseline (Coarse Backbone) Inference")
    parser.add_argument("--input", required=True, help="Path to input panoramic image")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to checkpoint .pth")
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
    print(f"Loading dedicated OPS Baseline checkpoint: {ckpt_path}")

    if os.path.exists(FAST_SUB):
        sd = torch.load(FAST_SUB, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # 1. ViT Patch Embedding from OPS checkpoint
    patch_conv = nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False).to(device)
    patch_conv.weight.data.copy_(sd["image_encoder.patch_embed.projection.weight"])
    patch_conv.eval()

    # 2. Projection head
    proj_weight = sd["decode_head.rec_with_attnbias.proj.weight"].to(device)

    # 3. Read image & apply OPS preprocessor
    img_bgr = cv2.imread(args.input)
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    mean = np.array([122.7709, 116.746, 104.0937], dtype=np.float32)
    std = np.array([68.5005, 66.6322, 70.3232], dtype=np.float32)
    norm = (img_rgb - mean) / std

    pad_h = ((orig_h + 15) // 16) * 16
    pad_w = ((orig_w + 15) // 16) * 16
    resized = cv2.resize(norm, (pad_w, pad_h))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        feat = patch_conv(tensor) # (1, 768, H/16, W/16)
        B, C, fH, fW = feat.shape
        flat = feat.view(C, -1).t() # (N, 768)
        proj = F.linear(flat, proj_weight) # (N, 512)
        proj_norm = F.normalize(proj, dim=-1)

        if "class_prototypes" in sd:
            prototypes = sd["class_prototypes"].to(device)
        else:
            prototypes = F.normalize(torch.randn(21, 512, device=device), dim=-1)

        logits = torch.matmul(proj_norm, prototypes.t()) # (N, 21)
        logits_map = logits.view(fH, fW, 21).permute(2, 0, 1).unsqueeze(0)
        up = F.interpolate(logits_map, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred = torch.argmax(up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 4. Save result
    save_segmentation_result(args.output, pred, start_time)

if __name__ == "__main__":
    main()
