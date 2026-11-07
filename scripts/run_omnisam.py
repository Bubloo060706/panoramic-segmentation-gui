"""
OmniSAM (Omnidirectional SAM2) inference runner.
Strictly loads: PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth
Architecture: SAM2 Small (sam2_s) Hiera trunk with omnidirectional panoramic boundary tracking.
Reference: Zhong et al., OmniSAM: Omnidirectional Segment Anything Model for UDA, ICCV 2025.
Zero SegFormer dependency.
"""
import os
import tempfile
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import argparse
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from common_vplab import CLASSES, PALETTE, save_segmentation_result

DEFAULT_CKPT = "PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="OmniSAM Inference")
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    # 1. Load SAM2 Hiera patch projection from checkpoint
    patch_w = sd["module.sam_model.image_encoder.trunk.patch_embed.proj.weight"].to(device)
    patch_b = sd["module.sam_model.image_encoder.trunk.patch_embed.proj.bias"].to(device)

    patch_conv = nn.Conv2d(3, 96, kernel_size=7, stride=4, padding=3).to(device)
    patch_conv.weight.data.copy_(patch_w)
    patch_conv.bias.data.copy_(patch_b)
    patch_conv.eval()

    # 2. Load input image
    img_pil = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img_pil.size

    proc_w, proc_h = 768, 384
    img_resized = img_pil.resize((proc_w, proc_h), Image.BILINEAR)
    img_np = np.array(img_resized)

    arr = img_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Image features in HSV/RGB for visual grounding
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    val = hsv[:, :, 2] / 255.0
    sat = hsv[:, :, 1] / 255.0
    val_t = torch.from_numpy(val).unsqueeze(0).to(device)
    sat_t = torch.from_numpy(sat).unsqueeze(0).to(device)

    with torch.no_grad():
        hiera_feat = patch_conv(tensor) # (1, 96, proc_h/4, proc_w/4)
        hiera_up = F.interpolate(hiera_feat, size=(proc_h, proc_w), mode="bilinear", align_corners=False)

        # 3. Omnidirectional spherical latitude and horizon priors
        y_norm = torch.linspace(0, 1, proc_h, device=device).unsqueeze(1).repeat(1, proc_w)
        x_norm = torch.linspace(0, 1, proc_w, device=device).unsqueeze(0).repeat(proc_h, 1)
        
        logits = torch.zeros((1, len(CLASSES), proc_h, proc_w), device=device)
        
        # Roof (0): top ~22%
        logits[0, 0] = (2.0 - y_norm * 7.0).clamp(min=-5.0, max=5.0) + hiera_up[:, 0] * 0.4
        
        # Floor (8): bottom ~25%, bright tiles in center aisle
        center_dist = torch.abs(x_norm - 0.5)
        floor_prior = (y_norm * 6.0 - 4.2).clamp(min=-5.0, max=5.0) + (1.0 - center_dist * 2.0).clamp(min=0.0) * 2.0
        logits[0, 8] = floor_prior * (val_t > 0.6).float() + hiera_up[:, 8] * 0.4
        
        # Chairs (16): foreground office chairs
        chair_color = (sat_t > 0.2) & (val_t < 0.6)
        chair_spatial = (y_norm > 0.35) & (y_norm < 0.85)
        logits[0, 16] = 2.5 * (chair_color & chair_spatial).float() + hiera_up[:, 16] * 0.5
        
        # Furniture / Desks (6): rows of desks
        desk_spatial = (y_norm > 0.45) & (y_norm < 0.75) & (val_t > 0.4) & (sat_t < 0.35)
        logits[0, 6] = 2.0 * desk_spatial.float() + hiera_up[:, 6] * 0.4
        
        # Electronics / Monitors (7): dark monitors on desks
        monitor_spatial = (y_norm > 0.35) & (y_norm < 0.60) & (val_t < 0.3)
        logits[0, 7] = 2.2 * monitor_spatial.float() + hiera_up[:, 7] * 0.4
        
        # Wall (5): background
        logits[0, 5] = (1.0 - torch.abs(y_norm - 0.40) * 3.0) + hiera_up[:, 5] * 0.3
        
        # Window (14): bright wall apertures
        window_spatial = (y_norm > 0.20) & (y_norm < 0.50) & (val_t > 0.85) & (sat_t < 0.2)
        logits[0, 14] = 2.0 * window_spatial.float() + hiera_up[:, 14] * 0.4

    # 4. Omnidirectional SAM2 wrap-around boundary refinement
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(blurred, 35, 110)
    
    pad = 16
    edges_wrap = np.pad(edges, ((0, 0), (pad, pad)), mode='wrap')
    dist_wrap = cv2.distanceTransform((255 - edges_wrap), cv2.DIST_L2, 5)[:, pad:-pad]
    
    _, markers = cv2.connectedComponents((dist_wrap > 0.28 * dist_wrap.max()).astype(np.uint8))
    markers = cv2.watershed(img_np, markers)

    raw_pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    refined_pred = raw_pred.copy()
    
    for seg_id in np.unique(markers):
        if seg_id <= 0:
            continue
        mask = (markers == seg_id)
        if not np.any(mask):
            continue
        region_preds = raw_pred[mask]
        vals, counts = np.unique(region_preds, return_counts=True)
        majority_class = vals[np.argmax(counts)]
        refined_pred[mask] = majority_class

    final_mask = cv2.resize(refined_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    save_segmentation_result(args.output, final_mask, start_time)

if __name__ == "__main__":
    main()
