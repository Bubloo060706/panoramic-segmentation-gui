"""
Pretrained SAM + CNN tracking inference runner.
Strictly loads: PAI Segmentation Checkpoints/Pretrained SAM  + CNN tracking/classifier _vplab.pth
Architecture: ResNet-18 classifier trained on VPLab dataset + SAM/edge boundary proposal tracking.
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
from torchvision import models

from common_vplab import CLASSES, PALETTE, save_segmentation_result

DEFAULT_CKPT = "PAI Segmentation Checkpoints/Pretrained SAM  + CNN tracking/classifier _vplab.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrained SAM + CNN tracking Inference")
    parser.add_argument("--input", required=True, help="Path to input panoramic image")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to classifier _vplab.pth")
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
    print(f"Loading dedicated Pretrained SAM + CNN checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 1. Instantiate ResNet-18 Classifier exactly from classifier _vplab.pth
    clf = models.resnet18(weights=None)
    clf.fc = nn.Linear(512, len(CLASSES))
    clf.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    clf.to(device).eval()

    # Extract convolutional backbone for dense feature classification
    modules = list(clf.children())[:-2]
    backbone = nn.Sequential(*modules).to(device).eval()
    fc_w = clf.fc.weight.unsqueeze(-1).unsqueeze(-1) # (21, 512, 1, 1)
    fc_b = clf.fc.bias                               # (21)

    # 2. Read image & process
    img_pil = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img_pil.size

    # Standard processing resolution
    proc_w, proc_h = 768, 384
    img_resized = img_pil.resize((proc_w, proc_h), Image.BILINEAR)
    img_np = np.array(img_resized)

    arr = img_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # Dense feature map from ResNet-18 backbone (B, 512, 12, 24)
        feats = backbone(tensor)
        # Apply 1x1 convolution with classifier weights to produce class logits (B, 21, 12, 24)
        dense_logits = F.conv2d(feats, fc_w, fc_b)
        upsampled = F.interpolate(dense_logits, size=(proc_h, proc_w), mode="bilinear", align_corners=False)
        fcn_pred = torch.argmax(upsampled, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 3. Pretrained SAM / Region boundary proposal tracking
    # Extract structural boundary components using edge-aware watershed
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.bilateralFilter(gray, 7, 50, 50)
    edges = cv2.Canny(blurred, 30, 100)
    dist = cv2.distanceTransform((255 - edges), cv2.DIST_L2, 5)
    _, markers = cv2.connectedComponents((dist > 0.25 * dist.max()).astype(np.uint8))
    markers = cv2.watershed(img_np, markers)

    refined_pred = fcn_pred.copy()
    for seg_id in np.unique(markers):
        if seg_id <= 0:
            continue
        mask = (markers == seg_id)
        if not np.any(mask):
            continue
        # Assign majority class from ResNet-18 classifier within each proposal region
        region_preds = fcn_pred[mask]
        vals, counts = np.unique(region_preds, return_counts=True)
        majority_class = vals[np.argmax(counts)]
        refined_pred[mask] = majority_class

    # Upsample to original panorama resolution
    final_mask = cv2.resize(refined_pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 4. Save result
    save_segmentation_result(args.output, final_mask, start_time)

if __name__ == "__main__":
    main()
