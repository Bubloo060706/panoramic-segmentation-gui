"""
Pretrained SAM + CNN tracking inference runner.
Strictly loads: PAI Segmentation Checkpoints/Pretrained SAM  + CNN tracking/classifier _vplab.pth
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

    # 2. Build Fully Convolutional Backbone for dense panoramic inference
    layers = [
        clf.conv1, clf.bn1, clf.relu, clf.maxpool,
        clf.layer1, clf.layer2, clf.layer3, clf.layer4
    ]
    backbone = nn.Sequential(*layers).to(device).eval()

    conv_fc = nn.Conv2d(512, len(CLASSES), kernel_size=1).to(device)
    conv_fc.weight.data.copy_(clf.fc.weight.data.unsqueeze(-1).unsqueeze(-1))
    conv_fc.bias.data.copy_(clf.fc.bias.data)
    conv_fc.eval()

    # 3. Read image & multi-scale dense classification
    img_bgr = cv2.imread(args.input)
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    total_logits = torch.zeros(1, len(CLASSES), orig_h, orig_w, device=device)
    with torch.no_grad():
        for scale in [0.75, 1.0, 1.25]:
            sw, sh = max(32, int(orig_w * scale)), max(32, int(orig_h * scale))
            scaled_img = cv2.resize(img_rgb, (sw, sh))
            norm = (scaled_img - mean) / std
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            feat = backbone(tensor)
            l = conv_fc(feat)
            up = F.interpolate(l, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            total_logits += up

    pred = torch.argmax(total_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 4. Save result
    save_segmentation_result(args.output, pred, start_time)

if __name__ == "__main__":
    main()
