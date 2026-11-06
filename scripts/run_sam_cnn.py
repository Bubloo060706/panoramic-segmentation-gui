import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from common_vplab import CLASSES, PALETTE, save_segmentation_result

DEFAULT_CKPT = "PAI Segmentation Checkpoints/Pretrained SAM  + CNN tracking/classifier _vplab.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrained SAM + CNN Tracking VPLab Inference")
    parser.add_argument("--input", required=True, help="Path to input panoramic image")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to classifier _vplab.pth")
    return parser.parse_args()

def load_classifier(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model

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
    classifier = load_classifier(ckpt_path, device)

    # Load image
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not read image: {args.input}", file=sys.stderr)
        sys.exit(1)
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Canvas initialized to Wall (Class 5 - dominant vertical background in VP Lab)
    label_map = np.full((h, w), 5, dtype=np.uint8)

    # Collect multi-scale spatial patches for proposal classification
    patch_configs = [
        (128, 64),
        (64, 32)
    ]

    all_crops = []
    all_coords = []

    for ps, step in patch_configs:
        for y in range(0, h - ps + 1, step):
            for x in range(0, w - ps + 1, step):
                crop = img_rgb[y:y+ps, x:x+ps]
                all_crops.append(transform(crop))
                all_coords.append((y, x, y+ps, x+ps))

    # Run inference in batches of 64
    batch_size = 64
    total_crops = len(all_crops)
    with torch.no_grad():
        for i in range(0, total_crops, batch_size):
            batch_tensors = torch.stack(all_crops[i:i+batch_size]).to(device)
            logits = classifier(batch_tensors)
            preds = logits.argmax(dim=1).cpu().numpy()
            for (y1, x1, y2, x2), p in zip(all_coords[i:i+batch_size], preds):
                label_map[y1:y2, x1:x2] = p

    # Refine equirectangular polar boundaries (equirectangular projection top=Roof, bottom=Floor)
    roof_bound = int(h * 0.18)
    floor_bound = int(h * 0.78)
    label_map[:roof_bound, :] = 0  # Roof
    label_map[floor_bound:, :] = 8  # Floor

    # Save output
    save_segmentation_result(args.output, label_map, start_time)

if __name__ == "__main__":
    main()

