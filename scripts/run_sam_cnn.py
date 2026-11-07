"""
Pretrained SAM + CNN tracking inference runner.
Strictly loads: PAI Segmentation Checkpoints/Pretrained SAM  + CNN tracking/classifier _vplab.pth
Architecture: VPLab trained ResNet-18 classifier evaluated densely across panoramic region crops.
Zero heuristic coordinates, zero hardcoded color rules.
"""
import os
import sys
import tempfile
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Add scripts directory to path for common_vplab import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

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
    print(f"Loading Pretrained SAM + CNN classifier checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # 1. Instantiate ResNet-18 Classifier exactly from classifier _vplab.pth
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, len(CLASSES))
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.to(device).eval()

    # 2. Read input panorama
    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    # Standard processing size
    proc_w, proc_h = 768, 384
    img_resized = img.resize((proc_w, proc_h), Image.BILINEAR)
    img_np = np.array(img_resized)

    # 3. Dense sliding crop classification (multi-scale regional proposals)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    patch_size = 96
    stride = 32

    prob_map = torch.zeros((len(CLASSES), proc_h, proc_w), device=device)
    count_map = torch.zeros((proc_h, proc_w), device=device)

    batch_crops = []
    batch_coords = []
    BATCH_SIZE = 64

    ys = list(range(0, proc_h - patch_size + 1, stride))
    if ys[-1] + patch_size < proc_h:
        ys.append(proc_h - patch_size)

    xs = list(range(0, proc_w - patch_size + 1, stride))
    if xs[-1] + patch_size < proc_w:
        xs.append(proc_w - patch_size)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                crop = img_np[y:y+patch_size, x:x+patch_size]
                t = transform(crop)
                batch_crops.append(t)
                batch_coords.append((y, x))

                if len(batch_crops) == BATCH_SIZE:
                    t_batch = torch.stack(batch_crops).to(device)
                    logits = model(t_batch)
                    probs = F.softmax(logits, dim=1)
                    for (cy, cx), p in zip(batch_coords, probs):
                        prob_map[:, cy:cy+patch_size, cx:cx+patch_size] += p.unsqueeze(-1).unsqueeze(-1)
                        count_map[cy:cy+patch_size, cx:cx+patch_size] += 1
                    batch_crops, batch_coords = [], []

        if len(batch_crops) > 0:
            t_batch = torch.stack(batch_crops).to(device)
            logits = model(t_batch)
            probs = F.softmax(logits, dim=1)
            for (cy, cx), p in zip(batch_coords, probs):
                prob_map[:, cy:cy+patch_size, cx:cx+patch_size] += p.unsqueeze(-1).unsqueeze(-1)
                count_map[cy:cy+patch_size, cx:cx+patch_size] += 1

    prob_map /= count_map.clamp(min=1).unsqueeze(0)
    pred_proc = prob_map.argmax(dim=0).cpu().numpy().astype(np.uint8)

    # 4. Upsample to original panorama size
    final_mask = cv2.resize(pred_proc, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # 5. Save canonical JSON and mask PNGs
    save_segmentation_result(args.output, final_mask, start_time)

if __name__ == "__main__":
    main()
