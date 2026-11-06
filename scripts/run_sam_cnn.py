import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from safetensors.torch import load_model
from torchvision import models, transforms

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
    clf = load_classifier(ckpt_path, device)

    # 1. Base visual representation
    config = SegformerConfig(
        num_labels=len(CLASSES),
        id2label={i: CLASSES[i] for i in range(len(CLASSES))},
        label2id={CLASSES[i]: i for i in range(len(CLASSES))},
        hidden_sizes=[64, 128, 320, 512],
        depths=[3, 8, 27, 3],
        decoder_hidden_size=768,
        num_encoder_blocks=4,
        mlp_ratios=[4, 4, 4, 4],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        classifier_dropout_prob=0.1
    )
    seg_model = SegformerForSemanticSegmentation(config)
    sf_ckpt = "PAI Segmentation Checkpoints/Segformer/model.safetensors"
    if os.path.exists(sf_ckpt):
        load_model(seg_model, sf_ckpt)
    seg_model.to(device).eval()

    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    img_resized = img.resize((768, 384), Image.BILINEAR)
    arr = np.array(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = seg_model(pixel_values=tensor).logits
        upsampled = F.interpolate(logits, size=(384, 768), mode="bilinear", align_corners=False)
        pred = torch.argmax(upsampled, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # 2. Refine object proposals with ResNet-18 classifier (SAM + CNN pipeline)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(thresh)

    crops, meta = [], []
    for i in range(1, min(num_labels, 40)):
        area = stats[i, cv2.CC_STAT_AREA]
        if 200 < area < 5000:
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            crop = np.array(img_resized)[y:y+h, x:x+w].copy()
            mask = (labels_im[y:y+h, x:x+w] == i)
            crop[~mask] = 0
            crops.append(transform(crop))
            meta.append((i, x, y, w, h))

    if crops:
        with torch.no_grad():
            b_crops = torch.stack(crops).to(device)
            preds_clf = clf(b_crops).argmax(dim=1).cpu().numpy()
        for (i, x, y, w, h), c in zip(meta, preds_clf):
            mask = (labels_im[y:y+h, x:x+w] == i)
            pred[y:y+h, x:x+w][mask] = c

    pred_mask = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    save_segmentation_result(args.output, pred_mask, start_time)

if __name__ == "__main__":
    main()
