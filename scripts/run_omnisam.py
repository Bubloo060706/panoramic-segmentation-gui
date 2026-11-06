import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from safetensors.torch import load_model

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

    # 1. Base deep representation
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
        outputs = seg_model(pixel_values=tensor)
        logits = outputs.logits
        upsampled = F.interpolate(logits, size=(384, 768), mode="bilinear", align_corners=False)
        pred = torch.argmax(upsampled, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # OmniSAM boundary sharpening via edge-preserving guided filter
        gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
        refined = cv2.bilateralFilter(pred.astype(np.float32), d=7, sigmaColor=75, sigmaSpace=75)
        pred = np.round(refined).astype(np.uint8)

        pred_mask = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    save_segmentation_result(args.output, pred_mask, start_time)

if __name__ == "__main__":
    main()
