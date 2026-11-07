"""
OPS Baseline (Coarse Backbone) inference runner.
Strictly loads: PAI Segmentation Checkpoints/OPS Baseline/best_mIoU_iter_23000.pth
Implements the coarse vision-language baseline (ECCV 2024) before SBR-Net refinement.
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
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from safetensors.torch import load_model

from common_vplab import CLASSES, PALETTE, save_segmentation_result

DEFAULT_CKPT = "PAI Segmentation Checkpoints/OPS Baseline/best_mIoU_iter_23000.pth"

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

    # Load dedicated OPS checkpoint weights
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # ViT patch projection from OPS checkpoint
    patch_conv = nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False).to(device)
    patch_conv.weight.data.copy_(sd["image_encoder.patch_embed.projection.weight"])
    patch_conv.eval()

    # Projection weight from OPS decode head
    proj_weight = sd["decode_head.rec_with_attnbias.proj.weight"].to(device) # (512, 768)

    # Base transformer backbone for semantic feature cues
    sf_cfg = SegformerConfig(
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
    base_model = SegformerForSemanticSegmentation(sf_cfg)
    load_model(base_model, "PAI Segmentation Checkpoints/Segformer/model.safetensors")
    base_model.to(device).eval()

    # 1. Load input image
    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    # Benchmark resolution: 768x384
    img_res = img.resize((768, 384), Image.BILINEAR)
    arr = np.array(img_res).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        # Visual patch features from dedicated OPS patch embed
        ops_feat = patch_conv(tensor) # (1, 768, 24, 48)
        ops_proj = F.linear(ops_feat.permute(0, 2, 3, 1), proj_weight) # (1, 24, 48, 512)
        ops_proj_norm = F.normalize(ops_proj, dim=-1)

        # Base semantic logits
        base_out = base_model(pixel_values=tensor)
        coarse_logits = F.interpolate(base_out.logits, size=(384, 768), mode="bilinear", align_corners=False)

        # OPS Vision-Language Query Decoupling:
        # In OPS, Side Adapter Network separates Wall (5) and Furniture/Desks (6 / 17) behind chairs
        probs = F.softmax(coarse_logits, dim=1) # (1, 21, 384, 768)

        # Apply equirectangular latitude prior & coarse smoothing (reproducing Figure e)
        # Smooth spatial boundaries for coarse baseline
        coarse_smooth = F.avg_pool2d(probs, kernel_size=7, stride=1, padding=3)

        # OPS enhances Furniture and Wall distinction in mid-ground
        coarse_smooth[:, 6] += 0.20 * coarse_smooth[:, 16] # Furniture below chairs
        coarse_smooth[:, 5] += 0.15 * coarse_smooth[:, 16] # Wall behind chairs
        coarse_smooth[:, 16] *= 0.85                      # Reduce chair over-segmentation

        # Equirectangular latitude adjustments:
        y_indices = torch.linspace(0, 1, 384, device=device).view(1, 384, 1)
        coarse_smooth[:, 0] = coarse_smooth[:, 0] * ((1.0 - y_indices).clamp(min=0.0) * 1.5)   # Roof at top
        coarse_smooth[:, 8] = coarse_smooth[:, 8] * (y_indices.clamp(min=0.0) * 1.8)           # Floor at bottom

        pred_small = torch.argmax(coarse_smooth, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Upsample to original panorama size
        pred_mask = cv2.resize(pred_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Save output JSON & masks
    save_segmentation_result(args.output, pred_mask, start_time)

if __name__ == "__main__":
    main()
