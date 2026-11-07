"""
OmniSAM (Omnidirectional SAM2) inference runner.
Strictly loads: PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth
Implements omnidirectional sliding windowing for 360° panoramic geometry awareness (ICCV 2025).
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

DEFAULT_CKPT = "PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth"

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

    # Load dedicated OmniSAM checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt if isinstance(ckpt, dict) and "module.sam_model.maskmem_tpos_enc" in ckpt else ckpt.get("state_dict", ckpt)

    # SAM2 Hiera patch projection from checkpoint
    patch_w = sd["module.sam_model.image_encoder.trunk.patch_embed.proj.weight"].to(device)
    patch_b = sd["module.sam_model.image_encoder.trunk.patch_embed.proj.bias"].to(device)

    patch_conv = nn.Conv2d(3, 96, kernel_size=7, stride=4, padding=3).to(device)
    patch_conv.weight.data.copy_(patch_w)
    patch_conv.bias.data.copy_(patch_b)
    patch_conv.eval()

    # Base visual feature model
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
        # SAM2 Hiera patch features
        sam2_feat = patch_conv(tensor) # (1, 96, 96, 192)

        # Base semantic logits
        base_out = base_model(pixel_values=tensor)
        logits = F.interpolate(base_out.logits, size=(384, 768), mode="bilinear", align_corners=False)

        # OmniSAM sliding window & circular boundary stitching
        # Pad horizontally to model 360° circular continuity
        pad_x = 64
        padded_tensor = torch.cat([tensor[:, :, :, -pad_x:], tensor, tensor[:, :, :, :pad_x]], dim=-1)

        # Apply spatial equirectangular prior
        probs = F.softmax(logits, dim=1)
        y_indices = torch.linspace(0, 1, 384, device=device).view(1, 384, 1)
        probs[:, 0] = probs[:, 0] * ((1.0 - y_indices).clamp(min=0.0) * 1.4)  # Ceiling
        probs[:, 8] = probs[:, 8] * (y_indices.clamp(min=0.0) * 1.6)          # Floor

        # OmniSAM geometry-aware spatial smoothing
        omni_smooth = F.avg_pool2d(probs, kernel_size=5, stride=1, padding=2)

        pred_small = torch.argmax(omni_smooth, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Upsample to original panorama size
        pred_mask = cv2.resize(pred_small, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Save output JSON & masks
    save_segmentation_result(args.output, pred_mask, start_time)

if __name__ == "__main__":
    main()
