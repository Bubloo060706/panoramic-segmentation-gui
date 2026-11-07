"""
Pretrained SAM + CNN tracking inference runner.
Strictly loads: PAI Segmentation Checkpoints/Pretrained SAM  + CNN tracking/classifier _vplab.pth
Implements proposal-level classification using the trained ResNet-18 classifier from the official pipeline.
Sorts proposals large-to-small so detailed object instances sit cleanly on top of backgrounds.
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
from torchvision import models, transforms
from PIL import Image
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from safetensors.torch import load_model

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

    # 2. Base proposal generator for class-agnostic regions
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

    # 3. Read image & process at standard benchmark resolution (768x384)
    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    proc_w, proc_h = 768, 384
    img_resized = img.resize((proc_w, proc_h), Image.BILINEAR)
    img_np = np.array(img_resized)

    arr = img_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Classifier transform
    clf_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        # Generate initial coarse proposal map
        base_out = base_model(pixel_values=tensor)
        logits = F.interpolate(base_out.logits, size=(proc_h, proc_w), mode="bilinear", align_corners=False)
        base_pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Extract connected component proposals
        proposals = []
        for class_id in np.unique(base_pred):
            binary = (base_pred == class_id).astype(np.uint8)
            num_cc, labels_cc, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            for c_id in range(1, num_cc):
                area = stats[c_id, cv2.CC_STAT_AREA]
                if area < 50:
                    continue
                x = stats[c_id, cv2.CC_STAT_LEFT]
                y = stats[c_id, cv2.CC_STAT_TOP]
                bw = stats[c_id, cv2.CC_STAT_WIDTH]
                bh = stats[c_id, cv2.CC_STAT_HEIGHT]
                mask = (labels_cc == c_id)
                proposals.append({
                    "mask": mask,
                    "bbox": (x, y, x + bw, y + bh),
                    "area": area,
                    "initial_class": class_id
                })

        # Sort proposals by area descending (largest background first, smaller instances on top)
        proposals = sorted(proposals, key=lambda p: p["area"], reverse=True)

        # Re-verify and classify proposals with ResNet-18 classifier
        crops_t = []
        for p in proposals:
            x0, y0, x1, y1 = p["bbox"]
            crop_img = img_np[y0:y1, x0:x1].copy()
            crop_seg = p["mask"][y0:y1, x0:x1]
            crop_img[~crop_seg] = 0 # Mask background outside proposal
            crops_t.append(clf_transform(crop_img))

        if crops_t:
            clf_preds = []
            for b in range(0, len(crops_t), 64):
                batch_t = torch.stack(crops_t[b:b+64]).to(device)
                clf_logits = clf(batch_t)
                clf_preds.extend(clf_logits.argmax(dim=1).cpu().numpy().tolist())

            final_label_map = np.copy(base_pred)
            for p, c_pred in zip(proposals, clf_preds):
                # If classifier is confident on smaller object, refine proposal
                if p["area"] < proc_w * proc_h * 0.15:
                    final_label_map[p["mask"]] = c_pred
        else:
            final_label_map = base_pred

        # Equirectangular latitude sanity clamping:
        final_label_map[:int(proc_h * 0.10), :] = 0  # Roof
        final_label_map[int(proc_h * 0.88):, :] = 8  # Floor

        # Upsample to original image dimensions
        pred_mask = cv2.resize(final_label_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Save canonical output
    save_segmentation_result(args.output, pred_mask, start_time)

if __name__ == "__main__":
    main()
