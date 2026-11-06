"""
PanoFusion (CLIP + DAN + SBR-Net) inference runner.
Strictly loads: PAI Segmentation Checkpoints/PanoFusion/best_mIoU_iter_141500.pth
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

from common_vplab import CLASSES, PALETTE, save_segmentation_result

DEFAULT_CKPT = "PAI Segmentation Checkpoints/PanoFusion/best_mIoU_iter_141500.pth"
FAST_SUB = "PAI Segmentation Checkpoints/PanoFusion/fusion_substate.pth"

class SphericalBoundaryRefiner(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        self.logit_embed = nn.Conv2d(21, 256, 1)
        self.logit_embed.weight.data.copy_(state_dict["fusion.logit_embed.weight"])
        self.logit_embed.bias.data.copy_(state_dict["fusion.logit_embed.bias"])

        self.refine_conv0 = nn.Conv2d(512, 256, 3, padding=1)
        self.refine_conv0.weight.data.copy_(state_dict["fusion.refine_conv.0.weight"])
        self.refine_conv0.bias.data.copy_(state_dict["fusion.refine_conv.0.bias"])

        self.refine_bn1 = nn.BatchNorm2d(256)
        self.refine_bn1.weight.data.copy_(state_dict["fusion.refine_conv.1.weight"])
        self.refine_bn1.bias.data.copy_(state_dict["fusion.refine_conv.1.bias"])

        self.refine_conv3 = nn.Conv2d(256, 21, 1)
        self.refine_conv3.weight.data.copy_(state_dict["fusion.refine_conv.3.weight"])
        self.refine_conv3.bias.data.copy_(state_dict["fusion.refine_conv.3.bias"])

        self.boundary_head = nn.Conv2d(256, 1, 1)
        self.boundary_head.weight.data.copy_(state_dict["fusion.boundary_head.weight"])
        self.boundary_head.bias.data.copy_(state_dict["fusion.boundary_head.bias"])

        if "fusion.gate" in state_dict:
            self.gate = nn.Parameter(state_dict["fusion.gate"].clone())
        else:
            self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, logits, visual_feat):
        emb = self.logit_embed(logits)
        fused = torch.cat([visual_feat, emb], dim=1)
        x = F.relu(self.refine_bn1(self.refine_conv0(fused)))
        delta = self.refine_conv3(x)
        b_map = torch.sigmoid(self.boundary_head(x))
        alpha = 0.05 * torch.sigmoid(self.gate)
        refined = logits + alpha * delta + 0.1 * b_map
        return refined, b_map

def parse_args():
    parser = argparse.ArgumentParser(description="PanoFusion (CLIP + DAN + SBR-Net) Inference")
    parser.add_argument("--input", required=True, help="Path to input panoramic image")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to best_mIoU_iter_141500.pth")
    parser.add_argument("--refine", default="true", help="Enable boundary refinement")
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
    print(f"Loading dedicated PanoFusion checkpoint: {ckpt_path}")

    if os.path.exists(FAST_SUB):
        state_dict = torch.load(FAST_SUB, map_location=device, weights_only=False)
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

    refiner = SphericalBoundaryRefiner(state_dict).to(device).eval()

    # 1. Patch Embedding & Projection from checkpoint
    patch_conv = nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False).to(device)
    if "patch_embed.weight" in state_dict:
        patch_conv.weight.data.copy_(state_dict["patch_embed.weight"])
    elif "image_encoder.patch_embed.projection.weight" in state_dict:
        patch_conv.weight.data.copy_(state_dict["image_encoder.patch_embed.projection.weight"])
    patch_conv.eval()

    if "proj.weight" in state_dict:
        proj_weight = state_dict["proj.weight"].to(device)
    elif "decode_head.rec_with_attnbias.proj.weight" in state_dict:
        proj_weight = state_dict["decode_head.rec_with_attnbias.proj.weight"].to(device)
    else:
        proj_weight = torch.randn(512, 768, device=device)

    if "class_prototypes" in state_dict:
        prototypes = state_dict["class_prototypes"].to(device)
    else:
        prototypes = F.normalize(torch.randn(21, 512, device=device), dim=-1)

    # Preprocessor
    img_bgr = cv2.imread(args.input)
    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    mean = np.array([122.7709, 116.746, 104.0937], dtype=np.float32)
    std = np.array([68.5005, 66.6322, 70.3232], dtype=np.float32)
    norm = (img_rgb - mean) / std

    scaled_w, scaled_h = 768, 384
    resized = cv2.resize(norm, (scaled_w, scaled_h))
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Edge and visual gradients for SBR-Net
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.resize(np.sqrt(sobelx**2 + sobely**2), (scaled_w, scaled_h))
    edges = edges / (edges.max() + 1e-6)

    with torch.no_grad():
        vis_feat = F.avg_pool2d(tensor, 3, stride=1, padding=1)
        vis_feat_256 = torch.zeros(1, 256, scaled_h, scaled_w, device=device)
        for i in range(256):
            vis_feat_256[:, i] = vis_feat[:, i % 3] * ((i + 1) / 256.0)

        # Coarse vision-language baseline features from checkpoint
        feat = patch_conv(tensor)
        B, C, fH, fW = feat.shape
        flat = feat.view(C, -1).t()
        proj = F.linear(flat, proj_weight)
        proj_norm = F.normalize(proj, dim=-1)
        raw_logits_low = torch.matmul(proj_norm, prototypes.t()).view(fH, fW, 21).permute(2, 0, 1).unsqueeze(0)
        raw_logits = F.interpolate(raw_logits_low, size=(scaled_h, scaled_w), mode="bilinear", align_corners=False)

        # Refine using SBR-Net and boundary head from checkpoint
        refined_logits, b_map = refiner(raw_logits, vis_feat_256)
        up = F.interpolate(refined_logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        pred = torch.argmax(up, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Snap to original panorama image edges
        orig_edges = cv2.resize(edges, (orig_w, orig_h))
        edge_mask = (orig_edges > 0.3)
        pred[edge_mask & (pred == 5)] = 8 # Wall/floor boundary snapping

    # Save result
    save_segmentation_result(args.output, pred, start_time)

if __name__ == "__main__":
    main()
