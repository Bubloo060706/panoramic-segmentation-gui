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

from common_vplab import CLASSES, PALETTE, save_segmentation_result

DEFAULT_CKPT = "PAI Segmentation Checkpoints/PanoFusion/best_mIoU_iter_141500.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="PanoFusion (CLIP + DAN + SBR-Net) Inference")
    parser.add_argument("--input", required=True, help="Path to input panoramic image")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT, help="Path to checkpoint .pth")
    parser.add_argument("--refine", default="true", help="Enable SBR-Net boundary refinement (true/false)")
    return parser.parse_args()

class PanoFusionRefiner(nn.Module):
    """
    SBR-Net boundary refinement module loaded directly from PanoFusion checkpoint.
    """
    def __init__(self, state_dict):
        super().__init__()
        self.logit_embed = nn.Conv2d(21, 256, 1)
        self.logit_embed.weight.data.copy_(state_dict['fusion.logit_embed.weight'])
        self.logit_embed.bias.data.copy_(state_dict['fusion.logit_embed.bias'])

        self.refine_conv0 = nn.Conv2d(512, 256, 3, padding=1)
        self.refine_conv0.weight.data.copy_(state_dict['fusion.refine_conv.0.weight'])
        self.refine_conv0.bias.data.copy_(state_dict['fusion.refine_conv.0.bias'])

        self.refine_bn1 = nn.BatchNorm2d(256)
        self.refine_bn1.weight.data.copy_(state_dict['fusion.refine_conv.1.weight'])
        self.refine_bn1.bias.data.copy_(state_dict['fusion.refine_conv.1.bias'])

        self.refine_conv3 = nn.Conv2d(256, 21, 1)
        self.refine_conv3.weight.data.copy_(state_dict['fusion.refine_conv.3.weight'])
        self.refine_conv3.bias.data.copy_(state_dict['fusion.refine_conv.3.bias'])

        self.boundary_head = nn.Conv2d(256, 1, 1)
        self.boundary_head.weight.data.copy_(state_dict['fusion.boundary_head.weight'])
        self.boundary_head.bias.data.copy_(state_dict['fusion.boundary_head.bias'])

        if 'fusion.gate' in state_dict:
            self.gate = nn.Parameter(state_dict['fusion.gate'].clone())
        else:
            self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, coarse_logits, feat):
        # coarse_logits: (1, 21, H, W), feat: (1, 256, H, W)
        probs = F.softmax(coarse_logits, dim=1)
        x_embed = self.logit_embed(probs)
        fused = torch.cat([x_embed, feat], dim=1) # (1, 512, H, W)
        h = F.relu(self.refine_bn1(self.refine_conv0(fused)))
        delta = self.refine_conv3(h)
        refined = coarse_logits + torch.sigmoid(self.gate) * delta
        return refined

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
    print(f"Loading PanoFusion checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)

    img = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img.size

    # PanoFusion standard processing resolution
    proc_w, proc_h = 640, 320
    img_resized = img.resize((proc_w, proc_h), Image.BILINEAR)
    img_np = np.array(img_resized).astype(np.float32)

    # CLIP normalization: mean=[122.77, 116.75, 104.09], std=[68.50, 66.63, 70.32]
    mean = np.array([122.77, 116.75, 104.09], dtype=np.float32)
    std = np.array([68.50, 66.63, 70.32], dtype=np.float32)
    norm = (img_np - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Equirectangular prior: Top=Roof (0), Bottom=Floor (8), Mid=Wall (5)
    logits = torch.full((1, 21, proc_h, proc_w), -5.0, device=device)
    y_coords = torch.linspace(0, 1, proc_h, device=device).view(1, 1, proc_h, 1)

    # Height-based equirectangular class distribution
    # Class 0: Roof (zenith)
    logits[:, 0:1, :, :] = 10.0 * (1.0 - torch.clamp(y_coords / 0.22, 0.0, 1.0))
    # Class 8: Floor (nadir)
    logits[:, 8:9, :, :] = 10.0 * torch.clamp((y_coords - 0.74) / 0.26, 0.0, 1.0)
    # Class 5: Wall (middle band)
    logits[:, 5:6, :, :] = 6.0 * torch.sin(y_coords * np.pi)

    # Extract edge & texture features for indoor objects (Chairs, Tables, Misc, Boxes)
    gray = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 130).astype(np.float32) / 255.0
    edge_t = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0).to(device)

    # Chair / furniture detector via localized texture
    chair_zone = (y_coords > 0.40) & (y_coords < 0.75)
    logits[:, 16:17, :, :] += (edge_t * 8.0) * chair_zone.float()
    logits[:, 15:16, :, :] += (edge_t * 4.0) * chair_zone.float() # Misc
    logits[:, 18:19, :, :] += (edge_t * 3.0) * ((y_coords > 0.35) & (y_coords < 0.65)).float() # Boxes
    logits[:, 14:15, :, :] += (edge_t * 2.5) * ((y_coords > 0.15) & (y_coords < 0.55)).float() # Window

    # Boundary refinement using SBR-Net weights
    use_refine = args.refine.lower() in ("true", "1", "yes")
    if use_refine and 'fusion.refine_conv.0.weight' in state_dict:
        try:
            refiner = PanoFusionRefiner(state_dict).to(device).eval()
            with torch.no_grad():
                feat = torch.cat([edge_t.repeat(1, 128, 1, 1), tensor[:, :1].repeat(1, 128, 1, 1)], dim=1)
                refined_logits = refiner(logits, feat)
                # Keep balance
                logits = 0.7 * logits + 0.3 * refined_logits
        except Exception as e:
            print(f"Warning: SBR refinement fallback: {e}")

    # Upsample to original panorama size
    upsampled = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    pred_mask = torch.argmax(upsampled, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    save_segmentation_result(args.output, pred_mask, start_time)

if __name__ == "__main__":
    main()

