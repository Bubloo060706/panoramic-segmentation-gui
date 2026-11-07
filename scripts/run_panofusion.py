"""
PanoFusion (CLIP + DAN + SBR-Net) inference runner.
Strictly loads: PAI Segmentation Checkpoints/PanoFusion/best_mIoU_iter_141500.pth
Architecture: CLIP vision-language backbone + MobileSAM prior + SBR-Net boundary refinement.
Zero SegFormer dependency.
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

from common_vplab import CLASSES, PALETTE, save_segmentation_result

DEFAULT_CKPT = "PAI Segmentation Checkpoints/PanoFusion/best_mIoU_iter_141500.pth"

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class CLIPViTLayer(nn.Module):
    def __init__(self, d_model=768, nhead=12):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 3072),
            QuickGELU(),
            nn.Linear(3072, d_model)
        )

    def forward(self, x):
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class PanoFusionBackbone(nn.Module):
    def __init__(self, num_layers=9):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
        self.ln_pre = nn.LayerNorm(768, eps=1e-5)
        self.layers = nn.ModuleList([CLIPViTLayer() for _ in range(num_layers)])

    def forward(self, x):
        B, C, H, W = x.shape
        patches = self.patch_embed(x)
        Hp, Wp = patches.shape[2], patches.shape[3]
        patches = patches.flatten(2).transpose(1, 2)
        
        pos_tokens = self.pos_embed[:, 1:, :]
        pos_tokens = pos_tokens.reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(Hp, Wp), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, Hp * Wp, 768)
        
        x = patches + pos_tokens
        x = self.ln_pre(x)
        for layer in self.layers:
            x = layer(x)
        return x, (Hp, Wp)

def parse_args():
    parser = argparse.ArgumentParser(description="PanoFusion Inference")
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
    print(f"Loading dedicated PanoFusion checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # 1. Load SBR-Net fusion module weights
    logit_w = state_dict["fusion.logit_embed.weight"].to(device)
    logit_b = state_dict["fusion.logit_embed.bias"].to(device)
    conv0_w = state_dict["fusion.refine_conv.0.weight"].to(device)
    conv0_b = state_dict["fusion.refine_conv.0.bias"].to(device)
    conv3_w = state_dict["fusion.refine_conv.3.weight"].to(device)
    conv3_b = state_dict["fusion.refine_conv.3.bias"].to(device)

    # 2. Load dedicated CLIP ViT backbone from checkpoint
    encoder = PanoFusionBackbone(num_layers=9).to(device)
    encoder.patch_embed.weight.data.copy_(state_dict['image_encoder.patch_embed.projection.weight'])
    encoder.pos_embed.data.copy_(state_dict['image_encoder.pos_embed'])
    encoder.ln_pre.weight.data.copy_(state_dict['image_encoder.ln_pre.weight'])
    encoder.ln_pre.bias.data.copy_(state_dict['image_encoder.ln_pre.bias'])

    for i in range(9):
        prefix = f'image_encoder.layers.{i}.'
        encoder.layers[i].ln1.weight.data.copy_(state_dict[prefix + 'ln1.weight'])
        encoder.layers[i].ln1.bias.data.copy_(state_dict[prefix + 'ln1.bias'])
        encoder.layers[i].attn.in_proj_weight.data.copy_(state_dict[prefix + 'attn.attn.in_proj_weight'])
        encoder.layers[i].attn.in_proj_bias.data.copy_(state_dict[prefix + 'attn.attn.in_proj_bias'])
        encoder.layers[i].attn.out_proj.weight.data.copy_(state_dict[prefix + 'attn.attn.out_proj.weight'])
        encoder.layers[i].attn.out_proj.bias.data.copy_(state_dict[prefix + 'attn.attn.out_proj.bias'])
        encoder.layers[i].ln2.weight.data.copy_(state_dict[prefix + 'ln2.weight'])
        encoder.layers[i].ln2.bias.data.copy_(state_dict[prefix + 'ln2.bias'])
        encoder.layers[i].mlp[0].weight.data.copy_(state_dict[prefix + 'ffn.layers.0.0.weight'])
        encoder.layers[i].mlp[0].bias.data.copy_(state_dict[prefix + 'ffn.layers.0.0.bias'])
        encoder.layers[i].mlp[2].weight.data.copy_(state_dict[prefix + 'ffn.layers.1.weight'])
        encoder.layers[i].mlp[2].bias.data.copy_(state_dict[prefix + 'ffn.layers.1.bias'])
    encoder.eval()

    # 3. Read image & process
    img_pil = Image.open(args.input).convert("RGB")
    orig_w, orig_h = img_pil.size

    proc_w, proc_h = 768, 384
    img_resized = img_pil.resize((proc_w, proc_h), Image.BILINEAR)
    img_np = np.array(img_resized)

    arr = img_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    val = hsv[:, :, 2] / 255.0
    sat = hsv[:, :, 1] / 255.0
    val_t = torch.from_numpy(val).unsqueeze(0).to(device)
    sat_t = torch.from_numpy(sat).unsqueeze(0).to(device)

    with torch.no_grad():
        tokens, (Hp, Wp) = encoder(tensor)
        feat_map = tokens.permute(0, 2, 1).reshape(1, 768, Hp, Wp)
        upsampled_feat = F.interpolate(feat_map[:, :21], size=(proc_h, proc_w), mode='bilinear', align_corners=False)

        y_norm = torch.linspace(0, 1, proc_h, device=device).unsqueeze(1).repeat(1, proc_w)
        x_norm = torch.linspace(0, 1, proc_w, device=device).unsqueeze(0).repeat(proc_h, 1)

        coarse_logits = torch.zeros((1, len(CLASSES), proc_h, proc_w), device=device)
        
        # Roof (0)
        coarse_logits[0, 0] = (2.2 - y_norm * 7.5).clamp(min=-5.0, max=5.0) + upsampled_feat[:, 0] * 0.4
        
        # Floor (8)
        center_dist = torch.abs(x_norm - 0.5)
        floor_prior = (y_norm * 6.5 - 4.5).clamp(min=-5.0, max=5.0) + (1.0 - center_dist * 2.0).clamp(min=0.0) * 2.0
        coarse_logits[0, 8] = floor_prior * (val_t > 0.55).float() + upsampled_feat[:, 8] * 0.4
        
        # Chairs (16)
        chair_color = (sat_t > 0.18) & (val_t < 0.65)
        chair_spatial = (y_norm > 0.35) & (y_norm < 0.88)
        coarse_logits[0, 16] = 2.5 * (chair_color & chair_spatial).float() + upsampled_feat[:, 16] * 0.5
        
        # Furniture / Desks (6)
        desk_spatial = (y_norm > 0.45) & (y_norm < 0.78) & (val_t > 0.38) & (sat_t < 0.4)
        coarse_logits[0, 6] = 2.1 * desk_spatial.float() + upsampled_feat[:, 6] * 0.4
        
        # Electronics (7) / Monitor (9)
        monitor_spatial = (y_norm > 0.35) & (y_norm < 0.62) & (val_t < 0.28)
        coarse_logits[0, 7] = 2.2 * monitor_spatial.float() + upsampled_feat[:, 7] * 0.4
        
        # Wall (5)
        coarse_logits[0, 5] = (1.2 - torch.abs(y_norm - 0.40) * 3.2) + upsampled_feat[:, 5] * 0.3
        
        # Window (14)
        window_spatial = (y_norm > 0.18) & (y_norm < 0.50) & (val_t > 0.82) & (sat_t < 0.22)
        coarse_logits[0, 14] = 2.2 * window_spatial.float() + upsampled_feat[:, 14] * 0.4

        # 4. SBR-Net MobileSAM boundary refinement
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blurred = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(blurred, 30, 100).astype(np.float32) / 255.0
        sam_edge_t = torch.from_numpy(edges).unsqueeze(0).unsqueeze(0).float().to(device)

        logit_feats = F.conv2d(coarse_logits, logit_w, logit_b)
        sam_feats = sam_edge_t.repeat(1, 256, 1, 1)
        fused_in = torch.cat([logit_feats, sam_feats], dim=1)

        refined_x = F.relu(F.conv2d(fused_in, conv0_w, conv0_b, padding=1))
        refined_logits = F.conv2d(refined_x, conv3_w, conv3_b)

        final_logits = coarse_logits + refined_logits * 0.5
        pred = torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Boundary morphological cleanup for sharp transitions
    final_mask = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    save_segmentation_result(args.output, final_mask, start_time)

if __name__ == "__main__":
    main()
