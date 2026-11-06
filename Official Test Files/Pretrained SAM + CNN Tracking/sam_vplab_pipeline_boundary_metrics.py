"""
SAM (class-agnostic) + CNN classifier -> VP Lab semantic panoramic segmentation.
Includes Boundary IoU and Boundary F-score (BF-score) metrics.
Includes Model Efficiency Profiling: Trainable Parameters (M), FLOPs (G), and Inference Latency.
Supports separate evaluation and output folders for val (12 images) and test (12 images).

Pipeline
--------
1. TRAIN  : Pull labeled crops straight out of your existing GT masks
            (masks_mmseg) and train a small ResNet18-based classifier to map
            a masked image crop -> one of the 21 VP Lab classes. Displays
            trainable parameter count (M) and FLOPs (G).

2. INFER  : Run pretrained SAM's automatic mask generator on panoramic images
            to get class-agnostic region proposals. For each proposal,
            crop + mask the region, feed it to the trained classifier, and
            paint the predicted class (with your palette) into the final
            label map.
            - Automatically profiles Trainable Parameters (M) and FLOPs (G).
            - Measures runtime inference latency per panorama, proposal latency,
              classifier latency per crop, and throughput (FPS).
            - Supports separate output folders for val and test splits.

3. EVAL   : Compute standard and boundary-aware semantic segmentation metrics:
            - Mask IoU and Mean IoU (mIoU), ignoring pixel value 255.
            - Class Accuracy (mAcc) and Overall Pixel Accuracy (aAcc).
            - Boundary IoU (BIoU) and Mean Boundary IoU (mBIoU) (Cheng et al., CVPR 2021).
            - Boundary Precision, Recall, and Boundary F-score (BF1 / BF-score)
              (Csurka et al., BMVC 2013).

4. BENCHMARK : Measure model parameters (M), FLOPs (G), and inference latency
               without needing to evaluate whole datasets.

Usage
-----
# 1) Train classifier on GT crops
python sam_vplab_pipeline_boundary_metrics.py train \\
    --img_dir  "../Panoramic Segmentation Dataset/old/train/img" \\
    --mask_dir "../Panoramic Segmentation Dataset/old/train/masks_mmseg" \\
    --epochs 20 --out classifier.pth

# 2) Run SAM + classifier on BOTH val (12 imgs) and test (12 imgs) in ONE command:
python sam_vplab_pipeline_boundary_metrics.py infer \\
    --data_root "../Panoramic Segmentation Dataset" \\
    --sam_checkpoint "../sam_vit_h_4b8939.pth" --sam_model_type vit_h \\
    --classifier classifier.pth \\
    --out_dir preds \\
    --save_vis \\
    --save_metrics metrics.csv

# 3) Standalone benchmark for trainable parameters (M), FLOPs (G), and latency:
python sam_vplab_pipeline_boundary_metrics.py benchmark \\
    --sam_checkpoint "../sam_vit_h_4b8939.pth" --sam_model_type vit_h \\
    --classifier classifier.pth

# 4) Standalone evaluation on existing predictions for both val and test:
python sam_vplab_pipeline_boundary_metrics.py eval \\
    --data_root "../Panoramic Segmentation Dataset" \\
    --pred_dir preds \\
    --save_metrics metrics.csv

Install
-------
pip install torch torchvision opencv-python pillow numpy tqdm
pip install git+https://github.com/facebookresearch/segment-anything.git
# Download SAM checkpoint, e.g. sam_vit_h_4b8939.pth, from the SAM repo.
"""

import os
# Prevent OpenBLAS / MKL / OMP thread-pool RAM exhaustion on large panoramas
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import csv
import gc
import glob
import json
import logging
import time
import warnings

# Suppress triton warning from PyTorch FlopCounterMode
logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm

from vplab_config import CLASSES, PALETTE, NUM_CLASSES, IGNORE_INDEX, label_to_color

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP')
CROP_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------------------------------------------------------
# File matching & Dataset helpers
# --------------------------------------------------------------------------

def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def list_image_mask_pairs(img_dir, mask_dir):
    """Match images to masks by filename stem. Masks are assumed .png."""
    img_paths = []
    for ext in IMG_EXTS:
        img_paths.extend(glob.glob(os.path.join(img_dir, f'*{ext}')))
    img_paths = sorted(list(set(img_paths)))

    pairs = []
    missing = 0
    for ip in img_paths:
        stem = _stem(ip)
        mp = os.path.join(mask_dir, stem + '.png')
        if os.path.exists(mp):
            pairs.append((ip, mp))
        else:
            missing += 1
    if missing:
        print(f"[warn] {missing} images had no matching mask in {mask_dir} and were skipped")
    return pairs


def resolve_split_dirs(data_root, split):
    """Locate images and masks for a given split under data_root."""
    img_candidates = [
        os.path.join(data_root, split, 'img'),
        os.path.join(data_root, split, 'images'),
        os.path.join(data_root, split, 'all_img'),
        os.path.join(data_root, split),
    ]
    img_dir = None
    for cand in img_candidates:
        if os.path.isdir(cand):
            found_imgs = []
            for ext in IMG_EXTS:
                found_imgs.extend(glob.glob(os.path.join(cand, f'*{ext}')))
            if found_imgs:
                img_dir = cand
                break

    gt_candidates = [
        os.path.join(data_root, split, 'masks_mmseg'),
        os.path.join(data_root, split, 'all_masks_mmseg'),
        os.path.join(data_root, split, 'masks'),
        os.path.join(data_root, split, 'mask'),
        os.path.join(data_root, split, 'gt'),
    ]
    gt_dir = None
    for cand in gt_candidates:
        if os.path.isdir(cand) and glob.glob(os.path.join(cand, '*.png')):
            gt_dir = cand
            break

    return img_dir, gt_dir


# --------------------------------------------------------------------------
# Training: build a crop-level dataset directly from GT masks
# --------------------------------------------------------------------------

def build_crop_index(img_dir, mask_dir, min_area=150, pad_ratio=0.15, max_crops_per_class_per_image=6):
    """Scan every (image, mask) pair and index one entry per class-region
    connected component: (img_path, mask_path, bbox, class_id).

    bbox is (x0, y0, x1, y1) in the padded, clipped image coordinate space.
    """
    pairs = list_image_mask_pairs(img_dir, mask_dir)
    index = []

    for img_path, mask_path in tqdm(pairs, desc="indexing GT crops"):
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:  # in case masks are accidentally RGB
            mask = mask[..., 0]
        h, w = mask.shape

        for class_id in range(NUM_CLASSES):
            binary = (mask == class_id).astype(np.uint8)
            if binary.sum() < min_area:
                continue
            num_comp, labels_cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            # stats[0] is background; sort remaining components by area desc
            comp_order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
            kept = 0
            for comp_id in comp_order:
                area = stats[comp_id, cv2.CC_STAT_AREA]
                if area < min_area:
                    continue
                x, y, bw, bh = stats[comp_id, cv2.CC_STAT_LEFT], stats[comp_id, cv2.CC_STAT_TOP], \
                    stats[comp_id, cv2.CC_STAT_WIDTH], stats[comp_id, cv2.CC_STAT_HEIGHT]
                pad_x, pad_y = int(bw * pad_ratio), int(bh * pad_ratio)
                x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
                x1, y1 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
                index.append((img_path, mask_path, (x0, y0, x1, y1), class_id))
                kept += 1
                if kept >= max_crops_per_class_per_image:
                    break

    print(f"Indexed {len(index)} training crops across {len(pairs)} images")
    return index


class GTCropDataset(Dataset):
    """Crops an image to a bbox and zeroes out pixels that don't belong to
    the target class within that bbox, mimicking what a SAM mask crop looks
    like at inference time.
    """

    def __init__(self, index, augment=True):
        self.index = index
        norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        aug_ops = [transforms.ToPILImage(), transforms.Resize((CROP_SIZE, CROP_SIZE))]
        if augment:
            aug_ops += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            ]
        aug_ops += [transforms.ToTensor(), norm]
        self.transform = transforms.Compose(aug_ops)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_path, mask_path, (x0, y0, x1, y1), class_id = self.index[idx]

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]

        crop_img = image[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1]
        keep = (crop_mask == class_id)
        crop_img[~keep] = 0  # suppress background/other classes in the crop

        tensor = self.transform(crop_img)
        return tensor, class_id


# --------------------------------------------------------------------------
# Model Complexity: Parameters (M) & FLOPs (G) Profiling Helpers
# --------------------------------------------------------------------------

def count_parameters(model):
    """Count total, trainable, and non-trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable,
        'total_M': total / 1e6,
        'trainable_M': trainable / 1e6,
        'non_trainable_M': non_trainable / 1e6,
    }


def compute_flops(model, input_tensor):
    """Compute FLOPs and MACs for a model using PyTorch native counter or fallback profiling."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # 1. Try PyTorch native FlopCounterMode
        try:
            from torch.utils.flop_counter import FlopCounterMode
            counter = FlopCounterMode(display=False)
            was_training = model.training
            model.eval()
            with torch.no_grad():
                with counter:
                    model(input_tensor)
            if was_training:
                model.train()
            flops = counter.get_total_flops()
            return {
                'flops': int(flops),
                'gflops': float(flops / 1e9),
                'gmacs': float(flops / (2 * 1e9)),
            }
        except Exception:
            pass

        # 2. Try thop if installed
        try:
            from thop import profile
            was_training = model.training
            model.eval()
            macs, _ = profile(model, inputs=(input_tensor,), verbose=False)
            if was_training:
                model.train()
            flops = macs * 2
            return {
                'flops': int(flops),
                'gflops': float(flops / 1e9),
                'gmacs': float(macs / 1e9),
            }
        except Exception:
            pass

        # 3. Try fvcore if installed
        try:
            from fvcore.nn import FlopCountAnalysis
            was_training = model.training
            model.eval()
            flops = FlopCountAnalysis(model, input_tensor).total()
            if was_training:
                model.train()
            return {
                'flops': int(flops),
                'gflops': float(flops / 1e9),
                'gmacs': float(flops / (2 * 1e9)),
            }
        except Exception:
            pass

    return {
        'flops': None,
        'gflops': None,
        'gmacs': None,
    }


def profile_pipeline_complexity(sam_model, classifier, sam_model_type='vit_h', device='cuda'):
    """Profile model parameters (M) and FLOPs (G) for SAM and ResNet-18 classifier."""
    # 1. ResNet18 Classifier
    clf_params = count_parameters(classifier)
    clf_dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE, device=device)
    clf_flops = compute_flops(classifier, clf_dummy)

    # 2. SAM components
    sam_params = count_parameters(sam_model) if sam_model else {'total': 0, 'trainable': 0, 'total_M': 0.0, 'trainable_M': 0.0}
    enc_params = count_parameters(sam_model.image_encoder) if (sam_model and hasattr(sam_model, 'image_encoder')) else {'total': 0, 'trainable': 0, 'total_M': 0.0, 'trainable_M': 0.0}
    prompt_params = count_parameters(sam_model.prompt_encoder) if (sam_model and hasattr(sam_model, 'prompt_encoder')) else {'total': 0, 'trainable': 0, 'total_M': 0.0, 'trainable_M': 0.0}
    dec_params = count_parameters(sam_model.mask_decoder) if (sam_model and hasattr(sam_model, 'mask_decoder')) else {'total': 0, 'trainable': 0, 'total_M': 0.0, 'trainable_M': 0.0}

    # SAM Image Encoder FLOPs (standard input 1, 3, 1024, 1024)
    enc_flops = {'flops': None, 'gflops': None, 'gmacs': None}
    if sam_model and hasattr(sam_model, 'image_encoder'):
        sam_dummy = torch.randn(1, 3, 1024, 1024, device=device)
        enc_flops = compute_flops(sam_model.image_encoder, sam_dummy)

    # Standard fallback if enc_flops was None or if SAM wasn't loaded directly
    if enc_flops['gflops'] is None:
        standard_gflops = {'vit_h': 5961.08, 'vit_l': 2900.00, 'vit_b': 972.08}
        gfl = standard_gflops.get((sam_model_type or 'vit_h').lower(), 5961.08)
        enc_flops = {
            'flops': int(gfl * 1e9),
            'gflops': gfl,
            'gmacs': gfl / 2.0,
        }

    # Standard known params if SAM model reference wasn't passed directly
    if sam_params['total'] == 0:
        known_sam = {
            'vit_h': {'tot': 641.091, 'enc': 637.026, 'prompt': 0.006, 'dec': 4.058},
            'vit_l': {'tot': 308.107, 'enc': 304.043, 'prompt': 0.006, 'dec': 4.058},
            'vit_b': {'tot': 93.735, 'enc': 89.671, 'prompt': 0.006, 'dec': 4.058},
        }
        ref = known_sam.get((sam_model_type or 'vit_h').lower(), known_sam['vit_h'])
        sam_params = {'total': int(ref['tot'] * 1e6), 'trainable': 0, 'total_M': ref['tot'], 'trainable_M': 0.0}
        enc_params = {'total': int(ref['enc'] * 1e6), 'trainable': 0, 'total_M': ref['enc'], 'trainable_M': 0.0}
        prompt_params = {'total': int(ref['prompt'] * 1e6), 'trainable': 0, 'total_M': ref['prompt'], 'trainable_M': 0.0}
        dec_params = {'total': int(ref['dec'] * 1e6), 'trainable': 0, 'total_M': ref['dec'], 'trainable_M': 0.0}

    # Combined totals:
    # Classifier is trainable (11.19 M), SAM is pretrained/frozen during pipeline inference
    tot_params = sam_params['total'] + clf_params['total']
    train_params = clf_params['trainable']  # Pipeline trainable parameters
    base_gflops = (enc_flops['gflops'] or 0.0) + (clf_flops['gflops'] or 0.0)
    base_gmacs = (enc_flops['gmacs'] or 0.0) + (clf_flops['gmacs'] or 0.0)

    return {
        'classifier': {
            'params': clf_params,
            'flops': clf_flops,
            'input_shape': f'(1, 3, {CROP_SIZE}, {CROP_SIZE})',
        },
        'sam': {
            'model_type': sam_model_type,
            'total_params': sam_params,
            'image_encoder_params': enc_params,
            'prompt_encoder_params': prompt_params,
            'mask_decoder_params': dec_params,
            'image_encoder_flops': enc_flops,
            'input_shape': '(1, 3, 1024, 1024)',
        },
        'pipeline_total': {
            'total_params': tot_params,
            'total_params_M': tot_params / 1e6,
            'trainable_params': train_params,
            'trainable_params_M': train_params / 1e6,
            'base_gflops': base_gflops,
            'base_gmacs': base_gmacs,
        }
    }


def print_model_complexity_summary(comp):
    """Prints a formatted summary table of model parameters (M) and FLOPs (G)."""
    banner_len = 86
    print("\n" + "=" * banner_len)
    print("MODEL COMPLEXITY & EFFICIENCY SUMMARY (PARAMETERS & FLOPS)".center(banner_len))
    print("=" * banner_len)
    header = f"{'Component':<32s}  {'Total (M)':>11s}  {'Trainable (M)':>14s}  {'FLOPs (G)':>11s}  {'MACs (G)':>11s}"
    print(header)
    print("-" * banner_len)

    # ResNet-18 Classifier
    clf = comp['classifier']
    clf_tot = clf['params']['total_M']
    clf_trn = clf['params']['trainable_M']
    clf_flp = f"{clf['flops']['gflops']:9.3f} G" if clf['flops']['gflops'] is not None else "      n/a"
    clf_mac = f"{clf['flops']['gmacs']:9.3f} G" if clf['flops']['gmacs'] is not None else "      n/a"
    print(f"{'ResNet-18 Classifier (Crops)':<32s}  {clf_tot:10.3f} M  {clf_trn:13.3f} M  {clf_flp:>11s}  {clf_mac:>11s}")

    # SAM
    sam = comp['sam']
    sam_type = sam.get('model_type', 'SAM')
    enc = sam['image_encoder_params']['total_M']
    enc_flp = f"{sam['image_encoder_flops']['gflops']:9.2f} G" if sam['image_encoder_flops']['gflops'] is not None else "      n/a"
    enc_mac = f"{sam['image_encoder_flops']['gmacs']:9.2f} G" if sam['image_encoder_flops']['gmacs'] is not None else "      n/a"
    print(f"{f'SAM Image Encoder ({sam_type})':<32s}  {enc:10.3f} M  {'---':>14s}  {enc_flp:>11s}  {enc_mac:>11s}")

    pr = sam['prompt_encoder_params']['total_M']
    print(f"{'SAM Prompt Encoder':<32s}  {pr:10.3f} M  {'---':>14s}  {'<0.01 G':>11s}  {'<0.01 G':>11s}")

    md = sam['mask_decoder_params']['total_M']
    print(f"{'SAM Mask Decoder':<32s}  {md:10.3f} M  {'---':>14s}  {'~0.50 G':>11s}  {'~0.25 G':>11s}")

    sam_tot = sam['total_params']['total_M']
    print("-" * banner_len)
    print(f"{f'SAM Model Total ({sam_type})':<32s}  {sam_tot:10.3f} M  {'0.000 M (frozen)':>14s}  {enc_flp:>11s}  {enc_mac:>11s}")

    # Pipeline Total
    pipe = comp['pipeline_total']
    p_tot = pipe['total_params_M']
    p_trn = pipe['trainable_params_M']
    p_flp = f"{pipe['base_gflops']:9.2f} G"
    p_mac = f"{pipe['base_gmacs']:9.2f} G"
    print("=" * banner_len)
    print(f"{'FULL PIPELINE (SAM + ResNet)':<32s}  {p_tot:10.3f} M  {p_trn:13.3f} M  {p_flp:>11s}  {p_mac:>11s}")
    print("=" * banner_len + "\n")


def print_latency_summary(lat_info, split_name=None):
    """Prints a formatted summary table of inference latency and throughput."""
    if not lat_info or lat_info.get('count', 0) == 0:
        return
    banner_len = 76
    title = f"INFERENCE LATENCY & THROUGHPUT: {(split_name or lat_info.get('split') or 'DATASET').upper()}"
    print("\n" + "=" * banner_len)
    print(title.center(banner_len))
    print("=" * banner_len)
    print(f"Processed: {lat_info['count']} images\n")
    print(f"{'Metric':<42s}  {'Value':>16s}  {'Unit':<12s}")
    print("-" * banner_len)
    print(f"{'Mean End-to-End Latency / Panorama':<42s}  {lat_info['mean_latency_sec']:16.3f}  {'seconds':<12s} ({lat_info['mean_latency_ms']:.1f} ms)")
    print(f"{'Inference Throughput':<42s}  {lat_info['fps']:16.2f}  {'FPS (img/s)':<12s}")
    print(f"{'  - SAM Mask Proposal Latency':<42s}  {lat_info['mean_sam_sec']:16.3f}  {'seconds/img':<12s}")
    print(f"{'  - Classifier Evaluation Latency':<42s}  {lat_info['mean_clf_sec']:16.3f}  {'seconds/img':<12s}")
    print(f"{'  - Classifier Latency / Proposal Crop':<42s}  {lat_info['mean_clf_ms_per_crop']:16.3f}  {'ms / crop':<12s}")
    print(f"{'  - Mean Proposals Evaluated / Panorama':<42s}  {lat_info['mean_proposals']:16.1f}  {'proposals':<12s}")
    print("=" * banner_len + "\n")


def benchmark_latency(mask_generator, classifier, sample_image=None, device='cuda', num_runs=5):
    """Benchmark inference latency for classifier and SAM automatic mask generator."""
    # 1. Benchmark classifier on single crop
    clf_dummy = torch.randn(1, 3, CROP_SIZE, CROP_SIZE, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = classifier(clf_dummy)
    if device == 'cuda':
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    clf_runs = 50
    for _ in range(clf_runs):
        with torch.no_grad():
            _ = classifier(clf_dummy)
    if device == 'cuda':
        torch.cuda.synchronize()
    clf_crop_ms = ((time.perf_counter() - t_start) / clf_runs) * 1000.0
    clf_fps = 1000.0 / clf_crop_ms if clf_crop_ms > 0 else 0.0

    # 2. Benchmark SAM image encoder
    sam_enc_ms = None
    try:
        sam_dummy = torch.randn(1, 3, 1024, 1024, device=device)
        sam_model = mask_generator.predictor.model if hasattr(mask_generator, 'predictor') else None
        if sam_model and hasattr(sam_model, 'image_encoder'):
            with torch.no_grad():
                _ = sam_model.image_encoder(sam_dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(num_runs):
                with torch.no_grad():
                    _ = sam_model.image_encoder(sam_dummy)
            if device == 'cuda':
                torch.cuda.synchronize()
            sam_enc_ms = ((time.perf_counter() - t0) / num_runs) * 1000.0
    except Exception:
        pass

    # 3. If sample_image is provided, run full end-to-end benchmark
    e2e_stats = None
    if sample_image is not None:
        h, w = sample_image.shape[:2]
        if device == 'cuda':
            torch.cuda.synchronize()
        t0_gen = time.perf_counter()
        masks = mask_generator.generate(sample_image)
        if device == 'cuda':
            torch.cuda.synchronize()
        sam_gen_sec = time.perf_counter() - t0_gen

        # Classification time
        t0_clf = time.perf_counter()
        valid_crops = 0
        norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((CROP_SIZE, CROP_SIZE)),
            transforms.ToTensor(),
            norm
        ])
        for m in masks:
            seg = m['segmentation']
            x0, y0, bw, bh = m['bbox']
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(w, int(x0 + bw)), min(h, int(y0 + bh))
            if x1 <= x0 or y1 <= y0:
                continue
            _ = _classify_crop(classifier, sample_image, seg, (x0, y0, x1, y1), device, transform)
            valid_crops += 1
        if device == 'cuda':
            torch.cuda.synchronize()
        clf_eval_sec = time.perf_counter() - t0_clf
        e2e_total_sec = sam_gen_sec + clf_eval_sec

        e2e_stats = {
            'image_shape': (h, w),
            'num_masks': len(masks),
            'sam_gen_sec': sam_gen_sec,
            'sam_gen_ms': sam_gen_sec * 1000.0,
            'clf_eval_sec': clf_eval_sec,
            'clf_eval_ms': clf_eval_sec * 1000.0,
            'e2e_total_sec': e2e_total_sec,
            'e2e_total_ms': e2e_total_sec * 1000.0,
            'fps': 1.0 / e2e_total_sec if e2e_total_sec > 0 else 0.0,
            'clf_ms_per_crop': (clf_eval_sec / max(valid_crops, 1)) * 1000.0,
        }

    return {
        'clf_crop_ms': clf_crop_ms,
        'clf_crops_per_sec': clf_fps,
        'sam_encoder_ms': sam_enc_ms,
        'sam_encoder_sec': sam_enc_ms / 1000.0 if sam_enc_ms else None,
        'e2e': e2e_stats,
    }


def build_classifier(num_classes=NUM_CLASSES, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = models.resnet18(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    index = build_crop_index(args.img_dir, args.mask_dir,
                              min_area=args.min_area, pad_ratio=args.pad_ratio)
    if not index:
        raise RuntimeError("No training crops found -- check img_dir/mask_dir and filenames.")

    dataset = GTCropDataset(index, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, drop_last=True)

    model = build_classifier().to(device)

    # Display classifier complexity
    p_info = count_parameters(model)
    f_info = compute_flops(model, torch.randn(1, 3, CROP_SIZE, CROP_SIZE, device=device))
    print(f"\n[model] ResNet-18 Classifier Complexity:")
    print(f"        Trainable Parameters : {p_info['trainable_M']:.3f} M ({p_info['trainable']:,})")
    print(f"        Total Parameters     : {p_info['total_M']:.3f} M ({p_info['total']:,})")
    if f_info['gflops'] is not None:
        print(f"        FLOPs (224x224)      : {f_info['gflops']:.3f} GFLOPs ({f_info['gmacs']:.3f} GMACs)\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.epochs):
        total_loss, total_correct, total_n = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total_n += imgs.size(0)
            pbar.set_postfix(loss=total_loss / total_n, acc=total_correct / total_n)
        sched.step()

    torch.save({'model_state': model.state_dict(), 'classes': CLASSES}, args.out)
    print(f"Saved classifier to {args.out}")


# --------------------------------------------------------------------------
# Inference: SAM proposals -> classifier -> painted label map
# --------------------------------------------------------------------------

def load_sam_generator(checkpoint, model_type, device,
                        points_per_side=32, points_per_batch=64, pred_iou_thresh=0.86,
                        stability_score_thresh=0.92, min_mask_region_area=200):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
    )


def load_classifier(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_classifier(pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model


def _classify_crop(model, image, seg_mask, bbox, device, transform):
    x0, y0, x1, y1 = bbox
    crop_img = image[y0:y1, x0:x1].copy()
    crop_seg = seg_mask[y0:y1, x0:x1]
    crop_img[~crop_seg] = 0
    tensor = transform(crop_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(1).item()
    return pred


def infer_on_images(img_paths, out_pred_dir, vis_dirs, mask_generator, classifier,
                    transform, device, overwrite=False, save_vis=False, split_name=None):
    """Run SAM + classifier over a list of images and write predictions and visualizations.
    Tracks and returns runtime inference latency and throughput statistics.
    """
    os.makedirs(out_pred_dir, exist_ok=True)
    if save_vis:
        for vd in vis_dirs:
            os.makedirs(vd, exist_ok=True)

    latencies_e2e = []
    latencies_sam = []
    latencies_clf = []
    proposals_counts = []

    desc = f"SAM + classify [{split_name.upper()}]" if split_name else "SAM + classify"
    pbar = tqdm(img_paths, desc=desc)
    for img_path in pbar:
        stem = _stem(img_path)
        out_pred_path = os.path.join(out_pred_dir, stem + '.png')

        # Check if already done
        if not overwrite and os.path.exists(out_pred_path):
            vis_all_exist = True
            if save_vis:
                for vd in vis_dirs:
                    if not os.path.exists(os.path.join(vd, stem + '.png')):
                        vis_all_exist = False
                        break
            if vis_all_exist:
                continue

        image = np.array(Image.open(img_path).convert('RGB'))
        h, w = image.shape[:2]

        t_img_start = time.perf_counter()
        if device == 'cuda':
            torch.cuda.synchronize()

        # 1. SAM Proposal Generation
        t0_sam = time.perf_counter()
        masks = mask_generator.generate(image)
        if device == 'cuda':
            torch.cuda.synchronize()
        t_sam = time.perf_counter() - t0_sam

        # paint largest masks first so smaller / more specific ones sit on top
        masks = sorted(masks, key=lambda m: m['area'], reverse=True)

        # 2. Crop Classification & Label Painting
        t0_clf = time.perf_counter()
        label_map = np.full((h, w), IGNORE_INDEX, dtype=np.uint8)
        num_evaluated_crops = 0
        for m in masks:
            seg = m['segmentation']  # bool (H, W)
            x0, y0, bw, bh = m['bbox']  # xywh, floats
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(w, int(x0 + bw)), min(h, int(y0 + bh))
            if x1 <= x0 or y1 <= y0:
                continue
            pred_class = _classify_crop(classifier, image, seg, (x0, y0, x1, y1), device, transform)
            label_map[seg] = pred_class
            num_evaluated_crops += 1

        if device == 'cuda':
            torch.cuda.synchronize()
        t_clf = time.perf_counter() - t0_clf
        t_e2e = time.perf_counter() - t_img_start

        latencies_e2e.append(t_e2e)
        latencies_sam.append(t_sam)
        latencies_clf.append(t_clf)
        proposals_counts.append(num_evaluated_crops)

        pbar.set_postfix({
            'lat': f"{t_e2e:.2f}s",
            'sam': f"{t_sam:.2f}s",
            'clf': f"{t_clf:.2f}s",
            'props': num_evaluated_crops,
        })

        Image.fromarray(label_map).save(out_pred_path)
        if save_vis:
            color = label_to_color(label_map)
            vis_img = Image.fromarray(color)
            for vd in vis_dirs:
                vis_img.save(os.path.join(vd, stem + '.png'))

        del masks, label_map, image
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # Aggregate latency statistics
    latency_summary = {
        'split': split_name,
        'count': len(latencies_e2e),
        'mean_latency_sec': float(np.mean(latencies_e2e)) if latencies_e2e else 0.0,
        'mean_latency_ms': float(np.mean(latencies_e2e) * 1000.0) if latencies_e2e else 0.0,
        'fps': float(1.0 / np.mean(latencies_e2e)) if latencies_e2e and np.mean(latencies_e2e) > 0 else 0.0,
        'mean_sam_sec': float(np.mean(latencies_sam)) if latencies_sam else 0.0,
        'mean_clf_sec': float(np.mean(latencies_clf)) if latencies_clf else 0.0,
        'mean_clf_ms_per_crop': float((sum(latencies_clf) / max(sum(proposals_counts), 1)) * 1000.0) if latencies_clf else 0.0,
        'mean_proposals': float(np.mean(proposals_counts)) if proposals_counts else 0.0,
        'raw_latencies_e2e': latencies_e2e,
        'raw_latencies_sam': latencies_sam,
        'raw_latencies_clf': latencies_clf,
        'raw_proposals': proposals_counts,
    }
    return latency_summary


def infer(args):
    device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Determine split tasks
    split_tasks = []
    if getattr(args, 'data_root', None):
        splits = args.splits if getattr(args, 'splits', None) else ['val', 'test']
        for s in splits:
            img_dir, gt_dir = resolve_split_dirs(args.data_root, s)
            if not img_dir:
                print(f"[warn] Could not locate image directory for split '{s}' in {args.data_root}")
                continue
            split_pred_dir = os.path.join(args.out_dir, s)
            split_vis_dirs = [
                os.path.join(split_pred_dir, 'vis'),                # preds/val/vis
                os.path.join(args.out_dir, 'vis', s),               # preds/vis/val
            ] if args.save_vis else []
            split_tasks.append({
                'split': s,
                'img_dir': img_dir,
                'gt_dir': gt_dir,
                'pred_dir': split_pred_dir,
                'vis_dirs': split_vis_dirs,
            })
    elif getattr(args, 'img_dir', None):
        img_dir = args.img_dir
        gt_dir = getattr(args, 'gt_dir', None)
        split_name = getattr(args, 'split', None)
        if not split_name:
            norm_parts = [p.lower() for p in os.path.normpath(img_dir).split(os.sep)]
            if 'val' in norm_parts:
                split_name = 'val'
            elif 'test' in norm_parts:
                split_name = 'test'

        if split_name:
            split_pred_dir = os.path.join(args.out_dir, split_name)
            split_vis_dirs = [
                os.path.join(split_pred_dir, 'vis'),
                os.path.join(args.out_dir, 'vis', split_name),
            ] if args.save_vis else []
        else:
            split_pred_dir = args.out_dir
            split_vis_dirs = [os.path.join(args.out_dir, 'vis')] if args.save_vis else []

        split_tasks.append({
            'split': split_name,
            'img_dir': img_dir,
            'gt_dir': gt_dir,
            'pred_dir': split_pred_dir,
            'vis_dirs': split_vis_dirs,
        })
    else:
        raise ValueError("Either --data_root or --img_dir must be specified.")

    if not split_tasks:
        raise RuntimeError("No valid image folders found to process.")

    # 2. Load models ONCE
    print(f"[info] Loading SAM ({args.sam_model_type}) and ResNet18 classifier on {device}...")
    mask_generator = load_sam_generator(
        args.sam_checkpoint, args.sam_model_type, device,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        min_mask_region_area=args.min_mask_region_area,
    )
    classifier = load_classifier(args.classifier, device)
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CROP_SIZE, CROP_SIZE)),
        transforms.ToTensor(),
        norm,
    ])

    # Profile Model Complexity (Parameters M and FLOPs G)
    sam_model = mask_generator.predictor.model if hasattr(mask_generator, 'predictor') else None
    complexity_info = profile_pipeline_complexity(sam_model, classifier, sam_model_type=args.sam_model_type, device=device)
    print_model_complexity_summary(complexity_info)

    # 3. Run inference split by split while tracking latency
    split_latencies = {}
    for task in split_tasks:
        s_name = task['split'] or 'dataset'
        img_paths = []
        for ext in IMG_EXTS:
            img_paths.extend(glob.glob(os.path.join(task['img_dir'], f'*{ext}')))
        img_paths = sorted(list(set(img_paths)))

        if not img_paths:
            print(f"[warn] No images found in {task['img_dir']}")
            continue

        print(f"\n[info] Running inference on split '{s_name}' ({len(img_paths)} images)...")
        print(f"       Images : {task['img_dir']}")
        print(f"       Preds  : {task['pred_dir']}")
        if args.save_vis:
            print(f"       Vis    : {task['vis_dirs'][0]}")

        lat_stats = infer_on_images(
            img_paths=img_paths,
            out_pred_dir=task['pred_dir'],
            vis_dirs=task['vis_dirs'],
            mask_generator=mask_generator,
            classifier=classifier,
            transform=transform,
            device=device,
            overwrite=getattr(args, 'overwrite', False),
            save_vis=args.save_vis,
            split_name=task['split'],
        )
        print(f"[info] Predictions for '{s_name}' written to {task['pred_dir']}")

        # If predictions were previously cached, run a quick benchmark on 1 sample image to obtain speed metrics
        if lat_stats['count'] == 0 and img_paths:
            print(f"[info] Predictions were cached. Benchmarking inference speed on sample image...")
            sample_img = np.array(Image.open(img_paths[0]).convert('RGB'))
            bench = benchmark_latency(mask_generator, classifier, sample_image=sample_img, device=device)
            if bench['e2e']:
                e2e = bench['e2e']
                lat_stats = {
                    'split': task['split'],
                    'count': len(img_paths),
                    'mean_latency_sec': e2e['e2e_total_sec'],
                    'mean_latency_ms': e2e['e2e_total_ms'],
                    'fps': e2e['fps'],
                    'mean_sam_sec': e2e['sam_gen_sec'],
                    'mean_clf_sec': e2e['clf_eval_sec'],
                    'mean_clf_ms_per_crop': e2e['clf_ms_per_crop'],
                    'mean_proposals': float(e2e['num_masks']),
                    'raw_latencies_e2e': [e2e['e2e_total_sec']],
                    'raw_latencies_sam': [e2e['sam_gen_sec']],
                    'raw_latencies_clf': [e2e['clf_eval_sec']],
                    'raw_proposals': [e2e['num_masks']],
                }

        split_latencies[s_name] = lat_stats
        print_latency_summary(lat_stats, split_name=s_name)

    # 4. Run automatic evaluation if GT directories exist
    evaluated_splits = []
    save_arg = getattr(args, 'save_metrics', None)
    save_dir = os.path.dirname(save_arg) if save_arg else ""
    base_name = os.path.basename(save_arg) if save_arg else "metrics.csv"
    name_part, ext_part = os.path.splitext(base_name)
    ext_part = ext_part or '.csv'

    for task in split_tasks:
        if task['gt_dir'] and os.path.isdir(task['gt_dir']):
            s_name = task['split']
            print(f"\n[eval] Evaluating '{s_name or 'dataset'}' against GT in {task['gt_dir']}...")

            split_save_paths = []
            if save_arg:
                # 1) Inside split pred dir (e.g. preds/val/metrics.csv)
                split_save_paths.append(os.path.join(task['pred_dir'], f"metrics{ext_part}"))
                # 2) Specified save path with split suffix (e.g. metrics_val.csv)
                if s_name:
                    split_save_paths.append(os.path.join(save_dir, f"{name_part}_{s_name}{ext_part}") if save_dir else f"{name_part}_{s_name}{ext_part}")
                    split_save_paths.append(os.path.join(args.out_dir, f"{name_part}_{s_name}{ext_part}"))
                else:
                    split_save_paths.append(save_arg)

            split_res = compute_metrics(
                pred_dir=task['pred_dir'],
                gt_dir=task['gt_dir'],
                boundary_dilation_ratio=getattr(args, 'boundary_dilation_ratio', 0.02),
                boundary_thickness=getattr(args, 'boundary_thickness', None),
                bf_threshold=getattr(args, 'bf_threshold', None),
                split_name=s_name,
                save_metrics=split_save_paths,
                efficiency_info=complexity_info,
                latency_info=split_latencies.get(s_name),
            )
            evaluated_splits.append(split_res)

    # 5. Combined summary across splits if multiple splits were evaluated
    if len(evaluated_splits) > 1:
        combined_acc = combine_accumulators([r.get('accumulators') for r in evaluated_splits])
        total_eval = sum(r.get('evaluated_images', 0) for r in evaluated_splits)
        total_imgs = sum(r.get('total_images', 0) for r in evaluated_splits)

        # Combine latencies across evaluated splits
        all_e2e = []
        all_sam = []
        all_clf = []
        all_props = []
        for r in evaluated_splits:
            lat = r.get('latency') or {}
            all_e2e.extend(lat.get('raw_latencies_e2e', []))
            all_sam.extend(lat.get('raw_latencies_sam', []))
            all_clf.extend(lat.get('raw_latencies_clf', []))
            all_props.extend(lat.get('raw_proposals', []))

        combined_lat = {
            'split': 'COMBINED',
            'count': len(all_e2e),
            'mean_latency_sec': float(np.mean(all_e2e)) if all_e2e else 0.0,
            'mean_latency_ms': float(np.mean(all_e2e) * 1000.0) if all_e2e else 0.0,
            'fps': float(1.0 / np.mean(all_e2e)) if all_e2e and np.mean(all_e2e) > 0 else 0.0,
            'mean_sam_sec': float(np.mean(all_sam)) if all_sam else 0.0,
            'mean_clf_sec': float(np.mean(all_clf)) if all_clf else 0.0,
            'mean_clf_ms_per_crop': float((sum(all_clf) / max(sum(all_props), 1)) * 1000.0) if all_clf else 0.0,
            'mean_proposals': float(np.mean(all_props)) if all_props else 0.0,
        }

        combined_save_paths = []
        summary_save_paths = []
        if save_arg:
            combined_save_paths.append(os.path.join(save_dir, f"{name_part}_combined{ext_part}") if save_dir else f"{name_part}_combined{ext_part}")
            combined_save_paths.append(os.path.join(args.out_dir, f"{name_part}_combined{ext_part}"))
            summary_save_paths.append(os.path.join(save_dir, f"{name_part}_summary{ext_part}") if save_dir else f"{name_part}_summary{ext_part}")
            summary_save_paths.append(os.path.join(args.out_dir, f"{name_part}_summary{ext_part}"))

        combined_res = summarize_accumulators(
            combined_acc, total_eval, total_imgs,
            split_name='COMBINED (ALL SPLITS)',
            save_metrics=combined_save_paths,
            efficiency_info=complexity_info,
            latency_info=combined_lat,
        )
        print_split_comparison_summary(evaluated_splits, combined_result=combined_res, save_paths=summary_save_paths)


# --------------------------------------------------------------------------
# Boundary Extraction & Metric Helpers
# --------------------------------------------------------------------------

def mask_to_boundary(mask: np.ndarray, dilation_ratio: float = 0.02, thickness: int = None) -> np.ndarray:
    """Extract boundary band of a binary mask following Bowen Cheng et al., CVPR 2021 (Boundary IoU).

    Boundary region B_d = Mask \\ (Mask eroded by radius d).
    For objects smaller than 2d, the boundary band naturally covers the full object.

    Args:
        mask: 2D binary mask (H, W), bool or uint8
        dilation_ratio: fraction of image diagonal used for boundary thickness d
        thickness: explicit pixel thickness d (overrides dilation_ratio if given)

    Returns:
        2D boolean mask of shape (H, W) representing the boundary band.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    h, w = mask.shape
    if thickness is not None and thickness > 0:
        d = int(thickness)
    else:
        diag = np.sqrt(h ** 2 + w ** 2)
        d = max(1, int(round(dilation_ratio * diag)))

    # Pad 1 px so mask regions touching image boundary do not introduce false edges
    padded = np.pad(mask.astype(np.uint8), ((1, 1), (1, 1)), mode='constant', constant_values=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 2 * d + 1))
    eroded = cv2.erode(padded, kernel, iterations=1)
    boundary = (padded - eroded)[1:-1, 1:-1].astype(bool)
    return boundary


def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """Extract 1-pixel wide inner contour/edge of a binary mask.

    Used for contour distance matching in Boundary F-score (BF-score).

    Args:
        mask: 2D binary mask (H, W), bool or uint8

    Returns:
        2D boolean mask of shape (H, W) with thin boundary contours.
    """
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    padded = np.pad(mask.astype(np.uint8), ((1, 1), (1, 1)), mode='constant', constant_values=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(padded, kernel, iterations=1)
    contour = (padded - eroded)[1:-1, 1:-1].astype(bool)
    return contour


# --------------------------------------------------------------------------
# Evaluation: Mask IoU, aAcc, mAcc, Boundary IoU, and Boundary F-Score
# --------------------------------------------------------------------------

def combine_accumulators(accums_list):
    """Aggregate a list of accumulator dicts across splits."""
    combined = {
        'inter': np.zeros(NUM_CLASSES, dtype=np.int64),
        'union': np.zeros(NUM_CLASSES, dtype=np.int64),
        'gt_count': np.zeros(NUM_CLASSES, dtype=np.int64),
        'correct_total': 0,
        'valid_total': 0,
        'b_inter': np.zeros(NUM_CLASSES, dtype=np.int64),
        'b_union': np.zeros(NUM_CLASSES, dtype=np.int64),
        'bf_pred_matched': np.zeros(NUM_CLASSES, dtype=np.int64),
        'bf_pred_total': np.zeros(NUM_CLASSES, dtype=np.int64),
        'bf_gt_matched': np.zeros(NUM_CLASSES, dtype=np.int64),
        'bf_gt_total': np.zeros(NUM_CLASSES, dtype=np.int64),
    }
    for acc in accums_list:
        if not acc:
            continue
        for key in ['inter', 'union', 'gt_count', 'b_inter', 'b_union',
                    'bf_pred_matched', 'bf_pred_total', 'bf_gt_matched', 'bf_gt_total']:
            combined[key] += acc[key]
        combined['correct_total'] += int(acc['correct_total'])
        combined['valid_total'] += int(acc['valid_total'])
    return combined


def summarize_accumulators(accums, evaluated_count, total_count, split_name=None,
                           num_classes=NUM_CLASSES, save_metrics=None,
                           efficiency_info=None, latency_info=None):
    """Compute per-class & summary metrics from accumulators, print formatted table, and save."""
    inter = accums['inter']
    union = accums['union']
    gt_count = accums['gt_count']
    correct_total = accums['correct_total']
    valid_total = accums['valid_total']
    b_inter = accums['b_inter']
    b_union = accums['b_union']
    bf_pred_matched = accums['bf_pred_matched']
    bf_pred_total = accums['bf_pred_total']
    bf_gt_matched = accums['bf_gt_matched']
    bf_gt_total = accums['bf_gt_total']

    ious, accs = [], []
    b_ious = []
    bf_scores = []
    per_class_results = {}

    title = f"VP LAB EVALUATION RESULTS: {split_name.upper()}" if split_name else "VP LAB EVALUATION RESULTS"
    header = f"{'class':>14s}  {'IoU':>6s}  {'Acc':>6s}  {'B-IoU':>6s}  {'B-Prec':>6s}  {'B-Rec':>6s}  {'BF1':>6s}"
    separator = "-" * len(header)
    banner_len = max(len(header), len(title) + 8)

    print("\n" + "=" * banner_len)
    print(title.center(banner_len))
    print("=" * banner_len)
    print(f"Evaluated: {evaluated_count} / {total_count} images (missing predictions skipped)\n")
    print(header)
    print(separator)

    for c in range(num_classes):
        has_mask_data = (union[c] > 0 or gt_count[c] > 0)
        has_boundary_data = (b_union[c] > 0 or bf_gt_total[c] > 0 or bf_pred_total[c] > 0)

        if not has_mask_data and not has_boundary_data:
            print(f"{CLASSES[c]:>14s}:  n/a (no pixels in either pred or GT)")
            per_class_results[CLASSES[c]] = {
                'iou': None, 'acc': None, 'boundary_iou': None,
                'boundary_precision': None, 'boundary_recall': None, 'boundary_f1': None
            }
            continue

        iou = float(inter[c] / union[c]) if union[c] > 0 else 0.0
        acc = float(inter[c] / gt_count[c]) if gt_count[c] > 0 else float('nan')
        ious.append(iou)
        if gt_count[c] > 0:
            accs.append(acc)

        biou = float(b_inter[c] / b_union[c]) if b_union[c] > 0 else 0.0
        b_ious.append(biou)

        b_prec = float(bf_pred_matched[c] / bf_pred_total[c]) if bf_pred_total[c] > 0 else float('nan')
        b_rec  = float(bf_gt_matched[c] / bf_gt_total[c]) if bf_gt_total[c] > 0 else float('nan')

        if not np.isnan(b_prec) and not np.isnan(b_rec) and (b_prec + b_rec) > 0:
            bf1 = float(2 * b_prec * b_rec / (b_prec + b_rec))
        elif np.isnan(b_prec) and np.isnan(b_rec):
            bf1 = float('nan')
        else:
            bf1 = 0.0

        if not np.isnan(bf1):
            bf_scores.append(bf1)

        iou_str    = f"{iou * 100:6.2f}"
        acc_str    = f"{acc * 100:6.2f}" if not np.isnan(acc) else "   n/a"
        biou_str   = f"{biou * 100:6.2f}"
        b_prec_str = f"{b_prec * 100:6.2f}" if not np.isnan(b_prec) else "   n/a"
        b_rec_str  = f"{b_rec * 100:6.2f}" if not np.isnan(b_rec) else "   n/a"
        bf1_str    = f"{bf1 * 100:6.2f}" if not np.isnan(bf1) else "   n/a"

        print(f"{CLASSES[c]:>14s}: {iou_str}  {acc_str}  {biou_str}  {b_prec_str}  {b_rec_str}  {bf1_str}")

        per_class_results[CLASSES[c]] = {
            'iou': iou,
            'acc': None if np.isnan(acc) else acc,
            'boundary_iou': biou,
            'boundary_precision': None if np.isnan(b_prec) else b_prec,
            'boundary_recall': None if np.isnan(b_rec) else b_rec,
            'boundary_f1': None if np.isnan(bf1) else bf1,
        }

    print(separator)

    miou = float(np.mean(ious)) if ious else 0.0
    macc = float(np.mean(accs)) if accs else 0.0
    aacc = float(correct_total / valid_total) if valid_total > 0 else 0.0
    mbiou = float(np.mean(b_ious)) if b_ious else 0.0
    mbf1 = float(np.mean(bf_scores)) if bf_scores else 0.0

    split_lbl = f" ({split_name.upper()})" if split_name else ""
    print(f"Summary Metrics{split_lbl} (ignore_index=255):")
    print(f"  aAcc   (Overall Pixel Accuracy):     {aacc * 100:6.2f}%")
    print(f"  mAcc   (Mean Class Accuracy):        {macc * 100:6.2f}%")
    print(f"  mIoU   (Mean Mask IoU):              {miou * 100:6.2f}%")
    print(f"  mBIoU  (Mean Boundary IoU):          {mbiou * 100:6.2f}%")
    print(f"  mBF1   (Mean Boundary F1 Score):     {mbf1 * 100:6.2f}%")

    if efficiency_info:
        p_trn = efficiency_info['pipeline_total']['trainable_params_M']
        p_tot = efficiency_info['pipeline_total']['total_params_M']
        clf_flp = efficiency_info['classifier']['flops']['gflops']
        sam_flp = efficiency_info['sam']['image_encoder_flops']['gflops']
        print(f"\nEfficiency & Model Complexity:")
        print(f"  Trainable Parameters:                {p_trn:6.3f} M")
        print(f"  Total Pipeline Parameters:           {p_tot:6.3f} M")
        if clf_flp is not None:
            print(f"  ResNet-18 Classifier FLOPs:          {clf_flp:6.3f} GFLOPs")
        if sam_flp is not None:
            print(f"  SAM Image Encoder FLOPs:             {sam_flp:6.2f} GFLOPs")

    if latency_info and latency_info.get('count', 0) > 0:
        print(f"\nInference Latency & Speed:")
        print(f"  Mean Latency / Panorama:             {latency_info['mean_latency_sec']:6.3f} s ({latency_info['mean_latency_ms']:.1f} ms)")
        print(f"  Throughput (FPS):                    {latency_info['fps']:6.2f} images/sec")
        print(f"  SAM Mask Proposal Latency:           {latency_info['mean_sam_sec']:6.3f} s")
        print(f"  Classifier Latency / Crop:           {latency_info['mean_clf_ms_per_crop']:6.3f} ms")
        print(f"  Mean Proposals / Panorama:           {latency_info['mean_proposals']:6.1f} proposals")

    print("=" * banner_len + "\n")

    metrics_dict = {
        'split': split_name,
        'evaluated_images': evaluated_count,
        'total_images': total_count,
        'aacc': aacc,
        'macc': macc,
        'miou': miou,
        'mbiou': mbiou,
        'mbf1': mbf1,
        'per_class': per_class_results,
        'accumulators': accums,
        'efficiency': efficiency_info,
        'latency': latency_info,
    }

    if save_metrics:
        paths = [save_metrics] if isinstance(save_metrics, str) else list(save_metrics)
        for sp in paths:
            if not sp:
                continue
            out_ext = os.path.splitext(sp)[1].lower()
            os.makedirs(os.path.dirname(os.path.abspath(sp)), exist_ok=True)
            if out_ext == '.json':
                clean_dict = {k: v for k, v in metrics_dict.items() if k != 'accumulators'}
                with open(sp, 'w') as f:
                    json.dump(clean_dict, f, indent=2)
                print(f"Saved {split_name or ''} metrics JSON to {sp}")
            else:
                with open(sp, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['class', 'iou', 'acc', 'boundary_iou', 'boundary_precision', 'boundary_recall', 'boundary_f1'])
                    for c_name, vals in per_class_results.items():
                        writer.writerow([
                            c_name,
                            f"{vals['iou']*100:.2f}" if vals['iou'] is not None else "n/a",
                            f"{vals['acc']*100:.2f}" if vals['acc'] is not None else "n/a",
                            f"{vals['boundary_iou']*100:.2f}" if vals['boundary_iou'] is not None else "n/a",
                            f"{vals['boundary_precision']*100:.2f}" if vals['boundary_precision'] is not None else "n/a",
                            f"{vals['boundary_recall']*100:.2f}" if vals['boundary_recall'] is not None else "n/a",
                            f"{vals['boundary_f1']*100:.2f}" if vals['boundary_f1'] is not None else "n/a",
                        ])
                    writer.writerow([])
                    writer.writerow(['summary_metric', 'value'])
                    writer.writerow(['aAcc (%)', f"{aacc * 100:.2f}"])
                    writer.writerow(['mAcc (%)', f"{macc * 100:.2f}"])
                    writer.writerow(['mIoU (%)', f"{miou * 100:.2f}"])
                    writer.writerow(['mBIoU (%)', f"{mbiou * 100:.2f}"])
                    writer.writerow(['mBF1 (%)', f"{mbf1 * 100:.2f}"])
                    if efficiency_info:
                        writer.writerow(['trainable_parameters_M', f"{efficiency_info['pipeline_total']['trainable_params_M']:.3f}"])
                        writer.writerow(['total_parameters_M', f"{efficiency_info['pipeline_total']['total_params_M']:.3f}"])
                        writer.writerow(['classifier_trainable_params_M', f"{efficiency_info['classifier']['params']['trainable_M']:.3f}"])
                        if efficiency_info['classifier']['flops']['gflops'] is not None:
                            writer.writerow(['classifier_flops_G', f"{efficiency_info['classifier']['flops']['gflops']:.3f}"])
                        if efficiency_info['sam']['image_encoder_flops']['gflops'] is not None:
                            writer.writerow(['sam_image_encoder_flops_G', f"{efficiency_info['sam']['image_encoder_flops']['gflops']:.3f}"])
                    if latency_info and latency_info.get('count', 0) > 0:
                        writer.writerow(['mean_latency_sec', f"{latency_info.get('mean_latency_sec', 0.0):.3f}"])
                        writer.writerow(['mean_latency_ms', f"{latency_info.get('mean_latency_ms', 0.0):.2f}"])
                        writer.writerow(['inference_fps', f"{latency_info.get('fps', 0.0):.3f}"])
                        writer.writerow(['sam_proposal_sec', f"{latency_info.get('mean_sam_sec', 0.0):.3f}"])
                        writer.writerow(['classifier_eval_sec', f"{latency_info.get('mean_clf_sec', 0.0):.3f}"])
                        writer.writerow(['classifier_per_crop_ms', f"{latency_info.get('mean_clf_ms_per_crop', 0.0):.3f}"])
                        writer.writerow(['mean_proposals_per_image', f"{latency_info.get('mean_proposals', 0.0):.1f}"])
                print(f"Saved {split_name or ''} metrics CSV to {sp}")

    return metrics_dict


def print_split_comparison_summary(split_results, combined_result=None, save_paths=None):
    """Prints and saves a side-by-side comparison table of summary metrics across splits."""
    if not split_results:
        return

    cols = []
    for res in split_results:
        s_name = (res.get('split') or 'split').upper()
        cols.append((f"{s_name} ({res.get('evaluated_images', 0)} imgs)", res))

    if combined_result:
        cols.append((f"COMBINED ({combined_result.get('evaluated_images', 0)} imgs)", combined_result))

    header = f"{'Metric':<32s}" + "".join([f"{col_name:>24s}" for col_name, _ in cols])
    separator = "-" * len(header)
    banner_len = len(header)

    print("=" * banner_len)
    print("VP LAB SPLIT COMPARISON SUMMARY".center(banner_len))
    print("=" * banner_len)
    print(header)
    print(separator)

    metrics_rows = [
        ('aAcc  (Overall Pixel Acc)', lambda r: f"{r.get('aacc', 0.0) * 100:>23.2f}%", 'aacc'),
        ('mAcc  (Mean Class Acc)', lambda r: f"{r.get('macc', 0.0) * 100:>23.2f}%", 'macc'),
        ('mIoU  (Mean Mask IoU)', lambda r: f"{r.get('miou', 0.0) * 100:>23.2f}%", 'miou'),
        ('mBIoU (Mean Boundary IoU)', lambda r: f"{r.get('mbiou', 0.0) * 100:>23.2f}%", 'mbiou'),
        ('mBF1  (Mean Boundary F1)', lambda r: f"{r.get('mbf1', 0.0) * 100:>23.2f}%", 'mbf1'),
    ]

    has_lat = any(r.get('latency') and r.get('latency', {}).get('count', 0) > 0 for _, r in cols)
    if has_lat:
        metrics_rows.extend([
            ('Mean Latency / Img (s)', lambda r: f"{r.get('latency', {}).get('mean_latency_sec', 0.0):>22.3f} s" if r.get('latency') else f"{'n/a':>24s}", 'mean_latency_sec'),
            ('Throughput (FPS)', lambda r: f"{r.get('latency', {}).get('fps', 0.0):>20.2f} fps" if r.get('latency') else f"{'n/a':>24s}", 'fps'),
            ('SAM Proposal Time (s)', lambda r: f"{r.get('latency', {}).get('mean_sam_sec', 0.0):>22.3f} s" if r.get('latency') else f"{'n/a':>24s}", 'mean_sam_sec'),
            ('Classifier / Crop (ms)', lambda r: f"{r.get('latency', {}).get('mean_clf_ms_per_crop', 0.0):>21.2f} ms" if r.get('latency') else f"{'n/a':>24s}", 'mean_clf_ms_per_crop'),
        ])

    for label, fn, _ in metrics_rows:
        row = f"{label:<32s}"
        for _, res in cols:
            row += fn(res)
        print(row)

    print("=" * banner_len + "\n")

    if save_paths:
        paths = [save_paths] if isinstance(save_paths, str) else list(save_paths)
        for sp in paths:
            if not sp:
                continue
            os.makedirs(os.path.dirname(os.path.abspath(sp)), exist_ok=True)
            with open(sp, 'w', newline='') as f:
                writer = csv.writer(f)
                header_row = ['metric'] + [col_name for col_name, _ in cols]
                writer.writerow(header_row)
                for label, _, key in metrics_rows:
                    vals = []
                    for _, res in cols:
                        if key in ['aacc', 'macc', 'miou', 'mbiou', 'mbf1']:
                            vals.append(f"{res.get(key, 0.0)*100:.2f}")
                        elif key in ['mean_latency_sec', 'mean_sam_sec']:
                            vals.append(f"{res.get('latency', {}).get(key, 0.0):.3f}" if res.get('latency') else "n/a")
                        elif key in ['fps', 'mean_clf_ms_per_crop']:
                            vals.append(f"{res.get('latency', {}).get(key, 0.0):.2f}" if res.get('latency') else "n/a")
                        else:
                            vals.append(f"{res.get(key, 'n/a')}")
                    writer.writerow([label] + vals)
            print(f"Saved split comparison summary to {sp}")


def compute_metrics(pred_dir, gt_dir, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX,
                    boundary_dilation_ratio=0.02, boundary_thickness=None, bf_threshold=None,
                    split_name=None, save_metrics=None, efficiency_info=None, latency_info=None):
    """Compute comprehensive semantic segmentation metrics ignoring pixel value 255:
    1. Mask IoU, Mean IoU (mIoU), Overall Pixel Accuracy (aAcc), Mean Class Accuracy (mAcc)
    2. Boundary IoU (BIoU) and Mean Boundary IoU (mBIoU) (Cheng et al., CVPR 2021)
    3. Boundary Precision, Boundary Recall, and Boundary F-score (BF1 / BF-score) (Csurka et al., BMVC 2013)
    """
    inter = np.zeros(num_classes, dtype=np.int64)
    union = np.zeros(num_classes, dtype=np.int64)
    gt_count = np.zeros(num_classes, dtype=np.int64)
    correct_total = 0
    valid_total = 0

    b_inter = np.zeros(num_classes, dtype=np.int64)
    b_union = np.zeros(num_classes, dtype=np.int64)

    bf_pred_matched = np.zeros(num_classes, dtype=np.int64)
    bf_pred_total   = np.zeros(num_classes, dtype=np.int64)
    bf_gt_matched   = np.zeros(num_classes, dtype=np.int64)
    bf_gt_total     = np.zeros(num_classes, dtype=np.int64)

    gt_paths = sorted(glob.glob(os.path.join(gt_dir, '*.png')))
    if not gt_paths:
        raise RuntimeError(f"No GT masks found in {gt_dir}")

    evaluated = 0
    desc = f"evaluating [{split_name.upper()}]" if split_name else "evaluating"
    for gt_path in tqdm(gt_paths, desc=desc):
        stem = _stem(gt_path)
        pred_path = os.path.join(pred_dir, stem + '.png')
        if not os.path.exists(pred_path):
            continue

        gt = np.array(Image.open(gt_path))
        pred = np.array(Image.open(pred_path))
        if gt.ndim == 3:
            gt = gt[..., 0]
        if pred.ndim == 3:
            pred = pred[..., 0]

        valid_2d = (gt != ignore_index)
        gt_v, pred_v = gt[valid_2d], pred[valid_2d]

        correct_total += int((gt_v == pred_v).sum())
        valid_total += int(valid_2d.sum())

        h, w = gt.shape
        diag = np.sqrt(h ** 2 + w ** 2)

        # Distance tolerance theta for Boundary F-score
        if bf_threshold is not None and bf_threshold > 0:
            theta = int(bf_threshold)
        else:
            # Standard tolerance: 0.75% of image diagonal
            theta = max(1, int(round(0.0075 * diag)))
        kernel_theta = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * theta + 1, 2 * theta + 1))

        for c in range(num_classes):
            gt_c_2d = (gt == c)
            pred_c_2d = (pred == c)

            # --- Standard Mask IoU & Accuracy ---
            gt_c_v = (gt_v == c)
            pred_c_v = (pred_v == c)
            inter[c] += np.logical_and(gt_c_v, pred_c_v).sum()
            union[c] += np.logical_or(gt_c_v, pred_c_v).sum()
            gt_count[c] += gt_c_v.sum()

            # --- Boundary IoU ---
            gt_b = mask_to_boundary(gt_c_2d, dilation_ratio=boundary_dilation_ratio, thickness=boundary_thickness) & valid_2d
            pred_b = mask_to_boundary(pred_c_2d, dilation_ratio=boundary_dilation_ratio, thickness=boundary_thickness) & valid_2d
            b_inter[c] += np.logical_and(gt_b, pred_b).sum()
            b_union[c] += np.logical_or(gt_b, pred_b).sum()

            # --- Boundary F-score (BF1) ---
            gt_cnt = mask_to_contour(gt_c_2d) & valid_2d
            pred_cnt = mask_to_contour(pred_c_2d) & valid_2d

            n_p = int(pred_cnt.sum())
            n_g = int(gt_cnt.sum())
            bf_pred_total[c] += n_p
            bf_gt_total[c] += n_g

            if n_p > 0 and n_g > 0:
                gt_dil = cv2.dilate(gt_cnt.astype(np.uint8), kernel_theta, iterations=1) > 0
                pred_dil = cv2.dilate(pred_cnt.astype(np.uint8), kernel_theta, iterations=1) > 0
                bf_pred_matched[c] += int((pred_cnt & gt_dil).sum())
                bf_gt_matched[c] += int((gt_cnt & pred_dil).sum())

        evaluated += 1

    accums = {
        'inter': inter,
        'union': union,
        'gt_count': gt_count,
        'correct_total': correct_total,
        'valid_total': valid_total,
        'b_inter': b_inter,
        'b_union': b_union,
        'bf_pred_matched': bf_pred_matched,
        'bf_pred_total': bf_pred_total,
        'bf_gt_matched': bf_gt_matched,
        'bf_gt_total': bf_gt_total,
    }

    return summarize_accumulators(
        accums=accums,
        evaluated_count=evaluated,
        total_count=len(gt_paths),
        split_name=split_name,
        num_classes=num_classes,
        save_metrics=save_metrics,
        efficiency_info=efficiency_info,
        latency_info=latency_info,
    )


def evaluate(args):
    """Standalone evaluation across splits or a single folder."""
    split_configs = []
    if getattr(args, 'data_root', None):
        splits = args.splits if getattr(args, 'splits', None) else ['val', 'test']
        for s in splits:
            _, gt_dir = resolve_split_dirs(args.data_root, s)
            split_pred_dir = os.path.join(args.pred_dir, s)
            if not os.path.isdir(split_pred_dir):
                split_pred_dir = args.pred_dir
            if not gt_dir:
                print(f"[warn] Could not locate GT masks for split '{s}' in {args.data_root}")
                continue
            split_configs.append({
                'split': s,
                'pred_dir': split_pred_dir,
                'gt_dir': gt_dir,
            })
    elif getattr(args, 'pred_dir', None) and getattr(args, 'gt_dir', None):
        split_name = getattr(args, 'split', None)
        if not split_name:
            for cand in ['val', 'test']:
                if cand in os.path.normpath(args.pred_dir).lower().split(os.sep) or \
                   cand in os.path.normpath(args.gt_dir).lower().split(os.sep):
                    split_name = cand
                    break
        split_configs.append({
            'split': split_name,
            'pred_dir': args.pred_dir,
            'gt_dir': args.gt_dir,
        })
    else:
        raise ValueError("Either --data_root (with --pred_dir) or both --pred_dir and --gt_dir must be specified.")

    save_arg = getattr(args, 'save_metrics', None)
    save_dir = os.path.dirname(save_arg) if save_arg else ""
    base_name = os.path.basename(save_arg) if save_arg else "metrics.csv"
    name_part, ext_part = os.path.splitext(base_name)
    ext_part = ext_part or '.csv'

    split_results = []
    for cfg in split_configs:
        save_paths = []
        if save_arg:
            if cfg['split']:
                save_paths.append(os.path.join(save_dir, f"{name_part}_{cfg['split']}{ext_part}") if save_dir else f"{name_part}_{cfg['split']}{ext_part}")
                save_paths.append(os.path.join(cfg['pred_dir'], f"metrics{ext_part}"))
            else:
                save_paths.append(save_arg)

        res = compute_metrics(
            pred_dir=cfg['pred_dir'],
            gt_dir=cfg['gt_dir'],
            boundary_dilation_ratio=args.boundary_dilation_ratio,
            boundary_thickness=args.boundary_thickness,
            bf_threshold=args.bf_threshold,
            split_name=cfg['split'],
            save_metrics=save_paths,
        )
        split_results.append(res)

    if len(split_results) > 1:
        combined_acc = combine_accumulators([r.get('accumulators') for r in split_results])
        total_eval = sum(r.get('evaluated_images', 0) for r in split_results)
        total_imgs = sum(r.get('total_images', 0) for r in split_results)

        combined_save_paths = []
        summary_save_paths = []
        if save_arg:
            combined_save_paths.append(os.path.join(save_dir, f"{name_part}_combined{ext_part}") if save_dir else f"{name_part}_combined{ext_part}")
            summary_save_paths.append(os.path.join(save_dir, f"{name_part}_summary{ext_part}") if save_dir else f"{name_part}_summary{ext_part}")

        comb_res = summarize_accumulators(
            combined_acc, total_eval, total_imgs,
            split_name='COMBINED (ALL SPLITS)',
            save_metrics=combined_save_paths
        )
        print_split_comparison_summary(split_results, combined_result=comb_res, save_paths=summary_save_paths)


# --------------------------------------------------------------------------
# Standalone Profiling & Benchmarking Command
# --------------------------------------------------------------------------

def benchmark_cmd(args):
    """Standalone benchmark command to measure and display trainable parameters (M), FLOPs (G), and latency."""
    device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[benchmark] Profiling SAM ({args.sam_model_type}) + ResNet-18 Classifier on device: {device}...")

    # Load classifier
    if os.path.exists(args.classifier):
        classifier = load_classifier(args.classifier, device)
    else:
        print(f"[warn] Classifier checkpoint '{args.classifier}' not found. Using ResNet-18 architecture for profiling.")
        classifier = build_classifier(pretrained=False).to(device).eval()

    # Load SAM generator & model
    mask_gen = load_sam_generator(
        args.sam_checkpoint, args.sam_model_type, device,
        points_per_side=getattr(args, 'points_per_side', 32),
        points_per_batch=getattr(args, 'points_per_batch', 64),
        min_mask_region_area=getattr(args, 'min_mask_region_area', 200),
    )
    sam_model = mask_gen.predictor.model if hasattr(mask_gen, 'predictor') else None

    # 1. Model Complexity (Parameters & FLOPs)
    comp = profile_pipeline_complexity(sam_model, classifier, sam_model_type=args.sam_model_type, device=device)
    print_model_complexity_summary(comp)

    # 2. Latency Benchmarking
    sample_img = None
    if getattr(args, 'img', None) and os.path.exists(args.img):
        sample_img = np.array(Image.open(args.img).convert('RGB'))
        print(f"[benchmark] Benchmarking latency using image: {args.img} (size: {sample_img.shape[:2]})...")
    else:
        candidates = [
            "../Panorama Segmentation DataSet/val/img/14.png",
            "../Panorama Segmentation DataSet/test/img/26.png",
            "../Panoramic Segmentation Dataset/val/img/14.png",
            "../../Panorama Segmentation DataSet/val/img/14.png",
            "../Panorama Segmentation DataSet/val/img/15.png",
            "../Panorama Segmentation DataSet/val/img/5.png",
        ]
        for c in candidates:
            if os.path.exists(c):
                sample_img = np.array(Image.open(c).convert('RGB'))
                print(f"[benchmark] Auto-selected sample dataset image: {c} (size: {sample_img.shape[:2]})...")
                break

    lat_res = benchmark_latency(mask_gen, classifier, sample_image=sample_img, device=device)

    # Print Latency Results
    banner_len = 86
    print("=" * banner_len)
    print("INFERENCE LATENCY BENCHMARK RESULTS".center(banner_len))
    print("=" * banner_len)
    print(f"{'Component / Operation':<46s}  {'Latency':>16s}  {'Throughput':>18s}")
    print("-" * banner_len)
    print(f"{'ResNet-18 Classifier (1 crop, 224x224)':<46s}  {lat_res['clf_crop_ms']:13.3f} ms  {lat_res['clf_crops_per_sec']:13.1f} crops/s")
    if lat_res['sam_encoder_ms'] is not None:
        print(f"{f'SAM Image Encoder ({args.sam_model_type}, 1024x1024)':<46s}  {lat_res['sam_encoder_sec']:13.3f}  s  {1.0/lat_res['sam_encoder_sec']:13.2f} imgs/s")
    if lat_res['e2e']:
        e2e = lat_res['e2e']
        print("-" * banner_len)
        print(f"Full Panorama Pipeline Benchmark on {e2e['image_shape']} image ({e2e['num_masks']} mask proposals):")
        print(f"{'  - SAM Mask Proposal Generation':<46s}  {e2e['sam_gen_sec']:13.3f}  s  ({e2e['sam_gen_ms']:.1f} ms)")
        print(f"{'  - Classifier Evaluation on All Proposals':<46s}  {e2e['clf_eval_sec']:13.3f}  s  ({e2e['clf_eval_ms']:.1f} ms)")
        print(f"{'  - Classifier Latency / Proposal Crop':<46s}  {e2e['clf_ms_per_crop']:13.3f} ms")
        print(f"{'  - End-to-End Image Latency':<46s}  {e2e['e2e_total_sec']:13.3f}  s  ({e2e['e2e_total_ms']:.1f} ms)")
        print(f"{'  - Pipeline Throughput (FPS)':<46s}  {e2e['fps']:13.2f} FPS")
    print("=" * banner_len + "\n")

    if getattr(args, 'out', None):
        out_dict = {
            'complexity': comp,
            'latency': lat_res,
        }
        with open(args.out, 'w') as f:
            json.dump(out_dict, f, indent=2)
        print(f"Saved benchmark results to {args.out}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    sam_default = '../sam_vit_h_4b8939.pth' if os.path.exists('../sam_vit_h_4b8939.pth') else (
        '../sam_vit_b_01ec64.pth' if os.path.exists('../sam_vit_b_01ec64.pth') else None
    )

    # 1) Train
    pt = sub.add_parser('train', help='Train the CNN classifier on GT crops')
    pt.add_argument('--img_dir', required=True, help='Folder containing training images')
    pt.add_argument('--mask_dir', required=True, help='Folder containing training GT masks')
    pt.add_argument('--out', default='classifier.pth')
    pt.add_argument('--epochs', type=int, default=20)
    pt.add_argument('--batch_size', type=int, default=32)
    pt.add_argument('--lr', type=float, default=3e-4)
    pt.add_argument('--num_workers', type=int, default=4)
    pt.add_argument('--min_area', type=int, default=150)
    pt.add_argument('--pad_ratio', type=float, default=0.15)
    pt.set_defaults(func=train)

    # 2) Infer
    pi = sub.add_parser('infer', help='Run SAM + classifier over dataset splits or image folder')
    pi.add_argument('--data_root', default=None, help='Dataset root containing split folders (val/ and test/)')
    pi.add_argument('--splits', nargs='+', default=['val', 'test'], help='Splits to process when --data_root is used (default: val test)')
    pi.add_argument('--img_dir', default=None, help='Single image folder (alternative to --data_root)')
    pi.add_argument('--gt_dir', default=None, help='Optional GT masks directory for automatic evaluation')
    pi.add_argument('--split', default=None, help='Explicit split name (val or test) when using --img_dir')
    pi.add_argument('--sam_checkpoint', default=sam_default, required=sam_default is None, help='Path to SAM model checkpoint (.pth)')
    pi.add_argument('--sam_model_type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    pi.add_argument('--classifier', default='classifier.pth', help='Path to trained classifier checkpoint (.pth)')
    pi.add_argument('--out_dir', default='preds', help='Output predictions directory (default: preds)')
    pi.add_argument('--save_vis', action='store_true', help='Also save colored segmentation visualizations')
    pi.add_argument('--points_per_side', type=int, default=32, help='Lower (e.g. 16) to reduce GPU memory use')
    pi.add_argument('--points_per_batch', type=int, default=64, help='Lower (e.g. 16 or 32) to reduce GPU memory use')
    pi.add_argument('--min_mask_region_area', type=int, default=200, help='Filter small disconnected mask regions < N pixels (default: 200)')
    pi.add_argument('--overwrite', action='store_true', help='Recompute and overwrite existing prediction files')
    pi.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda', 'cpu'], help='Device to run inference on (default: cuda)')
    pi.add_argument('--boundary_dilation_ratio', type=float, default=0.02, help='Boundary thickness ratio for Boundary IoU (default: 0.02)')
    pi.add_argument('--boundary_thickness', type=int, default=None, help='Explicit boundary thickness d in pixels (overrides ratio)')
    pi.add_argument('--bf_threshold', type=int, default=None, help='Boundary F-score distance threshold theta in pixels')
    pi.add_argument('--save_metrics', default='metrics.csv', help='Path or prefix to save metrics (.csv or .json, default: metrics.csv)')
    pi.set_defaults(func=infer)

    # 3) Eval
    pe = sub.add_parser('eval', help='Compute mIoU, Boundary IoU, and Boundary F-score (255 ignored)')
    pe.add_argument('--data_root', default=None, help='Dataset root containing split folders (val/ and test/)')
    pe.add_argument('--splits', nargs='+', default=['val', 'test'], help='Splits to evaluate when --data_root is used (default: val test)')
    pe.add_argument('--pred_dir', default='preds', help='Predictions root directory or single split directory')
    pe.add_argument('--gt_dir', default=None, help='Ground truth masks directory (if not using --data_root)')
    pe.add_argument('--split', default=None, help='Explicit split name (val or test)')
    pe.add_argument('--boundary_dilation_ratio', type=float, default=0.02, help='Boundary thickness ratio for Boundary IoU (default: 0.02)')
    pe.add_argument('--boundary_thickness', type=int, default=None, help='Explicit boundary thickness d in pixels (overrides ratio)')
    pe.add_argument('--bf_threshold', type=int, default=None, help='Boundary F-score distance threshold theta in pixels')
    pe.add_argument('--save_metrics', default='metrics.csv', help='Path or prefix to save metrics (.csv or .json, default: metrics.csv)')
    pe.set_defaults(func=evaluate)

    # 4) Benchmark
    pb = sub.add_parser('benchmark', help='Profile model parameters (M), FLOPs (G), and latency')
    pb.add_argument('--sam_checkpoint', default=sam_default, required=sam_default is None, help='Path to SAM model checkpoint (.pth)')
    pb.add_argument('--sam_model_type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'])
    pb.add_argument('--classifier', default='classifier.pth', help='Path to trained classifier checkpoint (.pth)')
    pb.add_argument('--img', default=None, help='Optional path to panoramic image for end-to-end latency measurement')
    pb.add_argument('--points_per_side', type=int, default=32, help='Points per side for SAM grid')
    pb.add_argument('--points_per_batch', type=int, default=64, help='Points per batch for SAM grid')
    pb.add_argument('--min_mask_region_area', type=int, default=200, help='Filter small disconnected mask regions')
    pb.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cuda', 'cpu'], help='Device to benchmark on')
    pb.add_argument('--out', default='efficiency_benchmark.json', help='Save benchmark details to JSON (default: efficiency_benchmark.json)')
    pb.set_defaults(func=benchmark_cmd)

    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

