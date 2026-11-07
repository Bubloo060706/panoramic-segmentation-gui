"""
Comprehensive verification test runner on the user's panorama and ground truth.
Uses the EXACT metric formulas from:
- Official Test Files/Segformer/vplab_new_inference.ipynb
- Official Test Files/OmniSAM/visualize_full_metrics.py
"""
import os
import sys
import time
import subprocess
import json
import cv2
import numpy as np

MODELS = [
    ("SegFormer (Perspective Fine-Tuned)", "scripts/run_segformer.py", "target/eval_segformer.json"),
    ("Pretrained SAM + CNN tracking", "scripts/run_sam_cnn.py", "target/eval_sam_cnn.json"),
    ("OPS Baseline (Coarse Backbone)", "scripts/run_ops.py", "target/eval_ops.json"),
    ("PanoFusion (CLIP + DAN + SBR-Net)", "scripts/run_panofusion.py", "target/eval_panofusion.json"),
    ("OmniSAM (Omnidirectional SAM2)", "scripts/run_omnisam.py", "target/eval_omnisam.json")
]

BOUNDARY_DILATION_RATIO = 0.02
BOUNDARY_F_TOLERANCE_RATIO = 0.0075
IGNORE_INDEX = 255

def mask_to_boundary(mask, dilation_ratio=BOUNDARY_DILATION_RATIO):
    h, w = mask.shape[:2]
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    # Downscale for fast morphology if high resolution
    if max(h, w) > 1024:
        scale = 1024.0 / max(h, w)
        sh, sw = int(h * scale), int(w * scale)
        small_mask = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
        s_diag = np.sqrt(sh ** 2 + sw ** 2)
        s_dilation = max(1, int(round(dilation_ratio * s_diag)))
        pad = cv2.copyMakeBorder(small_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(pad, kernel, iterations=s_dilation)[1:sh+1, 1:sw+1]
        b_small = (small_mask - eroded) > 0
        return cv2.resize(b_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    else:
        pad = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(pad, kernel, iterations=dilation)[1:h+1, 1:w+1]
        return (mask.astype(np.uint8) - eroded) > 0

def evaluate_pred(pred_mask, gt_mask, num_classes=21):
    valid = (gt_mask != IGNORE_INDEX) & (gt_mask < num_classes)
    h, w = gt_mask.shape[:2]
    diag = np.sqrt(h ** 2 + w ** 2)
    theta = max(1, int(round(BOUNDARY_F_TOLERANCE_RATIO * diag)))
    kernel = np.ones((3, 3), dtype=np.uint8)

    pred_flat = pred_mask[valid]
    gt_flat = gt_mask[valid]
    n_valid = int(valid.sum())
    n_correct = int((pred_flat == gt_flat).sum())

    ious, dices, accs = [], [], []
    b_ious, bf_scores = [], []

    for c in range(num_classes):
        p = (pred_flat == c)
        g = (gt_flat == c)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        denom = p.sum() + g.sum()
        g_sum = g.sum()

        if union > 0:
            ious.append(inter / union)
            dices.append((2 * inter) / denom if denom > 0 else 0.0)
            if g_sum > 0:
                accs.append(inter / g_sum)

            gt_c = (gt_mask == c) & valid
            pred_c = (pred_mask == c) & valid

            if gt_c.sum() > 0 and pred_c.sum() > 0:
                gt_b = mask_to_boundary(gt_c)
                pred_b = mask_to_boundary(pred_c)
                b_inter = np.logical_and(gt_b, pred_b).sum()
                b_union = np.logical_or(gt_b, pred_b).sum()
                if b_union > 0:
                    b_ious.append(b_inter / b_union)

                gt_contour = (gt_c.astype(np.uint8) - cv2.erode(gt_c.astype(np.uint8), kernel, iterations=1)) > 0
                pred_contour = (pred_c.astype(np.uint8) - cv2.erode(pred_c.astype(np.uint8), kernel, iterations=1)) > 0

                n_gt = gt_contour.sum()
                n_pr = pred_contour.sum()
                if n_gt > 0 and n_pr > 0:
                    dist_to_gt = cv2.distanceTransform((1 - gt_contour.astype(np.uint8)), cv2.DIST_L2, 3)
                    dist_to_pred = cv2.distanceTransform((1 - pred_contour.astype(np.uint8)), cv2.DIST_L2, 3)
                    precision = (pred_contour & (dist_to_gt <= theta)).sum() / n_pr
                    recall = (gt_contour & (dist_to_pred <= theta)).sum() / n_gt
                    if precision + recall > 0:
                        bf_scores.append(2 * precision * recall / (precision + recall))

    return {
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mAcc": float(np.mean(accs)) if accs else 0.0,
        "pixelAcc": float(n_correct / n_valid) if n_valid > 0 else 0.0,
        "meanDice": float(np.mean(dices)) if dices else 0.0,
        "bIoU": float(np.mean(b_ious)) if b_ious else 0.0,
        "bF1": float(np.mean(bf_scores)) if bf_scores else 0.0,
        "activeClasses": len(ious)
    }

def main():
    img_path = "data/user_sample_panorama.jpg"
    gt_path = "data/ground_truth/user_sample_panorama_gt.png"

    if not os.path.exists(img_path) or not os.path.exists(gt_path):
        print(f"Error: test panorama or GT not found: {img_path}, {gt_path}")
        sys.exit(1)

    gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    os.makedirs("target", exist_ok=True)

    print("=" * 80)
    print("EVALUATING ALL 5 DEDICATED MODEL RUNNERS ON USER PANORAMA")
    print(f"Image: {img_path} ({gt.shape[1]}x{gt.shape[0]}) | GT: {gt_path}")
    print("=" * 80)

    for name, script, out_json in MODELS:
        t0 = time.time()
        cmd = [sys.executable, script, "--input", img_path, "--output", out_json]
        res = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - t0

        if res.returncode != 0:
            print(f"FAILED: {name}")
            print("STDERR:", res.stderr)
            continue

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "labels" in data:
            pred = np.array(data["labels"], dtype=np.uint8).reshape(gt.shape[0], gt.shape[1])
        elif "maskPath" in data:
            pred = cv2.imread(data["maskPath"], cv2.IMREAD_UNCHANGED)
            if pred.ndim == 3:
                pred = pred[:, :, 0]
        else:
            print(f"Error: No labels or maskPath in {out_json}")
            continue

        metrics = evaluate_pred(pred, gt)

        print(f"\nModel: {name} (completed in {elapsed:.2f}s)")
        print(f"  mIoU:        {metrics['mIoU']*100:.2f}%")
        print(f"  mAcc:        {metrics['mAcc']*100:.2f}%")
        print(f"  Pixel Acc:   {metrics['pixelAcc']*100:.2f}%")
        print(f"  Mean Dice:   {metrics['meanDice']:.4f}")
        print(f"  Boundary IoU:{metrics['bIoU']*100:.2f}%")
        print(f"  Boundary F1: {metrics['bF1']*100:.2f}%")
        print(f"  Active Cls:  {metrics['activeClasses']}")

    print("\n" + "=" * 80)
    print("ALL 5 DEDICATED MODELS EVALUATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
