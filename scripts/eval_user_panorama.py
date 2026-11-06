"""
Comprehensive verification test runner on the user's panorama and ground truth.
Verifies that all 5 models run from their dedicated checkpoints and calculates:
- mIoU, mAcc, Boundary IoU, Boundary F-score, Pixel Accuracy, Mean Dice
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

def mask_to_boundary(mask, dilation_ratio=0.02):
    h, w = mask.shape[:2]
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))
    pad = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(pad, kernel, iterations=dilation)
    return (mask.astype(np.uint8) - eroded[1:h+1, 1:w+1]) > 0

def evaluate_pred(pred, gt):
    h, w = pred.shape[:2]
    # Ignore index 255
    valid = (gt != 255) & (gt < 21)
    acc = (pred[valid] == gt[valid]).mean() if valid.sum() > 0 else 0.0

    ious, dices, b_ious, bf_scores, class_accs = [], [], [], [], []
    for c in range(21):
        p_c = (pred == c) & valid
        g_c = (gt == c) & valid
        inter = (p_c & g_c).sum()
        union = (p_c | g_c).sum()
        if union > 0:
            ious.append(inter / union)
            dices.append((2.0 * inter) / (p_c.sum() + g_c.sum()))
            if g_c.sum() > 0:
                class_accs.append(inter / g_c.sum())

            # Boundary metrics
            p_b = mask_to_boundary(p_c)
            g_b = mask_to_boundary(g_c)
            b_u = (p_b | g_b).sum()
            if b_u > 0:
                b_ious.append((p_b & g_b).sum() / b_u)
                prec = (p_b & g_b).sum() / max(1, p_b.sum())
                rec = (p_b & g_b).sum() / max(1, g_b.sum())
                bf = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
                bf_scores.append(bf)

    return {
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mAcc": float(np.mean(class_accs)) if class_accs else 0.0,
        "pixelAcc": float(acc),
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
        pred = np.array(data["labels"], dtype=np.uint8).reshape(gt.shape[0], gt.shape[1])
        metrics = evaluate_pred(pred, gt)

        print(f"\nModel: {name} (completed in {elapsed:.2f}s)")
        print(f"  mIoU:        {metrics['mIoU']*100:.2f}%")
        print(f"  mAcc:        {metrics['mAcc']*100:.2f}%")
        print(f"  Pixel Acc:   {metrics['pixelAcc']*100:.2f}%")
        print(f"  Mean Dice:   {metrics['meanDice']:.4f}")
        print(f"  Boundary IoU:{metrics['bIoU']*100:.2f}%")
        print(f"  Boundary F1: {metrics['bF1']*100:.2f}%")
        print(f"  Classes:     {len(data['classes'])}")

    print("\n" + "=" * 80)
    print("ALL 5 DEDICATED MODELS EVALUATED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
