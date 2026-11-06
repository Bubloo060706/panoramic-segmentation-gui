"""
VP Lab OmniSAM — Val Set Inference & Visualization with Full Metric Suite
(mIoU, mAcc, Boundary IoU, Boundary F-score, Dice, Pixel Accuracy)

This reuses OmniSAM's panorama-stitching inference (same as visualize.py) but
computes metrics using the same formulas as the reference SegFormer notebook
(mask_to_boundary / compute_all_metrics_per_class), so numbers are directly
comparable across the two pipelines.

INPUT:  RGB panoramas in <data_root>/Panorama Segmentation DataSet/val/img
        + GT index masks in .../val/masks_mmseg
OUTPUT: Per-image panels (RGB | GT | Prediction) saved to --out_dir,
        a per-image CSV of all metrics, and printed dataset-level averages.
"""

import os
import argparse
import numpy as np
import cv2
import pandas as pd
import torch
import torch.distributed as dist
from torchvision import transforms
from PIL import Image, ImageDraw

from utils.build_ddp import DatasetBuilder, build_model, load_checkpoint, DATASET_NAME2CLASSES
from dataset.VPLAB import VPLAB21
from train.strategies import WithMemStrategy, NoMemStrategy

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

IGNORE_INDEX = 255
BOUNDARY_DILATION_RATIO = 0.02       # Boundary IoU dilation ratio (Cheng et al., CVPR 2021)
BOUNDARY_F_TOLERANCE_RATIO = 0.0075  # Boundary F-score tolerance ratio (Csurka et al., BMVC 2013)

VPLAB_PALETTE = np.array([
    [153,  51,   0], [  0, 142,  64], [255,   0,   0], [ 37, 150, 190],
    [240, 200,  60], [ 19,  27, 172], [  0, 255,   0], [150,  50, 145],
    [164, 244, 252], [153, 102,   0], [146, 146, 146], [  0, 204, 153],
    [255,  51, 153], [102,   0, 102], [220, 142, 205], [218, 240,  19],
    [ 45,   0, 246], [255, 255, 153], [160, 169,  94], [123,   0, 246],
    [255, 102,   0],
], dtype=np.uint8)


# ============== PALETTE / COLOR HELPERS ==============

def get_palette(dataset_name, num_classes):
    if dataset_name == 'VPLAB':
        palette = VPLAB_PALETTE
    else:
        rng = np.random.RandomState(42)
        palette = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    assert palette.shape[0] == num_classes, f"Palette has {palette.shape[0]} colors but dataset has {num_classes} classes"
    return palette


def colorize(label_mask, palette_np, ignore_index=IGNORE_INDEX):
    """Map a HxW index mask to an HxWx3 RGB image. Ignored pixels are drawn black."""
    safe_mask = label_mask.copy()
    safe_mask[safe_mask == ignore_index] = 0
    color = palette_np[safe_mask]
    color[label_mask == ignore_index] = [0, 0, 0]
    return color


def denorm(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * STD) + MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def save_legend(class_names, palette, out_path):
    row_h = 30
    img = Image.new('RGB', (300, row_h * len(class_names)), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for i, name in enumerate(class_names):
        color = tuple(int(c) for c in palette[i])
        draw.rectangle([10, i*row_h+5, 40, i*row_h+25], fill=color)
        draw.text((50, i*row_h+8), name, fill=(0, 0, 0))
    img.save(out_path)


# ============== METRIC HELPERS (ported from reference notebook) ==============

def mask_to_boundary(mask, dilation_ratio=BOUNDARY_DILATION_RATIO):
    """Convert binary mask (H, W) to boundary mask using morphological erosion.
    Follows official Boundary IoU paper (Cheng et al., CVPR 2021)."""
    h, w = mask.shape[:2]
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    pad_mask = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(pad_mask, kernel, iterations=dilation)
    mask_erode = eroded[1: h + 1, 1: w + 1]
    return (mask.astype(np.uint8) - mask_erode) > 0


def compute_all_metrics_per_class(
    pred_mask, gt_mask, num_classes,
    ignore_index=IGNORE_INDEX,
    dilation_ratio=BOUNDARY_DILATION_RATIO,
    bound_th=BOUNDARY_F_TOLERANCE_RATIO,
):
    """Per-class IoU, Dice, Accuracy (for mAcc), Boundary IoU, and Boundary F-score
    for a single image, ignoring `ignore_index` pixels."""
    valid = (gt_mask != ignore_index)
    h, w = gt_mask.shape[:2]
    diag = np.sqrt(h ** 2 + w ** 2)
    theta = max(1, int(round(bound_th * diag)))
    kernel = np.ones((3, 3), dtype=np.uint8)

    ious, dices, accs = [], [], []
    b_ious, bf_scores = [], []
    b_intersections, b_unions = [], []

    pred_flat = pred_mask[valid]
    gt_flat = gt_mask[valid]
    n_valid = int(valid.sum())
    n_correct = int((pred_flat == gt_flat).sum())

    for c in range(num_classes):
        p = (pred_flat == c)
        g = (gt_flat == c)
        inter = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        denom = p.sum() + g.sum()
        g_sum = g.sum()

        ious.append(np.nan if union == 0 else inter / union)
        dices.append(np.nan if denom == 0 else (2 * inter) / denom)
        accs.append(np.nan if g_sum == 0 else inter / g_sum)

        gt_c_2d = (gt_mask == c) & valid
        pred_c_2d = (pred_mask == c) & valid

        gt_empty = not np.any(gt_c_2d)
        pred_empty = not np.any(pred_c_2d)

        if gt_empty and pred_empty:
            b_ious.append(np.nan)
            bf_scores.append(np.nan)
            b_intersections.append(0)
            b_unions.append(0)
        elif gt_empty or pred_empty:
            b_ious.append(0.0)
            bf_scores.append(0.0)
            b_intersections.append(0)
            active_mask = gt_c_2d if not gt_empty else pred_c_2d
            b_active = mask_to_boundary(active_mask, dilation_ratio)
            b_unions.append(int(b_active.sum()))
        else:
            gt_b = mask_to_boundary(gt_c_2d, dilation_ratio)
            pred_b = mask_to_boundary(pred_c_2d, dilation_ratio)
            b_inter = int(np.logical_and(gt_b, pred_b).sum())
            b_union = int(np.logical_or(gt_b, pred_b).sum())
            b_iou = np.nan if b_union == 0 else (b_inter / b_union)
            b_ious.append(b_iou)
            b_intersections.append(b_inter)
            b_unions.append(b_union)

            gt_c_uint8 = gt_c_2d.astype(np.uint8)
            pred_c_uint8 = pred_c_2d.astype(np.uint8)
            gt_contour = (gt_c_uint8 - cv2.erode(gt_c_uint8, kernel, iterations=1)) > 0
            pred_contour = (pred_c_uint8 - cv2.erode(pred_c_uint8, kernel, iterations=1)) > 0

            n_gt_contour = gt_contour.sum()
            n_pred_contour = pred_contour.sum()

            if n_gt_contour == 0 and n_pred_contour == 0:
                bf_scores.append(np.nan)
            elif n_gt_contour == 0 or n_pred_contour == 0:
                bf_scores.append(0.0)
            else:
                dist_to_gt = cv2.distanceTransform((1 - gt_contour.astype(np.uint8)), cv2.DIST_L2, 3)
                dist_to_pred = cv2.distanceTransform((1 - pred_contour.astype(np.uint8)), cv2.DIST_L2, 3)

                precision = (pred_contour & (dist_to_gt <= theta)).sum() / n_pred_contour
                recall = (gt_contour & (dist_to_pred <= theta)).sum() / n_gt_contour
                if precision + recall > 0:
                    bf_scores.append(2 * precision * recall / (precision + recall))
                else:
                    bf_scores.append(0.0)

    return (
        np.array(ious), np.array(dices), np.array(accs),
        np.array(b_ious), np.array(bf_scores),
        np.array(b_intersections), np.array(b_unions),
        n_valid, n_correct,
    )


# ============== DATA LOADING ==============

def load_full_sample(img_path, mask_path, crop_size):
    image = Image.open(img_path).convert('RGB')
    label = Image.open(mask_path)
    image = image.resize(crop_size, Image.BICUBIC)
    label = label.resize(crop_size, Image.NEAREST)

    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    image_t = input_transform(image)
    label_np = np.array(label).astype('int32')
    return image_t, label_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="VPLAB")
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--backbone", type=str, default="sam2_s")
    parser.add_argument("--num_maskmem", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--patch_stride", type=int, default=128)
    parser.add_argument("--load_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="visualizations_full_metrics")
    parser.add_argument("--num_samples", type=int, default=9999,
                         help="How many panorama panels (RGB|GT|Pred) to save; metrics always run over the full split")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    class_names = DATASET_NAME2CLASSES[args.dataset]
    num_classes = len(class_names)
    palette = get_palette(args.dataset, num_classes)

    if args.split == "test":
        root_train = os.path.join(args.data_root, 'Panorama Segmentation DataSet')
        dataset_obj = VPLAB21(
            root_train, split='test',
            crop_size=(2048, 1024),
            sliding_window=(args.patch_stride, args.image_size),
        )
    else:
        data_builder = DatasetBuilder(args)
        train_loader, val_loader, train_sampler, val_sampler = data_builder.build_src()
        loader = val_loader if args.split == "val" else train_loader
        dataset_obj = loader.dataset

    model = build_model(args, num_classes, args.num_maskmem, device, local_rank)
    load_checkpoint(model, args.load_ckpt)
    model.eval()

    mem_strategy = WithMemStrategy() if args.dataset != 'Stanford2D3D' else NoMemStrategy()

    os.makedirs(args.out_dir, exist_ok=True)
    save_legend(class_names, palette, os.path.join(args.out_dir, "legend.png"))

    crop_w, crop_h = dataset_obj.crop_size
    window = dataset_obj.window_size
    stride = dataset_obj.stride
    frame_cnt = dataset_obj._get_frame_cnt()

    per_image_rows = []
    running_intersection = np.zeros(num_classes)
    running_union = np.zeros(num_classes)
    running_gt_pixels = np.zeros(num_classes)
    running_b_intersection = np.zeros(num_classes)
    running_b_union = np.zeros(num_classes)
    running_correct = 0
    running_total = 0
    saved = 0

    with torch.no_grad():
        for entry in dataset_obj.files:
            image_t, gt_mask = load_full_sample(entry["img"], entry["label"], (crop_w, crop_h))

            prob_canvas = torch.zeros((num_classes, crop_h, crop_w), device=device)
            count_canvas = torch.zeros((crop_h, crop_w), device=device)

            for frame_idx in range(frame_cnt):
                x = frame_idx * stride
                frame_img = image_t[:, :, x:x + window].unsqueeze(0).to(device)
                pred = model(frame_idx, frame_img, mem_type='source')
                prob = torch.softmax(pred, dim=1).squeeze(0)
                prob_canvas[:, :, x:x + window] += prob
                count_canvas[:, x:x + window] += 1

            mem_strategy.reset_memory(model)

            count_canvas = count_canvas.clamp(min=1)
            prob_canvas = prob_canvas / count_canvas.unsqueeze(0)
            pred_mask = torch.argmax(prob_canvas, dim=0).cpu().numpy().astype(np.uint8)
            gt_mask = gt_mask.astype(np.uint8)

            # ---- full metric suite for this image ----
            (
                ious, dices, accs,
                b_ious, bf_scores,
                b_inters, b_unis,
                n_valid, n_correct,
            ) = compute_all_metrics_per_class(pred_mask, gt_mask, num_classes)

            valid_ious = ious[~np.isnan(ious)]
            valid_dices = dices[~np.isnan(dices)]
            valid_accs = accs[~np.isnan(accs)]
            valid_b_ious = b_ious[~np.isnan(b_ious)]
            valid_bf_scores = bf_scores[~np.isnan(bf_scores)]

            img_miou = valid_ious.mean() if len(valid_ious) else np.nan
            img_macc = valid_accs.mean() if len(valid_accs) else np.nan
            img_biou = valid_b_ious.mean() if len(valid_b_ious) else np.nan
            img_bfscore = valid_bf_scores.mean() if len(valid_bf_scores) else np.nan
            img_mdice = valid_dices.mean() if len(valid_dices) else np.nan
            img_pacc = n_correct / n_valid if n_valid else np.nan

            per_image_rows.append({
                "image": entry["name"],
                "mIoU": img_miou,
                "mAcc": img_macc,
                "Boundary_IoU": img_biou,
                "Boundary_FScore": img_bfscore,
                "mDice": img_mdice,
                "pixel_acc": img_pacc,
            })

            valid_px = gt_mask != IGNORE_INDEX
            for c in range(num_classes):
                p = (pred_mask == c) & valid_px
                g = (gt_mask == c) & valid_px
                running_intersection[c] += np.logical_and(p, g).sum()
                running_union[c] += np.logical_or(p, g).sum()
                running_gt_pixels[c] += g.sum()

            running_b_intersection += b_inters
            running_b_union += b_unis
            running_correct += n_correct
            running_total += n_valid

            if saved < args.num_samples:
                rgb_np = denorm(image_t)
                gt_color = colorize(gt_mask, palette)
                pred_color = colorize(pred_mask, palette)
                panel = np.concatenate([rgb_np, gt_color, pred_color], axis=1)
                out_name = f"{entry['name']}_pano.png"
                Image.fromarray(panel).save(os.path.join(args.out_dir, out_name))
                if local_rank == 0:
                    print(
                        f"Saved {entry['name']:20s}  mIoU={img_miou:.4f}  mAcc={img_macc:.4f}  "
                        f"B-IoU={img_biou:.4f}  B-FScore={img_bfscore:.4f}  PixelAcc={img_pacc:.4f}"
                    )
            saved += 1

    if local_rank == 0:
        per_image_df = pd.DataFrame(per_image_rows)
        csv_path = os.path.join(args.out_dir, "per_image_metrics.csv")
        per_image_df.to_csv(csv_path, index=False)

        num_images = len(per_image_df)

        avg_of_image_miou = per_image_df["mIoU"].mean()
        avg_of_image_macc = per_image_df["mAcc"].mean()
        avg_of_image_biou = per_image_df["Boundary_IoU"].mean()
        avg_of_image_bfscore = per_image_df["Boundary_FScore"].mean()
        avg_of_image_mdice = per_image_df["mDice"].mean()
        avg_of_image_pacc = per_image_df["pixel_acc"].mean()

        with np.errstate(invalid="ignore"):
            class_iou_dataset = np.where(running_union > 0, running_intersection / running_union, np.nan)
            class_acc_dataset = np.where(running_gt_pixels > 0, running_intersection / running_gt_pixels, np.nan)
            class_biou_dataset = np.where(running_b_union > 0, running_b_intersection / running_b_union, np.nan)

        dataset_miou = np.nanmean(class_iou_dataset)
        dataset_macc = np.nanmean(class_acc_dataset)
        dataset_biou = np.nanmean(class_biou_dataset)
        dataset_pacc = running_correct / running_total

        print("=" * 65)
        print("AVERAGE OVER VAL FOLDER  (mean of each image's own score)")
        print("=" * 65)
        print(f"Number of images       : {num_images}")
        print(f"Average mIoU           : {avg_of_image_miou:.4f}")
        print(f"Average mAcc           : {avg_of_image_macc:.4f}")
        print(f"Average Boundary IoU   : {avg_of_image_biou:.4f}")
        print(f"Average Boundary F1    : {avg_of_image_bfscore:.4f}")
        print(f"Average mDice          : {avg_of_image_mdice:.4f}")
        print(f"Average Pixel Acc      : {avg_of_image_pacc:.4f}")

        print("\n" + "=" * 65)
        print("DATASET-LEVEL AGGREGATED (pool all pixels/boundaries first)")
        print("=" * 65)
        print(f"Number of images       : {num_images}")
        print(f"Aggregated mIoU        : {dataset_miou:.4f}")
        print(f"Aggregated mAcc        : {dataset_macc:.4f}")
        print(f"Aggregated Boundary IoU: {dataset_biou:.4f}")
        print(f"Aggregated Pixel Acc   : {dataset_pacc:.4f}")
        print(f"\nPer-image CSV saved to: {csv_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
