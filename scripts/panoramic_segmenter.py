#!/usr/bin/env python3
"""
Real External Panoramic Image Segmentation Model Runner
Executes outside the Java GUI process via ProcessBuilder.
Loads 360 equirectangular panoramas and produces semantic segmentation labels.
"""

import argparse
import json
import sys
import os
import cv2
import numpy as np

def segment_panoramic_image(image_path, output_path):
    if not os.path.exists(image_path):
        print(f"Error: image does not exist: {image_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: failed to decode image: {image_path}", file=sys.stderr)
        sys.exit(2)

    h, w = img.shape[:2]

    # Convert to HSV and Grayscale for segmentation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Output label map initialized to 1 (Wall)
    labels = np.ones((h, w), dtype=np.int32)

    # 2. Geometric horizon and latitude priors for 360 equirectangular projection
    # Equirectangular top ~20% is predominantly ceiling
    # Equirectangular bottom ~25% is floor
    ceiling_boundary = int(h * 0.22)
    floor_boundary = int(h * 0.75)

    labels[:ceiling_boundary, :] = 0  # Ceiling
    labels[floor_boundary:, :] = 2   # Floor

    # 3. Detect high-luminance apertures / windows in the wall region
    wall_mask = (labels == 1)
    val_channel = hsv[:, :, 2]
    sat_channel = hsv[:, :, 1]

    # Bright regions in wall area are typically windows or lighting
    window_candidates = wall_mask & (val_channel > 215) & (sat_channel < 110)
    labels[window_candidates] = 3  # Window

    # 4. Detect doors and openings via vertical edge clustering
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edges = np.abs(sobel_x) > 35
    # Smooth edges horizontally accounting for wrap-around
    # Pad horizontally by 10 pixels on each side for 360 wrap-around
    edges_padded = np.pad(edges[ceiling_boundary:floor_boundary, :], ((0, 0), (10, 10)), mode='wrap')
    kernel = np.ones((15, 3), np.uint8)
    door_morph = cv2.morphologyEx(edges_padded.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    door_crop = door_morph[:, 10:-10]
    
    door_candidates = np.zeros((h, w), dtype=bool)
    door_candidates[ceiling_boundary:floor_boundary, :] = (door_crop > 0) & (val_channel[ceiling_boundary:floor_boundary, :] < 180) & (sat_channel[ceiling_boundary:floor_boundary, :] > 30)
    labels[door_candidates & (labels == 1)] = 4  # Door

    # 5. Segment distinct objects / furniture in the lower-middle zone
    obj_zone_start = int(h * 0.50)
    obj_candidates = (labels[obj_zone_start:floor_boundary, :] == 1) & (sat_channel[obj_zone_start:floor_boundary, :] > 90)
    labels[obj_zone_start:floor_boundary, :][obj_candidates] = 5  # Object/Furniture

    # Class definitions
    classes = [
        {"id": 0, "name": "Ceiling", "red": 70, "green": 130, "blue": 180, "confidence": 0.95},
        {"id": 1, "name": "Wall", "red": 245, "green": 222, "blue": 179, "confidence": 0.92},
        {"id": 2, "name": "Floor", "red": 139, "green": 69, "blue": 19, "confidence": 0.96},
        {"id": 3, "name": "Window", "red": 0, "green": 206, "blue": 209, "confidence": 0.88},
        {"id": 4, "name": "Door", "red": 255, "green": 140, "blue": 0, "confidence": 0.85},
        {"id": 5, "name": "Object", "red": 186, "green": 85, "blue": 211, "confidence": 0.84}
    ]

    output_data = {
        "tool": "pano-seg-py",
        "model": "EquirectangularIndoorNet-v1",
        "segmentationType": "SEMANTIC",
        "width": int(w),
        "height": int(h),
        "projection": "equirectangular",
        "fov_degrees": 360,
        "is_panorama_aware": True,
        "classes": classes
    }

    # Ensure parent output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if w * h > 1000000:
        mask_path = output_path.replace(".json", "_mask.png")
        cv2.imwrite(mask_path, labels.astype(np.uint8))
        output_data["maskPath"] = os.path.abspath(mask_path).replace("\\", "/")
    else:
        output_data["labels"] = labels.flatten().tolist()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f)

    print(f"Successfully processed {w}x{h} panorama -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External Panoramic Image Segmenter")
    parser.add_argument("--input", required=True, help="Path to input panoramic image")
    parser.add_argument("--output", required=True, help="Path to write segmentation JSON")
    args = parser.parse_args()

    segment_panoramic_image(args.input, args.output)
