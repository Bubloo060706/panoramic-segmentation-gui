import os
import json
import time
import cv2
import numpy as np

CLASSES = [
    "Roof", "Bed", "Obstructions", "Refrigerator", "Phone", "Wall",
    "Furniture", "Electronics", "Floor", "Monitor", "Cabinet", "Door",
    "Curtain", "Pillow", "Window", "Misc.", "Chair", "Table", "Boxes",
    "Bag", "Bottles"
]

PALETTE = [
    [153,  51,   0],   # 0: Roof
    [  0, 142,  64],   # 1: Bed
    [255,   0,   0],   # 2: Obstructions
    [ 37, 150, 190],   # 3: Refrigerator
    [240, 200,  60],   # 4: Phone
    [ 19,  27, 172],   # 5: Wall
    [  0, 255,   0],   # 6: Furniture
    [150,  50, 145],   # 7: Electronics
    [164, 244, 252],   # 8: Floor
    [153, 102,   0],   # 9: Monitor
    [146, 146, 146],   # 10: Cabinet
    [  0, 204, 153],   # 11: Door
    [255,  51, 153],   # 12: Curtain
    [102,   0, 102],   # 13: Pillow
    [220, 142, 205],   # 14: Window
    [218, 240,  19],   # 15: Misc.
    [ 45,   0, 246],   # 16: Chair
    [255, 255, 153],   # 17: Table
    [160, 169,  94],   # 18: Boxes
    [123,   0, 246],   # 19: Bag
    [255, 102,   0],   # 20: Bottles
]

PALETTE_NP = np.array(PALETTE, dtype=np.uint8)

def save_segmentation_result(output_path, pred_mask, start_time):
    """
    Saves segmentation mask PNG and canonical JSON conforming to ProcessModelAdapter/ResultNormalizer.
    """
    h, w = pred_mask.shape[:2]
    pred_mask = np.clip(pred_mask, 0, 20).astype(np.uint8)
    
    # 1. Generate RGB colorized mask
    color_mask = PALETTE_NP[pred_mask]
    
    # Write mask image file
    mask_path = os.path.splitext(output_path)[0] + "_mask.png"
    cv2.imwrite(mask_path, cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
    
    # 2. Extract detected classes and pixel counts
    detected_classes = []
    unique_ids, counts = np.unique(pred_mask, return_counts=True)
    for c_id, cnt in zip(unique_ids, counts):
        if 0 <= c_id < len(CLASSES):
            r, g, b = PALETTE[c_id]
            detected_classes.append({
                "id": int(c_id),
                "name": CLASSES[c_id],
                "red": int(r),
                "green": int(g),
                "blue": int(b),
                "pixelCount": int(cnt),
                "confidence": 0.95
            })
            
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    out_dict = {
        "width": int(w),
        "height": int(h),
        "segmentationType": "SEMANTIC",
        "maskPath": mask_path.replace("\\", "/"),
        "labels": pred_mask.flatten().tolist(),
        "classes": detected_classes,
        "dataset": "VPLAB",
        "taxonomy": "Ind-PanSeg",
        "benchmark_mIoU": "54.94%",
        "metrics": {
            "inference_time_ms": elapsed_ms
        }
    }
    
    # Write JSON output
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)
    
    print(f"Inference completed in {elapsed_ms}ms. Detected {len(detected_classes)} classes. Saved to {output_path}")

