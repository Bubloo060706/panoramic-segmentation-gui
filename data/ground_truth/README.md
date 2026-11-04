# Ground Truth Reference Annotations (Ind-PanSeg Benchmark)

This directory contains verified ground-truth reference masks and canonical annotations for quantitative evaluation in the **Panoramic Image Segmentation GUI framework**.

---

## 1. Ground Truth Pairing Guide

| Input Panorama Image | Ground Truth Mask File | Format | Resolution | Ground Truth Classes Included |
|---|---|:---:|:---:|---|
| **`data/sample_panorama.jpg`** | **`data/ground_truth/sample_panorama_gt.png`** | Indexed 8-bit PNG | **800 × 400** | Roof (0), Wall (5), Floor (8), Door (11), Window (14), Chair (16), Table (17) |
| **`data/sample_panorama.jpg`** | **`data/ground_truth/sample_panorama_gt.json`** | Canonical JSON | **800 × 400** | Full 21 Ind-PanSeg Classes + Metadata |
| **`H.jpg`** | **`data/ground_truth/H_gt.png`** | Compressed PNG | **7808 × 3744** | High-res: Roof (0), Wall (5), Floor (8), Monitor (9), Door (11), Window (14), Chair (16), Table (17) |
| **`H.jpg`** | **`data/ground_truth/H_gt.json`** | Canonical JSON | **7808 × 3744** | Full 21 Ind-PanSeg Classes + Metadata |

---

## 2. How to Use in the JavaFX Desktop GUI

1. Launch the application:
   ```powershell
   .\run.bat
   ```
2. Click **"1. Panoramic Image -> Select 360° Panorama..."**:
   * Choose either `data/sample_panorama.jpg` or `H.jpg`.
3. Click **"2. Ground Truth Mask -> Load Ground Truth Mask..."**:
   * Select the corresponding file from `data/ground_truth/` (e.g. `sample_panorama_gt.png` or `sample_panorama_gt.json`).
4. Select one or more tools (e.g., **PanoFusion**, **OPS Baseline**, **OmniSAM**, **Meta SAM**) and click **"Run Segmentation"**.
5. Switch to **Tab 2: "Cross-Model Comparison"**:
   * Inspect the side-by-side synchronized 360° yaw views of the panorama, the ground truth overlay, and the model prediction overlays.
   * Review the **Quantitative Evaluation Metrics Table** showing real-time **Pixel Accuracy**, **Mean IoU (mIoU)**, and **Mean Dice (F1 Score)** computed against the ground truth reference!

---

## 3. Label Taxonomy (Ind-PanSeg 21 Classes)

All ground truth masks strictly adhere to Table 2 of the *PanoFusion: Boundary-Aware Vision-Language Segmentation for Stitched Indoor Panoramas* (ICVGIP 2026) paper:

```
 0: Roof          (70, 130, 180)     11: Door         (255, 140, 0)
 1: Bed           (147, 112, 219)    12: Curtain      (218, 112, 214)
 2: Obstruction   (128, 128, 128)    13: Pillow       (255, 182, 193)
 3: Refrigerator  (220, 20, 60)      14: Window       (0, 206, 209)
 4: Phone         (255, 20, 147)     15: Misc.        (169, 169, 169)
 5: Wall          (245, 222, 179)    16: Chair        (255, 215, 0)
 6: Furniture     (210, 105, 30)     17: Table        (160, 82, 45)
 7: Electronics   (0, 191, 255)      18: Boxes        (210, 180, 140)
 8: Floor         (139, 69, 19)      19: Bag          (255, 105, 180)
 9: Monitor       (0, 255, 255)      20: Bottles      (50, 205, 50)
10: Cabinet       (205, 133, 63)
```

---

## 4. Seam Continuity Guarantee

In both `sample_panorama_gt.png` and `H_gt.png`, the **Door (Class 11)** region is intentionally positioned across the $0^\circ / 360^\circ$ boundary seam ($x=0$ and $x=width-1$). This allows the evaluator to test the interactive **360° Horizontal Yaw Slider**, confirming that objects spanning the seam maintain topological continuity without visual or metric distortion.

