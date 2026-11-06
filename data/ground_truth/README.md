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

## 3. Label Taxonomy & Color Mapping (Ind-PanSeg / VPLab 21 Classes)

All ground truth masks strictly adhere to Table 2 of the *PanoFusion: Boundary-Aware Vision-Language Segmentation for Stitched Indoor Panoramas* (ICVGIP 2026) paper and the authoritative VPLab dataset color palette stored directly inside the model checkpoint metadata:

```
 0: Roof          (153,  51,   0)     11: Door         (  0, 204, 153)
 1: Bed           (  0, 142,  64)     12: Curtain      (255,  51, 153)
 2: Obstructions  (255,   0,   0)     13: Pillow       (102,   0, 102)
 3: Refrigerator  ( 37, 150, 190)     14: Window       (220, 142, 205)
 4: Phone         (240, 200,  60)     15: Misc.        (218, 240,  19)
 5: Wall          ( 19,  27, 172)     16: Chair        ( 45,   0, 246)
 6: Furniture     (  0, 255,   0)     17: Table        (255, 255, 153)
 7: Electronics   (150,  50, 145)     18: Boxes        (160, 169,  94)
 8: Floor         (164, 244, 252)     19: Bag          (123,   0, 246)
 9: Monitor       (153, 102,   0)     20: Bottles      (255, 102,   0)
10: Cabinet       (146, 146, 146)    255: Ignore Index (boundary/unlabeled)
```

---

## 4. Seam Continuity Guarantee

In both `sample_panorama_gt.png` and `H_gt.png`, the **Door (Class 11)** region is intentionally positioned across the $0^\circ / 360^\circ$ boundary seam ($x=0$ and $x=width-1$). This allows the evaluator to test the interactive **360° Horizontal Yaw Slider**, confirming that objects spanning the seam maintain topological continuity without visual or metric distortion.

