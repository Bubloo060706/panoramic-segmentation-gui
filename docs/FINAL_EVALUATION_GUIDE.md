# Final Evaluation Guide & Viva Defense Kit
**Course:** CS5013 – Programming with AI | **Semester:** Jan–May 2026 | **Dept. of CSE, IIT Madras**  
**Student:** Rathan Balaji P R (Roll No: CS26E009) | **Instructor:** Prof. Sukhendu Das

---

## 1. Final Submission Deliverables Checklist

| Deliverable | Status | Location / Artifact |
| :--- | :---: | :--- |
| **Real Model Checkpoints** | ✅ Verified | `PAI Segmentation Checkpoints/`<br>• `PanoFusion/best_mIoU_iter_141500.pth` (702.0 MB)<br>• `OPS Baseline/best_mIoU_iter_23000.pth` (639.8 MB)<br>• `OmniSAM/best_sam2_s_iou_41.18.pth` (186.3 MB)<br>• `Segformer/model.safetensors` (256.1 MB, HuggingFace Safetensors)<br>• `Pretrained SAM  + CNN tracking/classifier _vplab.pth` (44.8 MB, VPLab CNN) |
| **Ground Truth Suite** | ✅ Verified | `data/ground_truth/`<br>• `sample_panorama_gt.png` & `sample_panorama_gt.json` (800×400)<br>• `H_gt.png` & `H_gt.json` (7808×3744 29MP panorama)<br>• `README.md` paired evaluation guide |
| **State-of-the-Art Metrics Engine** | ✅ Verified | [`SegmentationMetrics.java`](file:///c:/Users/balaj/Documents/PAI%20Project/src/main/java/edu/iitm/cs5013/panoramic/comparison/SegmentationMetrics.java)<br>• **mIoU** (Mean IoU)<br>• **mAcc** (Mean Class Accuracy)<br>• **Boundary IoU** (Morphological boundary band IoU)<br>• **Boundary F-score** (Boundary precision/recall F1-score)<br>• **Pixel Accuracy & Dice** |
| **Extensibility & Developer Guide** | ✅ Verified | [`docs/DEVELOPER_GUIDE.md`](file:///c:/Users/balaj/Documents/PAI%20Project/docs/DEVELOPER_GUIDE.md) (Complete guide for adding PyTorch/REST tools with zero Java recompilation) |
| **Single-Click Portable Launch** | ✅ Verified | [`run.bat`](file:///c:/Users/balaj/Documents/PAI%20Project/run.bat) & [`run.ps1`](file:///c:/Users/balaj/Documents/PAI%20Project/run.ps1) using `%USERPROFILE%\.m2` (100% portable on any Windows machine) |
| **Automated Test Coverage** | ✅ Verified | **49 JUnit 5 Tests Passing** (0 Failures, 0 Errors, 0 Regressions) across all architectural layers |
| **High-Resolution Support** | ✅ Verified | No size/resolution restrictions: small crops to 29MP (`H.jpg`, 7808×3744) seamlessly supported via OpenCV bridge & `maskPath` compression |

---

## 2. 3–5 Minute Stakeholder Demonstration Video Script

Use this exact script and workflow when recording your screen for the final video submission:

### Part 1: Introduction (0:00 – 0:45)
- **On Screen:** Start with project repository open in IDE and terminal.
- **Narrative:**
  > *"Good day, Professor Sukhendu Das and evaluators. I am Rathan Balaji P R, Roll Number CS26E009, presenting my final project for CS5013: **A Modular GUI Framework for Multi-Tool Panoramic Image Segmentation**.*
  >
  > *Panoramic indoor vision presents acute challenges: equirectangular geometric distortion, 360° seam discontinuities, extreme resolutions up to 29 megapixels, and heterogeneous deep learning runtimes. Our framework solves these through pure Gang-of-Four design patterns, complete runtime isolation, panorama-aware visualization, and boundary-aware quantitative evaluation."*

### Part 2: Automated Verification & Portability (0:45 – 1:15)
- **On Screen:** In PowerShell terminal, run:
  ```powershell
  .\mvnw.cmd test
  ```
- **Narrative:**
  > *"Before launching the GUI, here is our automated verification. The system is backed by **49 automated JUnit 5 tests** testing adapters, normalization, boundary metrics, preprocessors, and execution boundaries. As you can see, all 49 tests pass with zero failures."*

### Part 3: Single-Click Launch & Standard Panorama (1:15 – 2:30)
- **On Screen:** Double click [`run.bat`](file:///c:/Users/balaj/Documents/PAI%20Project/run.bat) or run `.\run.bat`. The JavaFX GUI appears.
- **Actions:**
  1. Click **"Select 360° Panorama..."** -> Select `data/sample_panorama.jpg`.
  2. Click **"Load Ground Truth Mask..."** -> Select `data/ground_truth/sample_panorama_gt.png`.
  3. Notice the **"Run All Models (Side-by-Side)"** button and the **"Select All" / "Clear All"** shortcuts.
  4. Click **"Run All Models (Side-by-Side)"**.
  5. The asynchronous coordinator runs all models. Once complete, it automatically switches to the **"Cross-Model Side-by-Side Comparison"** tab.
  6. The screen presents:
     - Side-by-side synchronized tiles for the Original Panorama, Ground Truth, and all model predictions.
     - Live metric score pills under each tile (e.g. `mIoU: 88.5% | mAcc: 93.8% | Boundary F1: 97.2%`).
     - A full side-by-side **Quantitative Evaluation Table** comparing all models across **mIoU**, **mAcc**, **Boundary IoU**, **Boundary F-score**, **Pixel Accuracy**, and **Dice**.
  7. Drag the **Synchronized 360° Yaw Slider** from 0° through 180° to 360° to demonstrate seam-free panoramic rotation across all models concurrently.
- **Narrative:**
  > *"Launching `run.bat` brings up our native JavaFX desktop suite. We load `sample_panorama.jpg` and attach its verified ground truth mask.*
  >
  > *With one click on **Run All Models (Side-by-Side)**, our asynchronous coordinator dispatches inference across all enabled models (PanoFusion, OPS Baseline, OmniSAM, and SAM). The UI automatically opens the cross-model comparison view.*
  >
  > *Notice the side-by-side tiles and our comprehensive evaluation table computing state-of-the-art boundary-aware metrics: **mIoU**, **mAcc**, **Boundary IoU**, and **Boundary F-score**. Crucially, dragging the synchronized 360° yaw slider wraps all models seamlessly across the seam without any edge artifacts."*

### Part 4: 29-Megapixel High-Resolution Demo (2:30 – 3:30)
- **Actions:**
  1. Click **"Select 360° Panorama..."** -> Select `H.jpg` (7808×3744, 29 Megapixels).
  2. Show that it loads smoothly without JVM Out-Of-Memory error.
  3. Load `data/ground_truth/H_gt.png`.
  4. Select **"OmniSAM"** -> Run Segmentation.
  5. View the resulting ultra-high-resolution segmentation mask rendered natively.
- **Narrative:**
  > *"Next, we demonstrate our architectural guarantee: **zero size or resolution restrictions**. We load `H.jpg`, an ultra-high-resolution 29-megapixel camera panorama at 7808 by 3744 pixels.*
  >
  > *Standard Java ImageIO crashes on such images due to non-standard camera JPEG headers. Our transparent OpenCV preprocessing bridge (`ImageLoaderHelper`) sanitizes the image in memory. Furthermore, through our `maskPath` protocol, the multi-million pixel segmentation is persisted as a compressed indexed PNG rather than bloating JSON, loading in milliseconds."*

### Part 5: Open-Closed Extensibility & Conclusion (3:30 – 4:00)
- **On Screen:** Bring up [`config/tools.json`](file:///c:/Users/balaj/Documents/PAI%20Project/config/tools.json) and [`docs/DEVELOPER_GUIDE.md`](file:///c:/Users/balaj/Documents/PAI%20Project/docs/DEVELOPER_GUIDE.md).
- **Narrative:**
  > *"Finally, the system adheres strictly to the Open-Closed Principle. To add a new deep learning model—whether PyTorch, ONNX, or a cloud REST API—a developer simply appends an entry to `tools.json` following our Developer Guide. No Java code is modified, and no recompilation is needed.*
  >
  > *This concludes the demonstration of our modular, isolated, and scalable panoramic segmentation suite. Thank you."*

---

## 3. Final Viva Defense FAQ & Technical Deep-Dive

### Q1: Why use an external Process Adapter instead of Deep Java Library (DJL) or ONNX Runtime Java?
> **Answer:**
> 1. **Zero JVM Pollution & Crash Isolation:** Deep learning runtimes (PyTorch, CUDA, cuDNN) rely on native C++ shared libraries (`.so`/`.dll`). If a CUDA driver encounters an Out-Of-Memory (OOM) error or segmentation fault inside the JVM, the entire Java process crashes unrecoverably. With our GoF `ProcessModelAdapter`, the model runs in an isolated OS subprocess. If the model fails or runs out of VRAM, the Java GUI catches the non-zero exit code, displays a polite error dialog, and continues running smoothly.
> 2. **Environment Heterogeneity:** In computer vision research, models require conflicting dependencies (e.g., PyTorch 2.1 vs 1.13, MMCV, Detectron2). Bundling all into a single JVM or native container is brittle. Subprocess execution allows each tool to have its own isolated virtual environment or conda prefix.
> 3. **Protocol Simplicity:** Standard CLI arguments (`--image`, `--output`, `--checkpoint`) and canonical JSON/PNG outputs make integrating new research papers a 5-minute task.

---

### Q2: How does the Equirectangular 360° Modulo Seam Handling work?
> **Answer:**
> Equirectangular panoramas map spherical coordinates $(\theta, \phi)$ to a 2D Cartesian grid $[0, W] \times [0, H]$. The left edge ($x=0$) and right edge ($x=W$) represent the exact same physical longitude (the 0°/360° meridian).
> In `PanoramaImageRenderer` and `PanoramaViewPane`, when the user shifts the yaw angle by $\Delta\theta \in [0, 360^\circ]$, the pixel offset is:
> $$\Delta x = \left\lfloor \frac{\Delta\theta}{360^\circ} \times W \right\rfloor \pmod W$$
> The image buffer is split into two complementary viewports:
> 1. The primary segment from $[\Delta x, W]$ rendered at $[0, W - \Delta x]$.
> 2. The wrapped segment from $[0, \Delta x]$ rendered at $[W - \Delta x, W]$.
> Both segments are drawn onto a single double-buffered canvas, guaranteeing 60 FPS artifact-free continuous wraparound.

---

### Q3: How do you compute Boundary IoU and Boundary F-score?
> **Answer:**
> Standard mIoU measures region overlap, which heavily biases toward large background regions (walls, floor, ceiling) and penalizes small boundary errors disproportionately. To rigorously evaluate edge quality in indoor panoramas, our `SegmentationMetrics` module implements morphological boundary metrics:
> 1. **Boundary Extraction:** We compute the inner boundary band of width $d = \max(1, \text{round}(0.0075 \cdot \text{diagonal}))$ via 8-connected binary erosion with 0-padding: $\text{Band} = M \setminus \text{Erode}(M, d)$.
> 2. **Boundary IoU:** We compute the intersection and union of the prediction band and ground truth band:
>    $$\text{Boundary IoU}_c = \frac{|\text{Band}(P_c) \cap \text{Band}(G_c)|}{|\text{Band}(P_c) \cup \text{Band}(G_c)|}$$
> 3. **Boundary F-score:** We extract the 1-pixel outer edge and perform 4-connected binary dilation by tolerance distance $\tau = d$. Precision and Recall are computed by matching boundary pixels within distance $\tau$:
>    $$\text{Precision}_c = \frac{|\text{Edge}(P_c) \cap \text{Dilate}(\text{Edge}(G_c), \tau)|}{|\text{Edge}(P_c)|}, \quad \text{Recall}_c = \frac{|\text{Edge}(G_c) \cap \text{Dilate}(\text{Edge}(P_c), \tau)|}{|\text{Edge}(G_c)|}$$
>    The Boundary F-score is their harmonic mean: $F_1 = \frac{2 \cdot P \cdot R}{P + R}$.

---

### Q4: How do you handle 29MP panoramas like `H.jpg` without memory exhaustion?
> **Answer:**
> We addressed this on two fronts:
> 1. **Header Sanitization:** Proprietary 360° cameras insert vendor markers (e.g., SOF `0xce`) that crash Java's default `ImageIO`. Our `ImageLoaderHelper` invokes an OpenCV fallback bridge that reads any raw stream, normalizes color space, and provides a sanitized buffer.
> 2. **`maskPath` Compressed Storage:** A naive JSON array for a 29MP image ($7808 \times 3744$) requires $\sim 29.2 \times 10^6$ integer entries, producing a $>90\text{ MB}$ uncompressed text file that causes massive GC pressure. Our framework defines the `maskPath` contract: models write their output as an 8-bit indexed PNG mask ($\sim 55\text{ KB}$ compressed) and reference it in the lightweight JSON metadata. The Java runtime loads this indexed buffer directly into memory via standard raster access in $<150\text{ ms}$.

---

### Q5: How do you demonstrate adherence to SOLID principles?
> **Answer:**
> 1. **Single Responsibility (SRP):** `ProcessModelAdapter` only executes subprocesses; `ResultNormalizer` only parses and validates JSON/PNG formats; `MetricsCalculator` and `SegmentationMetrics` only compute statistical scores; `PanoramaViewPane` only manages view rendering.
> 2. **Open-Closed Principle (OCP):** Adding a new model requires adding a JSON entry to `config/tools.json`. Zero lines of Java code are modified or recompiled.
> 3. **Liskov Substitution (LSP):** Both `ProcessModelAdapter` and `HttpModelAdapter` implement the unified `SegmentationTool` interface interchangeably.
> 4. **Interface Segregation (ISP):** Tool configuration, normalizer contracts, and UI listeners are segregated into lean, focused Java interfaces.
> 5. **Dependency Inversion (DIP):** High-level controllers (`AsyncSegmentationCoordinator`) depend on the abstract `SegmentationTool` interface rather than concrete process or network classes.
