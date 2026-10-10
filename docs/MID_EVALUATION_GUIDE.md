# Mid-Evaluation Preparation Guide & Oral Defense Strategy
**Course:** CS5013 – Programming with AI | **Semester:** Jan–May 2026 | **Dept. of CSE, IIT Madras**  
**Student:** Rathan Balaji P R (Roll No: CS26E009) | **Instructor:** Dr. Sukhendu Das

---

## 1. Two-Minute Presentation Script (Verbatim, 120 Seconds)

> *"Good morning / afternoon. My project is a **Modular GUI Framework for Multi-Tool Panoramic Image Segmentation**, developed for Dr. Sukhendu Das's CS5013 course.*
>
> *Panoramic indoor segmentation presents two fundamental challenges that conventional vision tools cannot address:  
> First, **geometric distortion**: equirectangular 360° panoramas have severe pole stretching, extreme aspect ratios, and boundary continuity issues across the 0° and 360° seams.  
> Second, **runtime heterogeneity**: modern segmentation tools differ wildly—some are written in PyTorch with CUDA, some in C++, some run as REST APIs, and each outputs different mask formats.*
>
> *To solve this, I built a pure **Java 21 and JavaFX application** anchored on three core engineering principles:*
>
> 1. *First, the **Adapter Pattern for strict runtime isolation**: All deep learning models—including the **PanoFusion ICVGIP 2026 paper models** (CLIP + DAN + SBR-Net MobileSAM), the **OPS baseline**, and **SegFormer**—execute as isolated background subprocesses via Java `ProcessBuilder`. Heavy Python or CUDA dependencies never touch the Java Virtual Machine.*
> 2. *Second, **Canonical Normalization**: Outputs from any tool are parsed into an immutable `SegmentationResult` schema with 21 Ind-PanSeg classes, while any unmapped research metadata is preserved in an audit map.*
> 3. *Third, **Panorama-Aware Visualization & Quantitative Evaluation**: Our viewer features interactive 360° horizontal modulo yaw rotation—allowing users to spin panoramas seamlessly across the seam without visual artifacts—and a built-in metrics engine computing Pixel Accuracy, Mean IoU, and Dice score against ground truth.*
>
> *The entire system is backed by **35 automated JUnit 5 tests** and can be demonstrated live in one click. Thank you!"*

---

## 2. System Architecture & Flowchart (Soft Copy for Presentation)

### 2.1 Layered Architectural Diagram
```
+-----------------------------------------------------------------------------------+
|                        PRESENTATION LAYER (JavaFX 21)                             |
|  - PanoramicApp (Main Window)           - PanoramaViewPane (Wrap-around viewer)   |
|  - MainViewController (Event handlers)   - ComparisonViewPane (Side-by-side / diff)|
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     APPLICATION & CONTROLLER COORDINATION                         |
|  - AsyncSegmentationCoordinator (CompletableFuture, Thread Pool, Non-blocking UI) |
|  - ValidationService (Validates image integrity, 2:1 aspect ratio, file formats)   |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    SEGMENTATION ADAPTER BOUNDARY (GoF Adapter)                     |
|  - SegmentationTool Interface (Universal Tool Contract)                          |
|  - ToolRegistry & ToolConfigurationLoader (Dynamic discovery via tools.json)      |
|    +----------------------------+             +----------------------------+      |
|    |    ProcessModelAdapter     |             |      HttpModelAdapter      |      |
|    |  (ProcessBuilder + Streams)|             |  (REST Endpoint Adapter)   |      |
|    +-------------+--------------+             +--------------+-------------+      |
+------------------|-------------------------------------------|--------------------+
                   |                                           |
                   v (Isolated OS Subprocess)                  v (HTTP JSON)
+--------------------------------------+   +----------------------------------------+
|   EXTERNAL PYTHON / AI RUNTIMES      |   |        CLOUD INFERENCE SERVICES        |
| - PanoFusion (CLIP+DAN+SBR-Net)      |   | - Remote Segmentation REST APIs        |
| - OPS Baseline (ECCV 2024)           |   +----------------------------------------+
| - SegFormer Baseline (NeurIPS 2021)  |
| - Ind-PanSeg 21-Class Taxonomy       |
+------------------+-------------------+
                   |
                   v (Normalized Output JSON)
+-----------------------------------------------------------------------------------+
|                 CANONICAL NORMALIZATION & QUANTITATIVE COMPARISON                 |
|  - ResultNormalizer (Maps heterogeneous tool formats to canonical schema)         |
|  - SegmentationResult & MaskData (Immutable representations with Builder pattern) |
|  - MetricsCalculator (Computes Pixel Accuracy, mIoU, Dice / F1 Score)             |
|  - PanoramaImageRenderer (Alpha blend overlays & cyclic horizontal yaw modulo)    |
+-----------------------------------------------------------------------------------+
```

### 2.2 End-to-End Data Flow Sequence
```
[User] -> Selects Image -> [ValidationService] -> Validates 2:1 ratio, format
                                 |
                                 v
               [AsyncSegmentationCoordinator] -> Dispatches async workers
                                 |
        +------------------------+------------------------+
        |                                                 |
        v                                                 v
 [ProcessModelAdapter]                            [ProcessModelAdapter]
 (PanoFusion: SBR-Net Refined)                    (OPS: Coarse Baseline)
        |                                                 |
        v (ProcessBuilder)                                v (ProcessBuilder)
 [panofusion_segmenter.py]                        [panofusion_segmenter.py --refine false]
        |                                                 |
        +------------------------+------------------------+
                                 |
                                 v (Raw JSON output on disk)
                         [ResultNormalizer]
                                 |
                                 v (Immutable SegmentationResult)
                      [PanoramaImageRenderer] & [MetricsCalculator]
                                 |
                                 v
                      [JavaFX UI Updates] (Display Overlays, 360° Rotations, mIoU table)
```

---

## 3. Live Code Modification Cheat Sheet (Common TA Tasks & Instant Fixes)

During the evaluation, the TAs will ask you to perform a minor code modification live on the spot to test your understanding. Below are the most likely tasks, exactly where the code lives, and how to implement each in under 60 seconds:

### Task A: "Add a new class or change the color of an existing class"
- **Where:** Open [`config/tools.json`](file:///c:/Users/balaj/Documents/PAI%20Project/config/tools.json)
- **How:** Under `supportedClasses` for any tool, edit an entry or add:
  ```json
  { "id": 21, "name": "Whiteboard", "red": 255, "green": 255, "blue": 255 }
  ```
- **Explanation:** *"Our system uses dynamic tool configuration via Jackson JSON. Adding a class in `tools.json` is immediately picked up by `ToolConfigurationLoader` without recompiling the Java code."*

### Task B: "Add a new metric or modify metric calculation (e.g., compute Error Rate or Weighted Accuracy)"
- **Where:** Open [`src/main/java/edu/iitm/cs5013/panoramic/comparison/MetricsCalculator.java`](file:///c:/Users/balaj/Documents/PAI%20Project/src/main/java/edu/iitm/cs5013/panoramic/comparison/MetricsCalculator.java)
- **How:** In `evaluate()`, add:
  ```java
  double errorRate = 1.0 - pixelAccuracy;
  builder.addMetricDetail("ErrorRate", errorRate);
  ```
- **Explanation:** *"The `MetricsCalculator` computes confusion matrix statistics across all valid pixels. The `EvaluationResult.Builder` makes it trivial to attach custom metric details."*

### Task C: "Modify the image aspect ratio validation rule"
- **Where:** Open [`src/main/java/edu/iitm/cs5013/panoramic/validation/ValidationService.java`](file:///c:/Users/balaj/Documents/PAI%20Project/src/main/java/edu/iitm/cs5013/panoramic/validation/ValidationService.java)
- **How:** Look for line ~35:
  ```java
  double aspectRatio = (double) width / height;
  if (aspectRatio < 1.8 || aspectRatio > 2.2) {
      warnings.add("Image aspect ratio (" + String.format("%.2f", aspectRatio) +
              ":1) differs from standard equirectangular 2:1 ratio.");
  }
  ```
- **Explanation:** *"Validation rules are encapsulated in `ValidationService`. It emits non-fatal warnings for non-standard aspect ratios while allowing real stitched panoramas (which range up to 12:1 in Ind-PanSeg) to proceed."*

### Task D: "Change the default alpha opacity of the mask overlay"
- **Where:** Open [`src/main/java/edu/iitm/cs5013/panoramic/visualization/PanoramaImageRenderer.java`](file:///c:/Users/balaj/Documents/PAI%20Project/src/main/java/edu/iitm/cs5013/panoramic/visualization/PanoramaImageRenderer.java)
- **How:** The `renderOverlay(Image baseImage, SegmentationResult result, double opacity)` accepts the alpha level:
  ```java
  // Default is 0.50 (50%). Change to 0.70 (70%):
  renderer.renderOverlay(baseImage, result, 0.70);
  ```
- **Explanation:** *"Overlay rendering uses JavaFX `PixelWriter` with per-pixel alpha blending: `blended = base * (1 - alpha) + mask * alpha`."*

### Task E: "How does the 360° horizontal wrap-around rotation work?"
- **Where:** Open [`src/main/java/edu/iitm/cs5013/panoramic/visualization/PanoramaImageRenderer.java`](file:///c:/Users/balaj/Documents/PAI%20Project/src/main/java/edu/iitm/cs5013/panoramic/visualization/PanoramaImageRenderer.java)
- **Code:**
  ```java
  int shiftX = (int) Math.round((yawDegrees / 360.0) * width);
  int sourceX = (x + shiftX) % width;
  if (sourceX < 0) sourceX += width;
  ```
- **Explanation:** *"Equirectangular panoramas are topologically cylindrical along the x-axis. We map each pixel coordinate using modular arithmetic `(x + offset) % width`. This guarantees seamless cyclic translation across the 0° and 360° boundary."*

---

## 4. How to Use Gemini Pro During the Evaluation (Course Rules Compliance)

1. **Keep Prompts Specific & Constraint-Driven:**
   - *Example prompt if asked to add a metric:*  
     `"In MetricsCalculator.java, add a method calculating Frequency-Weighted IoU without breaking the existing EvaluationResult contract. Ensure no null pointer exceptions on empty masks."`
2. **Review Before Applying:**
   - Verify that package declarations, imports, and variable names match your existing classes.
3. **Run the Test Suite Immediately:**
   - After any modification, execute:
     ```powershell
     .\mvnw.cmd test
     ```
   - This proves to the TA that the modification compiles, integrates cleanly, and does not regress any of the 35 existing tests.
