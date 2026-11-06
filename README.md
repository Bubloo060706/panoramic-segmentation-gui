# GUI Framework for Multi-Tool Panoramic Image Segmentation

[![Java](https://img.shields.io/badge/Java-21%20LTS-orange.svg)](https://www.oracle.com/java/)
[![JavaFX](https://img.shields.io/badge/JavaFX-21-blue.svg)](https://openjfx.io/)
[![Maven](https://img.shields.io/badge/Maven-3.9+-red.svg)](https://maven.apache.org/)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-51%20Passed-brightgreen.svg)]()

**Course:** CS5013 – Programming with AI  
**Author:** Rathan Balaji P R (Roll No. CS26E009) – Department of Computer Science & Engineering, IIT Madras  
**Stakeholder:** Prof. Sukhendu Das, Department of Computer Science & Engineering, IIT Madras  

---

## 1. Project Purpose & Overview

Panoramic image segmentation assigns semantic or instance-level labels to regions of a 360° equirectangular image. In practice, modern segmentation tools differ substantially in their input specifications, execution environments (PyTorch, C++, Docker, REST APIs), output conventions (indexed arrays, color PNGs, polygon geometries), and supported classes. 

This project provides a clean, extensible Java desktop application with a JavaFX graphical interface that unifies multiple heterogeneous segmentation tools on panoramic images:
1. **Single Workflow:** Load a 360° equirectangular panoramic image, select one or multiple segmentation tools, run inference, and inspect normalized outputs in a common viewer.
2. **Model Adapter Boundary:** Keeps model-specific code and heavy Python/CUDA dependencies completely outside the Java GUI process. Local models are executed via Java `ProcessBuilder`, while remote models communicate over HTTP REST behind a common `SegmentationTool` interface.
3. **Canonical Result Normalization:** Converts heterogeneous tool outputs into a standardized `SegmentationResult` schema without silently dropping unsupported metadata (which is explicitly captured in an `unsupportedAttributes` map).
4. **Panorama-Aware Visualization:** Recognizes the cylindrical, equirectangular 360° geometry of the input, supporting interactive horizontal wrap-around rotation across the 0° / 360° boundary ($x' = (x + \Delta x) \pmod W$), alpha mask overlays, and pixel inspection.
5. **Cross-Model Side-by-Side Comparison & Metrics:** Provides synchronized multi-tool visual comparison and computes state-of-the-art evaluation metrics (**mIoU**, **mAcc**, **Boundary IoU**, **Boundary F-score**, **Pixel Accuracy**, and **Mean Dice**) against ground truth.

---

## 1.1 Reference Paper & Model Benchmark Support

This implementation integrates and benchmarks the models from the research paper:  
**"PanoFusion: Boundary-Aware Vision-Language Segmentation for Stitched Indoor Panoramas"**  
*(Proceedings of the 17th Indian Conference on Computer Vision, Graphics and Image Processing - ICVGIP 2026, Submission 214)*

### Supported Models & Checkpoints:
| Model ID | Model Name & Framework | Checkpoint File & Format | Ind-PanSeg mIoU | Ind-PanSeg mAcc |
|---|---|---|:---:|:---:|
| `panofusion` | **PanoFusion (Proposed)**<br>Frozen CLIP + DAN + SBR-Net MobileSAM gated cross-attention | `PAI Segmentation Checkpoints/PanoFusion/best_mIoU_iter_141500.pth`<br>(PyTorch `.pth`, 702.0 MB) | **54.94%** | **77.00%** |
| `ops-baseline` | **OPS Baseline [Zheng et al., ECCV 2024]**<br>Coarse vision-language backbone (CLIP + DAN) without SBR-Net | `PAI Segmentation Checkpoints/OPS Baseline/best_mIoU_iter_23000.pth`<br>(PyTorch `.pth`, 639.8 MB) | 44.12% | 66.46% |
| `omnisam-baseline` | **OmniSAM [Zhong et al., ICCV 2025]**<br>Omnidirectional Segment Anything Model with SAM2 memory adaptation | `PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth`<br>(PyTorch `.pth`, 186.3 MB) | 41.66% | 51.62% |
| `segformer-baseline` | **SegFormer [Xie et al., NeurIPS 2021]**<br>Transformer-based perspective baseline fine-tuned with focal loss | `PAI Segmentation Checkpoints/Segformer/model.safetensors`<br>(HuggingFace Safetensors, 256.1 MB) | 31.35% | 70.57% |
| `pretrained-sam-cnn` | **Pretrained SAM + CNN tracking**<br>Pretrained Segment Anything Model coupled with VPLab CNN tracking classifier | `PAI Segmentation Checkpoints/Pretrained SAM  + CNN tracking/classifier _vplab.pth`<br>(VPLab Classifier `.pth`, 44.8 MB) | 48.20% | 72.50% |

### Ind-PanSeg Authoritative Semantic Taxonomy & Color Palette (21 Classes)

Extracted directly from the training dataset metadata in `d['meta']['dataset_meta']`:

| Class ID | Semantic Class | Category | RGB Color | Hex Code | Swatch Preview |
|:---:|:---|:---|:---:|:---:|:---:|
| **0** | `Roof` | Structural | `[153, 51, 0]` | `#993300` | 🟫 Brown |
| **1** | `Bed` | Object | `[0, 142, 64]` | `#008E40` | 🟩 Forest Green |
| **2** | `Obstructions` | Structural | `[255, 0, 0]` | `#FF0000` | 🟥 Red |
| **3** | `Refrigerator` | Appliance | `[37, 150, 190]` | `#2596BE` | 🟦 Cyan/Teal |
| **4** | `Phone` | Object | `[240, 200, 60]` | `#F0C83C` | 🟨 Amber Gold |
| **5** | `Wall` | Structural | `[19, 27, 172]` | `#131BAC` | 🟦 Deep Blue |
| **6** | `Furniture` | Object | `[0, 255, 0]` | `#00FF00` | 🟩 Pure Green |
| **7** | `Electronics` | Appliance | `[150, 50, 145]` | `#963291` | 🟪 Purple Magenta |
| **8** | `Floor` | Structural | `[164, 244, 252]` | `#A4F4FC` | 🧊 Light Cyan |
| **9** | `Monitor` | Appliance | `[153, 102, 0]` | `#996600` | 🟤 Ochre Brown |
| **10** | `Cabinet` | Storage | `[146, 146, 146]` | `#929292` | ⬜ Gray |
| **11** | `Door` | Structural | `[0, 204, 153]` | `#00CC99` | 🟩 Mint Emerald |
| **12** | `Curtain` | Structural | `[255, 51, 153]` | `#FF3399` | 🌸 Hot Pink |
| **13** | `Pillow` | Object | `[102, 0, 102]` | `#660066` | 🟣 Dark Violet |
| **14** | `Window` | Structural | `[220, 142, 205]` | `#DC8ECD` | 🪻 Lilac |
| **15** | `Misc.` | Structural | `[218, 240, 19]` | `#DAF013` | 🟡 Lime Yellow |
| **16** | `Chair` | Furniture | `[45, 0, 246]` | `#2D00F6` | 🟦 Indigo Blue |
| **17** | `Table` | Furniture | `[255, 255, 153]` | `#FFFF99` | 🟨 Pale Yellow |
| **18** | `Boxes` | Storage | `[160, 169, 94]` | `#A0A95E` | 🫒 Olive Gray |
| **19** | `Bag` | Object | `[123, 0, 246]` | `#7B00F6` | 🟪 Violet |
| **20** | `Bottles` | Object | `[255, 102, 0]` | `#FF6600` | 🟧 Orange |
| **255** | `Ignore / Unlabeled` | Boundary / Seam | `[0, 0, 0]` | `#000000` | ⬛ Black (Excluded from mIoU/mAcc) |

---

## 2. Architecture & Layered Design

```
                     +---------------------------------------+
                     |       JavaFX Desktop Application      |
                     |  (PanoramaViewPane, ComparisonView)   |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |    Application / Controller Layer     |
                     | (MainViewController, AsyncCoordinator)|
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |       SegmentationTool Interface      |
                     |         (Common Tool Contract)        |
                     +-------------------+-------------------+
                                         |
                          +--------------+--------------+
                          |                             |
                          v                             v
           +-----------------------------+ +-----------------------------+
           |     ProcessModelAdapter     | |      HttpModelAdapter       |
           |  (Local Process Execution)  | |  (Remote Service Execution) |
           +--------------+--------------+ +--------------+--------------+
                          |                             |
                          v                             v
           +-----------------------------+ +-----------------------------+
           | External Subprocess Runtime | |    Remote HTTP Inference    |
           |   (scripts/*Segmenter.java) | |        REST Endpoint        |
           +--------------+--------------+ +--------------+--------------+
                          \                             /
                           \                           /
                            v                         v
                     +---------------------------------------+
                     |         Result Normalization          |
                     |     (Canonical SegmentationResult)    |
                     +-------------------+-------------------+
                                         |
                                         v
                     +---------------------------------------+
                     |  Panorama-Aware View & Evaluation     |
                     | (360° Modulo Yaw, mIoU, Boundary F1)  |
                     +---------------------------------------+
```

### Module Split
- **`model`:** Core data structures (`SegmentationRequest`, `SegmentationResult`, `MaskData`, `SemanticClass`, `ToolMetadata`, `ExecutionStatus`).
- **`segmentation`:** The universal `SegmentationTool` contract.
- **`adapter`:** Adapter implementations (`SegmentationAdapter`, `ProcessModelAdapter`, `HttpModelAdapter`).
- **`config`:** Jackson JSON loader (`ToolConfigurationLoader`) and dynamic `ToolRegistry` discovering tools via `config/tools.json`.
- **`execution`:** External process boundary (`ProcessExecutor`) and HTTP boundary (`HttpExecutor`).
- **`normalization`:** Standardized result normalization (`ResultNormalizer`, `NormalizationException`).
- **`visualization`:** Panorama-aware rendering (`PanoramaImageRenderer`) supporting horizontal cylindrical wrap-around modulo shifts and alpha overlays.
- **`comparison`:** State-of-the-art quantitative metrics engine (`SegmentationMetrics`, `MetricsCalculator`, `EvaluationResult`) computing mIoU, mAcc, Boundary IoU, Boundary F-score, Pixel Accuracy, and Mean Dice.
- **`validation`:** File, image aspect ratio, tool configuration, and dimension validation (`ValidationService`, `ValidationResult`).
- **`controller`:** Asynchronous thread pool coordinator (`AsyncSegmentationCoordinator`) with callback listeners and execution result caching.
- **`gui`:** JavaFX presentation layer (`PanoramicApp`, `MainViewController`, `PanoramaViewPane`, `ComparisonViewPane`).

---

## 3. Technology Stack & Requirements

- **JDK:** Java 21 LTS (`java version 21.0.1+`)
- **Build Tool:** Apache Maven 3.9+ (includes portable `mvnw.cmd` wrapper)
- **GUI Toolkit:** JavaFX 21 (`javafx-controls`, `javafx-fxml`, `javafx-swing`, `javafx-graphics`)
- **JSON Serialization:** Jackson Databind 2.17.0 & `jackson-datatype-jsr310`
- **Testing:** JUnit 5 Jupiter (`5.10.2`)
- **Checkpoints Format:** Supports PyTorch weights (`.pth`) and HuggingFace Safetensors (`.safetensors`).

---

## 4. How to Build, Test, and Run

### 4.1 Single-Click Desktop Launch (Recommended)
On Windows, simply run:
```powershell
.\run.bat
```
*(Automatically resolves OpenJFX and Jackson dependencies via `%USERPROFILE%\.m2` and boots the application).*

### 4.2 Running Automated Tests
Execute the complete test suite (**51 automated tests** covering unit, boundary, checkpoint parsing, metrics, and end-to-end workflows):
```powershell
.\mvnw.cmd test
```
Or on Linux/macOS:
```bash
mvn test
```

### 4.3 Running via Maven
```powershell
.\mvnw.cmd javafx:run
```

---

## 5. Adding a New Segmentation Tool (Open-Closed Principle)

Adding a new segmentation model does **not** require modifying or recompiling the core Java application. To integrate a new model:

### Step 1: Define Configuration in `config/tools.json`
Add a configuration entry for your tool:
```json
{
  "id": "my-new-panoramic-model",
  "name": "My New Panoramic Model",
  "description": "High-resolution indoor panorama segmenter",
  "executionType": "PROCESS",
  "commandTemplate": [
    "python",
    "scripts/my_model.py",
    "--image",
    "{IMAGE_PATH}",
    "--output",
    "{OUTPUT_PATH}",
    "--checkpoint",
    "PAI Segmentation Checkpoints/MyModel/weights.pth"
  ],
  "workingDirectory": ".",
  "timeoutSeconds": 60,
  "enabled": true,
  "parameters": {
    "num_classes": 21,
    "checkpoint_path": "PAI Segmentation Checkpoints/MyModel/weights.pth"
  },
  "supportedClasses": [
    { "id": 0, "name": "Roof", "red": 70, "green": 130, "blue": 180 },
    { "id": 5, "name": "Wall", "red": 245, "green": 222, "blue": 179 },
    { "id": 8, "name": "Floor", "red": 139, "green": 69, "blue": 19 }
  ]
}
```

For a remote cloud inference service, specify `"executionType": "HTTP"` and set `"endpointUrl": "https://api.example.com/segment"`.

For complete details, schemas, and Python/PyTorch templates, see [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md).

---

## 6. How External Model Execution Works

1. **Strict Runtime Isolation:** Deep learning runtimes (PyTorch, CUDA, C++) remain completely outside the Java Virtual Machine. The Java application communicates strictly through `ProcessBuilder` (standard streams and JSON/mask contracts) or HTTP REST.
2. **Template Substitution:** Command templates in `config/tools.json` support dynamic tokens (`{IMAGE_PATH}`, `{OUTPUT_PATH}`, `{TOOL_DIR}`).
3. **Deadlock-Free Stream Capture:** Standard output (`stdout`) and standard error (`stderr`) are captured on dedicated worker threads to prevent OS pipe buffer deadlocks.
4. **Timeout Enforcement:** Configurable timeouts ensure unresponsive external models are terminated forcibly and translated into controlled `SegmentationException` instances.
5. **No Hard-Coded Paths:** All paths and commands are relative or loaded from `config/tools.json`.

---

## 7. Submission & Evaluation Documentation

- **Final Evaluation Guide & Video Script:** [`docs/FINAL_EVALUATION_GUIDE.md`](docs/FINAL_EVALUATION_GUIDE.md)
- **Extensibility & Developer Guide:** [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md)
- **Ground Truth Evaluation Suite:** [`data/ground_truth/README.md`](data/ground_truth/README.md)
- **Design Document Principles & Practices:** [`docs/CS5013_Principles_and_Practices.md`](docs/CS5013_Principles_and_Practices.md)
