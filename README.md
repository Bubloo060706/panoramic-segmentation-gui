# GUI Framework for Multi-Tool Panoramic Image Segmentation

[![Java](https://img.shields.io/badge/Java-21%20LTS-orange.svg)](https://www.oracle.com/java/)
[![JavaFX](https://img.shields.io/badge/JavaFX-21-blue.svg)](https://openjfx.io/)
[![Maven](https://img.shields.io/badge/Maven-3.9+-red.svg)](https://maven.apache.org/)
[![Build](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-35%20Passed-brightgreen.svg)]()

**Course:** CS5013 – Programming with AI  
**Author:** Rathan Balaji P R (Roll No. CS26E009) – Department of Computer Science & Engineering, IIT Madras  
**Stakeholder:** Dr. Sukhendu Das, Professor, Dept. of Computer Science & Engineering, IIT Madras  

---

## 1. Project Purpose & Overview

Panoramic image segmentation assigns semantic or instance-level labels to regions of a 360° equirectangular image. In practice, modern segmentation tools differ substantially in their input specifications, execution environments (Python, PyTorch, C++, Docker, REST APIs), output conventions (indexed arrays, color PNGs, polygon geometries), and supported classes. 

This project provides a clean, extensible Java desktop application with a JavaFX graphical interface that unifies multiple heterogeneous segmentation tools on panoramic images:
1. **Single Workflow:** Load a 360° equirectangular panoramic image, select one or multiple segmentation tools, run inference, and inspect normalized outputs in a common viewer.
2. **Model Adapter Boundary:** Keeps model-specific code and heavy Python/CUDA dependencies completely outside the Java GUI process. Local models are executed via Java `ProcessBuilder`, while remote models communicate over HTTP REST behind a common `SegmentationTool` interface.
3. **Canonical Result Normalization:** Converts heterogeneous tool outputs into a standardized `SegmentationResult` schema without silently dropping unsupported metadata (which is explicitly captured in an `unsupportedAttributes` map).
4. **Panorama-Aware Visualization:** Recognizes the cylindrical, equirectangular 360° geometry of the input, supporting interactive horizontal wrap-around rotation across the 0° / 360° boundary, alpha mask overlays, and pixel inspection.
5. **Cross-Model Comparison & Metrics:** Provides side-by-side visual comparison and calculates quantitative metrics (Pixel Accuracy, Mean IoU / mIoU, Dice/F1, Per-class IoU) when valid ground truth and compatible label sets exist.

---

## 1.1 Reference Paper & Model Benchmark Support

This implementation integrates and benchmarks the models from the research paper:  
**"PanoFusion: Boundary-Aware Vision-Language Segmentation for Stitched Indoor Panoramas"**  
*(Proceedings of the 17th Indian Conference on Computer Vision, Graphics and Image Processing - ICVGIP 2026, Submission 214)*

### Supported Models (Table 3 in Paper):
| Model ID | Framework | Description | Ind-PanSeg mIoU | Ind-PanSeg mAcc |
|---|---|---|:---:|:---:|
| `panofusion` | **PanoFusion (Proposed)** | Frozen CLIP + Deformable Adapter Network + SBR-Net MobileSAM gated cross-attention refinement | **54.94%** | **77.00%** |
| `ops-baseline` | **OPS Baseline [Zheng et al., ECCV 2024]** | Coarse vision-language backbone (CLIP + DAN) without SBR-Net | 44.12% | 66.46% |
| `omnisam-baseline` | **OmniSAM [Zhong et al., ICCV 2025]** | Omnidirectional Segment Anything Model with SAM2 memory adaptation | 41.66% | 51.62% |
| `segformer-baseline` | **SegFormer [Xie et al., NeurIPS 2021]** | Transformer-based perspective baseline fine-tuned with focal loss | 31.35% | 70.57% |
| `deterministic-mock` | **Ind-PanSeg Ground Truth** | Deterministic reference adapter adhering to Ind-PanSeg 21-class schema | Reference | Reference |

### Ind-PanSeg Semantic Taxonomy (21 Classes, Table 2 in Paper):
- **Structural Classes:** `Roof` (0), `Wall` (5), `Floor` (8), `Door` (11), `Curtain` (12), `Window` (14), `Obstruction` (2), `Misc.` (15)
- **Object-Level Classes:** `Bed` (1), `Refrigerator` (3), `Phone` (4), `Furniture` (6), `Electronics` (7), `Monitor` (9), `Cabinet` (10), `Pillow` (13), `Chair` (16), `Table` (17), `Boxes` (18), `Bag` (19), `Bottles` (20)
- **Ignore Index:** `255` (unlabeled/invalid/stitched seam boundary regions)

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
           | External Python/CLI Runtime | |    Remote HTTP Inference    |
           |   (scripts/panoramic_*.py)  | |        REST Endpoint        |
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
                     | (360° Yaw Wrap-around, mIoU, Dice)    |
                     +---------------------------------------+
```

### Module Split
- **`model`:** Core data structures (`SegmentationRequest`, `SegmentationResult`, `MaskData`, `SemanticClass`, `ToolMetadata`, `ExecutionStatus`).
- **`segmentation`:** The common `SegmentationTool` contract.
- **`adapter`:** Adapter implementations (`SegmentationAdapter`, `ProcessModelAdapter`, `HttpModelAdapter`, `DeterministicTestAdapter`).
- **`config`:** Jackson JSON loader (`ToolConfigurationLoader`) and dynamic `ToolRegistry`.
- **`execution`:** External process boundary (`ProcessExecutor`) and HTTP boundary (`HttpExecutor`).
- **`normalization`:** Standardized result normalization (`ResultNormalizer`, `NormalizationException`).
- **`visualization`:** Panorama-aware rendering (`PanoramaImageRenderer`) supporting horizontal cylindrical wrap-around modulo shifts and alpha overlays.
- **`comparison`:** Quantitative metrics engine (`MetricsCalculator`, `EvaluationResult`) computing Pixel Accuracy, mIoU, Dice/F1, and Per-class IoU.
- **`validation`:** File, image aspect ratio, tool configuration, and dimension validation (`ValidationService`, `ValidationResult`).
- **`controller`:** Asynchronous thread pool coordinator (`AsyncSegmentationCoordinator`) with callback listeners and execution result caching.
- **`gui`:** JavaFX presentation layer (`PanoramicApp`, `MainViewController`, `PanoramaViewPane`, `ComparisonViewPane`).

---

## 3. Technology Stack & Requirements

- **JDK:** Java 21 LTS (`java version 21.0.1`)
- **Build Tool:** Apache Maven 3.9+ (includes `mvnw.cmd` wrapper)
- **GUI Toolkit:** JavaFX 21 (`javafx-controls`, `javafx-fxml`, `javafx-swing`, `javafx-graphics`)
- **JSON Serialization:** Jackson Databind 2.17.0 & `jackson-datatype-jsr310`
- **Testing:** JUnit 5 Jupiter (`5.10.2`)
- **External Model Runtime:** Python 3.9+ with OpenCV (`cv2`) and NumPy (optional: PyTorch) for the external Python model runner.

---

## 4. How to Build, Test, and Run

### 4.1 Running Automated Tests
Execute the comprehensive test suite (35 automated tests covering unit, integration, and end-to-end workflows):
```bash
.\mvnw.cmd test
```
Or on Linux/macOS:
```bash
mvn test
```

### 4.2 Running the Desktop Application
Launch the JavaFX desktop GUI application:
```bash
.\mvnw.cmd javafx:run
```
Or run the packaged application:
```bash
.\mvnw.cmd compile exec:java -Dexec.mainClass="edu.iitm.cs5013.panoramic.Main"
```

---

## 5. Adding a New Segmentation Tool

Adding a new segmentation model does **not** require modifying the core GUI or workflow layers. To integrate a new model:

### Step 1: Define Configuration in `config/tools.json`
Add a configuration entry for your tool:
```json
{
  "id": "my-panoramic-model",
  "name": "My Custom 360 Model",
  "description": "High-resolution equirectangular indoor segmenter",
  "executionType": "PROCESS",
  "commandTemplate": [
    "python",
    "path/to/my_inference.py",
    "--input",
    "{IMAGE_PATH}",
    "--output",
    "{OUTPUT_PATH}"
  ],
  "workingDirectory": ".",
  "timeoutSeconds": 60,
  "enabled": true,
  "parameters": {
    "confidence_threshold": 0.5
  },
  "supportedClasses": [
    { "id": 0, "name": "Ceiling", "red": 70, "green": 130, "blue": 180 },
    { "id": 1, "name": "Wall", "red": 245, "green": 222, "blue": 179 },
    { "id": 2, "name": "Floor", "red": 139, "green": 69, "blue": 19 }
  ]
}
```

For a remote cloud inference service, specify `"executionType": "HTTP"` and set `"endpointUrl": "https://api.example.com/segment"`.

### Step 2: Implement Adapter (If Custom Pre/Post Processing is Needed)
If the tool requires custom serialization, extend `SegmentationAdapter`:
```java
public class MyCustomModelAdapter implements SegmentationAdapter<String> {
    @Override
    public void preprocess(SegmentationRequest request) throws SegmentationException {
        // Prepare model-specific inputs
    }

    @Override
    public String executeRaw(SegmentationRequest request) throws SegmentationException {
        // Invoke external process or HTTP endpoint
    }

    @Override
    public SegmentationResult postprocess(String rawOutput, SegmentationRequest request) throws SegmentationException {
        // Convert to canonical SegmentationResult
    }

    @Override
    public SegmentationException translateError(Throwable t, SegmentationRequest request) {
        // Translate into domain-specific exception
    }
}
```

### Step 3: Register in ToolRegistry
Register the adapter in `ToolRegistry`:
```java
toolRegistry.registerTool(new MyCustomModelAdapter(...));
```
The GUI will automatically discover the newly registered tool, display it in the tool selection panel, and integrate it into comparison workflows.

---

## 6. How External Model Execution Works

1. **Isolation:** External Python scripts, PyTorch runtimes, or C++ binaries run outside the Java process. The Java application communicates strictly through `ProcessBuilder` (standard input/output and output files) or HTTP REST.
2. **Template Substitution:** Command templates in `config/tools.json` support dynamic tokens (`{IMAGE_PATH}`, `{OUTPUT_PATH}`, `{TOOL_DIR}`).
3. **Deadlock-Free Stream Capture:** Standard output (`stdout`) and standard error (`stderr`) are captured on dedicated worker threads to prevent OS pipe buffer deadlocks.
4. **Timeout Enforcement:** Configurable timeouts ensure unresponsive external models are terminated forcibly and translated into controlled `SegmentationException` instances.
5. **No Hard-Coded Paths:** All paths and commands are relative or loaded from `config/tools.json`.

---

## 7. Known Limitations & Plan B

1. **GPU Dependencies:** Deep learning models (e.g. HoHoNet) require CUDA-capable GPUs and specific Python environments. If GPU runtimes are unavailable on a target machine, the built-in lightweight Python runner (`scripts/panoramic_segmenter.py`) or the deterministic test adapter (`DeterministicTestAdapter`) allow full testing of the Java framework without modification.
2. **Ground Truth Compatibility:** Quantitative metrics (mIoU, Pixel Accuracy, Dice) are calculated only when a valid ground-truth mask with matching dimensions and compatible label definitions is provided. When ground truth is unavailable, visual side-by-side inspection remains active while quantitative calculation is gracefully marked as unavailable.
3. **Aspect Ratio:** Standard equirectangular panoramas follow a 2:1 aspect ratio ($360^\circ \times 180^\circ$). Non-standard aspect ratios trigger a validation warning in the interface while permitting execution.
