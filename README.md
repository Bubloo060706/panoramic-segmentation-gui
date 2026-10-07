# GUI Framework for Multi-Tool Panoramic Image Segmentation

**Course:** CS5013 – Programming with AI  
**Author:** Rathan Balaji P R (Roll No. CS26E009)  
**Stakeholder:** Dr. Sukhendu Das, Professor, Dept. of Computer Science & Engineering, IIT Madras  

---

## 1. Project Overview

Panoramic image segmentation aims to assign semantic or instance-level labels to regions of a 360° equirectangular image. Different segmentation tools and models differ substantially in their input conventions, execution environments (Python/PyTorch, Docker, REST services), output formats (color masks, indexed label maps, polygons), and supported tasks.

This project provides a modular Java desktop application with a JavaFX graphical interface that unifies multiple heterogeneous segmentation tools on panoramic images:
- **Unified Workflow:** Load 360° panoramas, select one or multiple tools, trigger inference, and inspect normalized outputs.
- **Model Adapter Boundary:** External models execute outside the Java GUI (via Java `ProcessBuilder` or HTTP REST) behind a stable `SegmentationTool` interface.
- **Canonical Result Normalization:** Converts model-specific output formats into a standardized `SegmentationResult` schema without silently dropping unsupported metadata.
- **Panorama-Aware Visualization:** Handles equirectangular 360° geometry, including horizontal wrap-around boundary alignment and overlays.
- **Comparison & Quantitative Evaluation:** Side-by-side visual comparison and calculation of metrics (Mean IoU, Pixel Accuracy, Dice/F1, Per-class IoU) when valid ground truth and compatible label sets exist.

---

## 2. System Architecture

```
        +-------------------------------------------------------+
        |                 JavaFX GUI (Presentation)             |
        |         Input, Tool Selection, Visualizations,        |
        |               Side-by-Side Comparison                 |
        +---------------------------+---------------------------+
                                    |
                                    v
        +-------------------------------------------------------+
        |            Application / Controller Layer             |
        |         Workflow, Validation, State Management        |
        +---------------------------+---------------------------+
                                    |
                                    v
        +-------------------------------------------------------+
        |             SegmentationTool Interface                |
        |              Common Adapter Contract                  |
        +---------------------------+---------------------------+
                                    |
                     +--------------+--------------+
                     |                             |
                     v                             v
        +-------------------------+   +-------------------------+
        |   ProcessBuilder Adapter|   |       HTTP Adapter      |
        |  (Local Model Execution)|   |   (Inference Service)   |
        +------------+------------+   +------------+------------+
                     |                             |
                     v                             v
        +-------------------------+   +-------------------------+
        | External Python Runtime |   |  Remote Model Server    |
        +-------------------------+   +-------------------------+
                     \                             /
                      \                           /
                       v                         v
        +-------------------------------------------------------+
        |                  Result Normalization                 |
        |             Canonical SegmentationResult              |
        +---------------------------+---------------------------+
                                    |
                                    v
        +-------------------------------------------------------+
        |            Panorama Visualization & Metrics           |
        |         Wrap-around Overlays, mIoU, Dice, Acc         |
        +-------------------------------------------------------+
```

---

## 3. Technology Stack & Requirements

- **JDK:** Java 21 LTS
- **Build System:** Apache Maven 3.9+
- **GUI Toolkit:** JavaFX 21 (controls, fxml, swing, graphics)
- **Serialization / Configuration:** Jackson Databind 2.17+
- **Testing:** JUnit 5 (Jupiter)
- **External Model Runtimes:** Python 3.9+ (optional for local adapters) / HTTP REST endpoints

---

## 4. Setup and Execution

### Build the Project
```bash
mvn clean compile
```

### Run Tests
```bash
mvn test
```

### Run the Desktop GUI Application
```bash
mvn javafx:run
```

---

## 5. Adding a New Segmentation Tool

To integrate an additional segmentation tool:
1. **Define Tool Configuration:** Add an entry in `config/tools.json` specifying the tool ID, display name, execution mode (`PROCESS` or `HTTP`), command line / endpoint, and expected input/output format.
2. **Implement Adapter (if custom handling is needed):** Implement `SegmentationAdapter` or extend `ProcessModelAdapter` / `HttpModelAdapter`.
3. **Register Adapter:** Register the adapter in `ToolRegistry`.
4. **Normalize Results:** Implement or configure `ResultNormalizer` mappings for the tool's output classes.
