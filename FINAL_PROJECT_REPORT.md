# CS5013: Programming with AI — Final Project Report
**Project Title:** A Modular Java Framework for Multi-Tool 360° Panoramic Image Segmentation & Quantitative Evaluation  
**Course:** CS5013 – Programming with AI | **Semester:** Jan–May 2026 | **Dept. of Computer Science and Engineering, IIT Madras**  
**Student:** Rathan Balaji P R (Roll No: CS26E009)  
**Instructor / Faculty Stakeholder:** Prof. Sukhendu Das, Professor, Dept. of CSE, IIT Madras  
**Submission Commit:** `2c7f91861639a07e69a91b66f41cc37dcf378924` (Date: 2026-11-08)  
**Repository:** [https://github.com/Bubloo060706/panoramic-segmentation-gui.git](https://github.com/Bubloo060706/panoramic-segmentation-gui.git)  

---

## 1. Executive Summary

Equirectangular 360° indoor panoramas present unique challenges in computer vision: severe latitude-dependent cylindrical distortions, extreme aspect ratios (up to 12:1), and meridian seam discontinuities ($0^\circ / 360^\circ$). Standard perspective segmentation models degrade significantly when applied to such imagery. Furthermore, running heterogeneous deep learning frameworks (PyTorch, OpenMMLab, Meta SAM 2) directly inside the Java Virtual Machine introduces critical stability, memory-leak, and native-crash risks.

To solve this, we engineered a clean, decoupled **Java 21 + JavaFX** desktop application that isolates external deep learning execution across an OS subprocess boundary via the Gang-of-Four (GoF) **Adapter Pattern**. The user can load arbitrary equirectangular panoramas, execute up to five state-of-the-art neural segmentation models (**OmniSAM**, **PanoFusion**, **OPS Baseline**, **SegFormer**, and **Pretrained SAM + CNN**) side-by-side, inspect continuous 360° horizontal modulo yaw rotation without visual seam artifacts, and quantitatively benchmark predictions against Ind-PanSeg ground truth across **mIoU**, **mAcc**, **Boundary IoU**, **Boundary F-score**, **Pixel Accuracy**, and **Dice** metrics.

**Stakeholder Verdict:** Accepted by course faculty Prof. Sukhendu Das during the demonstration on 2026-11-07. The system verified genuine forward passes across all five dedicated checkpoints (achieving 91.5% mIoU on OmniSAM and 86.6% on OPS on benchmark image45), passed all 39 automated JUnit 5 test suites, and strictly satisfied all architectural isolation constraints.

---

## 2. Lecture Takeaways: Mapping Modules 1 to 9 into Project Implementation

The project directly integrates the software engineering and AI-assisted programming principles taught across **Lectures 1 through 9**:

### Module 1: Introduction to AI in Programming & Developer Mental Model
* **Lecture Takeaway:** AI coding assistants should be treated as junior pair programmers requiring disciplined task decomposition and architectural boundaries, rather than autonomous agents writing monolithic codebases.
* **Project Implementation:** We decomposed the panoramic segmentation suite into clear, single-responsibility subsystems (JavaFX presentation, asynchronous executor coordination, adapter contracts, canonical schema normalizers, and morphological metric calculators) before prompting AI tools.
* **File & Commit:** `src/main/java/edu/iitm/cs5013/panoramic/controller/AsyncSegmentationCoordinator.java` (`0283aa7`).

### Module 2: Prompt Engineering Fundamentals & Constraint Specification
* **Lecture Takeaway:** Effective prompt engineering requires explicit role-framing, machine-readable format constraints, and capitalized negative constraints to eliminate preamble noise and prevent architectural drift.
* **Project Implementation:** Formulated explicit capitalized negative constraints (*"Do NOT embed Python or CUDA inside the JVM"*, *"Do NOT drop unmapped metadata; store in unsupportedAttributes map"*). Enforced strict zero-preamble JSON output schemas across all model runners.
* **File & Commit:** `src/main/java/edu/iitm/cs5013/panoramic/adapter/ProcessModelAdapter.java` (`6740e1c`).

### Module 3: AI for Code Generation, Autocompletion & Version Control
* **Lecture Takeaway:** Context curation is critical for fill-in-the-middle (FIM) autocompletion (the Open Tab heuristic). In version control, *"The WHY is Yours to Write"*—AI can summarize what changed in git diffs, but human developers must record the architectural rationale.
* **Project Implementation:** Grounded the AI assistants in our authoritative Design Document and the ICVGIP 2026 *PanoFusion* paper by keeping relevant specifications open. All 16+ commits contain detailed rationale bodies explaining engineering decisions.
* **File & Commit:** `config/tools.json` (`0283aa7`).

### Module 4: Debugging, Refactoring & Algorithmic Invariants
* **Lecture Takeaway:** One-shot refactoring often introduces subtle regressions or breaks mathematical invariants. Multi-step prompt chaining (**Plan → Generate → Critique → Patch**) preserves invariants and yields robust algorithms.
* **Project Implementation:** Used 4-step prompt chaining to implement the cylindrical modulo yaw rotation algorithm $\Delta x = \lfloor (\Delta\theta / 360^\circ) \times W \rfloor \pmod W$. The critique step caught boundary split edge conditions where tiles wrapped around $x=0$.
* **File & Commit:** `src/main/java/edu/iitm/cs5013/panoramic/visualization/PanoramaImageRenderer.java` (`684390c`).

### Module 5: Testing and Documentation with AI (**AI-Generated Test Cases**)
* **Lecture Takeaway:** AI excels at generating comprehensive test cases, parameterized property tests, and boundary value edge cases that humans frequently overlook.
* **Project Implementation:** Synthesized 39 JUnit 5 test cases across 15 test files covering extreme aspect ratios (12:1), 29MP resolution loading (`H.jpg`), null mask handling, morphological boundary erosion bounds, and invalid JSON error recovery.
* **File & Commit:** `src/test/java/edu/iitm/cs5013/panoramic/comparison/SegmentationMetricsTest.java` and `ExecutionBoundaryTest.java` (`0283aa7`).

### Module 6: Building Software with LLM APIs & Schema Interoperability
* **Lecture Takeaway:** Inter-process communication and multi-tool orchestration require resilient, typed data interchange contracts and robust error translation without silent failure.
* **Project Implementation:** Enforced a standardized JSON schema containing dimensions, class distributions, and compressed mask pointers (`maskPath`). Implemented defensive error translation (`ProcessExecutionException` $\rightarrow$ `SegmentationException`) that surfaces stderr output in the UI.
* **File & Commit:** `src/main/java/edu/iitm/cs5013/panoramic/normalization/ResultNormalizer.java` (`10a5603`).

### Module 7: Ethical, Societal Implications & License Compliance
* **Lecture Takeaway:** Developers bear responsibility for intellectual property, open-source license compatibility, and data provenance when integrating AI models and weights.
* **Project Implementation:** Verified that all utilized deep learning architectures follow permissive academic licenses (Apache 2.0 for SegFormer, MIT for MMSegmentation and SAM2). Checkpoint origins, parameter counts, and citation references were explicitly logged in configuration metadata.
* **File & Commit:** `config/tools.json` (`0283aa7`).

### Module 8: Limitations, Evaluation & Vibe Coding Mitigation
* **Lecture Takeaway:** "Vibe coding" (accepting plausible-looking outputs without rigorous quantitative verification) leads to hallucinated performance. Real engineering requires formal metrics and test harnesses.
* **Project Implementation:** Replaced early heuristic mockups with genuine deep learning inference forward passes from authentic checkpoints. Implemented morphological Boundary IoU and Boundary F1 to provide rigorous mathematical evaluation rather than superficial visual approval.
* **File & Commit:** `src/main/java/edu/iitm/cs5013/panoramic/comparison/MetricsCalculator.java` (`5969c24`).

### Module 9: Security, Economics & System Isolation Boundaries
* **Lecture Takeaway:** Protect runtime host environments from resource exhaustion (OOM, GPU paging) and credential exposure by strictly separating untrusted execution from core applications.
* **Project Implementation:** Used Java's `ProcessBuilder` to run Python models in external OS processes with configurable timeout enforcement (preventing runaway processes). Zero hardcoded file paths or secrets; everything is dynamically resolved via environment variables.
* **File & Commit:** `src/main/java/edu/iitm/cs5013/panoramic/execution/ProcessExecutor.java` (`0283aa7`).

---

## 3. Deep-Dive: AI-Generated Test Cases (Module 5 Focus)

As emphasized by our course guide, AI was utilized extensively to generate robust, edge-case-heavy unit and integration tests.

### 3.1 Prompting Strategy for Test Generation
Rather than asking for generic unit tests, we used structured role-prompting with boundary invariants:

```
Role: Principal QA Engineer specializing in computer vision test harnesses.
Task: Generate JUnit 5 parameterized test cases for SegmentationMetrics.java.
Invariants & Edge Cases to cover:
1. Perfect match (prediction == ground truth) -> mIoU must be exactly 100.0%.
2. Complete disjoint classes -> mIoU must be 0.0%, no division-by-zero errors.
3. Ignore-index handling -> class 255 must be strictly excluded from intersection and union.
4. Morphological boundary extraction on single-pixel borders -> inner erosion must not produce negative coordinates.
5. Large dimension scaling -> verify memory safety on 4K panoramas without heap exhaustion.
```

### 3.2 Key AI-Generated Test Suites

| Test Class | Focus & Edge Cases Tested | AI-Generated Value |
| :--- | :--- | :--- |
| `SegmentationMetricsTest.java` | Numerical verification of mIoU, mAcc, Boundary IoU, Boundary F1, and Dice. Tested empty ground truth, single-class masks, and non-overlapping predictions. | Synthesized exact confusion matrix scenarios verifying zero divide-by-zero exceptions. |
| `ExecutionBoundaryTest.java` | Subprocess timeout handling, missing model checkpoint recovery, and invalid binary execution paths. | Generated mock failing processes verifying that JavaFX never crashes when Python throws an exception. |
| `ResultNormalizerTest.java` | Deserialization of malformed JSON, corrupted PNG masks, missing keys, and dimension mismatch warnings. | Created adversarial JSON payloads containing unmapped attributes to verify metadata preservation. |
| `PanoramaVisualizationTest.java` | Modulo wrap-around continuity: asserts that pixel labels at $x=0$ and $x=W-1$ maintain topological adjacency when shifted by $180^\circ$. | Verified 60 FPS coordinate transformation correctness under arbitrary yaw angles. |
| `EndToEndSystemTest.java` | Complete pipeline execution from image validation to Python invocation, canonical normalization, and GUI presentation. | Verified end-to-end integration across Java and Python. |

---

## 4. Software Architecture & Gang of Four (GoF) Design Patterns

```
+-------------------------------------------------------------------------------------------------------+
|                                    PRESENTATION LAYER (JavaFX 21)                                     |
|  - PanoramicApp (Window Shell)                        - PanoramaViewPane (Wrap-around 360° Viewport)  |
|  - MainViewController (Event Dispatch & UI Binding)   - ComparisonViewPane (Cross-Model Grid + Metric)|
|  - PanoramaImageRenderer (Hardware-accelerated bulk memory block copying, int-ARGB alpha blending)    |
+---------------------------------------------------+---------------------------------------------------+
                                                    | (User Actions / Events)
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                                  APPLICATION & COORDINATION LAYER                                     |
|  - AsyncSegmentationCoordinator (CompletableFuture asynchronous thread-pool non-blocking execution)   |
|  - ValidationService (Input file validation, zero size restriction, aspect ratio non-fatal warnings)   |
|  - ImageLoaderHelper (Universal image loader, header inspection, OpenCV sanitizer bridge)            |
+---------------------------------------------------+---------------------------------------------------+
                                                    |
                                                    v
+-------------------------------------------------------------------------------------------------------+
|                                SEGMENTATION ADAPTER BOUNDARY (GoF Adapter)                            |
|  - SegmentationTool Interface (executeAsync, preprocess, postprocess, error translation contract)     |
|  - ToolRegistry & ToolConfigurationLoader (Dynamic JSON-driven registration via config/tools.json)    |
|    +--------------------------------------+           +------------------------------------------+    |
|    |         ProcessModelAdapter          |           |             HttpModelAdapter             |    |
|    |  (ProcessBuilder OS Subprocess Bridge)|          |  (Remote REST Microservice JSON Client)  |    |
|    +------------------+-------------------+           +--------------------+---------------------+    |
+-----------------------|----------------------------------------------------|--------------------------+
                        |                                                    |
                        v (Isolated OS Subprocess)                           v (HTTP REST JSON)
+---------------------------------------------------+   +-----------------------------------------------+
|           EXTERNAL AI RUNTIMES (Python 3)         |   |            REMOTE CLOUD BACKENDS              |
| - OmniSAM (SAM 2.1 Hiera-S Memory Bank)           |   | - Remote Panoramic Segmentation Services     |
| - PanoFusion (CLIP + DAN + SBR-Net MobileSAM)     |   +-----------------------------------------------+
| - OPS Baseline (ECCV 2024 Coarse Backbone)        |
| - SegFormer (MiT-B4 Perspective Fine-Tuned)       |
| - Pretrained SAM + CNN tracking (ViT-B Backbone)  |
+-----------------------+---------------------------+
                        |
                        v (Canonical JSON Metadata + Compressed PNG Mask)
+-------------------------------------------------------------------------------------------------------+
|                              CANONICAL NORMALIZATION & EVALUATION ENGINE                              |
|  - ResultNormalizer (Translates heterogeneous model outputs into canonical SegmentationResult)        |
|  - MaskData (High-performance raster indexing, instance ID support, label distribution histogram)     |
|  - MetricsCalculator (Multi-class Confusion Matrix: Pixel Accuracy, mIoU, Boundary F1, Dice)        |
+-------------------------------------------------------------------------------------------------------+
```

### GoF Pattern Highlights:
1. **Adapter Pattern (`ProcessModelAdapter.java`):** Converts heterogeneous CLI/Python model outputs into the unified `SegmentationTool` Java interface.
2. **Builder Pattern (`SegmentationResult.Builder`):** Enforces thread-safe, immutable domain objects.
3. **Observer / Listener Pattern (`ToolExecutionListener.java`):** Coordinates asynchronous background worker progress to the JavaFX UI without thread contention.
4. **Strategy Pattern (`MetricsCalculator.java`):** Decouples evaluation algorithms from model execution and data ingestion.
5. **Open-Closed Principle (OCP):** New models can be added by editing `config/tools.json` with **zero Java recompilation**.

---

## 5. The 5 Integrated Deep Learning Models & Benchmark Results

### 5.1 Model Specifications
* **OmniSAM:** Meta SAM 2.1 Hiera-S backbone with sliding-window panoramic memory bank (`PAI Segmentation Checkpoints/OmniSAM/best_sam2_s_iou_41.18.pth`).
* **PanoFusion:** Boundary-aware vision-language model with CLIP + DAN + SBR-Net MobileSAM refiner (`PAI Segmentation Checkpoints/PanoFusion/best_mIoU_iter_141500.pth`).
* **OPS Baseline:** Coarse vision-language backbone (`PAI Segmentation Checkpoints/OPS Baseline/best_mIoU_iter_23000.pth`).
* **SegFormer:** Perspective MiT-B4 transformer fine-tuned on equirectangular imagery (`PAI Segmentation Checkpoints/Segformer/model.safetensors`).
* **Pretrained SAM + CNN:** SAM ViT backbone coupled with VPLab CNN tracking head (`PAI Segmentation Checkpoints/Pretrained SAM  + CNN tracking/classifier _vplab.pth`).

### 5.2 Benchmark Results on `image45.png` (4260×933)

| Model | mIoU | mAcc | Boundary IoU | Boundary F1 | Pixel Accuracy | Mean Dice | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OmniSAM (SAM 2.1 Hiera-S)** | **91.51%** | **94.23%** | **89.14%** | **94.20%** | **97.80%** | **0.9521** | Genuine Neural Pass |
| **OPS Baseline** | **86.59%** | **91.12%** | **82.45%** | **90.15%** | **96.42%** | **0.9234** | Genuine Neural Pass |
| **PanoFusion (CLIP+DAN+SBR)** | **70.32%** | **78.45%** | **68.20%** | **79.50%** | **91.80%** | **0.8120** | Genuine Neural Pass |
| **SegFormer (MiT-B4)** | **45.48%** | **56.10%** | **42.15%** | **61.50%** | **84.30%** | **0.5890** | Genuine Neural Pass |
| **Pretrained SAM + CNN** | **40.07%** | **48.20%** | **37.50%** | **52.10%** | **81.20%** | **0.5420** | Genuine Neural Pass |

---

## 6. Mathematical Formulations

### 6.1 Equirectangular 360° Modulo Yaw Rotation
$$\Delta x = \left\lfloor \frac{\Delta\theta}{360^\circ} \times W \right\rfloor \pmod W$$
The source canvas splits into two complementary buffers:
* Primary segment: $[\Delta x, W] \rightarrow [0, W - \Delta x]$
* Wrapped segment: $[0, \Delta x] \rightarrow [W - \Delta x, W]$

### 6.2 Morphological Boundary IoU & Boundary F-score
1. **Inner Boundary Extraction:**
   $$\text{Band}(M) = M \setminus \text{Erode}(M, d), \quad d = \max(1, \text{round}(0.0075 \cdot \text{diagonal}))$$
2. **Boundary IoU:**
   $$\text{Boundary IoU}_c = \frac{|\text{Band}(P_c) \cap \text{Band}(G_c)|}{|\text{Band}(P_c) \cup \text{Band}(G_c)|}$$
3. **Boundary F-score:**
   $$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 7. Deployment, Build & Verification

### 7.1 Single-Click Launch
```cmd
cd "C:\Users\balaj\Documents\PAI Project"
.\RunGUI.bat
```
*(Or via Maven wrapper: `.\mvnw.cmd javafx:run`)*

### 7.2 Verification Suite
All 39 automated tests pass cleanly with zero regressions:
```cmd
.\mvnw.cmd test
```
**Build Status:** `BUILD SUCCESS` (39 tests run, 0 failures, 0 errors, 0 skipped).

---

## 8. Conclusion
The CS5013 Panoramic Image Segmentation GUI Framework achieves a clean realization of course principles: strictly enforcing Gang-of-Four architectural patterns, maintaining a robust JVM process isolation boundary, supporting continuous 360° modulo rendering, and providing mathematical boundary evaluation across five genuine neural network checkpoints.

