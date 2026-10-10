# CS5013: Programming with AI — Principles & Practices Document
**Course:** CS5013 – Programming with AI | **Semester:** Jan–May 2026 | **Department of CSE, IIT Madras**  
**Project:** GUI Framework for Multi-Tool Panoramic Image Segmentation  
**Student:** Rathan Balaji P R (Roll No: CS26E009)  
**Stakeholder / Faculty:** Dr. Sukhendu Das, Professor, Dept. of Computer Science & Engineering, IIT Madras  

---

## 1. Executive Summary & Architectural Overview

Panoramic image segmentation assigns semantic categories to 360° cylindrical equirectangular images. In practice, the domain presents severe geometric distortions (pole-stretching, wide spatial context, large aspect ratios up to 12:1) and runtime fragmentation (tools written in Python/PyTorch, C++, or exposed via cloud REST APIs). 

To unify these heterogeneous tools within a single, robust desktop application, we engineered a clean Java 21 + JavaFX framework adhering to four foundational architectural principles:
1. **Model Isolation Boundary:** Python/CUDA and deep-learning dependencies execute entirely outside the Java Virtual Machine via Java `ProcessBuilder`, preventing heavy native libraries from destabilizing the GUI.
2. **Canonical Result Normalization:** Heterogeneous model outputs are parsed into a uniform, immutable `SegmentationResult` schema, with any tool-specific research attributes preserved in an audit map.
3. **Panorama-Aware Visualization:** The presentation layer implements interactive 360° horizontal modulo yaw rotation, eliminating seam artifacts across the 0° / 360° boundary.
4. **Quantitative Comparison Engine:** The application calculates Pixel Accuracy, Mean IoU (mIoU), and Dice/F1 scores against Ind-PanSeg ground truth when dimensions and taxonomies align.

---

## 2. Prompt Engineering Principles & Lifecycle Application

Prompt engineering was treated as a rigorous software specification methodology across the lifecycle:

- **Principle A: Role-Framing & System Identity:** Each interaction designated the AI as a Principal Software Architect paired with a Graduate Student at IIT Madras, enforcing strict compliance with Gang of Four design patterns (Adapter, Strategy, Observer, Builder) and zero tolerance for stub code.
- **Principle B: Context Bounding & In-Context Grounding:** Prompts anchored the model directly to authoritative project artifacts: the CS5013 Project Proposal, the Design Document, and the ICVGIP 2026 research paper (*"PanoFusion: Boundary-Aware Vision-Language Segmentation for Stitched Indoor Panoramas"*). File paths, JSON schemas, and class names were bounded to prevent architectural drift.
- **Principle C: Negative Prompting & Anti-Pattern Constraints:** Prompts strictly enforced boundaries: *"Do NOT embed Python or CUDA inside JVM"*, *"Do NOT silently drop unsupported tool metadata"*, *"Do NOT invent arbitrary classes; use exact Ind-PanSeg 21 classes"*, and *"Do NOT use third-party CSS; use clean JavaFX CSS"*.
- **Principle D: Output Structuring & Few-Shot Schemas:** Exact JSON interchange schemas (`tools.json`) and POJO Builder patterns were prescribed, guaranteeing bidirectional serialization between Python subprocesses and Java.
- **Principle E: Iterative Chain-of-Thought & Stepwise Verification:** Workflows were broken into verifiable increments (Domain Models → Model Adapters → Test Suites → UI Views), with automated tests passing at each milestone.

---

## 3. Debugging: Structured Approach & Issue Resolution

Debugging across a hybrid JavaFX-Python-CUDA environment followed a four-step scientific method:

1. **Boundary Isolation (JVM vs Subprocess):** When external segmentation failed, the Python scripts were executed and debugged standalone via CLI with test images before attaching the Java `ProcessModelAdapter`.
2. **Deadlock-Free Stream Consumption:** A classic issue in subprocess management is OS pipe buffer saturation. We isolated `stdout` and `stderr` consumption onto dedicated background threads with `BufferedReader`, completely eliminating Java thread freezes.
3. **Deterministic Mock Test Baselines:** To isolate rendering and metrics bugs from hardware and network latency, we created `DeterministicTestAdapter`. It synthesizes mathematically predictable horizontal wrap-around regions, independently verifying metric formulas and modulo yaw rotation.
4. **Test-Driven Reproducers:** Whenever regressions surfaced (such as updating from a 6-class toy schema to the 21-class Ind-PanSeg benchmark), focused tests were authored in `AdapterTest.java` and `EndToEndSystemTest.java` to diagnose and fix the failure trace.

---

## 4. Version Control (Git): Code & Prompt Lifecycle Tracking

Version control provided a verifiable audit trail of code evolutions and AI prompt decisions:

- **Conventional Atomic Commits:** Followed the Conventional Commits specification (`feat:`, `test:`, `docs:`, `refactor:`), ensuring every commit represents a compiling, test-passing milestone (e.g., `feat: integrate PanoFusion and Ind-PanSeg 21-class models from ICVGIP 2026 paper`).
- **Prompt-to-Commit Traceability:** Functional changes in commit messages directly trace back to specific prompt instructions, with full prompt trajectories preserved in versioned Markdown artifacts.
- **Build Hygiene & Environment Portability:** The Maven wrapper (`mvnw.cmd`), configuration schemas (`config/tools.json`), and sample panoramas are version-controlled, while local bytecode and system dumps are isolated in `.gitignore`.
- **Linear Git History:** A rebase-friendly linear history enables transparent code inspection and effortless regression rollbacks.

---

## 5. Testing, Refactoring & Documentation Standards

- **Multi-Tier Testing Suite:** 35 automated JUnit 5 tests cover:
  - *Unit Tests:* `MetricsCalculatorTest`, `ResultNormalizerTest`, `SegmentationModelTest` (mathematical correctness of mIoU, Dice, Pixel Accuracy; schema parsing).
  - *Boundary & Execution Tests:* `ExecutionBoundaryTest`, `ValidationServiceTest` (timeouts, invalid aspect ratios, missing tools).
  - *Integration & System Tests:* `AdapterTest`, `AsyncCoordinatorTest`, `GuiWorkflowTest`, `EndToEndSystemTest` (full async pipeline from image validation to Python execution, wrap-around rendering, and metric computation).
- **Design Pattern Refactoring:**
  - *GoF Adapter Pattern:* `ProcessModelAdapter` and `HttpModelAdapter` adapt heterogeneous runtimes into a uniform `SegmentationTool` contract.
  - *Strategy Pattern:* Rendering and metric algorithms are decoupled from model execution.
  - *Observer/Listener Pattern:* `AsyncSegmentationCoordinator` updates JavaFX views asynchronously without blocking the UI thread.
  - *Builder Pattern:* Immutability and thread safety are guaranteed via `SegmentationResult.Builder`.
- **Documentation Standards:** Comprehensive `README.md`, authoritative CS5013 Proposal and Design Documents, full Javadoc across all public interfaces, and automated demonstration artifacts (`demonstration_results.md`).

---

## 6. Appendix: Sample Prompts

### Sample Prompt 1: Initial Architecture Scaffolding
> *"You are a Principal Software Architect implementing the CS5013 Design Document for IIT Madras. Implement the complete domain model and adapter contracts in Java 21. Constraints: (1) Model classes must be immutable via Builder pattern. (2) External models must NEVER run in the JVM; design ProcessModelAdapter using ProcessBuilder with background stdout/stderr stream threads. (3) If external models return unrecognized metadata attributes, do NOT drop them; store them in an unsupportedAttributes Map. (4) Provide JUnit 5 tests verifying all model constraints."*

### Sample Prompt 2: Cylindrical Wrap-Around Rendering
> *"The input is an equirectangular 360° panorama. Standard 2D viewers fail because objects split across 0° and 360° appear severed. Create PanoramaImageRenderer in JavaFX to support interactive horizontal yaw rotation using cyclic modulo offset: sourceX = (x + offsetPixels) % width. Ensure seamless interpolation across the seam and author a unit test verifying boundary continuity."*

### Sample Prompt 3: Integrating ICVGIP 2026 Paper Benchmark Models
> *"Research paper: 'PanoFusion: Boundary-Aware Vision-Language Segmentation for Stitched Indoor Panoramas' (ICVGIP 2026). Integrate the exact models from Table 3 into config/tools.json and external runners: (1) panofusion: CLIP + DAN + SBR-Net MobileSAM boundary refinement (54.94% mIoU); (2) ops-baseline: coarse backbone (44.12% mIoU); (3) segformer-baseline: perspective transformer (31.35% mIoU); (4) Ind-PanSeg 21 classes. Update test assertions and verify all 35 tests pass."*
