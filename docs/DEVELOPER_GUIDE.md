# Developer & Extensibility Guide: Integrating New Tools & Models
**Course:** CS5013 – Programming with AI | **Dept. of Computer Science & Engineering, IIT Madras**  
**Instructor:** Prof. Sukhendu Das | **Student:** Rathan Balaji P R (Roll No: CS26E009)  
**System:** Modular GUI Framework for Multi-Tool Panoramic Image Segmentation

---

## 1. Architectural Philosophy: The Open-Closed Principle (OCP)

A primary architectural requirement of this framework (CS5013 Design Document, Section 7) is **extensibility**:
> *"The segmentation model is treated as a replaceable service behind a stable Java interface. Adding a new segmentation model must NEVER require modifying the core GUI, workflow, or comparison code."*

To achieve this, the system decouples tool execution using the **Gang of Four (GoF) Adapter Pattern**:

```
+---------------------------------------------------------------------------------------+
|                                    JavaFX GUI & COORDINATOR                           |
|  MainViewController  <--->  AsyncSegmentationCoordinator  <--->  ToolRegistry        |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                         SegmentationTool Interface (Common Contract)                  |
|  + getMetadata(): ToolMetadata                                                        |
|  + getConfiguration(): ToolConfig                                                     |
|  + isAvailable(): boolean                                                             |
|  + segment(request: SegmentationRequest): SegmentationResult                         |
+---------------------------------------------------------------------------------------+
             |                                                |
             v                                                v
+---------------------------------------+   +-------------------------------------------+
|          ProcessModelAdapter          |   |             HttpModelAdapter              |
|  - Spawns OS Subprocess via           |   |  - Sends JSON POST request to             |
|    ProcessBuilder                     |   |    remote REST microservice               |
|  - Manages timeouts & stream buffers  |   |  - Handles HTTP status codes & timeouts   |
+---------------------------------------+   +-------------------------------------------+
             |                                                |
             v                                                v
    [ Python / PyTorch / Java ]                         [ Cloud / FastAPI ]
```

Because all model configurations are externalized in [**`config/tools.json`**](file:///c:/Users/balaj/Documents/PAI%20Project/config/tools.json), integrating a new tool requires **zero Java recompilation**.

---

## 2. Standard 3-Step Integration Procedure

Integrating any new panoramic segmentation model (whether written in Python, PyTorch, C++, or exposed via a REST API) involves three simple steps:

```
Step 1: Create Runner Script  --->  Step 2: Add Entry in tools.json  --->  Step 3: Launch GUI & Verify
```

---

### Step 1: Implement the Model Runner Script

The model runner must accept input/output parameters from the command line and produce a standardized output format.

#### CLI Contract Requirements
Your executable script must accept the following command-line flags:
* `--input <path>`: Path to the equirectangular panorama image (JPG, PNG).
* `--output <path>`: Path where the model must write its output JSON file.
* `--checkpoint <path>` *(optional)*: Path to the trained model weights file (`.pth`, `.pt`, `.onnx`).

#### Canonical Output JSON Schema
When inference finishes, your runner must write a JSON file to `--output`:

```json
{
  "tool": "my-panoramic-model",
  "modelName": "My Panoramic Network v1",
  "description": "Custom horizon-aware indoor panorama segmenter",
  "segmentationType": "SEMANTIC",
  "width": 800,
  "height": 400,
  "classes": [
    { "id": 0, "name": "Roof", "red": 70, "green": 130, "blue": 180 },
    { "id": 1, "name": "Wall", "red": 245, "green": 222, "blue": 179 },
    { "id": 2, "name": "Floor", "red": 139, "green": 69, "blue": 19 }
  ],
  "labels": [0, 0, 1, 1, 2, 2, ...],
  "customResearchMetric": 0.892
}
```

#### High-Resolution Panorama Support (`maskPath`)
For ultra-high-resolution panoramas (e.g., $7808 \times 3744$ like `H.jpg`), serializing 29 million integers into a JSON array creates large files (>90 MB). To ensure maximum performance, output a compressed 8-bit or 16-bit PNG mask and reference it via `"maskPath"`:

```json
{
  "tool": "my-panoramic-model",
  "modelName": "My High-Res Model",
  "segmentationType": "SEMANTIC",
  "width": 7808,
  "height": 3744,
  "classes": [ ... ],
  "maskPath": "target/cache/my_model_mask.png"
}
```

> **Attribute Preservation Rule:** Any additional fields you output (such as `customResearchMetric`, `depthMapProjection`, or `fovPitch`) are never silently dropped; the framework's `ResultNormalizer` captures them automatically in the `unsupportedAttributes` map for diagnostics and auditing.

---

### Step 2: Register the Tool in `config/tools.json`

Open [**`config/tools.json`**](file:///c:/Users/balaj/Documents/PAI%20Project/config/tools.json) and add a new tool object to the array:

```json
{
  "id": "my-panoramic-model",
  "name": "My Panoramic Model",
  "description": "Horizon-aware indoor equirectangular segmenter",
  "executionType": "PROCESS",
  "commandTemplate": [
    "python",
    "scripts/my_model_runner.py",
    "--input",
    "{IMAGE_PATH}",
    "--output",
    "{OUTPUT_PATH}",
    "--checkpoint",
    "PAI Segmentation Checkpoints/MyModel/weights.pth"
  ],
  "workingDirectory": ".",
  "timeoutSeconds": 120,
  "enabled": true,
  "parameters": {
    "framework": "PyTorch",
    "backbone": "ResNet-50",
    "checkpoint_path": "PAI Segmentation Checkpoints/MyModel/weights.pth"
  },
  "supportedClasses": [
    { "id": 0, "name": "Roof", "red": 70, "green": 130, "blue": 180 },
    { "id": 1, "name": "Wall", "red": 245, "green": 222, "blue": 179 },
    { "id": 2, "name": "Floor", "red": 139, "green": 69, "blue": 19 }
  ]
}
```

#### Template Variables Supported
Inside `"commandTemplate"`, the framework automatically interpolates the following variables:
* `{IMAGE_PATH}`: Absolute path to the selected panoramic image.
* `{OUTPUT_PATH}`: Unique path to a temporary output file managed by the framework.
* `{TOOL_DIR}`: Path specified in `"workingDirectory"`.

---

### Step 3: Launch and Verify in the GUI

Run the application:
```powershell
.\run.bat
```

1. The new tool will appear automatically in the **"Select Segmentation Tools"** sidebar checkboxes.
2. It will appear in the **"Active Display Tool"** selector dropdown.
3. When executed alongside other tools, it will appear in the **"Cross-Model Comparison"** multi-tile inspection grid and quantitative evaluation table.

---

## 3. Concrete Python / PyTorch Integration Example

Here is a minimal, complete template for a PyTorch segmentation runner (`scripts/my_pytorch_segmenter.py`):

```python
#!/usr/bin/env python3
import argparse
import json
import os
import cv2
import numpy as np
import torch

def run_inference(image_path, output_path, checkpoint_path=None):
    # 1. Load image (using OpenCV for robust non-standard JPEG compatibility)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    
    h, w = img.shape[:2]

    # 2. Load model weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}")
        # model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    # 3. Model forward pass (example placeholder)
    # output = model(torch_tensor)
    # labels = torch.argmax(output, dim=1).cpu().numpy()
    labels = np.zeros((h, w), dtype=np.uint8)
    labels[:h//4, :] = 0  # Roof
    labels[h//4:3*h//4, :] = 1  # Wall
    labels[3*h//4:, :] = 2  # Floor

    # 4. Handle output storage
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    result_payload = {
        "tool": "my-pytorch-tool",
        "modelName": "Custom PyTorch Segmenter",
        "segmentationType": "SEMANTIC",
        "width": int(w),
        "height": int(h),
        "classes": [
            {"id": 0, "name": "Roof", "red": 70, "green": 130, "blue": 180},
            {"id": 1, "name": "Wall", "red": 245, "green": 222, "blue": 179},
            {"id": 2, "name": "Floor", "red": 139, "green": 69, "blue": 19}
        ]
    }

    # For high-res images, write PNG mask; otherwise inline array
    if w * h > 1000000:
        mask_png = output_path.replace(".json", "_mask.png")
        cv2.imwrite(mask_png, labels)
        result_payload["maskPath"] = os.path.abspath(mask_png).replace("\\", "/")
    else:
        result_payload["labels"] = labels.flatten().tolist()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_payload, f)

    print(f"Inference complete: {w}x{h} -> {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    run_inference(args.input, args.output, args.checkpoint)
```

---

## 4. Integrating Remote HTTP / Cloud Microservices

If you deploy your model as a cloud inference API (e.g. using FastAPI or Flask), set `"executionType": "HTTP"`:

```json
{
  "id": "cloud-panoseg-api",
  "name": "Cloud PanoSeg Microservice",
  "description": "GPU-accelerated remote segmentation API",
  "executionType": "HTTP",
  "endpointUrl": "http://10.42.0.1:8000/api/v1/segment",
  "timeoutSeconds": 90,
  "enabled": true,
  "parameters": {
    "device": "cuda:0",
    "batch_size": 1
  },
  "supportedClasses": [ ... ]
}
```

The framework's `HttpModelAdapter` will automatically send a JSON POST request:
```json
{
  "imagePath": "C:/path/to/panorama.jpg",
  "parameters": {
    "device": "cuda:0",
    "batch_size": 1
  }
}
```
And expects the canonical JSON schema in the HTTP response body.

---

## 5. Integrating Mock Tools for Automated Testing

For unit testing, CI pipelines, or machines without GPU/Python environments, set `"executionType": "MOCK"`:

```json
{
  "id": "my-mock-tool",
  "name": "Deterministic CI Mock",
  "description": "Mock model producing predictable geometric layers",
  "executionType": "MOCK",
  "commandTemplate": [],
  "workingDirectory": ".",
  "timeoutSeconds": 5,
  "enabled": true,
  "parameters": {},
  "supportedClasses": [ ... ]
}
```

The framework routes mock tools to `DeterministicTestAdapter`, generating instant synthetic outputs with zero process overhead.

---

## 6. Developer Checklist & Best Practices

1. **Seam Wrap-Around Handling:**
   * When implementing boundary refinement in your model, ensure feature padding wraps cyclically across the horizontal boundary so objects spanning $0^\circ$ and $360^\circ$ are segmented as single continuous regions.
2. **Subprocess Pipe Deadlock Prevention:**
   * The framework's `ProcessExecutor` captures `stdout` and `stderr` asynchronously on dedicated worker threads. Ensure your script flushes its logs (`sys.stdout.flush()`) or prints upon exit.
3. **Timeout Protection:**
   * Always configure realistic `"timeoutSeconds"` in `tools.json`. If a model hangs or runs out of GPU memory, the framework automatically terminates the process cleanly and displays a red error badge in the GUI without freezing the user interface.
4. **Validating Changes:**
   * Before committing, verify the test suite:
     ```powershell
     .\mvnw.cmd test
     ```
   * All unit, boundary, and end-to-end tests must remain green (`BUILD SUCCESS`).
