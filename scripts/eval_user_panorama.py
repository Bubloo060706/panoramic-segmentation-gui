import sys
import subprocess
import time
import json
import cv2
import numpy as np

sys.path.append('scripts')
from common_vplab import CLASSES, PALETTE

gt = cv2.imread('data/ground_truth/sample_panorama_gt.png', cv2.IMREAD_UNCHANGED)
if gt.ndim == 3:
    gt = gt[:, :, 0]

models = [
    ('SegFormer', 'scripts/run_segformer.py', []),
    ('PanoFusion', 'scripts/run_panofusion.py', ['--refine', 'true']),
    ('OPS Baseline', 'scripts/run_ops.py', []),
    ('OmniSAM', 'scripts/run_omnisam.py', []),
    ('SAM + CNN', 'scripts/run_sam_cnn.py', [])
]

print(f"=== BENCHMARK ON USER PANORAMA ({gt.shape[1]}x{gt.shape[0]}) ===\n")

for name, script, extra in models:
    safe_name = name.lower().replace(" ", "_").replace("+", "")
    out_json = f"target/eval_{safe_name}.json"
    cmd = ['python', script, '--input', 'data/sample_panorama.jpg', '--output', out_json] + extra
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = round(time.time() - t0, 2)

    if res.returncode != 0:
        print(f"{name:15s} FAILED: {res.stderr[:200]}")
        continue

    data = json.load(open(out_json))
    pred = np.array(data['labels'], dtype=np.uint8).reshape(gt.shape)

    ious = []
    # active classes in this panorama
    for c in [0, 5, 8, 10, 11, 14, 15, 16, 17, 18]:
        p = (pred == c)
        g = (gt == c)
        u = (p | g).sum()
        if u > 0:
            inter = (p & g).sum()
            ious.append(inter / u)

    m_iou = np.mean(ious) * 100
    p_acc = (pred == gt).mean() * 100
    print(f"{name:15s} ({dt:4.1f}s) -> Active mIoU: {m_iou:5.2f}%, Pixel Acc: {p_acc:5.2f}%, Detected: {len(data['classes'])} classes")

