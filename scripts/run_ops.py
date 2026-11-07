"""
OPS Baseline (Coarse Backbone) inference runner.
Loads: PAI Segmentation Checkpoints/OPS Baseline/best_mIoU_iter_23000.pth
Architecture: Vision-Language Coarse Baseline (CLIP-ViT-B/16 + DAN without SBR-Net).
100% Genuine Neural Network inference.
"""
import os
import sys
import subprocess
import time
import argparse

VENV_PYTHON = r"C:\Users\balaj\Documents\nighil_rathan_dl_models\venv\Scripts\python.exe"

def parse_args():
    parser = argparse.ArgumentParser(description="OPS Baseline Inference")
    parser.add_argument("--input", required=True, help="Path to input panoramic image")
    parser.add_argument("--output", required=True, help="Path to write JSON output")
    parser.add_argument("--checkpoint", default="PAI Segmentation Checkpoints/OPS Baseline/best_mIoU_iter_23000.pth", help="Path to checkpoint .pth")
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_args()
    start_time = time.time()

    # Relaunch under Python 3.9 venv if needed (where mmseg & mmcv are natively compiled)
    if os.path.exists(VENV_PYTHON) and sys.executable.lower() != os.path.abspath(VENV_PYTHON).lower():
        cmd = [VENV_PYTHON, os.path.abspath(__file__)] + sys.argv[1:]
        result = subprocess.run(cmd)
        sys.exit(result.returncode)

    # In Python 3.9 venv with mmseg
    import torch
    import numpy as np
    from PIL import Image

    sys.path.insert(0, os.path.abspath("."))
    sys.path.insert(0, os.path.abspath("scripts"))

    from mmengine.registry import init_default_scope
    try:
        init_default_scope('mmseg')
    except Exception:
        pass

    import ops
    from mmseg.apis import init_model, inference_model
    from common_vplab import save_segmentation_result

    ckpt_path = args.checkpoint
    cfg_path = os.path.abspath("scripts/ops_vplab_cfg.py")

    print(f"Loading OPS Baseline from: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = init_model(cfg_path, ckpt_path, device=device)

    print(f"Running OPS inference on {args.input}...")
    res = inference_model(model, args.input)
    pred_mask = res.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)

    save_segmentation_result(args.output, pred_mask, start_time)
    print(f"Saved OPS segmentation to: {args.output}")

if __name__ == "__main__":
    main()
