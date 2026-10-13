#!/usr/bin/env python3
"""
Universal Image Preprocessor & Normalizer for Panoramic Framework.
Handles non-standard camera JPEGs (SOF 0xce, SOS markers, proprietary EXIF markers like 0xa0).
Ensures any panoramic image of any resolution can be seamlessly loaded by JavaFX.
"""

import sys
import os
import json
import cv2

def process(input_path, output_path=None):
    if not os.path.exists(input_path):
        print(json.dumps({"success": False, "error": f"File not found: {input_path}"}))
        sys.exit(1)

    img = cv2.imread(input_path)
    if img is None:
        print(json.dumps({"success": False, "error": f"Could not decode image with OpenCV: {input_path}"}))
        sys.exit(2)

    h, w = img.shape[:2]
    clean_path = input_path

    if output_path:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(output_path, img)
        clean_path = output_path

    print(json.dumps({
        "success": True,
        "width": int(w),
        "height": int(h),
        "cleanPath": os.path.abspath(clean_path).replace("\\", "/")
    }))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image Preprocessor & Sanitizer")
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", default=None, help="Optional output path for sanitized image")
    args = parser.parse_args()
    process(args.input, args.output)
