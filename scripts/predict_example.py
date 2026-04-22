"""
Predict resistance for a single new image.

Usage
-----
    python scripts/predict_example.py

Edit IMAGE_PATH and CONDITIONS below to match your new image.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from im2quant.inference import predict_single_image

if __name__ == "__main__":

    CHECKPOINT = os.path.join(
        PROJECT_ROOT, "runs", "platinum", "best_yolo26n_tuned.pt"
    )

    # ── Edit these two things for each new image ───────────────────────────────

    IMAGE_PATH = os.path.join(PROJECT_ROOT, "examples", "batch3.1.jpg")   # ← your image

    CONDITIONS = {                     # ← print parameters used for that image
        "printlayers":      48,
        "printmaxpower(%)": 13,
        "printspeed(mm/s)": 12,
        "polishlayers":     3,
        "polishmaxpower(%)": 19,
    }

    # ── Run prediction ────────────────────────────────────────────────────────
    result = predict_single_image(
        checkpoint_path=CHECKPOINT,
        image_path=IMAGE_PATH,
        conditions=CONDITIONS,
    )

    print(f"\nImage    : {os.path.basename(IMAGE_PATH)}")
    print(f"R        : {result['R_ohm']:.3f}  Ω")
    print(f"log10(R) : {result['log10R']:.3f}")
    print(f"P(LowCoV): {result['p_low_cov']:.3f}  "
          f"({'good quality' if result['p_low_cov'] > 0.5 else 'high variability'})")
