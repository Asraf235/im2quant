"""
Full training example for im2quant.

Usage
-----
    python scripts/train_example.py
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path so im2quant package is found
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from im2quant.config import Config
from im2quant.pipeline import load_csv, build_metadata, stratified_split
from im2quant.train import tune_hyperparameters, train

if __name__ == "__main__":
    # ── Configuration ──────────────────────────────────────────────────────────
    cfg = Config(
        image_dir=os.path.join(PROJECT_ROOT, "Platinum_images"),
        csv_file=os.path.join(PROJECT_ROOT, "Platinum_results.csv"),
        output_dir=os.path.join(PROJECT_ROOT, "runs", "platinum"),
        backbone="yolo26n",
        epochs=100,
        patience=30,
        n_trials=100,
    )

    # ── Load data ───────────────────────────────────────────────────────────────
    df = load_csv(cfg)
    print(f"CSV loaded: {df.shape}")

    meta = build_metadata(cfg, df)
    print(f"Metadata: {meta.shape}")

    splits = stratified_split(meta, cfg)
    print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")

    # Save metadata for reference
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    meta.to_csv(f"{cfg.output_dir}/metadata.csv", index=False)

    # ── Hyperparameter tuning ───────────────────────────────────────────────────
    # You already have best_params_yolo26n.json in the project root from the
    # notebook — copy it to runs/platinum/ to skip re-tuning:
    #   cp best_params_yolo26n.json runs/platinum/best_params_yolo26n.json

    params_file = Path(cfg.output_dir) / f"best_params_{cfg.backbone}.json"
    if params_file.exists():
        with open(params_file) as f:
            best_params = json.load(f)
        print(f"Loaded existing best_params: {best_params}")
    else:
        print("Running Optuna tuning ...")
        best_params = tune_hyperparameters(cfg, splits)

    # ── Train final model ───────────────────────────────────────────────────────
    ckpt_path = train(cfg, splits, best_params)
    print(f"\nDone! Checkpoint saved to: {ckpt_path}")
