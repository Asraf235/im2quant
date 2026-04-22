"""Configuration dataclass for im2quant."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ── Required paths ────────────────────────────────────────────────────────
    image_dir: str = ""
    csv_file: str = ""
    output_dir: str = "./runs/im2quant"

    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone: str = "yolo26n"
    freeze_backbone: bool = False

    # ── Data quality filters ──────────────────────────────────────────────────
    cov_threshold: float = 0.5          # Low CoV < cov_threshold
    r_min: float = 0.1                  # Minimum resistance (Ohm)
    r_max: float = 1e6                  # Maximum resistance (Ohm)
    train_low_cov_only: bool = True     # Exclude high-CoV batches from training
    z_score_threshold: float = 3.0     # Per-image z-score filter within batch

    # ── Print condition columns ───────────────────────────────────────────────
    condition_cols: List[str] = field(default_factory=lambda: [
        "printlayers",
        "printmaxpower(%)",
        "printspeed(mm/s)",
        "polishlayers",
        "polishmaxpower(%)",
    ])

    # ── Training hyperparameters ──────────────────────────────────────────────
    epochs: int = 100
    patience: int = 30
    lambda_weight: float = 0.4   # 0.4 * MSE + 0.6 * BCE
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15

    # ── Image processing ──────────────────────────────────────────────────────
    crop_half_frac: float = 0.20     # Crop +/- this fraction of H around wire
    img_size: Tuple[int, int] = (224, 224)
    log_transform: bool = True       # Predict log10(R) instead of R

    # ── Optuna tuning ─────────────────────────────────────────────────────────
    n_trials: int = 100
    n_tune_epochs: int = 30
    tune_patience: int = 30
