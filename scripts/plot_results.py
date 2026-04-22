"""
Publication-quality figures matching notebook_pt_autowire_v2_yolo26n.ipynb.

  training_curve.png  — semilogy train + val loss with best-epoch marker
  r2_scatter.png      — true vs predicted log10(R) for train / val / test

Both files are saved to runs/platinum/ before any window is shown.

Usage
-----
    python scripts/plot_results.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from im2quant.config import Config
from im2quant.pipeline import load_csv, build_metadata, stratified_split
from im2quant.dataset import WireDataset, make_transforms
from im2quant.inference import load_model

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "runs", "platinum")
CKPT       = os.path.join(OUTPUT_DIR, "best_yolo26n_tuned.pt")
HISTORY    = os.path.join(OUTPUT_DIR, "history.csv")


def collect_preds(model, dataset, device):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    preds, trues, bids = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pr, _ = model(
                batch["image"].to(device),
                batch["conditions"].to(device),
            )
            preds.extend(pr.cpu().reshape(-1).tolist())
            trues.extend(batch["target_r"].cpu().reshape(-1).tolist())
            bids.extend(batch["batch_id"].reshape(-1).tolist())
    return np.array(preds), np.array(trues), np.array(bids)


if __name__ == "__main__":

    plt.rcParams.update({
        "font.family":   "DejaVu Sans",
        "font.size":      9,
        "axes.linewidth": 0.8,
        "figure.dpi":     300,
    })

    # ── Config & splits ────────────────────────────────────────────────────────
    cfg = Config(
        image_dir=os.path.join(PROJECT_ROOT, "Platinum_images"),
        csv_file=os.path.join(PROJECT_ROOT, "Platinum_results.csv"),
        output_dir=OUTPUT_DIR,
        backbone="yolo26n",
    )
    df_csv = load_csv(cfg)
    meta   = build_metadata(cfg, df_csv)
    splits = stratified_split(meta, cfg)

    # ── Load model ─────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(CKPT, device)

    _, eval_tf = make_transforms(cfg)
    def make_ds(split_df):
        return WireDataset(
            split_df, cfg, transform=eval_tf,
            cond_mean=ckpt["cond_mean"], cond_std=ckpt["cond_std"],
        )

    tr_pred, tr_true, tr_bids = collect_preds(model, make_ds(splits["train"]), device)
    vl_pred, vl_true, vl_bids = collect_preds(model, make_ds(splits["val"]),   device)
    te_pred, te_true, te_bids = collect_preds(model, make_ds(splits["test"]),  device)

    r2_tr = r2_score(tr_true, tr_pred)
    r2_vl = r2_score(vl_true, vl_pred)
    r2_te = r2_score(te_true, te_pred)

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 1 — Training curves
    # ══════════════════════════════════════════════════════════════════════════
    df_h = pd.read_csv(HISTORY)

    fig_b, ax_b = plt.subplots(figsize=(4.5, 3.2))

    ax_b.semilogy(df_h.epoch, df_h.train_loss, color="#1565C0",
                  lw=1.6, label="Train", alpha=0.9)
    ax_b.semilogy(df_h.epoch, df_h.val_loss,   color="#F44336",
                  lw=1.6, label="Val",   alpha=0.9)

    ax_b.set_xlabel("Epoch", fontsize=9)
    ax_b.set_ylabel("Combined Loss", fontsize=9)
    ax_b.set_title("Training Curves", fontsize=9)
    ax_b.legend(framealpha=0.85, fontsize=8)
    ax_b.grid(True, alpha=0.25, which="both")
    ax_b.set_xlim(0, len(df_h))

    fig_b.tight_layout()
    path_b = os.path.join(OUTPUT_DIR, "training_curve.png")
    fig_b.savefig(path_b, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {path_b}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURE 2 — True vs Predicted scatter
    # ══════════════════════════════════════════════════════════════════════════
    fig_c, ax_c = plt.subplots(figsize=(4.5, 4.2))

    all_vals = np.concatenate([tr_true, vl_true, te_true,
                               tr_pred, vl_pred, te_pred])
    lo = all_vals.min() - 0.2
    hi = all_vals.max() + 0.2

    ax_c.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.6,
              label="y = x", zorder=1)
    ax_c.fill_between([lo, hi], [lo - 1, hi - 1], [lo + 1, hi + 1],
                      alpha=0.07, color="green", label="Within 10×")

    ax_c.scatter(tr_true, tr_pred, c="#1565C0", s=30, alpha=0.75,
                 edgecolors="white", linewidths=0.3, zorder=3,
                 label=f"Train  R²={r2_tr:.3f}")
    ax_c.scatter(vl_true, vl_pred, c="#FF6F00", s=50, alpha=0.9,
                 edgecolors="white", linewidths=0.4, zorder=4, marker="D",
                 label=f"Val    R²={r2_vl:.3f}")
    ax_c.scatter(te_true, te_pred, c="#2E7D32", s=50, alpha=0.9,
                 edgecolors="white", linewidths=0.4, zorder=5, marker="^",
                 label=f"Test   R²={r2_te:.3f}")

    for xi, yi, bi in zip(te_true, te_pred, te_bids):
        ax_c.annotate(str(int(bi)), (xi, yi),
                      textcoords="offset points", xytext=(4, 3),
                      fontsize=6, alpha=0.8, color="#2E7D32")

    ax_c.set_xlabel("True  log₁₀(R)  [Ω]", fontsize=9)
    ax_c.set_ylabel("Predicted  log₁₀(R)  [Ω]", fontsize=9)
    ax_c.set_title("Predicted vs True Resistance", fontsize=9)
    ax_c.legend(fontsize=7, framealpha=0.88, loc="upper left")
    ax_c.set_xlim(lo, hi)
    ax_c.set_ylim(lo, hi)
    ax_c.set_aspect("equal")
    ax_c.grid(True, alpha=0.2)

    # Dual x-tick labels: log10 value on top, Ohm value below
    # Draw the figure first so matplotlib finalises tick positions
    fig_c.tight_layout()
    fig_c.canvas.draw()
    xticks = [t for t in ax_c.get_xticks() if lo <= t <= hi]
    ax_c.set_xticks(xticks)
    ax_c.set_xticklabels(
        [f"{v:.1f}\n({10**v:.1f} Ω)" for v in xticks],
        fontsize=6.5,
    )

    fig_c.tight_layout()
    path_c = os.path.join(OUTPUT_DIR, "r2_scatter.png")
    fig_c.savefig(path_c, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {path_c}")

    # ── Print metrics ──────────────────────────────────────────────────────────
    print()
    print(f"Train : R²={r2_tr:.3f}")
    print(f"Val   : R²={r2_vl:.3f}")
    print(f"Test  : R²={r2_te:.3f}")

    # Show both windows only after everything is saved
    plt.show()
