"""
Evaluate the trained model on the test set.
Prints R² and saves predictions to runs/platinum/test_predictions.csv
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from im2quant.config import Config
from im2quant.pipeline import load_csv, build_metadata, stratified_split
from im2quant.dataset import WireDataset, make_transforms
from im2quant.inference import load_model

import torch
from torch.utils.data import DataLoader

if __name__ == "__main__":
    cfg = Config(
        image_dir=os.path.join(PROJECT_ROOT, "Platinum_images"),
        csv_file=os.path.join(PROJECT_ROOT, "Platinum_results.csv"),
        output_dir=os.path.join(PROJECT_ROOT, "runs", "platinum"),
        backbone="yolo26n",
    )

    CKPT = os.path.join(PROJECT_ROOT, "runs", "platinum", "best_yolo26n_tuned.pt")

    # ── Rebuild same splits ────────────────────────────────────────────────────
    df = load_csv(cfg)
    meta = build_metadata(cfg, df)
    splits = stratified_split(meta, cfg)

    # ── Load model + normalisation stats from checkpoint ──────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(CKPT, device)

    # ── Build test dataset using training normalisation stats ──────────────────
    _, eval_tf = make_transforms(cfg)
    test_ds = WireDataset(
        splits["test"], cfg, transform=eval_tf,
        cond_mean=ckpt["cond_mean"], cond_std=ckpt["cond_std"],
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # ── Collect predictions ────────────────────────────────────────────────────
    preds_log, trues_log, batch_ids, p_low_covs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            pred_r, logits = model(
                batch["image"].to(device), batch["conditions"].to(device)
            )
            preds_log.extend(pred_r.cpu().reshape(-1).tolist())
            trues_log.extend(batch["target_r"].cpu().reshape(-1).tolist())
            batch_ids.extend(batch["batch_id"].reshape(-1).tolist())
            p_low_covs.extend(
                torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            )

    preds_log = np.array(preds_log)
    trues_log = np.array(trues_log)

    # ── Metrics ───────────────────────────────────────────────────────────────
    ss_res = np.sum((trues_log - preds_log) ** 2)
    ss_tot = np.sum((trues_log - trues_log.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\n── Test set results ───────────────────────────")
    print(f"  N images : {len(preds_log)}")
    print(f"  R²       : {r2:.4f}")
    print(f"  Factor-2 accuracy: "
          f"{np.mean(np.abs(preds_log - trues_log) < np.log10(2)):.1%}")

    # ── Save predictions ───────────────────────────────────────────────────────
    out = pd.DataFrame({
        "batch_id":    batch_ids,
        "true_log10R": trues_log,
        "pred_log10R": preds_log,
        "true_R_ohm":  10 ** trues_log,
        "pred_R_ohm":  10 ** preds_log,
        "p_low_cov":   p_low_covs,
    })
    out_path = os.path.join(cfg.output_dir, "test_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"\n  Saved predictions → {out_path}")
