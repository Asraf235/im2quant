"""Data loading, metadata construction, and stratified splitting."""

import math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config
from .utils import parse_batch_sample


def load_csv(cfg: Config) -> pd.DataFrame:
    """
    Load the results CSV and add a binary CoV label.

    The CSV is expected to have a ``Batch`` column and per-image resistance
    columns named ``Line_1_R`` … ``Line_5_R``, plus ``CoV_R``.

    Returns a DataFrame with ``batch_id`` (renamed from ``Batch``) and a
    ``label`` column: 1 = Low CoV, 0 = High CoV.
    """
    df = pd.read_csv(cfg.csv_file)
    df.rename(columns={"Batch": "batch_id"}, inplace=True)
    for col in cfg.condition_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["label"] = (df["CoV_R"] < cfg.cov_threshold).astype(int)
    return df


def build_metadata(cfg: Config, df_batch: pd.DataFrame) -> pd.DataFrame:
    """
    Scan *image_dir* and build a per-image metadata DataFrame.

    Applies three quality filters:

    1. **Physical range**: ``r_min ≤ per_image_R ≤ r_max``
    2. **CoV filter**: when ``train_low_cov_only=True``, skip batches with
       High CoV (label == 0).
    3. **Z-score filter**: skip images whose resistance deviates more than
       ``z_score_threshold`` standard deviations from the batch mean.

    Parameters
    ----------
    cfg : Config
    df_batch : DataFrame returned by :func:`load_csv`.

    Returns
    -------
    DataFrame with one row per retained image.
    """
    image_dir = Path(cfg.image_dir)
    records = []
    skipped = {"r_filter": 0, "cov": 0, "zscore": 0, "parse": 0}

    for batch_id in sorted(df_batch["batch_id"].unique()):
        mask = df_batch["batch_id"] == batch_id
        if not mask.any():
            continue
        row = df_batch[mask].iloc[0]

        for fpath in sorted(image_dir.glob(f"batch{batch_id}.*")):
            parsed = parse_batch_sample(fpath.name)
            if parsed is None:
                skipped["parse"] += 1
                continue
            _, sample_id = parsed

            col_name = f"Line_{sample_id}_R"
            if col_name not in df_batch.columns:
                continue
            per_image_r = float(row[col_name])

            # Filter 1: physical resistance range
            if per_image_r < cfg.r_min or per_image_r > cfg.r_max:
                skipped["r_filter"] += 1
                continue

            # Filter 2: CoV-based quality gate
            if cfg.train_low_cov_only and int(row["label"]) == 0:
                skipped["cov"] += 1
                continue

            # Filter 3: per-image z-score within batch
            avg_r = float(row["Average_R"])
            std_r = float(row["StdDev_R"])
            if std_r > 0:
                z = abs(per_image_r - avg_r) / std_r
                if z > cfg.z_score_threshold:
                    skipped["zscore"] += 1
                    continue

            record = {col: row[col] for col in cfg.condition_cols}
            record.update(
                {
                    "batch_id": int(batch_id),
                    "sample_id": int(sample_id),
                    "image_path": str(fpath),
                    "per_image_R": per_image_r,
                    "Average_R": avg_r,
                    "StdDev_R": std_r,
                    "CoV_R": float(row["CoV_R"]),
                    "label": int(row["label"]),
                }
            )
            records.append(record)

    meta_df = pd.DataFrame(records)
    print(
        f"build_metadata: {len(meta_df)} images retained | "
        f"skipped r_filter={skipped['r_filter']}  cov={skipped['cov']}  "
        f"zscore={skipped['zscore']}  parse={skipped['parse']}"
    )
    return meta_df


def stratified_split(meta_df: pd.DataFrame, cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Split at the **batch** level, stratified by log10(Average_R) decade.

    Rules
    -----
    * Batches are grouped by ``math.floor(log10(avg_R))``.
    * Bins with **< 3 batches** → all batches go to train.
    * Bins with **≥ 3 batches** → 70 / 15 / 15 train / val / test split.
    * No batch appears in more than one split (asserted).

    Returns
    -------
    ``{'train': df, 'val': df, 'test': df}``
    """
    batch_avg = meta_df.groupby("batch_id")["Average_R"].mean()

    def _decade_bin(avg_r: float) -> int:
        if avg_r <= 0:
            return -1
        return math.floor(math.log10(avg_r))

    batch_bins = batch_avg.apply(_decade_bin)

    rng = np.random.RandomState(cfg.seed)
    train_batches: list = []
    val_batches: list = []
    test_batches: list = []

    for _bin_val, group in batch_bins.groupby(batch_bins):
        batches = group.index.tolist()
        shuffled = rng.permutation(batches).tolist()
        n = len(shuffled)

        if n < 3:
            train_batches.extend(shuffled)
        else:
            n_test = max(1, int(round(n * cfg.test_split)))
            n_val = max(1, min(int(round(n * cfg.val_split)), n - n_test - 1))
            test_batches.extend(shuffled[:n_test])
            val_batches.extend(shuffled[n_test : n_test + n_val])
            train_batches.extend(shuffled[n_test + n_val :])

    train_set = set(train_batches)
    val_set = set(val_batches)
    test_set = set(test_batches)

    assert not (train_set & val_set), "Batch overlap between train and val!"
    assert not (train_set & test_set), "Batch overlap between train and test!"
    assert not (val_set & test_set), "Batch overlap between val and test!"

    splits = {
        "train": meta_df[meta_df["batch_id"].isin(train_set)].reset_index(drop=True),
        "val": meta_df[meta_df["batch_id"].isin(val_set)].reset_index(drop=True),
        "test": meta_df[meta_df["batch_id"].isin(test_set)].reset_index(drop=True),
    }

    for name, df in splits.items():
        batches_in_split = df["batch_id"].nunique()
        print(f"  {name:5s}: {len(df):4d} images, {batches_in_split} batches")

    return splits
