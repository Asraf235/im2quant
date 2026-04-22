"""PyTorch Dataset and transform factory for im2quant."""

from typing import List, Optional

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .config import Config
from .utils import crop_central_line

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class WireDataset(Dataset):
    """
    Per-image resistance dataset for dual-head model training.

    Regression target
    -----------------
    ``per_image_R`` (the individual ``Line_N_R`` measurement), **not**
    the batch-average ``Average_R``.  All images contribute to the
    regression loss; the CoV label is used only by the classifier head.

    Normalisation
    -------------
    Condition columns are z-score normalised.  When ``cond_mean`` /
    ``cond_std`` are provided (e.g. from the training split) those
    statistics are used for val/test datasets so no information leaks.

    Parameters
    ----------
    df : DataFrame returned by :func:`~im2quant.pipeline.build_metadata`.
    cfg : :class:`~im2quant.config.Config`
    transform : torchvision transform applied to the cropped, resized PIL image.
    cond_mean : Optional pre-computed condition mean (length = n_conditions).
    cond_std : Optional pre-computed condition std (length = n_conditions).
    """

    def __init__(
        self,
        df,
        cfg: Config,
        transform=None,
        cond_mean: Optional[np.ndarray] = None,
        cond_std: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform
        self._resize = T.Resize(cfg.img_size, antialias=True)

        cond_vals = df[cfg.condition_cols].values.astype(np.float32)
        if cond_mean is None:
            self._cond_mean = cond_vals.mean(axis=0)
            self._cond_std = cond_vals.std(axis=0)
        else:
            self._cond_mean = np.array(cond_mean, dtype=np.float32)
            self._cond_std = np.array(cond_std, dtype=np.float32)
        # Avoid division by zero for constant columns
        self._cond_std[self._cond_std == 0] = 1.0

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        # ── Image ─────────────────────────────────────────────────────────────
        img = Image.open(row["image_path"]).convert("RGB")
        img = crop_central_line(img, crop_half_frac=self.cfg.crop_half_frac)
        img = self._resize(img)
        img = self.transform(img) if self.transform else TF.to_tensor(img)

        # ── Conditions (z-score normalised) ────────────────────────────────────
        cond = (
            row[self.cfg.condition_cols].values.astype(np.float32) - self._cond_mean
        ) / self._cond_std

        # ── Regression target: per-image resistance ────────────────────────────
        r = float(row["per_image_R"])
        target_r = np.log10(r) if self.cfg.log_transform else r

        # ── Classifier label: Low CoV (1) / High CoV (0) ──────────────────────
        label = int(row["label"])

        return {
            "image": img,
            "conditions": torch.tensor(cond, dtype=torch.float32),
            "target_r": torch.tensor([target_r], dtype=torch.float32),
            "label": torch.tensor([label], dtype=torch.long),
            "batch_id": int(row["batch_id"]),
            "sample_id": int(row["sample_id"]),
            "per_image_R": r,
        }


def make_transforms(cfg: Config):
    """
    Return ``(train_transform, eval_transform)`` with ImageNet normalisation.

    Training augmentation: random-resized crop, horizontal flip, small
    rotation, and colour jitter.  Evaluation: tensor conversion + normalise
    only (no spatial augmentation).
    """
    train_tf = T.Compose(
        [
            T.RandomResizedCrop(
                cfg.img_size[0], scale=(0.85, 1.0), ratio=(0.9, 1.1), antialias=True
            ),
            T.RandomHorizontalFlip(),
            T.RandomRotation(5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf
