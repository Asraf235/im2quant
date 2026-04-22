"""Model loading and single-image / batch inference for im2quant."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .model import TunableDualHeadModel
from .utils import crop_central_line

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_EVAL_TRANSFORM = T.Compose(
    [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
)


def load_model(
    checkpoint_path: str, device: Optional[str] = None
) -> Tuple[TunableDualHeadModel, dict]:
    """
    Load a trained model from a checkpoint file.

    All architecture parameters and normalisation statistics are read
    **from the checkpoint** — no training data is required at inference time.

    Parameters
    ----------
    checkpoint_path : str or Path
    device : ``'cuda'`` or ``'cpu'``.  Auto-detected if not given.

    Returns
    -------
    (model, ckpt_dict) where ``ckpt_dict`` contains ``cond_mean``,
    ``cond_std``, ``condition_cols``, ``log_transform``, etc.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    shared_layers = ckpt.get("shared_layers", [256])
    reg_layers = ckpt.get("reg_layers", [64])
    cls_layers = ckpt.get("cls_layers", [64])
    backbone = ckpt.get("backbone", "yolo26n")
    condition_cols = ckpt.get(
        "condition_cols",
        [
            "printlayers",
            "printmaxpower(%)",
            "printspeed(mm/s)",
            "polishlayers",
            "polishmaxpower(%)",
        ],
    )

    model = TunableDualHeadModel(
        backbone_name=backbone,
        n_conditions=len(condition_cols),
        shared_layers=shared_layers,
        reg_layers=reg_layers,
        cls_layers=cls_layers,
        dropout=0.0,          # No dropout during inference
        freeze_backbone=False,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _preprocess(
    image_path: str,
    img_size: Tuple[int, int] = (224, 224),
    crop_half_frac: float = 0.20,
) -> torch.Tensor:
    """Crop, resize, normalise → [1, 3, H, W] tensor."""
    img = Image.open(image_path).convert("RGB")
    img = crop_central_line(img, crop_half_frac=crop_half_frac)
    resize = T.Resize(img_size, antialias=True)
    return _EVAL_TRANSFORM(resize(img)).unsqueeze(0)  # [1, 3, H, W]


def _normalise_conditions(
    conditions: Dict[str, float],
    condition_cols: List[str],
    cond_mean: np.ndarray,
    cond_std: np.ndarray,
) -> torch.Tensor:
    """Return a [1, n_conditions] normalised condition tensor."""
    cond_vec = np.array([conditions[c] for c in condition_cols], dtype=np.float32)
    safe_std = np.where(cond_std == 0, 1.0, cond_std)
    cond_norm = (cond_vec - cond_mean) / safe_std
    return torch.tensor(cond_norm, dtype=torch.float32).unsqueeze(0)  # [1, n_cond]


def predict_single_image(
    checkpoint_path: str,
    image_path: str,
    conditions: Dict[str, float],
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Predict resistance for a single microscope image.

    Parameters
    ----------
    checkpoint_path :
        Path to ``.pt`` checkpoint saved by :func:`~im2quant.train.train`.
    image_path :
        Path to the image file.
    conditions :
        Dict mapping condition column names to their numeric values.
        Keys must match ``condition_cols`` stored in the checkpoint.
    device :
        ``'cuda'`` or ``'cpu'``.  Auto-detected if not given.

    Returns
    -------
    ``{'R_ohm': float, 'log10R': float, 'p_low_cov': float}``

    * ``R_ohm``    — predicted resistance in Ohm.
    * ``log10R``   — predicted log10(R) (raw model output).
    * ``p_low_cov``— probability that this wire has Low CoV (good quality).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_model(checkpoint_path, device)

    cond_mean = np.array(ckpt["cond_mean"], dtype=np.float32)
    cond_std = np.array(ckpt["cond_std"], dtype=np.float32)
    condition_cols: List[str] = ckpt.get("condition_cols", list(conditions.keys()))
    log_transform: bool = ckpt.get("log_transform", True)

    img_tensor = _preprocess(image_path).to(device)
    cond_tensor = _normalise_conditions(conditions, condition_cols, cond_mean, cond_std).to(device)

    with torch.no_grad():
        pred_r, logits = model(img_tensor, cond_tensor)

    log10r: float = pred_r.item()
    r_ohm: float = 10 ** log10r if log_transform else log10r
    p_low_cov: float = torch.softmax(logits, dim=1)[0, 1].item()

    return {"R_ohm": r_ohm, "log10R": log10r, "p_low_cov": p_low_cov}


def predict_batch(
    checkpoint_path: str,
    image_paths: List[str],
    conditions_list: List[Dict[str, float]],
    device: Optional[str] = None,
) -> List[Dict[str, float]]:
    """
    Predict resistance for a list of images.

    The model is loaded once and reused for all images.

    Parameters
    ----------
    checkpoint_path :
        Path to ``.pt`` checkpoint.
    image_paths :
        List of image file paths.
    conditions_list :
        List of condition dicts, one per image.
    device :
        ``'cuda'`` or ``'cpu'``.  Auto-detected if not given.

    Returns
    -------
    List of ``{'R_ohm': float, 'log10R': float, 'p_low_cov': float}`` dicts.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_model(checkpoint_path, device)

    cond_mean = np.array(ckpt["cond_mean"], dtype=np.float32)
    cond_std = np.array(ckpt["cond_std"], dtype=np.float32)
    condition_cols: List[str] = ckpt.get("condition_cols", list(conditions_list[0].keys()))
    log_transform: bool = ckpt.get("log_transform", True)

    results: List[Dict[str, float]] = []

    for image_path, conditions in zip(image_paths, conditions_list):
        img_tensor = _preprocess(image_path).to(device)
        cond_tensor = _normalise_conditions(
            conditions, condition_cols, cond_mean, cond_std
        ).to(device)

        with torch.no_grad():
            pred_r, logits = model(img_tensor, cond_tensor)

        log10r = pred_r.item()
        r_ohm = 10 ** log10r if log_transform else log10r
        p_low_cov = torch.softmax(logits, dim=1)[0, 1].item()
        results.append({"R_ohm": r_ohm, "log10R": log10r, "p_low_cov": p_low_cov})

    return results
