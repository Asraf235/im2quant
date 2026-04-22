"""Training and Optuna hyperparameter tuning for im2quant."""

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .dataset import WireDataset, make_transforms
from .model import TunableDualHeadModel, make_decreasing_layers


# ── Internal helpers ──────────────────────────────────────────────────────────


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train: bool,
    lambda_weight: float,
    device: str,
    cls_criterion: nn.Module,
) -> dict:
    """
    One training or validation epoch.

    Loss = lambda_weight * MSE(ALL) + (1 - lambda_weight) * BCE(ALL).
    Regression is computed on ALL samples — no CoV-based masking.
    """
    model.train(train)
    tot_loss = tot_mse = tot_bce = tot_acc = 0.0
    n = 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            imgs = batch["image"].to(device)
            conds = batch["conditions"].to(device)
            tgts_r = batch["target_r"].to(device)
            labels = batch["label"].squeeze(1).to(device)

            pred_r, logits = model(imgs, conds)
            mse_loss = nn.functional.mse_loss(pred_r, tgts_r)
            bce_loss = cls_criterion(logits, labels)
            loss = lambda_weight * mse_loss + (1 - lambda_weight) * bce_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            predicted = logits.argmax(dim=1)
            acc = (predicted == labels).float().mean().item()
            tot_loss += loss.item()
            tot_mse += mse_loss.item()
            tot_bce += bce_loss.item()
            tot_acc += acc
            n += 1

    return {
        "loss": tot_loss / n,
        "mse": tot_mse / n,
        "bce": tot_bce / n,
        "acc": tot_acc / n,
    }


def _resolve_architecture(cfg: Config, best_params: Optional[dict]) -> tuple:
    """Return (shared_layers, reg_layers, cls_layers, lr, dropout, lambda_weight)."""
    if best_params is not None:
        shared_layers = make_decreasing_layers(
            best_params["n_shared_layers"], best_params["shared_first"]
        )
        reg_layers = make_decreasing_layers(
            best_params["n_reg_layers"], best_params["reg_first"]
        )
        cls_layers = make_decreasing_layers(
            best_params["n_cls_layers"], best_params["cls_first"]
        )
        lr = float(best_params.get("lr", cfg.lr))
        dropout = float(best_params.get("dropout", 0.3))
        lambda_weight = float(best_params.get("lambda", cfg.lambda_weight))
    else:
        shared_layers = [256]
        reg_layers = [64]
        cls_layers = [64]
        lr = cfg.lr
        dropout = 0.3
        lambda_weight = cfg.lambda_weight

    return shared_layers, reg_layers, cls_layers, lr, dropout, lambda_weight


# ── Public API ────────────────────────────────────────────────────────────────


def train(
    cfg: Config,
    splits: Dict[str, pd.DataFrame],
    best_params: Optional[dict] = None,
    device: Optional[str] = None,
) -> Path:
    """
    Train the dual-head model and save the best checkpoint.

    Parameters
    ----------
    cfg : Config
    splits : dict returned by :func:`~im2quant.pipeline.stratified_split`.
    best_params : Optuna best-trial dict (from :func:`tune_hyperparameters`)
        or ``None`` to use default architecture ``[256] / [64] / [64]``.
    device : ``'cuda'`` or ``'cpu'``.  Auto-detected if not given.

    Returns
    -------
    Path to the saved ``.pt`` checkpoint.

    Side effects
    ------------
    Writes ``history.csv`` and the checkpoint to ``cfg.output_dir``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = make_transforms(cfg)

    train_ds = WireDataset(splits["train"], cfg, transform=train_tf)
    val_ds = WireDataset(
        splits["val"],
        cfg,
        transform=eval_tf,
        cond_mean=train_ds._cond_mean,
        cond_std=train_ds._cond_std,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    shared_layers, reg_layers, cls_layers, lr, dropout, lambda_weight = (
        _resolve_architecture(cfg, best_params)
    )

    model = TunableDualHeadModel(
        backbone_name=cfg.backbone,
        n_conditions=len(cfg.condition_cols),
        shared_layers=shared_layers,
        reg_layers=reg_layers,
        cls_layers=cls_layers,
        dropout=dropout,
        freeze_backbone=cfg.freeze_backbone,
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=lr * 0.01
    )
    cls_criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_count = 0
    history = []
    ckpt_path = output_dir / f"best_{cfg.backbone}_tuned.pt"

    for epoch in range(cfg.epochs):
        tr = _run_epoch(model, train_loader, optimizer, True, lambda_weight, device, cls_criterion)
        vl = _run_epoch(model, val_loader, optimizer, False, lambda_weight, device, cls_criterion)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "train_mse": tr["mse"],
                "train_bce": tr["bce"],
                "train_acc": tr["acc"],
                "val_loss": vl["loss"],
                "val_mse": vl["mse"],
                "val_bce": vl["bce"],
                "val_acc": vl["acc"],
                "lr": scheduler.get_last_lr()[0],
            }
        )

        print(
            f"Epoch {epoch + 1:>3}/{cfg.epochs}  "
            f"train_loss={tr['loss']:.4f}  val_loss={vl['loss']:.4f}  "
            f"val_acc={vl['acc']:.3f}"
        )

        if vl["loss"] < best_val_loss:
            best_val_loss = vl["loss"]
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    # ── Inference metadata ────────────────────────────────────
                    "backbone": cfg.backbone,
                    "condition_cols": cfg.condition_cols,
                    "log_transform": cfg.log_transform,
                    "cond_mean": train_ds._cond_mean,
                    "cond_std": train_ds._cond_std,
                    "lambda": lambda_weight,
                    "best_params": best_params,
                    "shared_layers": shared_layers,
                    "reg_layers": reg_layers,
                    "cls_layers": cls_layers,
                },
                ckpt_path,
            )
        else:
            patience_count += 1

        if patience_count >= cfg.patience:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    print(f"\nBest val loss: {best_val_loss:.4f}  |  Checkpoint: {ckpt_path}")
    return ckpt_path


def tune_hyperparameters(
    cfg: Config,
    splits: Dict[str, pd.DataFrame],
    device: Optional[str] = None,
) -> dict:
    """
    Run Optuna hyperparameter search and return the best-trial params dict.

    Search space
    ------------
    * ``lr``: 1e-5 – 1e-3 (log scale)
    * ``dropout``: 0.1 – 0.5
    * ``lambda``: 0.2 – 0.8
    * ``n_shared_layers`` / ``shared_first``: 1–2 layers, 64–512 nodes
    * ``n_reg_layers`` / ``reg_first``: 1–2 layers, 8–64 nodes
    * ``n_cls_layers`` / ``cls_first``: 1–2 layers, 8–64 nodes

    The best params are also saved to ``cfg.output_dir/best_params_<backbone>.json``.

    Parameters
    ----------
    cfg : Config
    splits : dict returned by :func:`~im2quant.pipeline.stratified_split`.
    device : ``'cuda'`` or ``'cpu'``.  Auto-detected if not given.

    Returns
    -------
    ``best_params`` dict suitable for passing directly to :func:`train`.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as exc:
        raise ImportError("Install optuna: pip install optuna") from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_tf, eval_tf = make_transforms(cfg)
    train_ds = WireDataset(splits["train"], cfg, transform=train_tf)
    val_ds = WireDataset(
        splits["val"],
        cfg,
        transform=eval_tf,
        cond_mean=train_ds._cond_mean,
        cond_std=train_ds._cond_std,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        lam = trial.suggest_float("lambda", 0.2, 0.8)

        n_shared = trial.suggest_int("n_shared_layers", 1, 2)
        shared_first = trial.suggest_categorical("shared_first", [64, 128, 256, 512])
        shared_layers = make_decreasing_layers(n_shared, shared_first)

        n_reg = trial.suggest_int("n_reg_layers", 1, 2)
        reg_first = trial.suggest_categorical("reg_first", [8, 16, 32, 64])
        reg_layers = make_decreasing_layers(n_reg, reg_first)

        n_cls = trial.suggest_int("n_cls_layers", 1, 2)
        cls_first = trial.suggest_categorical("cls_first", [8, 16, 32, 64])
        cls_layers = make_decreasing_layers(n_cls, cls_first)

        try:
            m = TunableDualHeadModel(
                backbone_name=cfg.backbone,
                n_conditions=len(cfg.condition_cols),
                shared_layers=shared_layers,
                reg_layers=reg_layers,
                cls_layers=cls_layers,
                dropout=dropout,
                freeze_backbone=cfg.freeze_backbone,
            ).to(device)
        except Exception:
            raise optuna.exceptions.TrialPruned()

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, m.parameters()),
            lr=lr,
            weight_decay=cfg.weight_decay,
        )
        cls_crit = nn.CrossEntropyLoss()

        best_vl = float("inf")
        pat_count = 0

        for epoch in range(cfg.n_tune_epochs):
            _run_epoch(m, train_loader, opt, True, lam, device, cls_crit)
            vl_dict = _run_epoch(m, val_loader, opt, False, lam, device, cls_crit)
            vl = vl_dict["loss"]

            if vl < best_vl:
                best_vl = vl
                pat_count = 0
            else:
                pat_count += 1

            trial.report(vl, epoch)
            if trial.should_prune() or pat_count >= cfg.tune_patience:
                del m
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise optuna.exceptions.TrialPruned()

        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return best_vl

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=True)

    best_params = study.best_params
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params_path = output_dir / f"best_params_{cfg.backbone}.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nBest Optuna params saved to {params_path}")
    print(f"Best trial val loss: {study.best_value:.4f}")
    print(f"Best params: {best_params}")
    return best_params
