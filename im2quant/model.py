"""Model architecture: YOLO feature extractor and dual-head MLP."""

from typing import List

import torch
import torch.nn as nn


def make_decreasing_layers(n_layers: int, first_size: int) -> List[int]:
    """
    Build a decreasing layer-size list.

    Examples
    --------
    >>> make_decreasing_layers(2, 128)
    [128, 32]
    >>> make_decreasing_layers(1, 512)
    [512]

    Each subsequent layer is ~4× smaller than the previous, with a
    minimum of 8 nodes.
    """
    sizes: List[int] = []
    current = first_size
    for _ in range(n_layers):
        sizes.append(current)
        current = max(8, current // 4)
    return sizes


class YOLOFeatureExtractor(nn.Module):
    """
    YOLO backbone + neck used as a frozen or trainable feature extractor.

    The YOLO detection head is stripped automatically by iterating through
    layers with a ``try/except`` block — the loop stops the first time a
    layer raises an exception (i.e. when it hits the Detect head which
    expects a list of inputs rather than a single tensor).

    The last 4-D spatial feature map that was successfully produced is
    captured, then reduced to a 1-D vector via Global Average Pooling.

    This exact pattern must be preserved: do **not** simplify the loop.
    """

    def __init__(self, model_name: str, freeze: bool = False):
        super().__init__()
        from ultralytics import YOLO

        print(f"  Loading {model_name}.pt ...")
        yolo = YOLO(f"{model_name}.pt")
        self.layers = yolo.model.model

        # Probe feature dimension with a dummy forward pass
        with torch.no_grad():
            feat = self._extract(torch.zeros(1, 3, 224, 224))
        self.feature_dim: int = feat.shape[1]
        print(f"  Feature dim: {self.feature_dim}")

        if freeze:
            for p in self.layers.parameters():
                p.requires_grad_(False)
            print("  Backbone frozen.")

    def _extract(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone + neck layers, stopping at Detect head.

        Keeps track of the last 4-D spatial tensor produced.  After the loop,
        applies Global Average Pool to collapse H×W → scalar per channel.
        """
        last_spatial = None
        for layer in self.layers:
            try:
                x = layer(x)
                if isinstance(x, torch.Tensor) and x.dim() == 4:
                    last_spatial = x
            except Exception:
                break

        feat = last_spatial if last_spatial is not None else x
        if isinstance(feat, torch.Tensor) and feat.dim() == 4:
            feat = feat.mean(dim=[2, 3])  # Global Average Pool → [B, C]
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._extract(x)


class TunableDualHeadModel(nn.Module):
    """
    YOLO backbone → shared MLP neck → two parallel output heads:

    1. **Regression head** → log10(R)  (trained on ALL samples)
    2. **Classifier head** → Low CoV (1) / High CoV (0)

    The MLP head architecture is fully configurable via
    ``shared_layers``, ``reg_layers``, and ``cls_layers``, which are
    lists of integer node counts.  Use :func:`make_decreasing_layers` to
    generate these lists from Optuna suggestions.

    Parameters
    ----------
    backbone_name :
        YOLO model name without extension, e.g. ``"yolo26n"``.
    n_conditions :
        Number of print-condition scalars concatenated to backbone features.
    shared_layers :
        Node counts for the shared MLP neck, e.g. ``[256]`` or ``[512, 128]``.
    reg_layers :
        Node counts for regression head hidden layers, e.g. ``[64]``.
    cls_layers :
        Node counts for classifier head hidden layers, e.g. ``[64]``.
    dropout :
        Dropout probability applied after each shared-neck layer.
    freeze_backbone :
        If True, backbone weights are frozen during training.
    """

    def __init__(
        self,
        backbone_name: str,
        n_conditions: int,
        shared_layers: List[int],
        reg_layers: List[int],
        cls_layers: List[int],
        dropout: float,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = YOLOFeatureExtractor(backbone_name, freeze=freeze_backbone)
        feat_dim = self.backbone.feature_dim
        in_dim = feat_dim + n_conditions

        # ── Shared neck ──────────────────────────────────────────────────────
        shared_blocks: List[nn.Module] = []
        prev = in_dim
        for nodes in shared_layers:
            shared_blocks += [
                nn.Linear(prev, nodes),
                nn.LayerNorm(nodes),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = nodes
        self.shared_neck = nn.Sequential(*shared_blocks)
        shared_out = shared_layers[-1]

        # ── Regression head → log10(R) ────────────────────────────────────────
        reg_blocks: List[nn.Module] = []
        prev = shared_out
        for nodes in reg_layers:
            reg_blocks += [nn.Linear(prev, nodes), nn.ReLU()]
            prev = nodes
        reg_blocks.append(nn.Linear(prev, 1))
        self.regression_head = nn.Sequential(*reg_blocks)

        # ── Classifier head → Low CoV (1) / High CoV (0) ─────────────────────
        cls_blocks: List[nn.Module] = []
        prev = shared_out
        for nodes in cls_layers:
            cls_blocks += [nn.Linear(prev, nodes), nn.ReLU()]
            prev = nodes
        cls_blocks.append(nn.Linear(prev, 2))
        self.classifier_head = nn.Sequential(*cls_blocks)

    def forward(self, image: torch.Tensor, conditions: torch.Tensor):
        """
        Parameters
        ----------
        image : [B, 3, H, W]
        conditions : [B, n_conditions]  (z-score normalised)

        Returns
        -------
        pred_r : [B, 1]   log10(R) prediction
        logits : [B, 2]   classifier logits
        """
        feat = self.backbone(image)                      # [B, feat_dim]
        x = torch.cat([feat, conditions], dim=1)         # [B, feat_dim + n_cond]
        shared = self.shared_neck(x)                     # [B, shared_out]
        pred_r = self.regression_head(shared)            # [B, 1]
        logits = self.classifier_head(shared)            # [B, 2]
        return pred_r, logits
