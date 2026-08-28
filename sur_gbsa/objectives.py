# objectives.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


def masked_mean(loss: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-8) -> torch.Tensor:
    if mask is None:
        return loss.mean()
    mask = mask.float()
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    return (loss * mask).sum() / mask.sum().clamp_min(eps)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = (pred - target).pow(2)
    if loss.dim() > 2:
        loss = loss.sum(dim=-1)
    return masked_mean(loss, mask)


def next_timestep_coordinate_prediction(coord_pred, coord_target, mask=None):
    return masked_mse(coord_pred, coord_target, mask)


def coordinate_denoising(coord_pred, coord_target, mask=None):
    return masked_mse(coord_pred, coord_target, mask)


def pairwise_snapshot_ordering_logits(order_logits, order_target):
    return F.binary_cross_entropy_with_logits(order_logits.float(), order_target.float())


def rmsf_prediction(rmsf_pred, rmsf_target):
    return F.mse_loss(rmsf_pred.float(), rmsf_target.float())


def atomic_token_prediction(atom_logits, atom_targets, mask=None, ignore_index=-100):
    b, n, v = atom_logits.shape
    loss = F.cross_entropy(
        atom_logits.reshape(b * n, v),
        atom_targets.reshape(b * n),
        reduction="none",
        ignore_index=ignore_index,
    ).view(b, n)
    if mask is not None:
        valid = mask & (atom_targets != ignore_index)
        return masked_mean(loss, valid)
    return loss.mean()


def edge_token_prediction(edge_logits, edge_targets, mask=None, ignore_index=-100):
    b, n, v = edge_logits.shape
    loss = F.cross_entropy(
        edge_logits.reshape(b * n, v),
        edge_targets.reshape(b * n),
        reduction="none",
        ignore_index=ignore_index,
    ).view(b, n)
    if mask is not None:
        valid = mask & (edge_targets != ignore_index)
        return masked_mean(loss, valid)
    return loss.mean()


@dataclass
class MultiTaskLossWeights:
    coord: float = 1.0
    ordering: float = 1.0
    denoise: float = 1.0
    rmsf: float = 1.0
    atom: float = 1.0
    edge: float = 1.0


def parse_loss_weights(spec: str) -> MultiTaskLossWeights:
    if not spec:
        return MultiTaskLossWeights()
    out = MultiTaskLossWeights()
    for chunk in spec.split(","):
        if not chunk.strip():
            continue
        key, value = chunk.split("=")
        key = key.strip()
        value = float(value)
        if not hasattr(out, key):
            raise ValueError(f"Unknown loss weight key: {key}")
        setattr(out, key, value)
    return out


def compute_objective(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weights=None):
    weights = weights or MultiTaskLossWeights()
    losses = {}

    if "coord_pred" in outputs and "coord_target" in batch:
        losses["coord"] = weights.coord * next_timestep_coordinate_prediction(
            outputs["coord_pred"],
            batch["coord_target"],
            batch.get("coord_mask"),
        )

    if "denoise_pred" in outputs and "denoise_target" in batch:
        losses["denoise"] = weights.denoise * coordinate_denoising(
            outputs["denoise_pred"],
            batch["denoise_target"],
            batch.get("denoise_mask"),
        )

    if "order_logits" in outputs and "order_target" in batch:
        losses["ordering"] = weights.ordering * pairwise_snapshot_ordering_logits(
            outputs["order_logits"],
            batch["order_target"],
        )

    if "rmsf_pred" in outputs and "rmsf_target" in batch:
        losses["rmsf"] = weights.rmsf * rmsf_prediction(
            outputs["rmsf_pred"],
            batch["rmsf_target"],
        )

    if "atom_logits" in outputs and "atom_targets" in batch:
        losses["atom"] = weights.atom * atomic_token_prediction(
            outputs["atom_logits"],
            batch["atom_targets"],
            batch.get("atom_mask"),
        )

    if "edge_logits" in outputs and "edge_targets" in batch:
        losses["edge"] = weights.edge * edge_token_prediction(
            outputs["edge_logits"],
            batch["edge_targets"],
            batch.get("edge_mask"),
        )

    if losses:
        losses["total"] = sum(losses.values())
    else:
        device = next(iter(outputs.values())).device
        losses["total"] = torch.tensor(0.0, device=device)

    return losses