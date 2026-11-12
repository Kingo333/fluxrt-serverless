"""Garment-weighted losses for the 0811 LoRA (step 3a). Pure array maths: the same functions
take numpy arrays (tests) or torch tensors (train.py), shape (..., H, W) on the last two axes.

Owner targets (step 1 grids on live0811): keep exact print, trims (ruffles, bows, collars,
necklines), knit/rib texture and hem details. A plain MSE over the whole frame mostly rewards
the right average colour and the background; these two terms put the effort on the garment:

- weighted_mse: MSE with weight `inside` (AZ_MASK_WEIGHT, 3.0) on garment pixels and 1.0
  elsewhere (Glance agnostic mask, downsampled to the latent grid), normalised by the mean
  weight so the loss scale stays comparable to plain MSE.
- edge_l1: L1 between the horizontal and vertical finite differences of prediction and target
  inside the garment area (AZ_EDGE_WEIGHT, 0.5). Ruffles, ribs, bow edges and print outlines are
  edges, so smoothing them away is penalised directly.
Items without a mask use weight 1 everywhere (plain MSE + edge term over the whole frame).
"""

from __future__ import annotations


def mask_to_grid(mask_img, h: int, w: int):
    """PIL mask (white = garment) -> float numpy array (h, w) in 0..1, area-averaged (BOX), so a
    latent cell half covered by the garment gets 0.5."""
    import numpy as np
    from PIL import Image
    m = mask_img.convert("L").resize((w, h), Image.BOX)
    return np.asarray(m, dtype=np.float32) / 255.0


def weight_map(mask, inside: float):
    """1 outside, `inside` on the garment, linear in between (mask in 0..1)."""
    return 1.0 + (float(inside) - 1.0) * mask


def weighted_mse(pred, target, weight):
    """mean(w * (p - t)^2) / mean(w); weight broadcasts over the leading axes."""
    diff2 = (pred - target) ** 2
    w = weight + diff2 * 0  # broadcast to diff2's shape (works for numpy and torch)
    return (w * diff2).mean() / w.mean()


def edge_l1(pred, target, mask=None):
    """Mean |d(p) - d(t)| over x and y finite differences, weighted by the mask (garment area).
    mask None = the whole frame."""
    dxp, dxt = pred[..., :, 1:] - pred[..., :, :-1], target[..., :, 1:] - target[..., :, :-1]
    dyp, dyt = pred[..., 1:, :] - pred[..., :-1, :], target[..., 1:, :] - target[..., :-1, :]
    ex, ey = abs(dxp - dxt), abs(dyp - dyt)
    if mask is None:
        return (ex.mean() + ey.mean()) / 2
    mx = mask[..., :, 1:] * mask[..., :, :-1] + ex * 0
    my = mask[..., 1:, :] * mask[..., :-1, :] + ey * 0
    den = mx.sum() + my.sum()
    if float(den) <= 1e-6:
        return (ex.mean() + ey.mean()) / 2
    return ((ex * mx).sum() + (ey * my).sum()) / den


def garment_loss(pred, target, mask=None, inside: float = 3.0, edge_weight: float = 0.5) -> dict:
    """{'total', 'mse', 'edge'}: total = weighted MSE + edge_weight * edge L1."""
    w = 1.0 if mask is None else weight_map(mask, inside)
    mse = weighted_mse(pred, target, w) if mask is not None else ((pred - target) ** 2).mean()
    edge = edge_l1(pred, target, mask) if edge_weight > 0 else mse * 0
    return {"total": mse + edge_weight * edge, "mse": mse, "edge": edge}
