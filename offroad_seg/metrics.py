"""
Segmentation metrics. Background (class 0) is excluded everywhere.

Two IoU aggregations are provided:
  * per-image mean IoU (the evaluation protocol of the project): IoU of each
    class on each image, NaN when the class is absent from both masks, then
    the mean over images and classes ignoring NaNs;
  * dataset-level IoU (used during training): intersections and unions are
    accumulated over all pixels of the set, then IoU per class.
"""

import numpy as np
import torch

FOREGROUND = range(1, 9)


def image_ious(true_mask, pred_mask):
    """IoU of classes 1..8 on one image (NaN where the class is in neither mask)."""
    ious = np.full(8, np.nan)
    for i, c in enumerate(FOREGROUND):
        t, p = true_mask == c, pred_mask == c
        union = np.logical_or(t, p).sum()
        if union > 0:
            ious[i] = np.logical_and(t, p).sum() / union
    return ious


class IoUMetric:
    """Dataset-level IoU accumulator for classes 1..8 (torch tensors of class IDs)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(8, dtype=torch.float64)
        self.union = torch.zeros(8, dtype=torch.float64)

    def update(self, predictions, targets):
        for i, c in enumerate(FOREGROUND):
            p, t = predictions == c, targets == c
            self.intersection[i] += (p & t).sum().item()
            self.union[i] += (p | t).sum().item()

    def compute(self):
        ious = torch.full((8,), float("nan"), dtype=torch.float64)
        valid = self.union > 0
        ious[valid] = self.intersection[valid] / self.union[valid]
        present = ious[~torch.isnan(ious)]
        mean = present.mean().item() if len(present) else 0.0
        return mean, ious.numpy()


def batch_image_ious(predictions, targets):
    """Per-image IoU of classes 1..8 for a batch of class masks (torch tensors) -> numpy [B, 8]."""
    out = np.full((predictions.shape[0], 8), np.nan)
    for i, c in enumerate(FOREGROUND):
        p, t = predictions == c, targets == c
        inter = (p & t).flatten(1).sum(1).cpu().numpy()
        union = (p | t).flatten(1).sum(1).cpu().numpy()
        valid = union > 0
        out[valid, i] = inter[valid] / union[valid]
    return out
