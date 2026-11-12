"""Loss functions."""

import torch.nn.functional as F
from torch import nn


class DiceLoss(nn.Module):
    """
    Soft Dice loss over the foreground classes (1..C-1), averaged over the
    classes present in the ground truth of the batch. It optimises
    the overlap directly, so small and rare classes weigh as much as large ones.
    """

    def __init__(self, num_classes, eps=1.0):
        super().__init__()
        self.num_classes, self.eps = num_classes, eps

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1:]
        onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()[:, 1:]
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        denom = probs.sum(dims) + onehot.sum(dims)
        present = onehot.sum(dims) > 0
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice[present].mean() if present.any() else logits.sum() * 0


class CEDiceLoss(nn.Module):
    """Class-weighted cross-entropy + soft Dice."""

    def __init__(self, class_weights, num_classes, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)
