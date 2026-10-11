"""Loss functions for segmentation probabilities and image residuals."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss_function(preds, targets):
    """Return mean per-sample soft Dice loss with smoothing of 1e-4."""
    smooth = 1e-4
    n = preds.size(0)
    iflat = preds.view(n, -1)
    tflat = targets.view(n, -1)
    intersection = (iflat * tflat).sum(1)
    loss = 1 - ((2. * intersection + smooth) /
                (iflat.sum(1) + tflat.sum(1) + smooth))
    return loss.mean()


class DiceLoss(nn.Module):
    """Compute mean per-sample soft Dice loss with smoothing of 1e-6."""
    def __init__(self):
        super().__init__()
        self.smooth = 1e-6

    def forward(self, preds, targets):
        n = preds.size(0)
        iflat = preds.view(n, -1)
        tflat = targets.view(n, -1)
        intersection = (iflat * tflat).sum(1)
        loss = 1 - ((2. * intersection + self.smooth) /
                    (iflat.sum(1) + tflat.sum(1) + self.smooth))
        return loss.mean()


class CharbonnierLoss(nn.Module):
    """Compute the mean Charbonnier penalty sqrt((X - Y)^2 + eps)."""
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
