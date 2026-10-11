"""Crack segmentation metrics with explicit probability handling."""

import torch
import cv2
import numpy as np
from sklearn.metrics import f1_score

def Get_F1_score(target, predict):

    """Compute binary F1 after NumPy rounding of predictions and targets."""
    target = target.flatten()
    predict = predict.flatten()

    target = np.round(target)
    predict = np.round(predict)

    predict = predict.astype(int)
    target = target.astype(int)

    f1 = f1_score(predict,target)

    return f1


def get_Dice(gt, predict):

    """Compute soft Dice over all elements without thresholding predictions."""
    intersection = torch.sum(predict*gt)
    epsilon = 1e-6
    Dice = (2.*intersection + epsilon) / (torch.sum(predict) + torch.sum(gt) + epsilon)

    return Dice


def get_IoU(image_mask, predict):

    """Compute binary IoU after thresholding both inputs at 0.5."""
    predict = predict.ge(0.5).float()

    image_mask = image_mask.ge(0.5).float()

    epsilon = 1e-6
    interArea = predict*image_mask
    tem = predict + image_mask
    unionArea = tem - interArea
    inter = torch.sum(interArea)
    union = torch.sum(unionArea)
    IoU = (inter + epsilon) / (union + epsilon)

    return IoU
