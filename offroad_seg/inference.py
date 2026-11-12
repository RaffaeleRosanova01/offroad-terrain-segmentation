"""Inference: batch prediction, color coding and overlays."""

import cv2
import numpy as np
import torch

from offroad_seg.config import CLASS_COLORS, CLASS_TO_LABEL_INDEX, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE


# Lookup table: class ID -> palette index of the label files
LABEL_INDEX_LUT = np.array([CLASS_TO_LABEL_INDEX[c] for c in range(len(CLASS_COLORS))], dtype=np.uint8)


class Segmenter:
    """
    Wraps the trained network with its pre- and post-processing.

    predict_classes(X): X uint8 [B, H, W, 3] (any H, W) -> class IDs (config.CLASS_NAMES)
                        uint8 [B, H, W, 1] at the input resolution.
    predict(X):         same, but encoded as the palette indices of the label
                        files, the output format required by the project.
    """

    def __init__(self, model, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def preprocess(self, X):
        h, w = INPUT_SIZE
        # Same resizing as in training (OpenCV bilinear)
        resized = np.stack([cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) for img in X])
        x = torch.from_numpy(resized).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        return (x - self.mean) / self.std

    @torch.no_grad()
    def predict_classes(self, X):
        X = np.asarray(X, dtype=np.uint8)
        if X.ndim == 3:
            X = X[None]
        out_h, out_w = X.shape[1:3]
        classes = self.model(self.preprocess(X)).argmax(1).byte().cpu().numpy()
        # Nearest-neighbour upsampling of the class map back to the input size
        classes = np.stack([cv2.resize(c, (out_w, out_h), interpolation=cv2.INTER_NEAREST) for c in classes])
        return classes[..., None]

    def predict(self, X):
        return LABEL_INDEX_LUT[self.predict_classes(X)]


def colorize(class_mask):
    """Class IDs [H, W] -> color-coded RGB label image [H, W, 3]."""
    out = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        out[class_mask == class_id] = color
    return out


def overlay(rgb, class_mask, alpha=0.45):
    """Blends the color-coded prediction over the image."""
    return cv2.addWeighted(rgb, 1 - alpha, colorize(class_mask), alpha, 0)
