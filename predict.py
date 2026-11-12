"""
Segments images and saves the color-coded masks and overlays.

Usage:
    python predict.py --weights weights/resnet34_unet.pth --images img1.jpg img2.jpg --output-dir predictions
"""

import argparse
import os

import cv2
import torch

from offroad_seg.data import read_rgb
from offroad_seg.inference import Segmenter, colorize, overlay
from offroad_seg.model import load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--images", nargs="+", required=True)
    p.add_argument("--output-dir", default="predictions")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    segmenter = Segmenter(load_model(args.weights, device), device)
    os.makedirs(args.output_dir, exist_ok=True)

    for path in args.images:
        rgb = read_rgb(path)
        classes = segmenter.predict_classes(rgb[None])[0, ..., 0]
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem == "rgb":  # dataset layout <sample>/rgb.jpg: name the outputs after the sample folder
            stem = os.path.basename(os.path.dirname(os.path.abspath(path)))
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_mask.png"), cv2.cvtColor(colorize(classes), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(args.output_dir, f"{stem}_overlay.jpg"), cv2.cvtColor(overlay(rgb, classes), cv2.COLOR_RGB2BGR))
        print(f"[INFO] {path} -> {args.output_dir}/{stem}_mask.png, {stem}_overlay.jpg")


if __name__ == "__main__":
    main()
