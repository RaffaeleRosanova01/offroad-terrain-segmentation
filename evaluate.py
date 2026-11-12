"""
Evaluates a trained model on a set of labelled samples.

Usage:
    # Validation split used in training
    python evaluate.py --weights weights/resnet34_unet.pth --data-root /path/to/train --split outputs/split.json

    # Every sample in a folder (e.g. the official YCOR validation set)
    python evaluate.py --weights weights/resnet34_unet.pth --data-root /path/to/valid

Reports, at the original image resolution:
  * per-image mean IoU (project protocol): IoU per class per image, NaN for
    classes absent from both masks, averaged ignoring NaNs;
  * per-class IoU with both aggregations (per-image mean, dataset-level);
  * inference speed (frames per second, batch 1, pre- and post-processing
    included) and peak GPU memory.
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from tqdm import tqdm

from offroad_seg.config import CLASS_NAMES
from offroad_seg.data import list_samples, read_rgb, rgb_to_class_mask
from offroad_seg.inference import Segmenter, overlay
from offroad_seg.metrics import image_ious
from offroad_seg.model import load_model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--split", default=None, help="split.json written by train.py (uses its 'val' list)")
    p.add_argument("--output", default=None, help="optional JSON file for the results")
    p.add_argument("--examples-dir", default=None, help="save image / ground truth / prediction overlays here")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.split:
        with open(args.split) as f:
            names = json.load(f)["val"]
    else:
        names = list_samples(args.data_root)

    device = torch.device(args.device)
    segmenter = Segmenter(load_model(args.weights, device), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    per_image, inter, union, times = [], np.zeros(8), np.zeros(8), []
    in_truth = np.zeros(8, dtype=int)  # images whose ground truth contains the class
    for name in tqdm(names, desc="Evaluating"):
        rgb = read_rgb(os.path.join(args.data_root, name, "rgb.jpg"))
        truth = rgb_to_class_mask(read_rgb(os.path.join(args.data_root, name, "labels.png")))

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = segmenter.predict_classes(rgb[None])[0, ..., 0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

        per_image.append(image_ious(truth, pred))
        in_truth += [(truth == c).any() for c in range(1, 9)]
        for i, c in enumerate(range(1, 9)):
            inter[i] += np.logical_and(truth == c, pred == c).sum()
            union[i] += np.logical_or(truth == c, pred == c).sum()

        if args.examples_dir:
            os.makedirs(args.examples_dir, exist_ok=True)
            panel = np.concatenate([rgb, overlay(rgb, truth), overlay(rgb, pred)], axis=1)
            cv2.imwrite(os.path.join(args.examples_dir, f"{name}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    per_image = np.array(per_image)
    per_class_image = np.nanmean(per_image, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        per_class_dataset = np.where(union > 0, inter / union, np.nan)
    fps = 1.0 / np.mean(times[1:] if len(times) > 1 else times)  # first call excluded (warm-up)
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if device.type == "cuda" else None

    results = {
        "samples": len(names),
        "mean_iou_per_image": float(np.nanmean(per_image)),
        "mean_iou_dataset": float(np.nanmean(per_class_dataset)),
        "per_class": {
            CLASS_NAMES[c]: {"per_image": float(a), "dataset": float(b),
                             "images_present": int((~np.isnan(per_image[:, i])).sum()),
                             "images_in_ground_truth": int(in_truth[i])}
            for i, (c, a, b) in enumerate(zip(range(1, 9), per_class_image, per_class_dataset))
        },
        "fps": float(fps),
        "peak_gpu_memory_gb": peak_gb,
        "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
    }

    print(f"\nSamples: {len(names)}")
    print(f"Mean IoU (per image, project protocol): {results['mean_iou_per_image']:.4f}")
    print(f"Mean IoU (dataset-level):               {results['mean_iou_dataset']:.4f}\n")
    print(f"{'class':<32}{'per-image':>10}{'dataset':>10}{'in GT':>7}{'GT or pred':>11}")
    for name, r in results["per_class"].items():
        print(f"{name:<32}{r['per_image']:>10.4f}{r['dataset']:>10.4f}"
              f"{r['images_in_ground_truth']:>7}{r['images_present']:>11}")
    mem = f", peak GPU memory {peak_gb:.2f} GB" if peak_gb is not None else ""
    print(f"\nSpeed: {fps:.1f} FPS on {results['device']} (batch 1, pre/post-processing included){mem}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
