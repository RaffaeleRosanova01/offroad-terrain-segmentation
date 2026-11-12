"""
Trains the segmentation model (ResNet34 + U-Net or SegFormer) on the YCOR training images.

Two recipes are available:

  --recipe baseline   the original course-project setup
      stratified 80/20 split on the dominant class, offline class-targeted
      augmentation (344 extra samples), class-weighted cross-entropy,
      model selection on the dataset-level validation IoU, early stopping
      on the validation loss, 20 epochs, Adam lr 5e-5.

  --recipe v2         the revised setup
      scene-aware split (near-duplicate frames never cross the split),
      online augmentation at every epoch, class-weighted cross-entropy +
      Dice loss, model selection and early stopping on the per-image
      validation IoU (the evaluation protocol), up to 40 epochs, AdamW lr 1e-4.

Models: --model resnet34_unet (default) | segformer-b1 | segformer-b2
(SegFormer is initialised from Cityscapes weights and uses lr 6e-5).

Usage:
    python train.py --recipe v2 --data-root /path/to/train --output-dir outputs/v2
    python train.py --recipe v2 --model segformer-b2 --data-root /path/to/train --output-dir outputs/v2_segformer

Training resumes automatically from <output-dir>/last.pth. The best
weights are saved to <output-dir>/best_model.pth with the training history.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from offroad_seg.augmentation import generate_augmented_samples
from offroad_seg.config import CLASS_NAMES, NUM_CLASSES
from offroad_seg.data import (OffRoadDataset, grouped_split, list_samples, load_augmented_folder,
                              stratified_split)
from offroad_seg.losses import CEDiceLoss
from offroad_seg.metrics import IoUMetric, batch_image_ious
from offroad_seg.model import MODELS, build_model

RECIPES = {
    "baseline": dict(split="stratified", offline_aug=True, online_aug=False, dice=False,
                     select="dataset_iou", stop_on="val_loss", epochs=20, lr=5e-5, optimizer="adam", patience=6,
                     class_weights="inverse"),
    "v2": dict(split="grouped", offline_aug=False, online_aug=True, dice=True,
               select="image_iou", stop_on="image_iou", epochs=40, lr=1e-4, optimizer="adamw", patience=8,
               class_weights="inverse_sqrt"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recipe", choices=RECIPES, default="v2")
    p.add_argument("--model", choices=MODELS, default="resnet34_unet")
    p.add_argument("--data-root", required=True, help="folder with one sub-folder (rgb.jpg, labels.png) per sample")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--epochs", type=int, default=None, help="override the recipe")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--limit", type=int, default=None, help="use only the first N samples (smoke tests)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true", help="random encoder initialisation (tests only)")
    p.add_argument("--amp", action="store_true", help="mixed precision (float16) on GPU: less memory, faster")
    return p.parse_args()


def compute_class_weights(dataset, mode="inverse"):
    """
    Class weights for the cross-entropy from the pixel frequencies f_c of the training set.

    inverse:       1/f_c rescaled to sum to the number of classes, clamped to
                   [0.1, 10] (original recipe). With a class as rare as puddle
                   (<0.1% of the pixels) every common class ends up at 0.1.
    inverse_sqrt:  1/sqrt(f_c) rescaled to mean 1, clamped to [0.1, 10]: a
                   smoother reweighting that keeps the ordering of all classes.
    """
    counts = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for i in tqdm(range(len(dataset)), desc="Class weights", leave=False):
        _, mask = dataset[i]
        counts += torch.bincount(mask.flatten(), minlength=NUM_CLASSES).double()
    eps = 1e-10
    freq = counts / (counts.sum() + eps)
    present = counts > 0  # classes absent from the training set get weight 1
    weights = torch.ones(NUM_CLASSES, dtype=torch.float64)
    if mode == "inverse":
        inverse = 1.0 / freq[present]
        weights[present] = inverse / inverse.sum() * present.sum()
    else:
        inverse = 1.0 / torch.sqrt(freq[present])
        weights[present] = inverse / inverse.mean()
    return torch.clamp(weights, 0.1, 10.0).float()


def run_epoch(model, loader, criterion, device, optimizer=None, desc="", scaler=None):
    """
    One pass over the loader; trains when an optimizer is given.
    Returns loss, dataset-level mean IoU, per-image mean IoU and dataset-level per-class IoU.
    """
    training = optimizer is not None
    model.train(training)
    metric, total, image_ious = IoUMetric(), 0.0, []
    with torch.set_grad_enabled(training):
        for images, masks in tqdm(loader, desc=desc, leave=False):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=scaler is not None):
                outputs = model(images)
                loss = criterion(outputs.float(), masks)
            if training:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total += loss.item()
            preds = outputs.argmax(1)
            metric.update(preds.cpu(), masks.cpu())
            image_ious.append(batch_image_ious(preds, masks))
    mean_iou, class_ious = metric.compute()
    return total / len(loader), mean_iou, float(np.nanmean(np.concatenate(image_ious))), class_ious


def main():
    args = parse_args()
    cfg = dict(RECIPES[args.recipe])
    if args.epochs:
        cfg["epochs"] = args.epochs
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[INFO] Recipe '{args.recipe}': {cfg}")

    # --- Split (computed once and saved) ---
    split_path = os.path.join(args.output_dir, "split.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        train_names, val_names = split["train"], split["val"]
    else:
        names = list_samples(args.data_root)[: args.limit]
        if cfg["split"] == "stratified":
            val_size = 180 if args.limit is None else max(2, len(names) // 5)
            train_names, val_names = stratified_split(args.data_root, names, val_size=val_size, seed=args.seed)
        else:
            train_names, val_names = grouped_split(args.data_root, names, seed=args.seed)
        with open(split_path, "w") as f:
            json.dump({"train": train_names, "val": val_names}, f, indent=1)
    print(f"[INFO] train {len(train_names)}, val {len(val_names)}")

    # --- Offline augmentation (baseline recipe, training images only) ---
    augmented = []
    if cfg["offline_aug"]:
        aug_dir = os.path.join(args.output_dir, "augmentation")
        if os.path.isdir(aug_dir) and os.listdir(aug_dir):
            augmented = load_augmented_folder(aug_dir)
        else:
            augmented = generate_augmented_samples(args.data_root, train_names, output_folder=aug_dir)
        print(f"[INFO] {len(augmented)} offline augmented samples")

    train_set = OffRoadDataset(args.data_root, train_names, augmented, online_augmentation=cfg["online_aug"])
    val_set = OffRoadDataset(args.data_root, val_names)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # --- Model, loss, optimizer ---
    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    print(f"[INFO] Model {args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    if args.model.startswith("segformer"):
        cfg["lr"] = 6e-5  # standard fine-tuning learning rate for SegFormer
        print(f"[INFO] Learning rate set to {cfg['lr']} for SegFormer")
    weights_path = os.path.join(args.output_dir, "class_weights.npy")
    if os.path.exists(weights_path):
        class_weights = torch.from_numpy(np.load(weights_path))
    else:
        class_weights = compute_class_weights(OffRoadDataset(args.data_root, train_names, augmented), cfg["class_weights"])
        np.save(weights_path, class_weights.numpy())
    print("[INFO] Class weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(CLASS_NAMES, class_weights)))
    class_weights = class_weights.to(device)
    criterion = CEDiceLoss(class_weights, NUM_CLASSES) if cfg["dice"] else nn.CrossEntropyLoss(weight=class_weights)

    opt_cls = torch.optim.AdamW if cfg["optimizer"] == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-7)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    state = {"epoch": 0, "best_score": -1.0, "best_epoch": 0, "stop_best": None, "stop_counter": 0,
             "history": {k: [] for k in ("train_loss", "train_iou", "val_loss", "val_iou", "val_image_iou", "lr")}}
    best_state = None

    # --- Resume ---
    last_path = os.path.join(args.output_dir, "last.pth")
    if os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_state = ckpt["best_model"]
        state = ckpt["state"]
        print(f"[INFO] Resumed from epoch {state['epoch']}")

    for epoch in range(state["epoch"] + 1, cfg["epochs"] + 1):
        start = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tr_loss, tr_iou, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, f"train {epoch}", scaler)
        va_loss, va_iou, va_img_iou, va_class = run_epoch(model, val_loader, criterion, device, desc=f"val {epoch}", scaler=scaler)

        score = va_iou if cfg["select"] == "dataset_iou" else va_img_iou
        scheduler.step(score)
        h = state["history"]
        for k, v in zip(h, (tr_loss, tr_iou, va_loss, va_iou, va_img_iou, optimizer.param_groups[0]["lr"])):
            h[k].append(v)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} IoU {tr_iou:.4f} | val loss {va_loss:.4f} "
              f"IoU {va_iou:.4f} (per-image {va_img_iou:.4f}) | {time.time() - start:.0f}s"
              + (f" | peak GPU {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB" if device.type == "cuda" else ""))
        print("   " + "  ".join(f"{n[:12]}={v:.3f}" for n, v in zip(CLASS_NAMES[1:], va_class)))

        if score > state["best_score"]:
            state.update(best_score=score, best_epoch=epoch)
            best_state = copy.deepcopy(model.state_dict())
            state["best_val_class_ious"] = va_class.tolist()
            torch.save({"model_state_dict": best_state, "training_history": h, "recipe": args.recipe,
                        "model_name": args.model,
                        "model_config": model.config_dict() if hasattr(model, "config_dict") else None,
                        "best_epoch": epoch, "best_val_class_ious": state["best_val_class_ious"]},
                       os.path.join(args.output_dir, "best_model.pth"))

        # Early stopping: validation loss must decrease, or the per-image IoU must increase
        monitored = -va_loss if cfg["stop_on"] == "val_loss" else va_img_iou
        delta = 0.001
        if state["stop_best"] is None or monitored > state["stop_best"] + delta:
            state.update(stop_best=monitored, stop_counter=0)
        else:
            state["stop_counter"] += 1

        state["epoch"] = epoch
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "best_model": best_state, "state": state}, last_path)

        if state["stop_counter"] >= cfg["patience"]:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    print(f"[INFO] Best validation score {state['best_score']:.4f} at epoch {state['best_epoch']} "
          f"-> {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
