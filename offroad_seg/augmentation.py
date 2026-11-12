"""
Offline, class-targeted data augmentation.

The three rarest classes (by number of training images that contain them)
receive extra augmented samples:
    level 1 (rarest):        all 5 transformations on every image with the class
    level 2 (second rarest): one random transformation (no zoom) per image, 1x the images
    level 3 (third rarest):  one random transformation (no zoom), 0.5x the images
Transformations are mild (no strong color shifts or geometry changes) so
that the scene stays plausible for a camera mounted on the vehicle.
"""

import os
import random
from collections import defaultdict

import cv2
import numpy as np

from offroad_seg.data import classes_in_label, load_sample, read_rgb, rgb_to_class_mask

DEFAULT_CONFIG = {
    "zoom_crop_factor": 0.8,
    "brightness_range": (-0.2, 0.2),
    "contrast_range": (0.8, 1.2),
    "rotation_range": (-5, 5),
}


# ---------------------------------------------------------------------------
# Transformations (image and label are RGB uint8 arrays)
# ---------------------------------------------------------------------------

def zoom_crop(rgb, label, target_class, crop_factor=0.8):
    """Crop of crop_factor x the image, centred on the target class pixels when there are enough of them."""
    h, w = rgb.shape[:2]
    ch, cw = int(h * crop_factor), int(w * crop_factor)
    target = rgb_to_class_mask(label) == target_class
    if target.sum() > 30:
        ys, xs = np.where(target)
        y0 = max(0, min(h - ch, int(ys.mean()) - ch // 2))
        x0 = max(0, min(w - cw, int(xs.mean()) - cw // 2))
    else:
        y0, x0 = (h - ch) // 2, (w - cw) // 2
    return rgb[y0:y0 + ch, x0:x0 + cw], label[y0:y0 + ch, x0:x0 + cw]


def brightness(rgb, value_range=(-0.2, 0.2)):
    shift = random.uniform(*value_range) * 255
    return np.clip(rgb.astype(np.float32) + shift, 0, 255).astype(np.uint8)


def contrast(rgb, factor_range=(0.8, 1.2)):
    factor = random.uniform(*factor_range)
    return np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def horizontal_flip(rgb, label):
    return cv2.flip(rgb, 1), cv2.flip(label, 1)


def rotation(rgb, label, angle_range=(-5, 5)):
    angle = random.uniform(*angle_range)
    h, w = rgb.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rgb = cv2.warpAffine(rgb, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    label = cv2.warpAffine(label, m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    return rgb, label


def _transformations(config):
    return [
        ("zoom_crop", lambda rgb, lab, c: zoom_crop(rgb, lab, c, config["zoom_crop_factor"])),
        ("brightness", lambda rgb, lab, c: (brightness(rgb, config["brightness_range"]), lab)),
        ("flip", lambda rgb, lab, c: horizontal_flip(rgb, lab)),
        ("rotation", lambda rgb, lab, c: rotation(rgb, lab, config["rotation_range"])),
        ("contrast", lambda rgb, lab, c: (contrast(rgb, config["contrast_range"]), lab)),
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def class_rarity(root, names):
    """Classes sorted by the number of images that contain them (rarest first)."""
    counts = defaultdict(int)
    for name in names:
        for c in classes_in_label(read_rgb(os.path.join(root, name, "labels.png")), downsample=128):
            counts[c] += 1
    return sorted(counts.items(), key=lambda kv: kv[1])


def generate_augmented_samples(root, train_names, config=None, output_folder=None):
    """
    Generates the augmented samples from the training images only.
    Returns a list of (rgb, label_rgb) pairs at the network resolution and,
    if output_folder is given, also saves them as <folder>/<name>/{rgb.jpg, labels.png}.
    """
    config = {**DEFAULT_CONFIG, **(config or {})}
    rarest = [c for c, _ in class_rarity(root, train_names)[:3]]
    if len(rarest) < 3:
        return []

    images_with = {c: [] for c in rarest}
    for name in train_names:
        present = classes_in_label(read_rgb(os.path.join(root, name, "labels.png")))
        for c in rarest:
            if c in present:
                images_with[c].append(name)

    all_t = _transformations(config)
    no_zoom = [t for t in all_t if t[0] != "zoom_crop"]
    samples = []

    def add(rgb, lab, tag):
        samples.append((rgb, lab))
        if output_folder:
            save_augmentation(rgb, lab, os.path.join(output_folder, tag))

    # Level 1: every transformation on every image of the rarest class
    for i, name in enumerate(images_with[rarest[0]]):
        rgb, lab = load_sample(root, name)
        for j, (t_name, t) in enumerate(all_t):
            add(*t(rgb, lab, rarest[0]), f"level1_{t_name}_{i:03d}_{j}")

    # Levels 2 and 3: random transformations until the target count is reached
    for level, (class_id, factor) in enumerate([(rarest[1], 1.0), (rarest[2], 0.5)], start=2):
        target, created = int(len(images_with[class_id]) * factor), 0
        while created < target and images_with[class_id]:
            for name in images_with[class_id]:
                if created >= target:
                    break
                rgb, lab = load_sample(root, name)
                t_name, t = random.choice(no_zoom)
                created += 1
                add(*t(rgb, lab, class_id), f"level{level}_{t_name}_{created:04d}")

    return samples


def save_augmentation(rgb, label, folder):
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, "rgb.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(folder, "labels.png"), cv2.cvtColor(label, cv2.COLOR_RGB2BGR))
