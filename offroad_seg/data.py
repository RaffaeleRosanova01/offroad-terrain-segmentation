"""
Dataset loading, label decoding and the train/validation split.

Each sample is a folder with `rgb.jpg` (1024x544) and `labels.png`
(color-coded classes, see config.CLASS_COLORS).
"""

import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from offroad_seg.config import COLOR_TO_CLASS, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, NUM_CLASSES


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def list_samples(root):
    """Sorted names of the sample folders that contain both rgb.jpg and labels.png."""
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "rgb.jpg")) and os.path.isfile(os.path.join(root, d, "labels.png"))
    )


def read_rgb(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _color_codes(label_rgb):
    rgb = label_rgb.astype(np.int32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


_CODE_TO_CLASS = {(r << 16) | (g << 8) | b: c for (r, g, b), c in COLOR_TO_CLASS.items()}


def rgb_to_class_mask(label_rgb):
    """Converts a color-coded label image (H, W, 3) to class IDs (H, W). Unknown colors become 0."""
    codes = _color_codes(label_rgb)
    uniq, inverse = np.unique(codes, return_inverse=True)
    lut = np.array([_CODE_TO_CLASS.get(int(u), 0) for u in uniq], dtype=np.uint8)
    return lut[inverse].reshape(codes.shape)


def load_sample(root, name, size=INPUT_SIZE):
    """
    Loads a sample resized to `size` (height, width): bilinear for the image,
    nearest neighbour for the labels so that no new colors are created.
    Returns (rgb uint8 [H, W, 3], label_rgb uint8 [H, W, 3]).
    """
    folder = os.path.join(root, name)
    rgb = read_rgb(os.path.join(folder, "rgb.jpg"))
    label = read_rgb(os.path.join(folder, "labels.png"))
    if size is not None:
        h, w = size
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    return rgb, label


def classes_in_label(label_rgb, downsample=None):
    """Set of class IDs present in a label image (optionally on a downsampled copy, as in the analysis)."""
    if downsample is not None:
        label_rgb = cv2.resize(label_rgb, (downsample, downsample), interpolation=cv2.INTER_NEAREST)
    return {_CODE_TO_CLASS[int(u)] for u in np.unique(_color_codes(label_rgb)) if int(u) in _CODE_TO_CLASS}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OffRoadDataset(Dataset):
    """
    Original samples (read from disk) followed by optional pre-generated
    augmented samples (list of (rgb, label_rgb) arrays).
    Returns a normalised image tensor [3, H, W] and a class mask [H, W].
    With online_augmentation=True a random transformation is applied every
    time a sample is read (see online_augment).
    """

    def __init__(self, root, names, augmented=None, online_augmentation=False):
        self.root = root
        self.names = list(names)
        self.augmented = augmented or []
        self.online_augmentation = online_augmentation
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.names) + len(self.augmented)

    def __getitem__(self, idx):
        if idx < len(self.names):
            rgb, label = load_sample(self.root, self.names[idx])
        else:
            rgb, label = self.augmented[idx - len(self.names)]
            h, w = INPUT_SIZE
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
        mask = rgb_to_class_mask(label)
        if self.online_augmentation:
            rgb, mask = online_augment(rgb, mask)
        return self.transform(rgb), torch.from_numpy(mask).long()


def online_augment(rgb, mask):
    """
    Random augmentation applied at every epoch (v2 recipe):
    horizontal flip (p=0.5); random crop of 75-100% of the frame resized back
    (p=0.5), which also zooms on small objects; small rotation of +-5 degrees
    (p=0.3); brightness, contrast and saturation jitter.
    Geometric transforms use nearest-neighbour interpolation on the mask.
    """
    h, w = mask.shape
    if random.random() < 0.5:
        rgb, mask = rgb[:, ::-1], mask[:, ::-1]
    if random.random() < 0.5:
        scale = random.uniform(0.75, 1.0)
        ch, cw = int(h * scale), int(w * scale)
        y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
        rgb = cv2.resize(np.ascontiguousarray(rgb[y0:y0 + ch, x0:x0 + cw]), (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(np.ascontiguousarray(mask[y0:y0 + ch, x0:x0 + cw]), (w, h), interpolation=cv2.INTER_NEAREST)
    if random.random() < 0.3:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-5, 5), 1.0)
        rgb = cv2.warpAffine(np.ascontiguousarray(rgb), m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(np.ascontiguousarray(mask), m, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    img = rgb.astype(np.float32)
    img = img * random.uniform(0.8, 1.2) + random.uniform(-0.15, 0.15) * 255      # contrast, brightness
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * random.uniform(0.8, 1.2)                          # saturation
    return np.clip(img, 0, 255).astype(np.uint8), np.ascontiguousarray(mask)


def load_augmented_folder(folder):
    """Loads augmented samples saved by augmentation.save_augmentation."""
    names = list_samples(folder) if os.path.isdir(folder) else []
    return [load_sample(folder, n, size=None) for n in names]


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def stratified_split(root, names, val_size=180, train_ratio=0.8, seed=23):
    """
    Split stratified on the dominant non-background class of each image
    (computed on a 32x32 nearest-neighbour thumbnail of the labels).

    Groups are processed from the largest to the smallest; each contributes
    ~20% of its images to validation (at least one), and the last group
    fills the remaining budget so that the validation set has exactly
    `val_size` images.
    """
    random.seed(seed)
    np.random.seed(seed)

    groups = defaultdict(list)
    for name in names:
        label = read_rgb(os.path.join(root, name, "labels.png"))
        thumb = cv2.resize(label, (32, 32), interpolation=cv2.INTER_NEAREST)
        counts = np.bincount(rgb_to_class_mask(thumb).ravel(), minlength=NUM_CLASSES)
        foreground = {c: counts[c] for c in range(1, NUM_CLASSES) if counts[c] > 0}
        dominant = max(foreground, key=foreground.get) if foreground else int(np.argmax(counts))
        groups[dominant].append(name)

    train, val = [], []
    sizes = sorted(((c, len(groups[c])) for c in range(NUM_CLASSES) if groups[c]), key=lambda x: x[1], reverse=True)
    taken = 0
    for i, (class_id, total) in enumerate(sizes):
        remaining_groups = len(sizes) - i - 1
        budget = val_size - taken
        if remaining_groups == 0:
            n_val = budget
        else:
            n_val = max(1, min(round(total * (1 - train_ratio)), budget - remaining_groups))
        folders = groups[class_id].copy()
        random.shuffle(folders)
        val.extend(folders[:n_val])
        train.extend(folders[n_val:])
        taken += n_val

    return sorted(train), sorted(val)


# ---------------------------------------------------------------------------
# Scene-aware split (v2)
# ---------------------------------------------------------------------------

def _thumbnail_features(root, names):
    """Normalised 32x17 grayscale thumbnails: a cheap descriptor to find near-duplicate frames."""
    feats = []
    for name in names:
        gray = cv2.cvtColor(read_rgb(os.path.join(root, name, "rgb.jpg")), cv2.COLOR_RGB2GRAY)
        a = cv2.resize(gray, (32, 17), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        a = (a - a.mean()) / (a.std() + 1e-6)
        feats.append(a / np.linalg.norm(a))
    return np.stack(feats)


def scene_groups(root, names, sim_near=0.80, sim_any=0.95, max_id_gap=3):
    """
    Groups frames of the same scene. Two images are linked when their
    thumbnail correlation is above `sim_any`, or above `sim_near` and their
    folder numbers differ by at most `max_id_gap` (frames close in the
    recording). Groups are the connected components of this graph.
    """
    feats = _thumbnail_features(root, names)
    sim = feats @ feats.T
    ids = np.array([int(n) for n in names])
    parent = list(range(len(names)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(names)):
        near = np.abs(ids - ids[i]) <= max_id_gap
        linked = np.where(((sim[i] > sim_near) & near) | (sim[i] > sim_any))[0]
        for j in linked[linked > i]:
            parent[find(i)] = find(j)

    groups = defaultdict(list)
    for i, name in enumerate(names):
        groups[find(i)].append(name)
    return list(groups.values())


def grouped_split(root, names, val_fraction=0.2, seed=23, max_val_group=30):
    """
    Train/validation split in which every scene group (see scene_groups) is
    entirely on one side, so that no validation image has a near-duplicate
    in training. Groups containing the rare classes (puddle, obstacle) are
    assigned first, smallest first, so that validation gets about
    val_fraction of their images;
    the remaining budget is filled with random groups. Very large groups
    stay in training.
    """
    rng = random.Random(seed)
    groups = scene_groups(root, names)
    rng.shuffle(groups)

    present = {n: classes_in_label(read_rgb(os.path.join(root, n, "labels.png"))) for n in names}
    target = int(round(len(names) * val_fraction))
    val, used = [], set()

    def add(i):
        used.add(i)
        val.extend(groups[i])

    rare_classes = (7, 8)  # puddle, obstacle
    quota = {c: val_fraction * sum(c in present[n] for n in names) for c in rare_classes}

    def rare_in_val(c):
        return sum(c in present[n] for n in val)

    def allowed(i):
        # a group may not add a rare class whose validation quota is already reached
        g = groups[i]
        return i not in used and len(g) <= max_val_group and not any(
            rare_in_val(c) >= quota[c] and any(c in present[n] for n in g) for c in rare_classes)

    for rare in rare_classes:
        count = {i: sum(rare in present[n] for n in g) for i, g in enumerate(groups)}
        # smallest groups first, so that validation does not overshoot its share of the rare class
        for i in sorted((i for i in count if count[i] > 0), key=lambda i: count[i]):
            if rare_in_val(rare) >= quota[rare]:
                break
            if allowed(i):
                add(i)
    for i, g in enumerate(groups):
        if len(val) >= target:
            break
        if allowed(i) and len(val) + len(g) <= target + 5:
            add(i)

    val_set = set(val)
    return sorted(n for n in names if n not in val_set), sorted(val)
