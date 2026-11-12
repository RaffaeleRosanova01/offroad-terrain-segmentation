"""Classes, label colors and preprocessing constants of the Yamaha-CMU Off-Road (YCOR) data."""

CLASS_NAMES = [
    "background",                      # 0
    "sky",                             # 1
    "rough_trail",                     # 2
    "smooth_trail",                    # 3
    "traversable_grass",               # 4
    "high_vegetation",                 # 5
    "non_traversable_low_vegetation",  # 6
    "puddle",                          # 7
    "obstacle",                        # 8
]
NUM_CLASSES = len(CLASS_NAMES)

# RGB color of each class in the color-coded label images.
CLASS_COLORS = {
    0: (255, 255, 255),
    1: (1, 88, 255),
    2: (156, 76, 30),
    3: (178, 176, 153),
    4: (128, 255, 0),
    5: (40, 80, 0),
    6: (0, 160, 0),
    7: (255, 0, 128),
    8: (255, 0, 0),
}
COLOR_TO_CLASS = {color: cid for cid, color in CLASS_COLORS.items()}

# The label PNG files are palette images whose index order differs from the
# class IDs above. Palette of the label files (index -> color):
LABEL_PALETTE = [
    (255, 255, 255),  # 0 background
    (178, 176, 153),  # 1 smooth_trail
    (128, 255, 0),    # 2 traversable_grass
    (156, 76, 30),    # 3 rough_trail
    (255, 0, 128),    # 4 puddle
    (255, 0, 0),      # 5 obstacle
    (0, 160, 0),      # 6 non_traversable_low_vegetation
    (40, 80, 0),      # 7 high_vegetation
    (1, 88, 255),     # 8 sky
]
CLASS_TO_LABEL_INDEX = {cid: LABEL_PALETTE.index(color) for cid, color in CLASS_COLORS.items()}

# Images are 1024x544; the network works at 768x416 (both multiples of 32).
ORIGINAL_SIZE = (544, 1024)   # (height, width)
INPUT_SIZE = (416, 768)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
