"""
SegFormer (Xie et al., NeurIPS 2021) through Hugging Face transformers.

A hierarchical Transformer encoder (Mix Transformer) with a lightweight
all-MLP decoder. The weights are initialised from the checkpoints
fine-tuned on Cityscapes (urban driving scenes), so that both encoder and
decoder start from a segmentation model; only the 19-class head is replaced
by a 9-class one.
"""

import torch.nn.functional as F
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from offroad_seg.config import NUM_CLASSES

PRETRAINED = {
    "segformer-b1": "nvidia/segformer-b1-finetuned-cityscapes-1024-1024",
    "segformer-b2": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
}
# Architecture hyper-parameters, used when no pretrained weights are loaded
ARCHS = {
    "segformer-b1": dict(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=256),
    "segformer-b2": dict(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512], decoder_hidden_size=768),
}


class SegFormer(nn.Module):
    """SegFormer returning logits at the input resolution (the decoder works at 1/4)."""

    def __init__(self, variant="segformer-b2", pretrained=True, config_dict=None):
        super().__init__()
        self.variant = variant
        if config_dict is not None:
            self.net = SegformerForSemanticSegmentation(SegformerConfig(**config_dict))
        elif pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                PRETRAINED[variant], num_labels=NUM_CLASSES, ignore_mismatched_sizes=True)
        else:
            self.net = SegformerForSemanticSegmentation(SegformerConfig(num_labels=NUM_CLASSES, **ARCHS[variant]))

    def forward(self, x):
        logits = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)

    def config_dict(self):
        return self.net.config.to_dict()
