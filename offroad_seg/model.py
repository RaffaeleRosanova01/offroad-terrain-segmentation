"""
ResNet34 encoder + U-Net decoder.

The encoder is an ImageNet-pretrained ResNet34 without the classification
head; it returns the bottleneck (1/32 resolution, 512 channels) and five
skip connections (1/16 ... 1/2). The decoder upsamples with transposed
convolutions, concatenates the skip connection of the same resolution and
refines with two 3x3 conv + BN + ReLU; a final 1x1 convolution gives the
logits of the 9 classes at the input resolution.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from offroad_seg.config import INPUT_SIZE, NUM_CLASSES


class ResNet34Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        c1 = self.relu(self.bn1(self.conv1(x)))  # 1/2,  64 ch
        c2 = self.maxpool(c1)                     # 1/4,  64 ch
        c3 = self.layer1(c2)                      # 1/4,  64 ch
        c4 = self.layer2(c3)                      # 1/8,  128 ch
        c5 = self.layer3(c4)                      # 1/16, 256 ch
        c6 = self.layer4(c5)                      # 1/32, 512 ch (bottleneck)
        return c6, [c5, c4, c3, c2, c1]


class UNetDecoder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, target_height=INPUT_SIZE[0], target_width=INPUT_SIZE[1]):
        super().__init__()
        self.target_height = target_height
        self.target_width = target_width

        self.upconv6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6 = self._make_conv_block(256 + 256, 256)
        self.upconv7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7 = self._make_conv_block(128 + 128, 128)
        self.upconv8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8 = self._make_conv_block(64 + 64, 64)
        self.upconv9 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv9 = self._make_conv_block(64 + 64, 64)
        self.upconv10 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv10 = self._make_conv_block(32 + 64, 32)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    @staticmethod
    def _make_conv_block(in_channels, out_channels):
        """Two 3x3 convolutions with BatchNorm and ReLU (standard U-Net block)."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _up(x, upconv, skip, conv):
        # Upsample, align to the skip size (odd sizes), concatenate, refine.
        x = F.interpolate(upconv(x), size=skip.shape[2:], mode="bilinear", align_corners=False)
        return conv(torch.cat([x, skip], dim=1))

    def forward(self, bottleneck, skip_connections):
        c5, c4, c3, c2, c1 = skip_connections
        x = self._up(bottleneck, self.upconv6, c5, self.conv6)
        x = self._up(x, self.upconv7, c4, self.conv7)
        x = self._up(x, self.upconv8, c3, self.conv8)
        x = self._up(x, self.upconv9, c2, self.conv9)
        x = self._up(x, self.upconv10, c1, self.conv10)
        # Back to the input resolution (the last skip is at 1/2)
        x = F.interpolate(x, size=(self.target_height, self.target_width), mode="bilinear", align_corners=False)
        return self.final_conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, input_height=INPUT_SIZE[0], input_width=INPUT_SIZE[1]):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = ResNet34Encoder(pretrained=pretrained)
        self.decoder = UNetDecoder(num_classes, input_height, input_width)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        return self.decoder(bottleneck, skips)


MODELS = ("resnet34_unet", "segformer-b1", "segformer-b2")


def build_model(name="resnet34_unet", pretrained=True, config_dict=None):
    if name == "resnet34_unet":
        return ResNetUNet(pretrained=pretrained)
    from offroad_seg.segformer import SegFormer  # requires transformers
    return SegFormer(name, pretrained=pretrained, config_dict=config_dict)


def load_model(weights_path, device="cpu"):
    """
    Loads a trained model. Accepts a plain ResNet-UNet state dict or a
    checkpoint with 'model_state_dict' (and, for SegFormer, 'model_name'
    and 'model_config', so that no download is needed).
    """
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    name = checkpoint.get("model_name", "resnet34_unet") if isinstance(checkpoint, dict) else "resnet34_unet"
    model = build_model(name, pretrained=False, config_dict=checkpoint.get("model_config"))
    model.load_state_dict(state_dict)
    return model.to(device).eval()
