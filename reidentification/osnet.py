"""OSNet (Omni-Scale Network) architecture for person re-identification.

Standalone PyTorch implementation extracted from KaiyangZhou/deep-person-reid.
License: MIT (Copyright (c) 2018 Kaiyang Zhou).

References:
    - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
    - Zhou et al. Learning Generalisable Omni-Scale Representations
      for Person Re-Identification. TPAMI, 2021.

Source: https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/models/osnet.py
Weights: https://huggingface.co/kaiyangzhou/osnet (MIT license)
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["OSNet", "osnet_x1_0", "osnet_x0_75", "osnet_x0_5", "osnet_x0_25"]


# ---------------------------------------------------------------------------
# Basic layers
# ---------------------------------------------------------------------------
class ConvLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride=1, padding=0, groups=1, IN=False):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.InstanceNorm2d(out_c, affine=True) if IN else nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1(nn.Module):
    def __init__(self, in_c, out_c, stride=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False, groups=out_c)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


# ---------------------------------------------------------------------------
# Omni-scale building blocks
# ---------------------------------------------------------------------------
class ChannelGate(nn.Module):
    def __init__(self, in_c, num_gates=None, return_gates=False, gate_activation="sigmoid", reduction=16, layer_norm=False):
        super().__init__()
        if num_gates is None:
            num_gates = in_c
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_c, in_c // reduction, 1, bias=True)
        self.norm1 = nn.LayerNorm((in_c // reduction, 1, 1)) if layer_norm else None
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_c // reduction, num_gates, 1, bias=True)
        if gate_activation == "sigmoid":
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == "relu":
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == "linear":
            self.gate_activation = None
        else:
            raise RuntimeError(f"Unknown gate activation: {gate_activation}")

    def forward(self, x):
        inp = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        return x if self.return_gates else inp * x


class OSBlock(nn.Module):
    def __init__(self, in_c, out_c, IN=False, bottleneck_reduction=4, **kwargs):
        super().__init__()
        mid = out_c // bottleneck_reduction
        self.conv1 = Conv1x1(in_c, mid)
        self.conv2a = LightConv3x3(mid, mid)
        self.conv2b = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2c = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.conv2d = nn.Sequential(LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid), LightConv3x3(mid, mid))
        self.gate = ChannelGate(mid)
        self.conv3 = Conv1x1Linear(mid, out_c)
        self.downsample = Conv1x1Linear(in_c, out_c) if in_c != out_c else None
        self.IN = nn.InstanceNorm2d(out_c, affine=True) if IN else None

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)
        x2 = self.gate(self.conv2a(x1)) + self.gate(self.conv2b(x1)) + self.gate(self.conv2c(x1)) + self.gate(self.conv2d(x1))
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return F.relu(out)


# ---------------------------------------------------------------------------
# OSNet main network
# ---------------------------------------------------------------------------
class OSNet(nn.Module):
    """Omni-Scale Network for person re-identification.

    In eval mode, ``forward()`` returns a 512-dim (or ``feature_dim``) L2-ready
    appearance embedding per input crop.
    """

    def __init__(self, num_classes, blocks, layers, channels, feature_dim=512, loss="softmax", IN=False, **kwargs):
        super().__init__()
        assert len(blocks) == len(layers) == len(channels) - 1
        self.loss = loss
        self.feature_dim = feature_dim

        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3, IN=IN)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = self._make_layer(blocks[0], layers[0], channels[0], channels[1], reduce=True, IN=IN)
        self.conv3 = self._make_layer(blocks[1], layers[1], channels[1], channels[2], reduce=True)
        self.conv4 = self._make_layer(blocks[2], layers[2], channels[2], channels[3], reduce=False)
        self.conv5 = Conv1x1(channels[3], channels[3])
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc(feature_dim, channels[3])
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self._init_params()

    @staticmethod
    def _make_layer(block, n_layers, in_c, out_c, reduce=False, IN=False):
        layers = [block(in_c, out_c, IN=IN)]
        for _ in range(1, n_layers):
            layers.append(block(out_c, out_c, IN=IN))
        if reduce:
            layers.append(nn.Sequential(Conv1x1(out_c, out_c), nn.AvgPool2d(2, stride=2)))
        return nn.Sequential(*layers)

    def _construct_fc(self, dim, input_dim):
        if dim is None or dim < 0:
            self.feature_dim = input_dim
            return None
        self.feature_dim = dim
        return nn.Sequential(nn.Linear(input_dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True))

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.training:
            return v
        y = self.classifier(v)
        if self.loss == "softmax":
            return y
        elif self.loss == "triplet":
            return y, v
        else:
            raise KeyError(f"Unsupported loss: {self.loss}")


# ---------------------------------------------------------------------------
# Weight loading utilities
# ---------------------------------------------------------------------------
def load_pretrained_weights(model: OSNet, weight_path: str | Path) -> None:
    """Load pretrained weights into an OSNet model, tolerating mismatched layers."""
    state_dict = torch.load(str(weight_path), map_location="cpu", weights_only=True)
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)


# ---------------------------------------------------------------------------
# Factory functions (match torchreid API)
# ---------------------------------------------------------------------------
def osnet_x1_0(num_classes=1000, pretrained_path=None, loss="softmax", **kw):
    m = OSNet(num_classes, [OSBlock] * 3, [2, 2, 2], [64, 256, 384, 512], loss=loss, **kw)
    if pretrained_path:
        load_pretrained_weights(m, pretrained_path)
    return m


def osnet_x0_75(num_classes=1000, pretrained_path=None, loss="softmax", **kw):
    m = OSNet(num_classes, [OSBlock] * 3, [2, 2, 2], [48, 192, 288, 384], loss=loss, **kw)
    if pretrained_path:
        load_pretrained_weights(m, pretrained_path)
    return m


def osnet_x0_5(num_classes=1000, pretrained_path=None, loss="softmax", **kw):
    m = OSNet(num_classes, [OSBlock] * 3, [2, 2, 2], [32, 128, 192, 256], loss=loss, **kw)
    if pretrained_path:
        load_pretrained_weights(m, pretrained_path)
    return m


def osnet_x0_25(num_classes=1000, pretrained_path=None, loss="softmax", **kw):
    m = OSNet(num_classes, [OSBlock] * 3, [2, 2, 2], [16, 64, 96, 128], loss=loss, **kw)
    if pretrained_path:
        load_pretrained_weights(m, pretrained_path)
    return m
