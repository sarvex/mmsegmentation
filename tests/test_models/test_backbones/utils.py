# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch.nn.modules import GroupNorm
from torch.nn.modules.batchnorm import _BatchNorm

from mmseg.models.backbones.resnet import BasicBlock, Bottleneck
from mmseg.models.backbones.resnext import Bottleneck as BottleneckX


def is_block(modules):
    """Check if is ResNet building block."""
    return isinstance(modules, (BasicBlock, Bottleneck, BottleneckX))


def is_norm(modules):
    """Check if is one of the norms."""
    return isinstance(modules, (GroupNorm, _BatchNorm))


def all_zeros(modules):
    """Check if the weight(and bias) is all zero."""
    weight_zero = torch.allclose(modules.weight.data,
                                 torch.zeros_like(modules.weight.data))
    if hasattr(modules, 'bias'):
        bias_zero = torch.allclose(modules.bias.data,
                                   torch.zeros_like(modules.bias.data))
    else:
        bias_zero = True

    return weight_zero and bias_zero


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    return not any(
        isinstance(mod, _BatchNorm) and mod.training != train_state
        for mod in modules
    )
