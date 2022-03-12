import torch.nn as nn
from ..data import annotools
from ..metrics import metrics
from ..transforms import transforms

from typing import Optional

def freeze_bn(module: nn.Module) -> None:
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.eval()
        module.weight.requires_grad = False
        module.bias.requires_grad = False


def activate_bn(module: nn.Module) -> None:
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        module.train()
        module.weight.requires_grad = True
        module.bias.requires_grad = True


def freeze_module(module: Optional[nn.Module] = None):
    if module is None:
        return
    for para in module.parameters():
        para.requires_grad = False
    for m in module.modules():
        freeze_bn(module=m)
