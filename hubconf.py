# Optional list of dependencies required by the package
dependencies = ["torch", "torchvision"]

from torchlm.models import (
    pipnet_resnet18_10x68x32x256_300w,
    pipnet_resnet50_10x68x32x256_300w,
    pipnet_resnet101_10x68x32x256_300w,
    pipnet_resnet18_10x19x32x256_aflw,
    pipnet_resnet50_10x19x32x256_aflw,
    pipnet_resnet101_10x19x32x256_aflw,
    pipnet_resnet18_10x29x32x256_cofw,
    pipnet_resnet50_10x29x32x256_cofw,
    pipnet_resnet101_10x29x32x256_cofw,
    pipnet_resnet18_10x98x32x256_wflw,
    pipnet_resnet50_10x98x32x256_wflw,
    pipnet_resnet101_10x98x32x256_wflw
)
