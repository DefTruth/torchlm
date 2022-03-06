# Optional list of dependencies required by the package
dependencies = ["torch", "torchvision"]

from torchlm.models import (
    pipnet,
    pipnet_resnet18_10x68x32x256_300w,
    pipnet_resnet50_10x68x32x256_300w,
    pipnet_resnet101_10x68x32x256_300w
)
