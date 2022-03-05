# Optional list of dependencies required by the package
dependencies = ["torch", "torchvision"]

from torchlm.models import (
    pipnet_resnet18_10x68x32x256,
    pipnet_resnet50_10x68x32x256,
    pipnet_resnet101_10x68x32x256,
    pipnet_mobilenetv2_10x68x32x256
)
