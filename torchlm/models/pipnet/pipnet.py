"""
A PyTorch Re-implementation of PIPNet with all-in-one style, include
model definition, loss computation, training, inference, exporting
and evaluating processes in a one class.
Reference: https://github.com/jhb86253817/PIPNet/blob/master/lib/networks.py
"""
import warnings

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet, MobileNetV2
from typing import Union, Optional, Any

from ._impls import (
    _PIPNetImpl,
    _PIPNet_Output_Type
)
from .._utils import freeze_module, freeze_bn

__all__ = [
    "pipnet_resnet18_10x68x32x256_300w",
    "pipnet_resnet101_10x68x32x256_300w",
    "pipnet_resnet18_10x19x32x256_aflw",
    "pipnet_resnet101_10x19x32x256_aflw",
    "pipnet_resnet18_10x29x32x256_cofw",
    "pipnet_resnet101_10x29x32x256_cofw",
    "pipnet_resnet18_10x98x32x256_wflw",
    "pipnet_resnet101_10x98x32x256_wflw",
    "pipnet"
]

model_urls_root_r16 = "https://github.com/DefTruth/torchlm/releases/download/torchlm-0.1.6-alpha/"

model_urls = {
    # Paths for torchlm v0.1.6 PIPNet's pretrained weights
    'pipnet_resnet18_10x68x32x256_300w': f'{model_urls_root_r16}/pipnet_resnet18_10x68x32x256_300w.pth',
    'pipnet_resnet101_10x68x32x256_300w': f'{model_urls_root_r16}/pipnet_resnet101_10x68x32x256_300w.pth',
    'pipnet_resnet18_10x19x32x256_aflw': f'{model_urls_root_r16}/pipnet_resnet18_10x19x32x256_aflw.pth',
    'pipnet_resnet101_10x19x32x256_aflw': f'{model_urls_root_r16}/pipnet_resnet101_10x19x32x256_aflw.pth',
    'pipnet_resnet18_10x29x32x256_cofw': f'{model_urls_root_r16}/pipnet_resnet18_10x29x32x256_cofw.pth',
    'pipnet_resnet101_10x29x32x256_cofw': f'{model_urls_root_r16}/pipnet_resnet101_10x29x32x256_cofw.pth',
    'pipnet_resnet18_10x98x32x256_wflw': f'{model_urls_root_r16}/pipnet_resnet18_10x98x32x256_wflw.pth',
    'pipnet_resnet101_10x98x32x256_wflw': f'{model_urls_root_r16}/pipnet_resnet101_10x98x32x256_wflw.pth'
}


class PIPNetResNet(_PIPNetImpl):
    def __init__(
            self,
            resnet: ResNet,
            num_nb: int = 10,
            num_lms: int = 68,
            input_size: int = 256,
            net_stride: int = 32,
            expansion: int = 4,
            meanface_type: Optional[str] = None
    ):
        """
        :param resnet: specific ResNet backbone from torchvision.models, such as resnet18/34/50/101/...
        :param num_nb: the number of Nearest-neighbor landmarks for NRM, default 10
        :param num_lms: the number of input/output landmarks, default 68.
        :param input_size: input size for PIPNet, default 256.
        :param net_stride: net stride for PIPNet, default 32, should be one of (32,64,128).
        :param expansion: expansion ratio for ResNet backbone, 1 or 4
        :param meanface_type: meanface type for PIPNet, AFLW/WFLW/COFW/300W/300W_CELEBA/300W_COFW_WFLW
        The relationship of net_stride and the output size of feature map is:
            # net_stride output_size
            # 128        2x2
            # 64         4x4
            # 32         8x8
        """
        super(PIPNetResNet, self).__init__(
            num_nb=num_nb,
            num_lms=num_lms,
            input_size=input_size,
            net_stride=net_stride,
            meanface_type=meanface_type
        )
        # 1: ResNet18/34, 4:ResNet50/101/150/...
        assert expansion in (1, 4)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.maxpool = resnet.maxpool
        self.sigmoid = nn.Sigmoid()
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        # calculate inplane & plane
        self.inplane = 512 * expansion  # 512/2048
        self.plane = self.inplane // (net_stride // 32)  # 32
        # setup extra layers
        self._make_extra_layers(inplane=self.inplane, plane=self.plane)
        # setup det headers
        self._make_det_headers(plane=self.plane)

    def set_custom_meanface(
            self,
            custom_meanface_file_or_string: str
    ) -> bool:
        """
        :param custom_meanface_file_or_string: a long string or a file contains normalized
        or un-normalized meanface coords, the format is "x0,y0,x1,y1,x2,y2,...,xn-1,yn-1".
        :return: status, True if successful.
        """
        old_num_lms = self.num_lms
        if super(PIPNetResNet, self).set_custom_meanface(
            custom_meanface_file_or_string=custom_meanface_file_or_string
        ):
            # update detect headers and extra layers according to new num_lms
            if self.num_lms != old_num_lms:
                # setup extra layers
                self._make_extra_layers(inplane=self.inplane, plane=self.plane)
                # setup det headers
                self._make_det_headers(plane=self.plane)
                warnings.warn(f"update detect headers and extra"
                              f" layers according to new num_lms: "
                              f"{self.num_lms}, the old num_lms "
                              f"is: {old_num_lms}")
            return True
        return False

    def apply_freezing(
            self,
            backbone: Optional[bool] = True,
            heads: Optional[bool] = False,
            extra: Optional[bool] = False
    ):
        if backbone:
            freeze_module(self.layer1)
            freeze_module(self.layer2)
            freeze_module(self.layer3)
            freeze_module(self.layer4)
        if heads:
            freeze_module(self.cls_layer)
            freeze_module(self.x_layer)
            freeze_module(self.y_layer)
            freeze_module(self.nb_x_layer)
            freeze_module(self.nb_y_layer)
        if extra:
            if self.net_stride == 128:
                freeze_module(self.layer5)
                freeze_bn(self.bn5)
                freeze_module(self.layer6)
                freeze_bn(self.bn6)
            elif self.net_stride == 64:
                freeze_module(self.layer5)
                freeze_bn(self.bn5)

    def _make_extra_layers(
            self,
            inplane: int = 2048,
            plane: int = 2048
    ):
        assert self.net_stride in (32, 64, 128)
        if self.net_stride == 128:
            self.layer5 = nn.Conv2d(inplane, plane, kernel_size=(3, 3),
                                    stride=(2, 2), padding=(1, 1))
            self.bn5 = nn.BatchNorm2d(plane)
            self.layer6 = nn.Conv2d(plane, plane, kernel_size=(3, 3),
                                    stride=(2, 2), padding=(1, 1))
            self.bn6 = nn.BatchNorm2d(plane)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)

            nn.init.normal_(self.layer6.weight, std=0.001)
            if self.layer6.bias is not None:
                nn.init.constant_(self.layer6.bias, 0)
            nn.init.constant_(self.bn6.weight, 1)
            nn.init.constant_(self.bn6.bias, 0)
        elif self.net_stride == 64:
            self.layer5 = nn.Conv2d(
                inplane, plane, kernel_size=(3, 3),
                stride=(2, 2), padding=(1, 1)
            )
            self.bn5 = nn.BatchNorm2d(plane)
            # init
            nn.init.normal_(self.layer5.weight, std=0.001)
            if self.layer5.bias is not None:
                nn.init.constant_(self.layer5.bias, 0)
            nn.init.constant_(self.bn5.weight, 1)
            nn.init.constant_(self.bn5.bias, 0)

    def _make_det_headers(
            self,
            plane: int = 2048
    ):
        # cls_layer: (68,8,8)
        self.cls_layer = nn.Conv2d(
            plane, self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # x_layer: (68,8,8)
        self.x_layer = nn.Conv2d(
            plane, self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # y_layer: (68,8,8)
        self.y_layer = nn.Conv2d(
            plane, self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # nb_x_layer: (68*10,8,8)
        self.nb_x_layer = nn.Conv2d(
            plane, self.num_nb * self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # nb_y_layer: (68*10,8,8)
        self.nb_y_layer = nn.Conv2d(
            plane, self.num_nb * self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def _forward_extra(self, x: Tensor) -> Tensor:
        if self.net_stride == 128:
            x = F.relu(self.bn5(self.layer5(x)))
            x = F.relu(self.bn6(self.layer6(x)))
        elif self.net_stride == 64:
            x = F.relu(self.bn5(self.layer5(x)))

        return x

    def _forward_impl(self, x: Tensor) -> _PIPNet_Output_Type:

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        self._forward_extra(x)
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5

    def forward(self, x: Tensor) -> _PIPNet_Output_Type:
        return self._forward_impl(x)


class PIPNetMobileNetV2(_PIPNetImpl):
    def __init__(
            self,
            mbnet: MobileNetV2,
            num_nb: int = 10,
            num_lms: int = 68,
            input_size: int = 256,
            net_stride: int = 32,
            meanface_type: Optional[str] = None
    ):
        """
        :param mbnet: specific MobileNetV2 backbone from torchvision.models,
        :param num_nb: the number of Nearest-neighbor landmarks for NRM, default 10
        :param num_lms: the number of input/output landmarks, default 68.
        :param input_size: input size for PIPNet, default 256.
        :param net_stride: net stride for PIPNet, only support 32
        :param meanface_type: meanface type for PIPNet, AFLW/WFLW/COFW/300W/300W_CELEBA/300W_COFW_WFLW
        The relationship of net_stride and the output size of feature map is:
            # net_stride output_size
            # 128        2x2
            # 64         4x4
            # 32         8x8
        """
        super(PIPNetMobileNetV2, self).__init__(
            num_nb=num_nb,
            num_lms=num_lms,
            input_size=input_size,
            net_stride=net_stride,
            meanface_type=meanface_type
        )
        assert net_stride == 32, "only support net_stride==32!"
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.features = mbnet.features
        self.sigmoid = nn.Sigmoid()
        # calculate plane
        self.plane = 1280
        # setup det headers
        self._make_det_headers(plane=self.plane)

    def set_custom_meanface(
            self,
            custom_meanface_file_or_string: str
    ) -> bool:
        """
        :param custom_meanface_file_or_string: a long string or a file contains normalized
        or un-normalized meanface coords, the format is "x0,y0,x1,y1,x2,y2,...,xn-1,yn-1".
        :return: status, True if successful.
        """
        old_num_lms = self.num_lms
        if super(PIPNetMobileNetV2, self).set_custom_meanface(
            custom_meanface_file_or_string=custom_meanface_file_or_string
        ):
            # update detect headers and extra layers according to new num_lms
            if self.num_lms != old_num_lms:
                # setup det headers
                self._make_det_headers(plane=self.plane)
                warnings.warn(f"update detect headers and extra"
                              f" layers according to new num_lms: "
                              f"{self.num_lms}, the old num_lms "
                              f"is: {old_num_lms}")
            return True
        return False

    def apply_freezing(
            self,
            backbone: Optional[bool] = True,
            heads: Optional[bool] = False
    ):
        if backbone:
            freeze_module(self.layer1)
            freeze_module(self.layer2)
            freeze_module(self.layer3)
            freeze_module(self.layer4)
        if heads:
            freeze_module(self.cls_layer)
            freeze_module(self.x_layer)
            freeze_module(self.y_layer)
            freeze_module(self.nb_x_layer)
            freeze_module(self.nb_y_layer)

    def _make_det_headers(
            self,
            plane: int = 1280
    ):
        # cls_layer: (68,8,8)
        self.cls_layer = nn.Conv2d(
            plane, self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # x_layer: (68,8,8)
        self.x_layer = nn.Conv2d(
            plane, self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # y_layer: (68,8,8)
        self.y_layer = nn.Conv2d(
            plane, self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # nb_x_layer: (68*10,8,8)
        self.nb_x_layer = nn.Conv2d(
            plane, self.num_nb * self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # nb_y_layer: (68*10,8,8)
        self.nb_y_layer = nn.Conv2d(
            plane, self.num_nb * self.num_lms,
            kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        nn.init.normal_(self.cls_layer.weight, std=0.001)
        if self.cls_layer.bias is not None:
            nn.init.constant_(self.cls_layer.bias, 0)

        nn.init.normal_(self.x_layer.weight, std=0.001)
        if self.x_layer.bias is not None:
            nn.init.constant_(self.x_layer.bias, 0)

        nn.init.normal_(self.y_layer.weight, std=0.001)
        if self.y_layer.bias is not None:
            nn.init.constant_(self.y_layer.bias, 0)

        nn.init.normal_(self.nb_x_layer.weight, std=0.001)
        if self.nb_x_layer.bias is not None:
            nn.init.constant_(self.nb_x_layer.bias, 0)

        nn.init.normal_(self.nb_y_layer.weight, std=0.001)
        if self.nb_y_layer.bias is not None:
            nn.init.constant_(self.nb_y_layer.bias, 0)

    def _forward_impl(self, x: Tensor) -> _PIPNet_Output_Type:

        x = self.features(x)
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5

    def forward(self, x: Tensor) -> _PIPNet_Output_Type:
        return self._forward_impl(x)


def pipnet(
        arch: Optional[str] = None,
        backbone: Optional[str] = "resnet18",
        pretrained: Optional[bool] = True,
        progress: Optional[bool] = True,
        num_nb: Optional[int] = 10,
        num_lms: Optional[int] = 68,
        net_stride: Optional[int] = 32,
        input_size: Optional[int] = 256,
        meanface_type: Optional[str] = "300w",
        backbone_pretrained: Optional[bool] = True,
        map_location: Optional[Union[str, torch.device]] = "cpu",
        checkpoint: Optional[str] = None,
        **kwargs: Any
) -> Union[PIPNetResNet, PIPNetMobileNetV2]:
    """
    :param arch: If arch is not None and pretrained is True, this function
     will try to load a pretrained PIPNet from torchlm's Github Repo. The format
     of arch must be:
     pipnet_backbone_(num_nb)x(num_lms)x(net_stride)x(input_size)_(meanface_type)
    :param backbone: Backbone type, one of ("resnet18", "resnet50", "resnet101", "mobilenet_v2")
    :param pretrained: If True and 'checkpoint' is None, we will try to load the pretrained
     PIPNet weights from torchlm's Github Repo, default True.
    :param progress: If True, displays a progress bar of the download to stderr
    :param num_nb: The number of Nearest-neighbor landmarks for NRM, default 10
    :param num_lms: The number of input/output landmarks, default 68.
    :param net_stride: net stride for PIPNet, default 32, should be one of (32,64,128).
    :param input_size: input size for PIPNet, default 256.
    :param meanface_type: meanface type for PIPNet, AFLW/WFLW/COFW/300W/300W_CELEBA/300W_COFW_WFLW
    :param backbone_pretrained: If True, try to load a backbone with pretrained weights.
    :param map_location: location to map the user's loading device.
    :param checkpoint: optional, local path to a pretrained weights, if not None,
     this method will try the pretrained weights from local path.
    :param kwargs: rest parameters for resnet and mobilenet_v2 from torchvision.
    :return:
    """
    import os

    # force map location to cpu if cuda is not available.
    map_location = map_location if torch.cuda.is_available() else "cpu"

    assert backbone in ("resnet18", "resnet50", "resnet101", "mobilenet_v2")

    if backbone != "mobilenet_v2":
        if backbone == "resnet18":
            expansion = 1
            resnet = models.resnet18(pretrained=backbone_pretrained, **kwargs)
        elif backbone == "resnet50":
            expansion = 4
            resnet = models.resnet50(pretrained=backbone_pretrained, **kwargs)
        else:
            expansion = 4
            resnet = models.resnet101(pretrained=backbone_pretrained, **kwargs)
        model = PIPNetResNet(
            resnet,
            num_nb=num_nb,
            num_lms=num_lms,
            net_stride=net_stride,
            input_size=input_size,
            expansion=expansion,
            meanface_type=meanface_type
        )
    else:
        mobilenetv2 = models.mobilenet_v2(pretrained=backbone_pretrained, **kwargs)
        model = PIPNetMobileNetV2(
            mobilenetv2,
            num_nb=num_nb,
            num_lms=num_lms,
            net_stride=net_stride,
            input_size=input_size,
            meanface_type=meanface_type
        )
    if pretrained and checkpoint is None:
        try:
            if arch is not None:
                # perform arch check before loading.
                assert arch in model_urls, f"Invalid arch: {arch}!"
                params = arch.strip(" ").split("_")
                _arch_backbone = params[1]  # e.g resnet18
                _arch_num_nb, _arch_num_lms, _arch_net_stride, _arch_input_size = \
                    [int(x) for x in params[2].split("x")]
                _arch_meanface_type = params[-1]
                assert all((
                    _arch_backbone == backbone, _arch_num_nb == num_nb,
                    _arch_num_lms == num_lms, _arch_net_stride == net_stride,
                    _arch_input_size == input_size, _arch_meanface_type == meanface_type
                )), "arch check failed!"
            else:
                # build arch from params
                arch = f"pipnet_{backbone}_{num_nb}x{num_lms}" \
                       f"x{net_stride}x{input_size}_{meanface_type}"
            # try to load pretrained weights
            state_dict = load_state_dict_from_url(model_urls[arch],
                                                  progress=progress,
                                                  map_location=map_location)
            model.load_state_dict(state_dict)
        except Exception as e:
            warnings.warn(f"Can not load pretrained weights from: {model_urls[arch]}"
                          f" for {arch} ! Skip this loading! \nError Info: {e}\n"
                          f"Note: The arch param should be one of:\n"
                          f" {model_urls.keys()}")

    # load a pretrained weights from local path
    if checkpoint is not None and os.path.exists(checkpoint):
        state_dict = torch.load(checkpoint, map_location=map_location)
        model.load_state_dict(state_dict)

    return model.to(map_location)


# Format: pipnet_backbone_(num_nb)x(num_lms)x(net_stride)x(input_size)_(meanface_type)
# 300W 68 landmarks
def pipnet_resnet18_10x68x32x256_300w(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNetImpl:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return pipnet("pipnet_resnet18_10x68x32x256_300w", "resnet18", pretrained, progress,
                   10, 68, 32, 256, "300w", **kwargs)

def pipnet_resnet101_10x68x32x256_300w(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNetImpl:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return pipnet("pipnet_resnet101_10x68x32x256_300w", "resnet101", pretrained, progress,
                   10, 68, 32, 256, "300w", **kwargs)


# AFLW 19 landmarks
def pipnet_resnet18_10x19x32x256_aflw(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNetImpl:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return pipnet("pipnet_resnet18_10x19x32x256_aflw", "resnet18", pretrained, progress,
                   10, 19, 32, 256, "aflw", **kwargs)


def pipnet_resnet101_10x19x32x256_aflw(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNetImpl:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return pipnet("pipnet_resnet101_10x19x32x256_aflw", "resnet101", pretrained, progress,
                   10, 19, 32, 256, "aflw", **kwargs)


# COFW 29 landmarks
def pipnet_resnet18_10x29x32x256_cofw(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNetImpl:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return pipnet("pipnet_resnet18_10x29x32x256_cofw", "resnet18", pretrained, progress,
                   10, 29, 32, 256, "cofw", **kwargs)


def pipnet_resnet101_10x29x32x256_cofw(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNetImpl:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return pipnet("pipnet_resnet101_10x29x32x256_cofw", "resnet101", pretrained, progress,
                   10, 29, 32, 256, "cofw", **kwargs)


# WFLW 98 landmarks
def pipnet_resnet18_10x98x32x256_wflw(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNetImpl:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return pipnet("pipnet_resnet18_10x98x32x256_wflw", "resnet18", pretrained, progress,
                   10, 98, 32, 256, "wflw", **kwargs)


def pipnet_resnet101_10x98x32x256_wflw(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNetImpl:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return pipnet("pipnet_resnet101_10x98x32x256_wflw", "resnet101", pretrained, progress,
                   10, 98, 32, 256, "wflw", **kwargs)
