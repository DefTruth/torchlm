"""
Reference: https://github.com/jhb86253817/PIPNet/blob/master/lib/networks.py
"""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet, MobileNetV2
from typing import Tuple, Union, Optional, Any

_PIPNet_Output_Type = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

__all__ = ["PIPNetResNet", "PIPNetMobileNetV2", "pipnet_resnet18_10x68x32x256",
           "pipnet_resnet50_10x68x32x256", "pipnet_resnet101_10x68x32x256",
           "pipnet_mobilenetv2_10x68x32x256"]

# TODO: update model_urls
model_urls = {
    'pipnet_resnet18_10x68x32x256': 'pipnet_resnet18_10x68x32x256.pth',
    'pipnet_resnet50_10x68x32x256': 'pipnet_resnet50_10x68x32x256.pth',
    'pipnet_resnet101_10x68x32x256': 'pipnet_resnet101_10x68x32x256.pth',
    'pipnet_mobilenetv2_10x68x32x256': 'pipnet_mobilenetv2_10x68x32x256.pth',
}


class PIPNetResNet(nn.Module):
    def __init__(
            self,
            resnet: ResNet,
            num_nb: int = 10,
            num_lms: int = 68,
            input_size: int = 256,
            net_stride: int = 32,
            expansion: int = 4
    ):
        """
        :param resnet: specific ResNet backbone from torchvision.models, such as resnet18/34/50/101/...
        :param num_nb: the number of Nearest-neighbor landmarks for NRM, default 10
        :param num_lms: the number of input/output landmarks, default 68.
        :param input_size: input size for PIPNet, default 256.
        :param net_stride: net stride for PIPNet, default 32, should be one of (32,64,128).
        :param expansion: expansion ratio for ResNet backbone, 1 or 4
        The relationship of net_stride and the output size of feature map is:
            # net_stride output_size
            # 128        2x2
            # 64         4x4
            # 32         8x8
        """
        super(PIPNetResNet, self).__init__()
        assert net_stride in (32, 64, 128)
        # 1: ResNet18/34, 4:ResNet50/101/150/...
        assert expansion in (1, 4)
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
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
        self._set_extra_layers(inplane=self.inplane, plane=self.plane)
        # setup det headers
        self._set_det_headers(plane=self.plane)

    def _set_extra_layers(
            self,
            inplane: int = 2048,
            plane: int = 2048
    ):
        assert self.net_stride in (64, 128)
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

    def _set_det_headers(
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

    def forward(
            self,
            x: Tensor
    ) -> _PIPNet_Output_Type:

        return self._forward_impl(x)

    def _forward_extra(
            self,
            x: Tensor
    ) -> Tensor:

        if self.net_stride == 128:
            x = F.relu(self.bn5(self.layer5(x)))
            x = F.relu(self.bn6(self.layer6(x)))
        elif self.net_stride == 64:
            x = F.relu(self.bn5(self.layer5(x)))

        return x

    def _forward_impl(
            self,
            x: Tensor
    ) -> _PIPNet_Output_Type:

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


class PIPNetMobileNetV2(nn.Module):
    def __init__(
            self,
            mbnet: MobileNetV2,
            num_nb: int = 10,
            num_lms: int = 68,
            input_size: int = 256,
            net_stride: int = 32
    ):
        """
        :param mbnet: specific MobileNetV2 backbone from torchvision.models,
        :param num_nb: the number of Nearest-neighbor landmarks for NRM, default 10
        :param num_lms: the number of input/output landmarks, default 68.
        :param input_size: input size for PIPNet, default 256.
        :param net_stride: net stride for PIPNet, only support 32
        The relationship of net_stride and the output size of feature map is:
            # net_stride output_size
            # 128        2x2
            # 64         4x4
            # 32         8x8
        """
        super(PIPNetMobileNetV2, self).__init__()
        assert net_stride == 32
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        self.features = mbnet.features
        self.sigmoid = nn.Sigmoid()
        # calculate plane
        self.plane = 1280
        # setup det headers
        self._set_det_headers(plane=self.plane)

    def _set_det_headers(
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

    def forward(
            self,
            x: Tensor
    ) -> _PIPNet_Output_Type:

        return self._forward_impl(x)

    def _forward_impl(
            self,
            x: Tensor
    ) -> _PIPNet_Output_Type:

        x = self.features(x)
        x1 = self.cls_layer(x)
        x2 = self.x_layer(x)
        x3 = self.y_layer(x)
        x4 = self.nb_x_layer(x)
        x5 = self.nb_y_layer(x)
        return x1, x2, x3, x4, x5


_PIPNet = Union[PIPNetResNet, PIPNetMobileNetV2]


def _pipnet(
        arch: str,
        backbone: Optional[str],
        pretrained: bool,
        progress: bool,
        num_nb: int,
        num_lms: int,
        net_stride: int,
        input_size: int,
        backbone_pretrained: Optional[bool] = True,
        map_location: Union[str, torch.device] = "cpu",
        **kwargs: Any
) -> _PIPNet:
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
            expansion=expansion
        )
    else:
        mobilenetv2 = models.mobilenet_v2(pretrained=backbone_pretrained, **kwargs)
        model = PIPNetMobileNetV2(
            mobilenetv2,
            num_nb=num_nb,
            num_lms=num_lms,
            net_stride=net_stride,
            input_size=input_size
        )
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress,
                                              map_location=map_location)
        model.load_state_dict(state_dict)
    return model

# TODO: add 21/98 landmarks models
# alias: pipnet backbone num_nb x num_lms x net_stride x input_size
def pipnet_resnet18_10x68x32x256(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNet:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return _pipnet("pipnet_resnet18_10x68x32x256", "resnet18", pretrained, progress,
                   10, 68, 32, 256, **kwargs)


def pipnet_resnet50_10x68x32x256(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNet:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return _pipnet("pipnet_resnet50_10x68x32x256", "resnet50", pretrained, progress,
                   10, 68, 32, 256, **kwargs)


def pipnet_resnet101_10x68x32x256(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNet:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return _pipnet("pipnet_resnet101_10x68x32x256", "resnet101", pretrained, progress,
                   10, 68, 32, 256, **kwargs)


def pipnet_mobilenetv2_10x68x32x256(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> _PIPNet:
    """
    :param pretrained: If True, returns a model pre-trained model
    :param progress: If True, displays a progress bar of the download to stderr
    """
    return _pipnet("pipnet_mobilenetv2_10x68x32x256", "mobilenet_v2", pretrained, progress,
                   10, 68, 32, 256, **kwargs)
