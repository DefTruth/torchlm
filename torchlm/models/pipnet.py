"""
A PyTorch Re-implementation of PIPNet with all-in-one style, include
model definition, loss computation, training, inference and exporting
processes in a single class.
Reference: https://github.com/jhb86253817/PIPNet/blob/master/lib/networks.py
"""
import os
import cv2
import torch
import warnings
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
from torchvision import models
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet, MobileNetV2
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, Optional, Any, List

from .base import BaseModel
from ..cfgs.pipnet import DEFAULT_MEANFACE_STRINGS

_PIPNet_Output_Type = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

__all__ = ["pipnet_resnet18_10x68x32x256", "pipnet_resnet50_10x68x32x256",
           "pipnet_resnet101_10x68x32x256", "pipnet_mobilenetv2_10x68x32x256",
           "pipnet"]

# TODO: update model_urls
model_urls = {
    'pipnet_resnet18_10x68x32x256': 'pipnet_resnet18_10x68x32x256.pth',
    'pipnet_resnet50_10x68x32x256': 'pipnet_resnet50_10x68x32x256.pth',
    'pipnet_resnet101_10x68x32x256': 'pipnet_resnet101_10x68x32x256.pth',
    'pipnet_mobilenetv2_10x68x32x256': 'pipnet_mobilenetv2_10x68x32x256.pth',
}


def _get_meanface(
        meanface_string: str,
        num_nb: int = 10
) -> Tuple[List[int], List[int], List[int], int, int]:
    """
    :param meanface_string: a long string contains normalized or un-normalized
     meanface coords, the format is "x0,y0,x1,y1,x2,y2,...,xn-1,yn-1".
    :param num_nb: the number of Nearest-neighbor landmarks for NRM, default 10
    :return: meanface_indices, reverse_index1, reverse_index2, max_len
    """
    meanface = meanface_string.strip("\n").strip(" ").split(" ")
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    meanface_lms = meanface.shape[0]
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i, :]
        dists = np.sum(np.power(pt - meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1 + num_nb])

    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[], []]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            # meanface_indices[i][0,1,2,...,9] -> [[i,i,...,i],[0,1,2,...,9]]
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len

    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0] * 10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1] * 10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    # [...,max_len,...,max_len*2,...]
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len, meanface_lms


def _normalize(
        img: np.ndarray
) -> Tensor:
    """
    :param img: source image, RGB with HWC and range [0,255]
    :return: normalized image CHW Tensor for PIPNet
    """
    img = img.astype(np.float32)
    img /= 255.
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225
    img = img.transpose((2, 0, 1))  # HWC->CHW
    return torch.from_numpy(img)


class _PIPNet(BaseModel):

    def __init__(
            self,
            num_nb: int = 10,
            num_lms: int = 68,
            input_size: int = 256,
            net_stride: int = 32,
            meanface_type: Optional[str] = None
    ):
        super(_PIPNet, self).__init__()
        assert net_stride in (32, 64, 128)
        self.num_nb = num_nb
        self.num_lms = num_lms
        self.input_size = input_size
        self.net_stride = net_stride
        # setup default meanface
        self.meanface_status = False
        self.meanface_type = meanface_type
        self.meanface_indices: List[int] = []
        self.reverse_index1: List[int] = []
        self.reverse_index2: List[int] = []
        self.max_len: int = -1
        self._set_default_meanface()

    def loss(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def detect(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def training(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def export(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def set_custom_meanface(
            self,
            custom_meanface_file_or_string: str
    ) -> bool:
        """
        :param custom_meanface_file_or_string: a long string or a file contains normalized
        or un-normalized meanface coords, the format is "x0,y0,x1,y1,x2,y2,...,xn-1,yn-1".
        :return: status, True if successful.
        """
        try:
            custom_meanface_type = "custom"
            if os.path.isfile(custom_meanface_file_or_string):
                with open(custom_meanface_file_or_string) as f:
                    custom_meanface_string = f.readlines()[0]
            else:
                custom_meanface_string = custom_meanface_file_or_string

            custom_meanface_indices, custom_reverse_index1, \
            custom_reverse_index2, custom_max_len, custom_meanface_lms = _get_meanface(
                meanface_string=custom_meanface_string, num_nb=self.num_nb)

            # check landmarks number
            if custom_meanface_lms != self.num_lms:
                warnings.warn(
                    f"custom_meanface_lms != self.num_lms, "
                    f"{custom_meanface_lms} != {self.num_lms}"
                    f"So, we will skip this setup for PIPNet meanface."
                    f"Please check and setup meanface carefully before"
                    f"running PIPNet ..."
                )
                self.meanface_status = False
            else:
                # replace if successful
                self.meanface_type = custom_meanface_type
                self.meanface_indices = custom_meanface_indices
                self.reverse_index1 = custom_reverse_index1
                self.reverse_index2 = custom_reverse_index2
                self.max_len = custom_max_len
                self.meanface_status = True
        except:
            self.meanface_status = False

        return self.meanface_status

    def _set_default_meanface(self):
        if self.meanface_type is not None:
            if self.meanface_type.upper() not in DEFAULT_MEANFACE_STRINGS:
                warnings.warn(
                    f"Can not found default dataset: {self.meanface_type.upper()}!"
                    f"So, we will skip this setup for PIPNet meanface."
                    f"Please check and setup meanface carefully before"
                    f"running PIPNet ..."
                )
                self.meanface_status = False
            else:
                meanface_string = DEFAULT_MEANFACE_STRINGS[self.meanface_type.upper()]
                meanface_indices, reverse_index1, reverse_index2, max_len, meanface_lms = \
                    _get_meanface(meanface_string=meanface_string, num_nb=self.num_nb)
                # check landmarks number
                if meanface_lms != self.num_lms:
                    warnings.warn(
                        f"meanface_lms != self.num_lms, {meanface_lms} != {self.num_lms}"
                        f"So, we will skip this setup for PIPNet meanface."
                        f"Please check and setup meanface carefully before"
                        f"running PIPNet ..."
                    )
                    self.meanface_status = False
                else:
                    self.meanface_indices = meanface_indices
                    self.reverse_index1 = reverse_index1
                    self.reverse_index2 = reverse_index2
                    self.max_len = max_len

                    self.meanface_status = True



@torch.no_grad()
def _detect_impl(
        net: _PIPNet,
        img: np.ndarray
) -> np.ndarray:
    """
    :param img: source face image without background, RGB with HWC and range [0,255]
    :return: detected landmarks coordinates, shape [num, 2]
    """
    if not net.meanface_status:
        raise RuntimeError(
            f"Can not found any meanface landmarks settings !"
            f"Please check and setup meanface carefully before"
            f"running PIPNet ..."
        )

    net.eval()

    height, width, _ = img.shape
    img: np.ndarray = cv2.resize(img, (net.input_size, net.input_size))  # 256, 256
    img: Tensor = _normalize(img=img).unsqueeze(0)  # (1,3,256,256)
    outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net.forward(img)
    # (1,68,8,8)
    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
    assert tmp_batch == 1

    outputs_cls = outputs_cls.view(tmp_batch * tmp_channel, -1)  # (68.64)
    max_ids = torch.argmax(outputs_cls, 1)  # (68,)
    max_ids = max_ids.view(-1, 1)  # (68,1)
    max_ids_nb = max_ids.repeat(1, net.num_nb).view(-1, 1)  # (68,10) -> (68*10,1)

    outputs_x = outputs_x.view(tmp_batch * tmp_channel, -1)  # (68,64)
    outputs_x_select = torch.gather(outputs_x, 1, max_ids)  # (68,1)
    outputs_x_select = outputs_x_select.squeeze(1)  # (68,)
    outputs_y = outputs_y.view(tmp_batch * tmp_channel, -1)
    outputs_y_select = torch.gather(outputs_y, 1, max_ids)
    outputs_y_select = outputs_y_select.squeeze(1)  # (68,)

    outputs_nb_x = outputs_nb_x.view(tmp_batch * net.num_nb * tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, max_ids_nb)  # (68*10,1)
    outputs_nb_x_select = outputs_nb_x_select.squeeze(1).view(-1, net.num_nb)  # (68,10)
    outputs_nb_y = outputs_nb_y.view(tmp_batch * net.num_nb * tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, max_ids_nb)
    outputs_nb_y_select = outputs_nb_y_select.squeeze(1).view(-1, net.num_nb)  # (68,10)

    # tmp_width=tmp_height=8 max_ids->[0,63] calculate grid center (cx,cy) in 8x8 map
    lms_pred_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_x_select.view(-1, 1)  # x=cx+offset_x
    lms_pred_y = (max_ids // tmp_width).view(-1, 1).float() + outputs_y_select.view(-1, 1)  # y=cy+offset_y
    lms_pred_x /= 1.0 * net.input_size / net.net_stride  # normalize coord (x*32)/256
    lms_pred_y /= 1.0 * net.input_size / net.net_stride  # normalize coord (y*32)/256

    lms_pred_nb_x = (max_ids % tmp_width).view(-1, 1).float() + outputs_nb_x_select  # (68,10)
    lms_pred_nb_y = (max_ids // tmp_width).view(-1, 1).float() + outputs_nb_y_select  # (68,10)
    lms_pred_nb_x = lms_pred_nb_x.view(-1, net.num_nb)  # (68,10)
    lms_pred_nb_y = lms_pred_nb_y.view(-1, net.num_nb)  # (68,10)
    lms_pred_nb_x /= 1.0 * net.input_size / net.net_stride  # normalize coord (nx*32)/256
    lms_pred_nb_y /= 1.0 * net.input_size / net.net_stride  # normalize coord (ny*32)/256

    # merge predictions
    tmp_nb_x = lms_pred_nb_x[net.reverse_index1, net.reverse_index2].view(net.num_lms, net.max_len)
    tmp_nb_y = lms_pred_nb_y[net.reverse_index1, net.reverse_index2].view(net.num_lms, net.max_len)
    tmp_x = torch.mean(torch.cat((lms_pred_x, tmp_nb_x), dim=1), dim=1).view(-1, 1)
    tmp_y = torch.mean(torch.cat((lms_pred_y, tmp_nb_y), dim=1), dim=1).view(-1, 1)
    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1)  # (68,2)
    lms_pred_merge = lms_pred_merge.cpu().numpy()  # (68,2)

    lms_pred_merge[:, 0] *= float(width)
    lms_pred_merge[:, 1] *= float(height)

    return lms_pred_merge


_PIPNet_Loss_Output_Type = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]


def _loss_impl(
        outputs_cls: Tensor,
        outputs_x: Tensor,
        outputs_y: Tensor,
        outputs_nb_x: Tensor,
        outputs_nb_y: Tensor,
        labels_cls: Tensor,
        labels_x: Tensor,
        labels_y: Tensor,
        labels_nb_x: Tensor,
        labels_nb_y: Tensor,
        criterion_cls: nn.Module,
        criterion_reg: nn.Module,
        num_nb: int = 10
) -> _PIPNet_Loss_Output_Type:
    """
    :param outputs_cls: output heatmap Tensor e.g (b,68,8,8)
    :param outputs_x: output x offsets Tensor e.g (b,68,8,8)
    :param outputs_y: output y offsets Tensor e.g (b,68,8,8)
    :param outputs_nb_x: output neighbor's x offsets Tensor e.g (b,68*10,8,8)
    :param outputs_nb_y: output neighbor's y offsets Tensor e.g (b,68*10,8,8)
    :param labels_cls: output heatmap Tensor e.g (b,68,8,8)
    :param labels_x: output x offsets Tensor e.g (b,68,8,8)
    :param labels_y: output y offsets Tensor e.g (b,68,8,8)
    :param labels_nb_x: output neighbor's x offsets Tensor e.g (b,68*10,8,8)
    :param labels_nb_y: output neighbor's y offsets Tensor e.g (b,68*10,8,8)
    :param criterion_cls: loss criterion for heatmap classification, e.g MSELoss
    :param criterion_reg: loss criterion for offsets regression, e.g L1Loss
    :param num_nb: the number of Nearest-neighbor landmarks for NRM
    :return: losses Tensor values without weighted.
    """

    tmp_batch, tmp_channel, tmp_height, tmp_width = outputs_cls.size()
    labels_cls = labels_cls.view(tmp_batch * tmp_channel, -1)
    labels_max_ids = torch.argmax(labels_cls, 1)
    labels_max_ids = labels_max_ids.view(-1, 1)
    labels_max_ids_nb = labels_max_ids.repeat(1, num_nb).view(-1, 1)

    outputs_x = outputs_x.view(tmp_batch * tmp_channel, -1)
    outputs_x_select = torch.gather(outputs_x, 1, labels_max_ids)
    outputs_y = outputs_y.view(tmp_batch * tmp_channel, -1)
    outputs_y_select = torch.gather(outputs_y, 1, labels_max_ids)
    outputs_nb_x = outputs_nb_x.view(tmp_batch * num_nb * tmp_channel, -1)
    outputs_nb_x_select = torch.gather(outputs_nb_x, 1, labels_max_ids_nb)
    outputs_nb_y = outputs_nb_y.view(tmp_batch * num_nb * tmp_channel, -1)
    outputs_nb_y_select = torch.gather(outputs_nb_y, 1, labels_max_ids_nb)

    labels_x = labels_x.view(tmp_batch * tmp_channel, -1)
    labels_x_select = torch.gather(labels_x, 1, labels_max_ids)
    labels_y = labels_y.view(tmp_batch * tmp_channel, -1)
    labels_y_select = torch.gather(labels_y, 1, labels_max_ids)
    labels_nb_x = labels_nb_x.view(tmp_batch * num_nb * tmp_channel, -1)
    labels_nb_x_select = torch.gather(labels_nb_x, 1, labels_max_ids_nb)
    labels_nb_y = labels_nb_y.view(tmp_batch * num_nb * tmp_channel, -1)
    labels_nb_y_select = torch.gather(labels_nb_y, 1, labels_max_ids_nb)

    labels_cls = labels_cls.view(tmp_batch, tmp_channel, tmp_height, tmp_width)
    loss_cls = criterion_cls(outputs_cls, labels_cls)
    loss_x = criterion_reg(outputs_x_select, labels_x_select)
    loss_y = criterion_reg(outputs_y_select, labels_y_select)
    loss_nb_x = criterion_reg(outputs_nb_x_select, labels_nb_x_select)
    loss_nb_y = criterion_reg(outputs_nb_y_select, labels_nb_y_select)

    return loss_cls, loss_x, loss_y, loss_nb_x, loss_nb_y


def _train_impl(
        net: _PIPNet,
        train_loader: DataLoader,
        criterion_cls: nn.Module = nn.MSELoss(),
        criterion_reg: nn.Module = nn.L1Loss(),
        learning_rate: float = 0.0001,
        cls_loss_weight: float = 10.,
        reg_loss_weight: float = 1.,
        num_nb: int = 10,
        num_epochs: int = 60,
        save_dir: Optional[str] = "./save",
        save_interval: Optional[int] = 10,
        save_prefix: Optional[str] = "",
        decay_steps: Optional[List[int]] = (30, 50),
        decay_gamma: Optional[float] = 0.1,
        device: Optional[Union[str, torch.device]] = "cuda"
):
    import logging

    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=decay_steps,
        gamma=decay_gamma
    )

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('-' * 10)

        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels_cls, labels_x, labels_y, labels_nb_x, labels_nb_y = data
            inputs = inputs.to(device)
            labels_cls = labels_cls.to(device)
            labels_x = labels_x.to(device)
            labels_y = labels_y.to(device)
            labels_nb_x = labels_nb_x.to(device)
            labels_nb_y = labels_nb_y.to(device)
            outputs_cls, outputs_x, outputs_y, outputs_nb_x, outputs_nb_y = net(inputs)
            loss_cls, loss_x, loss_y, loss_nb_x, loss_nb_y = net.loss(
                outputs_cls=outputs_cls,
                outputs_x=outputs_x,
                outputs_y=outputs_y,
                outputs_nb_x=outputs_nb_x,
                outputs_nb_y=outputs_nb_y,
                labels_cls=labels_cls,
                labels_x=labels_x,
                labels_y=labels_y,
                labels_nb_x=labels_nb_x,
                labels_nb_y=labels_nb_y,
                criterion_cls=criterion_cls,
                criterion_reg=criterion_reg,
                num_nb=num_nb
            )
            loss = cls_loss_weight * loss_cls + reg_loss_weight * loss_x \
                   + reg_loss_weight * loss_y + reg_loss_weight * loss_nb_x \
                   + reg_loss_weight * loss_nb_y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(
                    '[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <cls loss: {:.6f}> '
                    '<x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'
                        .format(epoch, num_epochs - 1, i, len(train_loader) - 1, loss.item(),
                                cls_loss_weight * loss_cls.item(), reg_loss_weight * loss_x.item(),
                                reg_loss_weight * loss_y.item(), reg_loss_weight * loss_nb_x.item(),
                                reg_loss_weight * loss_nb_y.item())
                )
                logging.info(
                    '[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> <cls loss: {:.6f}> '
                    '<x loss: {:.6f}> <y loss: {:.6f}> <nbx loss: {:.6f}> <nby loss: {:.6f}>'
                        .format(epoch, num_epochs - 1, i, len(train_loader) - 1, loss.item(),
                                cls_loss_weight * loss_cls.item(), reg_loss_weight * loss_x.item(),
                                reg_loss_weight * loss_y.item(), reg_loss_weight * loss_nb_x.item(),
                                reg_loss_weight * loss_nb_y.item())
                )
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        if epoch % (save_interval - 1) == 0 and epoch > 0:
            epoch_loss = np.round(epoch_loss, 4)
            filename = os.path.join(save_dir, f'{save_prefix}-epoch{epoch}-loss{epoch_loss}.pth')
            torch.save(net.state_dict(), filename)
            print(filename, ' saved')
        # adjust lr
        scheduler.step()

    return net


class PIPNetResNet(_PIPNet):
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

    def loss(self, *args, **kwargs) -> _PIPNet_Loss_Output_Type:
        return _loss_impl(*args, **kwargs)

    def detect(self, img: np.ndarray) -> np.ndarray:
        return _detect_impl(net=self, img=img)

    def training(self, *args, **kwargs) -> _PIPNet:
        pass

    def export(self, *args, **kwargs) -> Any:
        pass


class PIPNetMobileNetV2(_PIPNet):
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

    def loss(self, *args, **kwargs) -> _PIPNet_Loss_Output_Type:
        return _loss_impl(*args, **kwargs)

    def detect(self, img: np.ndarray) -> np.ndarray:
        return _detect_impl(net=self, img=img)

    def training(self, *args, **kwargs) -> _PIPNet:
        pass

    def export(self, *args, **kwargs) -> Any:
        pass


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


# general usage
def pipnet(*args, **kwargs) -> _PIPNet:
    return _pipnet(*args, **kwargs)


# TODO: add 19/29/68/98 landmarks models
# alias: pipnet backbone num_nb x num_lms x net_stride x input_size
# 68 landmarks
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
