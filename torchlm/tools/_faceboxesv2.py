import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from math import ceil
from pathlib import Path
from torch import Tensor
import torch.nn.functional as F
from itertools import product as product
from typing import Tuple, Union, List

from ..core import FaceDetBase

__all__ = ["FaceBoxesV2"]


class BasicConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Inception(nn.Module):

    def __init__(self):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch1x1_2 = BasicConv2d(128, 32, kernel_size=1, padding=0)
        self.branch3x3_reduce = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_reduce_2 = BasicConv2d(128, 24, kernel_size=1, padding=0)
        self.branch3x3_2 = BasicConv2d(24, 32, kernel_size=3, padding=1)
        self.branch3x3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        branch1x1 = self.branch1x1(x)

        branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch1x1_2 = self.branch1x1_2(branch1x1_pool)

        branch3x3_reduce = self.branch3x3_reduce(x)
        branch3x3 = self.branch3x3(branch3x3_reduce)

        branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
        branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
        branch3x3_3 = self.branch3x3_3(branch3x3_2)

        outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
        return torch.cat(outputs, 1)


class CRelu(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super(CRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = torch.cat([x, -x], 1)
        x = F.relu(x, inplace=True)
        return x


class FaceBoxesV2Impl(nn.Module):

    def __init__(self, phase: str = "test", num_classes: int = 2):
        super(FaceBoxesV2Impl, self).__init__()
        self.phase = phase
        self.num_classes = num_classes

        self.conv1 = BasicConv2d(3, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = BasicConv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv4 = BasicConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv5 = BasicConv2d(64, 128, kernel_size=3, stride=2, padding=1)

        self.inception1 = Inception()
        self.inception2 = Inception()
        self.inception3 = Inception()

        self.conv6_1 = BasicConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv6_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv7_1 = BasicConv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.conv7_2 = BasicConv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.loc, self.conf = self.multibox(self.num_classes)

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

        if self.phase == 'train':
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    if m.bias is not None:
                        nn.init.xavier_normal_(m.weight.data)
                        m.bias.data.fill_(0.02)
                    else:
                        m.weight.data.normal_(0, 0.01)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def multibox(self, num_classes: int) -> Tuple[nn.Sequential, nn.Sequential]:
        loc_layers = []
        conf_layers = []
        loc_layers += [nn.Conv2d(128, 21 * 4, kernel_size=(3, 3), padding=(1, 1))]
        conf_layers += [nn.Conv2d(128, 21 * num_classes, kernel_size=(3, 3), padding=(1, 1))]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=(3, 3), padding=(1, 1))]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=(3, 3), padding=(1, 1))]
        loc_layers += [nn.Conv2d(256, 1 * 4, kernel_size=(3, 3), padding=(1, 1))]
        conf_layers += [nn.Conv2d(256, 1 * num_classes, kernel_size=(3, 3), padding=(1, 1))]
        return nn.Sequential(*loc_layers), nn.Sequential(*conf_layers)

    # noinspection PyTypeChecker
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

        sources = list()
        loc = list()
        conf = list()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        sources.append(x)
        x = self.conv6_1(x)
        x = self.conv6_2(x)
        sources.append(x)
        x = self.conv7_1(x)
        x = self.conv7_2(x)
        sources.append(x)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (loc.view(loc.size()[0], -1, 4),
                      self.softmax(conf.view(-1, self.num_classes)))
        else:
            output = (loc.view(loc.size()[0], -1, 4),
                      conf.view(conf.size()[0], -1, self.num_classes))

        return output


class PriorBox(object):
    def __init__(self, cfg: dict, image_size: Tuple[int, int] = None):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        assert image_size is not None
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step),
                              ceil(self.image_size[1] / step)]
                             for step in self.steps]

    def forward(self) -> Tensor:
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in
                                    [j + 0, j + 0.25, j + 0.5, j + 0.75]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in
                                    [i + 0, i + 0.25, i + 0.5, i + 0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0, j + 0.5]]
                        dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0, i + 0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


# cpu NMS
def _nms(dets: np.ndarray, thresh: float) -> List[int]:
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


# Adapted from https://github.com/Hakuyume/chainer-ssd
def _decode(loc: np.ndarray, priors: np.ndarray, variances: List[float]) -> Tensor:
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class FaceBoxesV2(FaceDetBase):
    def __init__(self, device: Union[str, torch.device] = "cpu"):
        super(FaceBoxesV2).__init__()
        self.checkpoint_path = os.path.join(Path(__file__).parent, "assets/faceboxesv2.pth")
        self.net = FaceBoxesV2Impl(phase='test', num_classes=2)  # initialize detector
        self.device = device if torch.cuda.is_available() else "cpu"
        self.cfg = {
            'min_sizes': [[32, 64, 128], [256], [512]],
            'steps': [32, 64, 128],
            'variance': [0.1, 0.2],
            'clip': False,
            'loc_weight': 2.0,
            'gpu_train': True
        }

        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        self.net.load_state_dict(new_state_dict)
        self.net = self.net.to(self.device)
        self.net.eval()

    @torch.no_grad()
    def detect(
            self,
            image: np.ndarray,  # BGR
            thresh: float = 0.6,
            im_scale: float = None,
            top_k: int = 100
    ) -> np.ndarray:
        # auto resize for large images
        if im_scale is None:
            height, width, _ = image.shape
            if min(height, width) > 600:
                im_scale = 600. / min(height, width)
            else:
                im_scale = 1
        image_scale = cv2.resize(
            image, None, None, fx=im_scale,
            fy=im_scale, interpolation=cv2.INTER_LINEAR
        )

        scale = torch.Tensor(
            [image_scale.shape[1], image_scale.shape[0],
             image_scale.shape[1], image_scale.shape[0]]
        )
        image_scale = torch.from_numpy(image_scale.transpose(2, 0, 1)).to(self.device).int()
        mean_tmp = torch.IntTensor([104, 117, 123]).to(self.device)
        mean_tmp = mean_tmp.unsqueeze(1).unsqueeze(2)
        image_scale -= mean_tmp
        image_scale = image_scale.float().unsqueeze(0)
        scale = scale.to(self.device)

        # face detection
        out = self.net(image_scale)
        priorbox = PriorBox(self.cfg, image_size=(image_scale.size()[2], image_scale.size()[3]))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        loc, conf = out
        prior_data = priors.data
        boxes = _decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k * 3]
        boxes = boxes[order]
        scores = scores[order]

        # nms
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = _nms(dets, 0.3)
        dets = dets[keep, :]
        dets = dets[:top_k, :]  # x1,y1,x2,y2,score

        dets[:, :4] /= im_scale  # adapt bboxes to the original image size
        return dets
