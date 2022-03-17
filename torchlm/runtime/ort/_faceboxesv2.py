import os
import cv2
import numpy as np
from math import ceil
from pathlib import Path
import onnxruntime as ort
from itertools import product as product
from typing import List, Tuple

from .._runtime import FaceDetBase

__all__ = ["faceboxesv2_ort"]


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

    def forward(self) -> np.ndarray:
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
        # output = torch.Tensor(anchors).view(-1, 4)
        output = np.array(anchors).reshape(-1, 4)
        if self.clip:
            # output.clamp_(max=1, min=0)
            output = np.clip(anchors, 0., 1.)
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
def _decode(loc: np.ndarray, priors: np.ndarray, variances: List[float]) -> np.ndarray:
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

    # boxes = torch.cat((
    #     priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
    #     priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), 1
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


class _FaceBoxesV2ORT(FaceDetBase):

    def __init__(self):
        super(_FaceBoxesV2ORT, self).__init__()
        self.onnx_path = os.path.join(Path(__file__).parent, "assets/faceboxesv2-640x640.onnx")
        # ORT settings
        self.providers = ort.get_available_providers()
        self.device = ort.get_device()
        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]  # e.g 640
        self.input_width = self.input_shape[3]  # e.g 640
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.output_shapes = [x.shape for x in self.session.get_outputs()]
        # config for PriorBox
        self.cfg = {
            'min_sizes': [[32, 64, 128], [256], [512]],
            'steps': [32, 64, 128],
            'variance': [0.1, 0.2],
            'clip': False
        }
        # calculate only once
        self.priorbox = PriorBox(self.cfg, image_size=(self.input_height, self.input_width))
        self.priors = self.priorbox.forward()
        print(f"FaceBoxesV2ORT Running Device: {self.device},"
              f" Available Providers: {self.providers}\n"
              f"Model Loaded From: {self.onnx_path}\n"
              f"Input Name: {self.input_name},"
              f" Input Shape: {self.input_shape}"
              f" Output Names: {self.output_names},"
              f" Output Shapes: {self.output_shapes}")

    def apply_detecting(
            self,
            image: np.ndarray,  # BGR
            thresh: float = 0.3,
            top_k: int = 100
    ) -> np.ndarray:
        # original shape
        height, width, _ = image.shape

        image = cv2.resize(image, (self.input_width, self.input_height)).astype(np.uint8)
        image[:, :, 0] -= 104
        image[:, :, 1] -= 117
        image[:, :, 2] -= 123
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0).astype(np.float32)  # (1,3,H=640,W=640)

        # (1,?,4) (1,?,2)
        loc, conf = self.session.run(
            output_names=self.output_names, input_feed={self.input_name: image}
        )
        loc, conf = loc[0], conf[0]  # (?,4) (?,2)
        boxes = _decode(loc, self.priors, self.cfg["variance"])  # normalized coords 0~1
        scores = conf[:, 1]  # (?,)

        # ignore low scores
        inds = np.where(scores > thresh)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k * 3]
        boxes = boxes[order]
        scores = scores[order]
        # rescale to input size before nms
        boxes[:, 0] *= float(self.input_width)
        boxes[:, 1] *= float(self.input_height)
        boxes[:, 2] *= float(self.input_width)
        boxes[:, 3] *= float(self.input_height)

        # nms
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = _nms(dets, 0.3)
        dets = dets[keep, :]
        dets = dets[:top_k, :]  # x1,y1,x2,y2,score

        # rescale to original shape
        dets[:, 0] *= (float(width) / float(self.input_width))
        dets[:, 1] *= (float(height) / float(self.input_height))
        dets[:, 2] *= (float(width) / float(self.input_width))
        dets[:, 3] *= (float(height) / float(self.input_height))

        return dets


# Export Alias
faceboxesv2_ort = _FaceBoxesV2ORT
