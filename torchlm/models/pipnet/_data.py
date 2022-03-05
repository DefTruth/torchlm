import os
import cv2
import torch
import warnings
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, Optional, Any, List

from ..utlis import transforms


class _PIPDataset(Dataset):
    # TODO: 需要修改 根据input_size和归一化逻辑
    _default_transform_without_norm_resize = transforms.LandmarksCompose([
        # use native torchlm transforms
        transforms.LandmarksRandomScale(prob=0.5),
        transforms.LandmarksRandomTranslate(prob=0.5),
        transforms.LandmarksRandomShear(prob=0.5),
        transforms.LandmarksRandomMask(prob=0.5),
        transforms.LandmarksRandomBlur(kernel_range=(5, 25), prob=0.5),
        transforms.LandmarksRandomBrightness(prob=0.),
        transforms.LandmarksRandomRotate(40, prob=0.5, bins=8),
        transforms.LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.5)
    ])
    # warning: assume the default input size is 256x256
    _default_norm_resize_transform = transforms.LandmarksCompose([
        transforms.LandmarksResize((256, 256)),
        transforms.LandmarksNormalize(),
        transforms.LandmarksToTensor()
    ])

    def __init__(
            self,
            annotation_path: str,
            transform_without_norm_resize: Optional[transforms.LandmarksCompose] = None,
            norm_resize_transform: Optional[transforms.LandmarksCompose] = None
    ):
        super(_PIPDataset, self).__init__()
        self.annotation_path = annotation_path
        self.transform_without_norm_resize = transform_without_norm_resize
        self.norm_resize_transform = norm_resize_transform
        if self.transform_without_norm_resize is None:
            self.transform_without_norm_resize = self._default_transform_without_norm_resize
        if self.norm_resize_transform is None:
            self.norm_resize_transform = self._default_norm_resize_transform

    def __getitem__(self, index):
        ...

    def __len__(self):
        ...
