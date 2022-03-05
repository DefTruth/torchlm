import os
import cv2
import torch
import warnings
import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Union, Optional, Any, List

from ..utils import transforms
from ._utils import _normalize


# define a custom callable transform function
def _default_normalize(img: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img = _normalize(img=img)
    return img.astype(np.float32), landmarks.astype(np.float32)


class _PIPTrainDataset(Dataset):

    def __init__(
            self,
            annotation_path: str,
            input_size: int = 256,
            transform: Optional[transforms.LandmarksCompose] = None
    ):
        super(_PIPTrainDataset, self).__init__()
        self.annotation_path = annotation_path
        self.transform = transform
        if self.transform is None:
            # according to input size
            self.transform = transforms.LandmarksCompose([
                # use native torchlm transforms
                transforms.LandmarksRandomMaskMixUp(prob=0.25),
                transforms.LandmarksRandomBackgroundMixUp(prob=0.25),
                transforms.LandmarksRandomScale(prob=0.25),
                transforms.LandmarksRandomTranslate(prob=0.25),
                transforms.LandmarksRandomShear(prob=0.25),
                transforms.LandmarksRandomBlur(kernel_range=(5, 25), prob=0.25),
                transforms.LandmarksRandomBrightness(prob=0.25),
                transforms.LandmarksRandomRotate(30, prob=0.25, bins=8),
                transforms.LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.25),
                transforms.LandmarksResize((input_size, input_size)),
                transforms.bind(_default_normalize, transforms.BindEnum.Callable_Array),
                transforms.LandmarksToTensor()
            ])

    def __getitem__(self, index) -> Any:
        ...

    def __len__(self):
        ...


class _PIPEvalDataset(object):

    def __init__(
            self,
            annotation_path: str
    ):
        super(_PIPEvalDataset, self).__init__()
        self.annotation_path = annotation_path

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray]:  # img, lms_gt
        ...

    def __len__(self):
        ...
