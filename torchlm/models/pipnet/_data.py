import cv2
import torch
import numpy as np
from torch import Tensor
from math import floor
from torch.utils.data import Dataset
from typing import Tuple, Optional, List

from .._utils import transforms, annotools

_PIPTrainDataset_Output_Type = Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]


class _PIPTrainDataset(Dataset):

    def __init__(
            self,
            annotation_path: str,
            input_size: int = 256,
            num_lms: int = 68,
            net_stride: int = 32,
            meanface_indices: List[List[int]] = None,
            transform: Optional[transforms.LandmarksCompose] = None,
            coordinates_already_normalized: Optional[bool] = False
    ):
        super(_PIPTrainDataset, self).__init__()
        self.annotation_path = annotation_path
        self.input_size = input_size
        self.num_lms = num_lms
        self.net_stride = net_stride
        self.meanface_indices = meanface_indices
        self.transform = transform
        self.coordinates_already_normalized = coordinates_already_normalized
        self.to_tensor = transforms.LandmarksToTensor()
        if self.transform is None:
            # build default transform
            self.transform = transforms.build_default_transform(
                input_size=(input_size, input_size),
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                force_norm_before_mean_std=True,  # img/=255. first
                rotate=30,
                keep_aspect=False,
                to_tensor=False
            )

        if not bool(self.meanface_indices):
            raise RuntimeError(
                f"Can not found any meanface landmarks settings !"
                f"Please check and setup meanface carefully before"
                f"running _PIPTrainDataset ..."
            )

        self.num_nb = len(meanface_indices[0])
        self.annotations_info = \
            annotools.fetch_annotations(annotation_path=self.annotation_path)

    def __getitem__(self, index: int) -> _PIPTrainDataset_Output_Type:
        annotation_string = self.annotations_info[index]
        img_path, label = annotools.decode_annotation(annotation_string=annotation_string)
        img = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
        if not self.coordinates_already_normalized:
            h, w, _ = img.shape
            label[:, 0] /= w
            label[:, 1] /= h

        # online data augmentations
        img, label = self.transform(img, label)
        # converted to Tensor manually
        if isinstance(label, Tensor):
            label = label.cpu().numpy()

        grid_size = int(self.input_size / self.net_stride)
        label_cls = np.zeros((self.num_lms, grid_size, grid_size))
        label_x = np.zeros((self.num_lms, grid_size, grid_size))
        label_y = np.zeros((self.num_lms, grid_size, grid_size))
        label_nb_x = np.zeros((self.num_nb * self.num_lms, grid_size, grid_size))
        label_nb_y = np.zeros((self.num_nb * self.num_lms, grid_size, grid_size))
        label_cls, label_x, label_y, label_nb_x, label_nb_y = _generate_labels(
            normalized_label=label,
            meanface_indices=self.meanface_indices,
            label_cls=label_cls,
            label_x=label_x,
            label_y=label_y,
            label_nb_x=label_nb_x,
            label_nb_y=label_nb_y
        )

        if not isinstance(img, Tensor):
            img, label = self.to_tensor(img, label)  # HWC -> CHW

        label_cls = torch.from_numpy(label_cls)
        label_x = torch.from_numpy(label_x)
        label_y = torch.from_numpy(label_y)
        label_nb_x = torch.from_numpy(label_nb_x)
        label_nb_y = torch.from_numpy(label_nb_y)

        return img, label_cls, label_x, label_y, label_nb_x, label_nb_y

    def __len__(self) -> int:
        return len(self.annotations_info)


class _PIPEvalDataset(object):

    def __init__(
            self,
            annotation_path: str,
            coordinates_already_normalized: Optional[bool] = False
    ):
        super(_PIPEvalDataset, self).__init__()
        self.annotation_path = annotation_path
        self.coordinates_already_normalized = coordinates_already_normalized
        self.annotations_info = \
            annotools.fetch_annotations(annotation_path=self.annotation_path)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:  # img, lms_gt
        annotation_string = self.annotations_info[index]
        img_path, label = annotools.decode_annotation(annotation_string=annotation_string)
        img = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
        if self.coordinates_already_normalized:
            h, w, _ = img.shape
            label[:, 0] *= w
            label[:, 1] *= h
        return img, label

    def __len__(self) -> int:
        return len(self.annotations_info)


# Reference: https://github.com/jhb86253817/PIPNet/blob/master/lib/data_utils.py#L86
def _generate_labels(
        normalized_label: np.ndarray,
        meanface_indices: List[List[int]],
        label_cls: np.ndarray,
        label_x: np.ndarray,
        label_y: np.ndarray,
        label_nb_x: np.ndarray,
        label_nb_y: np.ndarray
):
    """
    :param normalized_label: np.ndarray normalized gt landmarks with shape (N,2)
    :param meanface_indices: nearest landmarks indexes for each gt landmarks, List of lists.
    :param label_cls: np.ndarray e.g (N, 8, 8)
    :param label_x: np.ndarray e.g (N, 8, 8)
    :param label_y: np.ndarray e.g (N, 8, 8)
    :param label_nb_x: np.ndarray e.g (N*10, 8, 8)
    :param label_nb_y: np.ndarray e.g (N*10, 8, 8)
    :return:
    """
    num_nb = len(meanface_indices[0])
    cls_channel, cls_height, cls_width = label_cls.shape  # (n,8,8)
    normalized_label = normalized_label.reshape(-1, 2)
    assert cls_channel == normalized_label.shape[0]

    for i in range(cls_channel):
        # find left-top corner of a anchor in 8x8 grid
        mu_x = int(floor(normalized_label[i][0] * cls_width))
        mu_y = int(floor(normalized_label[i][1] * cls_height))
        mu_x = max(0, mu_x)
        mu_y = max(0, mu_y)
        mu_x = min(mu_x, cls_width - 1)
        mu_y = min(mu_y, cls_height - 1)
        label_cls[i, mu_y, mu_x] = 1
        shift_x = normalized_label[i][0] * cls_width - mu_x
        shift_y = normalized_label[i][1] * cls_height - mu_y
        label_x[i, mu_y, mu_x] = shift_x
        label_y[i, mu_y, mu_x] = shift_y

        for j in range(num_nb):
            nb_x = normalized_label[meanface_indices[i][j]][0] * cls_width - mu_x
            nb_y = normalized_label[meanface_indices[i][j]][1] * cls_height - mu_y
            label_nb_x[num_nb * i + j, mu_y, mu_x] = nb_x
            label_nb_y[num_nb * i + j, mu_y, mu_x] = nb_y

    return label_cls, label_x, label_y, label_nb_x, label_nb_y
