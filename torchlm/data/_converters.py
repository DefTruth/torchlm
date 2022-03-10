import os
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, List, Any, Union


class BaseConverter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def convert(self, *args, **kwargs):
        raise NotImplementedError


class WFLWConverter(BaseConverter):
    def __init__(
            self,
            wflw_dir: Optional[str] = "./data/WFLW",
            save_dir: Optional[str] = "./data/WFLW/converted",
            extend: Optional[float] = 0.2
    ):
        super(WFLWConverter, self).__init__()
        self.wflw_dir = wflw_dir
        self.save_dir = save_dir
        self.scale = 1. + extend
        assert os.path.exists(wflw_dir), "WFLW dataset not found."
        os.makedirs(save_dir, exist_ok=True)
        self.wflw_images_dir = os.path.join(wflw_dir, "WFLW_images")
        self.wflw_annotation_dir = os.path.join(
            wflw_dir, "WFLW_annotations",
            "list_98pt_rect_attr_train_test"
        )
        self.save_train_image_dir = os.path.join(save_dir, "image/train")
        self.save_train_annotation_path = os.path.join(save_dir, "train.txt")
        self.save_test_image_dir = os.path.join(save_dir, "image/test")
        self.save_test_annotation_path = os.path.join(save_dir, "test.txt")
        os.makedirs(self.save_train_image_dir, exist_ok=True)
        os.makedirs(self.save_train_annotation_path, exist_ok=True)
        os.makedirs(self.save_test_image_dir, exist_ok=True)
        os.makedirs(self.save_test_annotation_path, exist_ok=True)
        self.source_train_annotation_path = os.path.join(
            self.wflw_annotation_dir, "list_98pt_rect_attr_train.txt"
        )
        self.source_test_annotation_path = os.path.join(
            self.wflw_annotation_dir, "list_98pt_rect_attr_train.txt"
        )

    def convert(self, *args, **kwargs):
        train_annotations, test_annotations = self._fetch_annotations()
        output_train_annotation = open(self.save_train_annotation_path, "w")
        output_test_annotation = open(self.save_test_annotation_path, "w")

    def process_wflw(self, anno: str):
        image_name = anno[-1]
        image_path = os.path.join(self.wflw_images_dir, image_name)
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        # 98 gt landmarks
        lms = anno[:196]
        lms = [float(x) for x in lms]
        lms_x = lms[0::2]
        lms_y = lms[1::2]
        # bounder check
        lms_x = [x if x >= 0 else 0 for x in lms_x]
        lms_x = [x if x <= image_width else image_width for x in lms_x]
        lms_y = [y if y >= 0 else 0 for y in lms_y]
        lms_y = [y if y <= image_height else image_height for y in lms_y]
        lms = [[x, y] for x, y in zip(lms_x, lms_y)]
        lms = [x for z in lms for x in z]
        bbox = anno[196:200]
        bbox = [float(x) for x in bbox]
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

        width = bbox_xmax - bbox_xmin
        height = bbox_ymax - bbox_ymin
        bbox_xmin -= width * (self.scale - 1) / 2
        bbox_ymin -= height * (self.scale - 1) / 2
        bbox_xmax += width * (self.scale - 1) / 2
        bbox_ymax += height * (self.scale - 1) / 2
        bbox_xmin = max(bbox_xmin, 0)
        bbox_ymin = max(bbox_ymin, 0)
        bbox_xmax = min(bbox_xmax, image_width - 1)
        bbox_ymax = min(bbox_ymax, image_height - 1)
        width = bbox_xmax - bbox_xmin
        height = bbox_ymax - bbox_ymin
        image_crop = image[int(bbox_ymin):int(bbox_ymax), int(bbox_xmin):int(bbox_xmax), :]

        tmp1 = [bbox_xmin, bbox_ymin] * 98
        tmp1 = np.array(tmp1)
        tmp2 = [width, height] * 98
        tmp2 = np.array(tmp2)
        lms = np.array(lms) - tmp1  # adjust according to left-top corner
        lms = lms / tmp2  # normalized
        lms = lms.tolist()
        lms = zip(lms[0::2], lms[1::2])
        return image_crop, list(lms)


    def _fetch_annotations(self) -> Tuple[List[str], List[str]]:
        assert os.path.exists(self.source_train_annotation_path)
        assert os.path.exists(self.source_test_annotation_path)
        train_annotations = []
        test_annotations = []

        with open(self.source_train_annotation_path, "r") as fin:
            train_annotations.extend(fin.readlines())

        with open(self.source_test_annotation_path, "r") as fin:
            test_annotations.extend(fin.readlines())

        return train_annotations, test_annotations


