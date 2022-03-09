import os
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, List, Any


class BaseConverter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def convert(self, *args, **kwargs):
        raise NotImplementedError


class WFLWConverter(BaseConverter):
    def __init__(
            self,
            wflw_dir: str = "./data/WFLW",
            save_dir: str = "./data/WFLW/converted"
    ):
        super(WFLWConverter, self).__init__()
        self.wflw_dir = wflw_dir
        self.save_dir = save_dir
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
        pass
