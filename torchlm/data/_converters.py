import os
import cv2
import tqdm
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, List, Union

from ..data import annotools


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
            self.wflw_annotation_dir, "list_98pt_rect_attr_test.txt"
        )

    def convert(self, *args, **kwargs):
        train_annotations, test_annotations = self._fetch_annotations()
        train_anno_file = open(self.save_train_annotation_path, "w")
        test_anno_file = open(self.save_test_annotation_path, "w")

        for anno in tqdm.tqdm(train_annotations):
            crop, landmarks, new_img_name = self._process_wflw(anno=anno)
            if crop is None or landmarks is None:
                continue
            new_img_path = os.path.join(self.save_train_image_dir, new_img_name)
            cv2.imwrite(new_img_path, crop)
            annotation_string = annotools.format_annotation(
                img_path=new_img_path, lms_gt=landmarks
            )
            train_anno_file.write(annotation_string + "\n")
        train_anno_file.close()

        for anno in tqdm.tqdm(test_anno_file):
            crop, landmarks, new_img_name = self._process_wflw(anno=anno)
            if crop is None or landmarks is None:
                continue
            new_img_path = os.path.join(self.save_test_image_dir, new_img_name)
            cv2.imwrite(new_img_path, crop)
            annotation_string = annotools.format_annotation(
                img_path=new_img_path, lms_gt=landmarks
            )
            test_anno_file.write(annotation_string + "\n")
        test_anno_file.close()

    def _process_wflw(self, anno: str) \
            -> Union[Tuple[np.ndarray, np.ndarray, str],
                     Tuple[None, None, None]]:
        anno = anno.strip("\n").strip(" ").split(" ")
        image_name = anno[-1]
        image_path = os.path.join(self.wflw_images_dir, image_name)
        if not os.path.exists(image_path):
            return None, None, None

        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        # 98 gt landmarks
        landmarks = anno[:196]
        landmarks = [float(x) for x in landmarks]
        landmarks = np.array(landmarks).reshape(-1, 2)  # (98,2)
        landmarks[:, 0] = np.minimum(np.maximum(0, landmarks[:, 0]), image_width)
        landmarks[:, 1] = np.minimum(np.maximum(0, landmarks[:, 1]), image_height)
        bbox = anno[196:200]
        bbox = [float(x) for x in bbox]
        xmin, ymin, xmax, ymax = bbox

        width = xmax - xmin
        height = ymax - ymin
        xmin -= width * (self.scale - 1) / 2
        ymin -= height * (self.scale - 1) / 2
        xmax += width * (self.scale - 1) / 2
        ymax += height * (self.scale - 1) / 2
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, image_width - 1)
        ymax = min(ymax, image_height - 1)
        crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
        # adjust according to left-top corner
        landmarks[:, 0] -= int(xmin)
        landmarks[:, 1] -= int(ymin)

        new_img_name = image_name.replace("/", "_")

        return crop, landmarks, new_img_name

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
