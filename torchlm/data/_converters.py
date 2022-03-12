import os
import cv2
import tqdm
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, List, Union

from ..data import annotools
from ..utils import draw_landmarks

__all__ = ["LandmarksWFLWConverter", "LandmarksALFWConverter",
           "Landmarks300WConverter", "LandmarksCOFWConverter"]


class BaseConverter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def convert(self, *args, **kwargs):
        """Convert the annotations to a standard format.
        "img0_path x0 y0 x1 y1 ... xn-1,yn-1"
        "img1_path x0 y0 x1 y1 ... xn-1,yn-1"
        "img2_path x0 y0 x1 y1 ... xn-1,yn-1"
        "img3_path x0 y0 x1 y1 ... xn-1,yn-1"
        ...
        """
        raise NotImplementedError

    @abstractmethod
    def show(self, *args, **kwargs):
        raise NotImplementedError


class LandmarksWFLWConverter(BaseConverter):
    def __init__(
            self,
            wflw_dir: Optional[str] = "./data/WFLW",
            save_dir: Optional[str] = "./data/WFLW/converted",
            extend: Optional[float] = 0.2,
            rebuild: Optional[bool] = True,
            force_normalize: Optional[bool] = False,
            force_absolute_path: Optional[bool] = True
    ):
        super(LandmarksWFLWConverter, self).__init__()
        self.wflw_dir = wflw_dir
        self.save_dir = save_dir
        self.scale = 1. + extend
        self.rebuild = rebuild
        self.force_normalize = force_normalize
        self.force_absolute_path = force_absolute_path
        assert os.path.exists(wflw_dir), "WFLW dataset not found."
        os.makedirs(save_dir, exist_ok=True)

        if self.force_absolute_path:
            wflw_dir = os.path.abspath(wflw_dir)
            save_dir = os.path.abspath(save_dir)

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
        os.makedirs(self.save_test_image_dir, exist_ok=True)
        self.source_train_annotation_path = os.path.join(
            self.wflw_annotation_dir, "list_98pt_rect_attr_train.txt"
        )
        self.source_test_annotation_path = os.path.join(
            self.wflw_annotation_dir, "list_98pt_rect_attr_test.txt"
        )

    def convert(self):
        train_annotations, test_annotations = self._fetch_annotations()
        print(f"Train annotations count: {len(train_annotations)}, \n"
              f"Test  annotations count: {len(test_annotations)}")

        train_anno_file = open(self.save_train_annotation_path, "w")
        test_anno_file = open(self.save_test_annotation_path, "w")

        for annotation in tqdm.tqdm(
                train_annotations,
                desc="converting train_annotations"
        ):
            crop, landmarks, new_img_name = self._process_wflw(annotation=annotation)
            if crop is None or landmarks is None:
                continue
            new_img_path = os.path.join(self.save_train_image_dir, new_img_name)
            if not self.rebuild:
                if not os.path.exists(new_img_path):
                    cv2.imwrite(new_img_path, crop)
            else:
                cv2.imwrite(new_img_path, crop)

            annotation_string = annotools.format_annotation(
                img_path=new_img_path, lms_gt=landmarks
            )
            train_anno_file.write(annotation_string + "\n")
        train_anno_file.close()

        for annotation in tqdm.tqdm(
                test_annotations,
                desc="converting test_annotations"
        ):
            crop, landmarks, new_img_name = self._process_wflw(annotation=annotation)
            if crop is None or landmarks is None:
                continue
            new_img_path = os.path.join(self.save_test_image_dir, new_img_name)
            if not self.rebuild:
                if not os.path.exists(new_img_path):
                    cv2.imwrite(new_img_path, crop)
            else:
                cv2.imwrite(new_img_path, crop)

            annotation_string = annotools.format_annotation(
                img_path=new_img_path, lms_gt=landmarks
            )
            test_anno_file.write(annotation_string + "\n")
        test_anno_file.close()

    def show(self, count: int = 10, show_dir: Optional[str] = None):
        assert os.path.exists(self.save_train_annotation_path), \
            f"{self.save_train_annotation_path} not found!"

        with open(self.save_train_annotation_path, "r") as fin:
            annotations = fin.readlines()[:count]

        assert len(annotations) >= 1, "no annotations!"

        if show_dir is None:
            show_dir = os.path.join(self.save_dir, "show")
            os.makedirs(show_dir, exist_ok=True)
        else:
            assert os.path.exists(show_dir)

        for annotation_string in annotations:
            img_path, lms_gt = annotools.decode_annotation(
                annotation_string=annotation_string)
            img_name = os.path.basename(img_path)
            out_path = os.path.join(show_dir, img_name)

            in_img: np.ndarray = cv2.imread(img_path)
            if in_img is not None:
                if self.force_normalize:
                    h, w, _ = in_img.shape
                    lms_gt[:, 0] *= w
                    lms_gt[:, 1] *= h
                out_img = draw_landmarks(in_img, landmarks=lms_gt)

                cv2.imwrite(out_path, out_img)

                print(f"saved show img to: {out_path} !")

    def _process_wflw(
            self,
            annotation: str
    ) -> Union[Tuple[np.ndarray, np.ndarray, str], Tuple[None, None, None]]:

        annotation = annotation.strip("\n").strip(" ").split(" ")
        image_name = annotation[-1]
        image_path = os.path.join(self.wflw_images_dir, image_name)
        if not os.path.exists(image_path):
            return None, None, None

        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        # 98 gt landmarks
        landmarks = annotation[:196]
        landmarks = [float(x) for x in landmarks]
        landmarks = np.array(landmarks).reshape(-1, 2)  # (98,2)
        landmarks[:, 0] = np.minimum(np.maximum(0, landmarks[:, 0]), image_width)
        landmarks[:, 1] = np.minimum(np.maximum(0, landmarks[:, 1]), image_height)
        bbox = annotation[196:200]
        bbox = [float(x) for x in bbox]
        xmin, ymin, xmax, ymax = bbox

        width = xmax - xmin
        height = ymax - ymin
        xmin -= width * (self.scale - 1.) / 2.
        ymin -= height * (self.scale - 1.) / 2.
        xmax += width * (self.scale - 1.) / 2.
        ymax += height * (self.scale - 1.) / 2.
        xmin = int(max(xmin, 0))
        ymin = int(max(ymin, 0))
        xmax = int(min(xmax, image_width - 1))
        ymax = int(min(ymax, image_height - 1))
        crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
        # adjust according to left-top corner
        landmarks[:, 0] -= float(xmin)
        landmarks[:, 1] -= float(ymin)

        if self.force_normalize:
            landmarks[:, 0] /= float(width)
            landmarks[:, 1] /= float(height)

        new_img_name = image_name.replace("/", "_")

        return crop, landmarks, new_img_name

    def _fetch_annotations(self) -> Tuple[List[str], List[str]]:
        print("Fetching annotations ...")
        assert os.path.exists(self.source_train_annotation_path)
        assert os.path.exists(self.source_test_annotation_path)
        train_annotations = []
        test_annotations = []

        with open(self.source_train_annotation_path, "r") as fin:
            train_annotations.extend(fin.readlines())

        with open(self.source_test_annotation_path, "r") as fin:
            test_annotations.extend(fin.readlines())

        return train_annotations, test_annotations


class Landmarks300WConverter:
    ...


class LandmarksCOFWConverter:
    ...


class LandmarksALFWConverter:
    ...
