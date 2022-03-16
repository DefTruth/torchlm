import os
import cv2
import tqdm
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, List, Union

from ..data import annotools
from ..utils import draw_landmarks, draw_bboxes
from ..transforms import LandmarksResize

__all__ = ["LandmarksWFLWConverter", "LandmarksAFLWConverter",
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
            data_dir: Optional[str] = "./data/WFLW",
            save_dir: Optional[str] = "./data/WFLW/converted",
            extend: Optional[float] = 0.2,
            target_size: Optional[int] = None,
            keep_aspect: Optional[bool] = False,
            rebuild: Optional[bool] = True,
            force_normalize: Optional[bool] = False,
            force_absolute_path: Optional[bool] = True
    ):
        super(LandmarksWFLWConverter, self).__init__()
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.scale = 1. + extend
        self.target_size = target_size
        self.rebuild = rebuild
        self.force_normalize = force_normalize
        self.force_absolute_path = force_absolute_path
        assert os.path.exists(self.data_dir), "WFLW dataset not found."
        os.makedirs(save_dir, exist_ok=True)

        self.resize_op = None
        if target_size is not None:
            self.resize_op = LandmarksResize((target_size, target_size), keep_aspect=keep_aspect)

        if self.force_absolute_path:
            self.data_dir = os.path.abspath(self.data_dir)
            self.save_dir = os.path.abspath(self.save_dir)

        self.wflw_images_dir = os.path.join(self.data_dir, "WFLW_images")
        self.wflw_annotation_dir = os.path.join(
            self.data_dir, "WFLW_annotations",
            "list_98pt_rect_attr_train_test"
        )
        self.save_train_image_dir = os.path.join(self.save_dir, "image/train")
        self.save_train_annotation_path = os.path.join(self.save_dir, "train.txt")
        self.save_test_image_dir = os.path.join(self.save_dir, "image/test")
        self.save_test_annotation_path = os.path.join(self.save_dir, "test.txt")
        os.makedirs(self.save_train_image_dir, exist_ok=True)
        os.makedirs(self.save_test_image_dir, exist_ok=True)
        self.source_train_annotation_path = os.path.join(
            self.wflw_annotation_dir, "list_98pt_rect_attr_train.txt"
        )
        self.source_test_annotation_path = os.path.join(
            self.wflw_annotation_dir, "list_98pt_rect_attr_test.txt"
        )
        self.train_annotations, self.test_annotations = self._fetch_annotations()
        print(f"Train annotations count: {len(self.train_annotations)}\n"
              f"Test  annotations count: {len(self.test_annotations)}")

    def convert(self):
        train_anno_file = open(self.save_train_annotation_path, "w")
        test_anno_file = open(self.save_test_annotation_path, "w")

        for annotation in tqdm.tqdm(
                self.train_annotations,
                colour="GREEN",
                desc="Converting WFLW Train Annotations"
        ):
            crop, landmarks, new_img_name = self._process_annotation(annotation=annotation)
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
                self.test_annotations,
                colour="GREEN",
                desc="Converting WFLW Test Annotations"
        ):
            crop, landmarks, new_img_name = self._process_annotation(annotation=annotation)
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

    def show(
            self,
            count: int = 10,
            show_dir: Optional[str] = None,
            original: Optional[bool] = False
    ):
        if show_dir is None:
            show_dir = os.path.join(self.save_dir, "show")
            os.makedirs(show_dir, exist_ok=True)
        else:
            assert os.path.exists(show_dir)

        if not original:
            assert os.path.exists(self.save_test_annotation_path), \
                f"{self.save_test_annotation_path} not found!"

            with open(self.save_test_annotation_path, "r") as fin:
                annotations = fin.readlines()[:count]

            assert len(annotations) >= 1, "no annotations!"

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
        else:
            assert len(self.test_annotations) >= 1
            # show original annotations without any process
            tmp_annotations = self.test_annotations[:count]

            for annotation in tmp_annotations:
                image, landmarks, bbox, new_img_name = self._get_annotation(annotation=annotation)
                bbox = np.expand_dims(np.array(bbox), axis=0)  # (1, 4)
                image = draw_bboxes(image, bboxes=bbox)
                image = draw_landmarks(image, landmarks=landmarks)
                out_path = os.path.join(show_dir, new_img_name)

                cv2.imwrite(out_path, image)

                print(f"saved show original img to: {out_path} !")

    def _get_annotation(self, annotation: str) \
            -> Union[Tuple[np.ndarray, np.ndarray, List[float], str],
                     Tuple[None, None, None, None]]:
        annotation = annotation.strip("\n").strip(" ").split(" ")
        image_name = annotation[-1]
        image_path = os.path.join(self.wflw_images_dir, image_name)
        if not os.path.exists(image_path):
            return None, None, None, None

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

        # the same image would contains more than 1 face.
        new_img_name = image_name.replace("/", "_").split(".")[0] + f"x{int(xmin)}y{int(ymin)}.jpg"

        return image, landmarks, bbox, new_img_name

    def _process_annotation(
            self,
            annotation: str
    ) -> Union[Tuple[np.ndarray, np.ndarray, str], Tuple[None, None, None]]:

        image, landmarks, bbox, new_img_name = self._get_annotation(annotation=annotation)

        if image is None:
            return None, None, None

        image_height, image_width, _ = image.shape

        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        # padding
        xmin -= width * (self.scale - 1.) / 2.
        ymin -= height * (self.scale - 1.) / 2.
        xmax += width * (self.scale - 1.) / 2.
        ymax += height * (self.scale - 1.) / 2.

        xmin = int(max(xmin, 0))
        ymin = int(max(ymin, 0))
        xmax = int(min(xmax, image_width - 1))
        ymax = int(min(ymax, image_height - 1))
        # update h and w
        width = xmax - xmin
        height = ymax - ymin
        # crop padded face
        crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
        # adjust according to left-top corner
        landmarks[:, 0] -= float(xmin)
        landmarks[:, 1] -= float(ymin)

        if self.target_size is not None and self.resize_op is not None:
            crop, landmarks = self.resize_op(crop, landmarks)
            # update h and w of resized crop
            height, width, _ = crop.shape

        if self.force_normalize:
            landmarks[:, 0] /= float(width)
            landmarks[:, 1] /= float(height)

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


class Landmarks300WConverter(BaseConverter):
    def __init__(
            self,
            data_dir: Optional[str] = "./data/300W",
            save_dir: Optional[str] = "./data/300W/converted",
            extend: Optional[float] = 0.2,
            target_size: Optional[int] = None,
            keep_aspect: Optional[bool] = False,
            rebuild: Optional[bool] = True,
            force_normalize: Optional[bool] = False,
            force_absolute_path: Optional[bool] = True
    ):
        super(Landmarks300WConverter, self).__init__()
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.scale = 1. + extend
        self.target_size = target_size
        self.rebuild = rebuild
        self.force_normalize = force_normalize
        self.force_absolute_path = force_absolute_path
        assert os.path.exists(self.data_dir), "300W dataset not found."
        os.makedirs(self.save_dir, exist_ok=True)

        self.resize_op = None
        if target_size is not None:
            self.resize_op = LandmarksResize((target_size, target_size), keep_aspect=keep_aspect)

        if self.force_absolute_path:
            self.data_dir = os.path.abspath(self.data_dir)
            self.save_dir = os.path.abspath(self.save_dir)

        self.source_train_folders = ['afw', 'helen/trainset', 'lfpw/trainset']
        self.source_test_folders = ['helen/testset', 'lfpw/testset', 'ibug']
        self.source_train_folders = [os.path.join(self.data_dir, x) for x in self.source_train_folders]
        self.source_test_folders = [os.path.join(self.data_dir, x) for x in self.source_test_folders]
        self.save_train_image_dir = os.path.join(self.save_dir, "image/train")
        self.save_train_annotation_path = os.path.join(self.save_dir, "train.txt")
        self.save_test_image_dir = os.path.join(self.save_dir, "image/test")
        self.save_test_annotation_path = os.path.join(self.save_dir, "test.txt")
        os.makedirs(self.save_train_image_dir, exist_ok=True)
        os.makedirs(self.save_test_image_dir, exist_ok=True)

        self.train_annotations, self.test_annotations = self._fetch_annotations()
        print(f"Train annotations count: {len(self.train_annotations)}\n"
              f"Test  annotations count: {len(self.test_annotations)}")

    def convert(self):
        train_anno_file = open(self.save_train_annotation_path, "w")
        test_anno_file = open(self.save_test_annotation_path, "w")

        for annotation in tqdm.tqdm(
                self.train_annotations,
                colour="GREEN",
                desc="Converting 300W Train Annotations"
        ):
            crop, landmarks, new_img_name = self._process_annotation(annotation=annotation)
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
                self.test_annotations,
                colour="GREEN",
                desc="Converting 300W Test Annotations"
        ):
            crop, landmarks, new_img_name = self._process_annotation(annotation=annotation)
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

    def show(
            self,
            count: int = 10,
            show_dir: Optional[str] = None,
            original: Optional[bool] = False
    ):
        if show_dir is None:
            show_dir = os.path.join(self.save_dir, "show")
            os.makedirs(show_dir, exist_ok=True)
        else:
            assert os.path.exists(show_dir)

        if not original:
            assert os.path.exists(self.save_test_annotation_path), \
                f"{self.save_test_annotation_path} not found!"

            with open(self.save_test_annotation_path, "r") as fin:
                annotations = fin.readlines()[:count]

            assert len(annotations) >= 1, "no annotations!"

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
        else:
            assert len(self.test_annotations) >= 1
            # show original annotations without any process
            tmp_annotations = self.test_annotations[:count]

            for annotation in tmp_annotations:
                image, landmarks, bbox, new_img_name = self._get_annotation(annotation=annotation)
                bbox = np.expand_dims(np.array(bbox), axis=0)  # (1, 4)
                image = draw_bboxes(image, bboxes=bbox)
                image = draw_landmarks(image, landmarks=landmarks)
                out_path = os.path.join(show_dir, new_img_name)
                cv2.imwrite(out_path, image)

                print(f"saved show original img to: {out_path} !")

    def _get_annotation(self, annotation: str) \
            -> Union[Tuple[np.ndarray, np.ndarray, List[float], str],
                     Tuple[None, None, None, None]]:
        annotation = annotation.strip("\n").split(" ")
        image_path = annotation[0]
        annotation_path = annotation[1]
        if not os.path.exists(image_path) or not os.path.exists(annotation_path):
            return None, None, None, None

        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        # 68 gt landmarks
        with open(annotation_path, "r") as fin:
            landmarks = fin.readlines()[3:-1]
            landmarks = [x.strip().split() for x in landmarks]
            landmarks = [[int(float(x[0])), int(float(x[1]))] for x in landmarks]  # [(x0,y0),...]
            landmarks = np.array(landmarks).reshape(-1, 2)  # (68,2)
            landmarks[:, 0] = np.minimum(np.maximum(0, landmarks[:, 0]), image_width)
            landmarks[:, 1] = np.minimum(np.maximum(0, landmarks[:, 1]), image_height)

        xmin = np.min(landmarks[:, 0]).item()
        ymin = np.min(landmarks[:, 1]).item()
        xmax = np.max(landmarks[:, 0]).item()
        ymax = np.max(landmarks[:, 1]).item()

        bbox = [xmin, ymin, xmax, ymax]
        # generate a new image name
        image_name = os.path.basename(image_path)
        # ['afw', 'helen/trainset', 'lfpw/trainset']
        # ['helen/testset', 'lfpw/testset', 'ibug']
        image_path_splits = image_path.split("/")
        if image_path_splits[-2].find("trainset") >= 0 \
                or image_path_splits[-2].find("testset") >= 0:
            # e.g 'helen_trainset' or 'helen_testset'
            image_path_prefix = image_path_splits[-3] + "_" + image_path_splits[-2]
        else:
            # e.g 'afw' or 'ibug'
            image_path_prefix = image_path_splits[-2]
        # e.g helen_trainset_xxx.jpg
        new_img_name = image_path_prefix + "_" + image_name.split(".")[0] + ".jpg"

        return image, landmarks, bbox, new_img_name

    def _process_annotation(
            self,
            annotation: str
    ) -> Union[Tuple[np.ndarray, np.ndarray, str], Tuple[None, None, None]]:

        image, landmarks, bbox, new_img_name = self._get_annotation(annotation=annotation)

        if image is None:
            return None, None, None

        image_height, image_width, _ = image.shape

        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        # padding
        xmin -= width * (self.scale - 1.) / 2.
        ymin -= height * (self.scale - 1.) / 2.
        xmax += width * (self.scale - 1.) / 2.
        ymax += height * (self.scale - 1.) / 2.

        xmin = int(max(xmin, 0))
        ymin = int(max(ymin, 0))
        xmax = int(min(xmax, image_width - 1))
        ymax = int(min(ymax, image_height - 1))
        # update h and w
        width = xmax - xmin
        height = ymax - ymin
        # crop padded face
        crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
        # adjust according to left-top corner
        landmarks[:, 0] -= float(xmin)
        landmarks[:, 1] -= float(ymin)

        if self.target_size is not None and self.resize_op is not None:
            crop, landmarks = self.resize_op(crop, landmarks)
            # update h and w of resized crop
            height, width, _ = crop.shape

        if self.force_normalize:
            landmarks[:, 0] /= float(width)
            landmarks[:, 1] /= float(height)

        return crop, landmarks, new_img_name

    def _fetch_annotations(self) -> Tuple[List[str], List[str]]:
        print("Fetching annotations ...")
        train_annotations = []
        test_annotations = []

        # fetch train annotations
        for source_train_folder in self.source_train_folders:
            all_files = sorted(os.listdir(source_train_folder))  # xxx.jpg/png xxx_mirror.jpg/png xxx.pts
            all_files = [x for x in all_files if x.lower().find("mirror") < 0]  # exclude mirror images
            img_files = [x for x in all_files if x.lower().endswith("jpg") or x.lower().endswith("png")]
            pts_files = [x.split(".")[0] + ".pts" for x in img_files]
            img_files = [os.path.join(source_train_folder, x) for x in img_files]
            pts_files = [os.path.join(source_train_folder, x) for x in pts_files]
            pts_files = [x for x in pts_files if os.path.exists(x)]
            assert len(img_files) == len(pts_files)
            # prepare annotation pairs: img_path pth_path
            for img_path, pts_path in zip(img_files, pts_files):
                annotation = img_path + " " + pts_path
                train_annotations.append(annotation)

        # fetch test annotations
        for source_test_folder in self.source_test_folders:
            all_files = sorted(os.listdir(source_test_folder))  # xxx.jpg/png xxx_mirror.jpg/png xxx.pts
            all_files = [x for x in all_files if x.lower().find("mirror") < 0]  # exclude mirror images
            img_files = [x for x in all_files if x.lower().endswith("jpg") or x.lower().endswith("png")]
            pts_files = [x.split(".")[0] + ".pts" for x in img_files]
            img_files = [os.path.join(source_test_folder, x) for x in img_files]
            pts_files = [os.path.join(source_test_folder, x) for x in pts_files]
            pts_files = [x for x in pts_files if os.path.exists(x)]
            assert len(img_files) == len(pts_files)
            # prepare annotation pairs: img_path pth_path
            for img_path, pts_path in zip(img_files, pts_files):
                annotation = img_path + " " + pts_path
                test_annotations.append(annotation)

        return train_annotations, test_annotations


class LandmarksCOFWConverter(BaseConverter):
    def __init__(
            self,
            data_dir: Optional[str] = "./data/COFW",
            save_dir: Optional[str] = "./data/COFW/converted",
            extend: Optional[float] = 0.2,
            target_size: Optional[int] = None,
            keep_aspect: Optional[bool] = False,
            rebuild: Optional[bool] = True,
            force_normalize: Optional[bool] = False,
            force_absolute_path: Optional[bool] = True
    ):
        super(LandmarksCOFWConverter, self).__init__()
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.scale = 1. + extend
        self.target_size = target_size
        self.rebuild = rebuild
        self.force_normalize = force_normalize
        self.force_absolute_path = force_absolute_path
        assert os.path.exists(self.data_dir), "COFW dataset not found."
        os.makedirs(self.save_dir, exist_ok=True)

        self.resize_op = None
        if target_size is not None:
            self.resize_op = LandmarksResize((target_size, target_size), keep_aspect=keep_aspect)

        if self.force_absolute_path:
            self.data_dir = os.path.abspath(self.data_dir)
            self.save_dir = os.path.abspath(self.save_dir)

        self.source_train_color_mat = os.path.join(self.data_dir, "COFW_train_color.mat")
        self.source_test_color_mat = os.path.join(self.data_dir, "COFW_test_color.mat")
        self.save_train_image_dir = os.path.join(self.save_dir, "image/train")
        self.save_train_annotation_path = os.path.join(self.save_dir, "train.txt")
        self.save_test_image_dir = os.path.join(self.save_dir, "image/test")
        self.save_test_annotation_path = os.path.join(self.save_dir, "test.txt")
        os.makedirs(self.save_train_image_dir, exist_ok=True)
        os.makedirs(self.save_test_image_dir, exist_ok=True)

        self.train_annotations, self.test_annotations = self._fetch_annotations()
        print(f"Train annotations count: {len(self.train_annotations)}\n"
              f"Test  annotations count: {len(self.test_annotations)}")

    def convert(self):
        train_anno_file = open(self.save_train_annotation_path, "w")
        test_anno_file = open(self.save_test_annotation_path, "w")

        # convert train annotations
        train_images = self.train_annotations["images"]
        train_bboxes = self.train_annotations["bboxes"]
        train_annos = self.train_annotations["annos"]
        train_num = train_images.shape[0]

        for i in tqdm.tqdm(
                range(train_num),
                colour="GREEN",
                desc="Converting COFW Train Annotations"
        ):
            image = train_images[i, 0]
            # grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # swap rgb channel to bgr
            else:
                image = image[:, :, ::-1]
            bbox = train_bboxes[i, :]
            anno = train_annos[i, :]
            crop, landmarks = self._process_annotation(
                image=image, bbox=bbox, anno=anno
            )
            if crop is None or landmarks is None:
                continue
            new_img_name = f"cofw_train_{i}.jpg"
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

        # convert test annotations
        test_images = self.test_annotations["images"]
        test_bboxes = self.test_annotations["bboxes"]
        test_annos = self.test_annotations["annos"]
        test_num = test_images.shape[0]

        for j in tqdm.tqdm(
                range(test_num),
                colour="GREEN",
                desc="Converting COFW Test Annotations"
        ):
            image = test_images[j, 0]
            # grayscale
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            # swap rgb channel to bgr
            else:
                image = image[:, :, ::-1]
            bbox = test_bboxes[j, :]
            anno = test_annos[j, :]
            crop, landmarks = self._process_annotation(
                image=image, bbox=bbox, anno=anno
            )
            if crop is None or landmarks is None:
                continue
            new_img_name = f"cofw_test_{j}.jpg"
            new_img_path = os.path.join(self.save_train_image_dir, new_img_name)
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

    def show(
            self,
            count: int = 10,
            show_dir: Optional[str] = None,
            original: Optional[bool] = False
    ):
        if show_dir is None:
            show_dir = os.path.join(self.save_dir, "show")
            os.makedirs(show_dir, exist_ok=True)
        else:
            assert os.path.exists(show_dir)

        if not original:
            assert os.path.exists(self.save_test_annotation_path), \
                f"{self.save_test_annotation_path} not found!"

            with open(self.save_test_annotation_path, "r") as fin:
                annotations = fin.readlines()[:count]

            assert len(annotations) >= 1, "no annotations!"

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
        else:
            test_images = self.test_annotations["images"]
            test_bboxes = self.test_annotations["bboxes"]
            test_annos = self.test_annotations["annos"]

            # show original annotations without any process
            for j in range(count):
                image = test_images[j, 0]
                # grayscale
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                # swap rgb channel to bgr
                else:
                    image = image[:, :, ::-1]
                bbox = test_bboxes[j, :]
                anno = test_annos[j, :]
                image, landmarks, bbox = self._get_annotation(
                    image=image, bbox=bbox, anno=anno
                )
                if image is None:
                    continue
                bbox = np.expand_dims(np.array(bbox), axis=0)  # (1, 4)
                image = draw_bboxes(image, bboxes=bbox)
                image = draw_landmarks(image, landmarks=landmarks)
                new_img_name = f"cofw_test_{j}.jpg"
                out_path = os.path.join(show_dir, new_img_name)

                cv2.imwrite(out_path, image)

                print(f"saved show original img to: {out_path} !")

    def _get_annotation(
            self,
            image: np.ndarray,
            bbox: np.ndarray,
            anno: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray, List[float]],
               Tuple[None, None, None]]:
        if image is None:
            return None, None, None
        image_height, image_width, _ = image.shape
        landmarks = np.zeros((29, 2))
        landmarks[:, 0] = anno[:29]
        landmarks[:, 1] = anno[29:58]
        landmarks[:, 0] = np.minimum(np.maximum(0, landmarks[:, 0]), image_width)
        landmarks[:, 1] = np.minimum(np.maximum(0, landmarks[:, 1]), image_height)
        xmin, ymin, width, height = bbox
        xmax = xmin + width - 1
        ymax = ymin + height - 1
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, image_width - 1)
        ymax = min(ymax, image_height - 1)
        bbox = [xmin, ymin, xmax, ymax]

        return image, landmarks, bbox

    def _process_annotation(
            self,
            image: np.ndarray,
            bbox: np.ndarray,
            anno: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:

        image, landmarks, bbox = self._get_annotation(
            image=image, bbox=bbox, anno=anno
        )

        if image is None:
            return None, None

        image_height, image_width, _ = image.shape

        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        # padding
        xmin -= width * (self.scale - 1.) / 2.
        ymin -= height * (self.scale - 1.) / 2.
        xmax += width * (self.scale - 1.) / 2.
        ymax += height * (self.scale - 1.) / 2.

        xmin = int(max(xmin, 0))
        ymin = int(max(ymin, 0))
        xmax = int(min(xmax, image_width - 1))
        ymax = int(min(ymax, image_height - 1))
        # update h and w
        width = xmax - xmin
        height = ymax - ymin
        # crop padded face
        crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
        # adjust according to left-top corner
        landmarks[:, 0] -= float(xmin)
        landmarks[:, 1] -= float(ymin)

        if self.target_size is not None and self.resize_op is not None:
            crop, landmarks = self.resize_op(crop, landmarks)
            # update h and w of resized crop
            height, width, _ = crop.shape

        if self.force_normalize:
            landmarks[:, 0] /= float(width)
            landmarks[:, 1] /= float(height)

        return crop, landmarks

    def _fetch_annotations(self) -> Tuple[dict, dict]:
        import hdf5storage
        print("Fetching annotations ...")
        train_annotations = {}
        test_annotations = {}

        # fetch train annotations
        train_mat = hdf5storage.loadmat(self.source_train_color_mat)
        train_images = train_mat['IsTr']
        train_bboxes = train_mat['bboxesTr']
        train_annos = train_mat['phisTr']

        train_annotations["images"] = train_images  # (b,1,h,w,3|1)
        train_annotations["bboxes"] = train_bboxes  # (b,4) xmin, ymin, width, height
        train_annotations["annos"] = train_annos  # (b,29*2) x0,x1,...,y0,y1,...

        # fetch test annotations
        test_mat = hdf5storage.loadmat(self.source_test_color_mat)
        test_images = test_mat['IsT']
        test_bboxes = test_mat['bboxesT']
        test_annos = test_mat['phisT']

        test_annotations["images"] = test_images  # (b,1,h,w,3|1)
        test_annotations["bboxes"] = test_bboxes  # (b,4) xmin, ymin, width, height
        test_annotations["annos"] = test_annos  # (b,29*2) x0,x1,...,y0,y1,...

        return train_annotations, test_annotations


class LandmarksAFLWConverter(BaseConverter):
    def __init__(
            self,
            data_dir: Optional[str] = "./data/AFLW",
            save_dir: Optional[str] = "./data/AFLW/converted",
            extend: Optional[float] = 0.2,
            target_size: Optional[int] = None,
            keep_aspect: Optional[bool] = False,
            rebuild: Optional[bool] = True,
            force_normalize: Optional[bool] = False,
            force_absolute_path: Optional[bool] = True,
            keep_pipnet_style: Optional[bool] = False,  # for pipnet
            train_count: Optional[int] = 20000
    ):
        super(LandmarksAFLWConverter, self).__init__()
        import warnings
        warnings.warn("Because the pre-processes of AFLW need SQLite,\n"
                      "Not all OS have install it by default! So, this"
                      "Converting process may failed.")
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.scale = 1. + extend
        self.target_size = target_size
        self.rebuild = rebuild
        self.force_normalize = force_normalize
        self.force_absolute_path = force_absolute_path
        self.keep_pipnet_style = keep_pipnet_style
        self.train_count = train_count
        assert os.path.exists(self.data_dir), "AFLW dataset not found."
        os.makedirs(self.save_dir, exist_ok=True)

        self.resize_op = None
        if target_size is not None:
            self.resize_op = LandmarksResize((target_size, target_size), keep_aspect=keep_aspect)

        if self.force_absolute_path:
            self.data_dir = os.path.abspath(self.data_dir)
            self.save_dir = os.path.abspath(self.save_dir)

        self.source_aflw_anno_sqlite = os.path.join(self.data_dir, "aflw.sqlite")
        self.source_aflw_image_dir = os.path.join(self.data_dir, "flickr")
        self.save_train_image_dir = os.path.join(self.save_dir, "image/train")
        self.save_train_annotation_path = os.path.join(self.save_dir, "train.txt")
        self.save_test_image_dir = os.path.join(self.save_dir, "image/test")
        self.save_test_annotation_path = os.path.join(self.save_dir, "test.txt")
        os.makedirs(self.save_train_image_dir, exist_ok=True)
        os.makedirs(self.save_test_image_dir, exist_ok=True)

        self.train_test_annotations = self._fetch_annotations()
        print(f"Train annotations count: {len(self.train_test_annotations['train_ids'])}\n"
              f"Test  annotations count: {len(self.train_test_annotations['test_ids'])}")

    def convert(self):

        train_anno_file = open(self.save_train_annotation_path, "w")
        test_anno_file = open(self.save_test_annotation_path, "w")

        train_ids = self.train_test_annotations["train_ids"]
        test_ids = self.train_test_annotations["test_ids"]

        # convert train annotations
        for i in tqdm.tqdm(
                train_ids,
                colour="GREEN",
                desc="Converting AFLW Train Annotations"
        ):
            # from face_id
            annotation = self.train_test_annotations[i]
            file_id = annotation["file_id"]
            bbox = annotation["bbox"]
            anno = annotation["anno"]
            crop, landmarks = self._process_annotation(
                file_id=file_id, bbox=bbox, anno=anno
            )
            if crop is None or landmarks is None:
                continue
            new_img_name = f"aflw_train_{file_id.split('.')[0]}_{i}.jpg"
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

        # convert test annotations
        for j in tqdm.tqdm(
                test_ids,
                colour="GREEN",
                desc="Converting AFLW Test Annotations"
        ):
            # from matlab index
            annotation = self.train_test_annotations[j]
            file_id = annotation["file_id"]
            bbox = annotation["bbox"]
            anno = annotation["anno"]
            crop, landmarks = self._process_annotation(
                file_id=file_id, bbox=bbox, anno=anno
            )
            if crop is None or landmarks is None:
                continue
            new_img_name = f"aflw_test_{file_id.split('.')[0]}_{j}.jpg"
            new_img_path = os.path.join(self.save_train_image_dir, new_img_name)
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

    def show(
            self,
            count: int = 10,
            show_dir: Optional[str] = None,
            original: Optional[bool] = False
    ):
        if show_dir is None:
            show_dir = os.path.join(self.save_dir, "show")
            os.makedirs(show_dir, exist_ok=True)
        else:
            assert os.path.exists(show_dir)

        if not original:
            assert os.path.exists(self.save_test_annotation_path), \
                f"{self.save_test_annotation_path} not found!"

            with open(self.save_test_annotation_path, "r") as fin:
                annotations = fin.readlines()[:count]

            assert len(annotations) >= 1, "no annotations!"

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
        else:
            face_ids = self.train_test_annotations["face_ids"][:count]

            # show original annotations without any process
            for j in face_ids:
                annotation = self.train_test_annotations[j]
                file_id = annotation["file_id"]
                bbox = annotation["bbox"]
                anno = annotation["anno"]
                image, landmarks, bbox = self._get_annotation(
                    file_id=file_id, bbox=bbox, anno=anno
                )
                if image is None or landmarks is None:
                    continue
                bbox = np.expand_dims(np.array(bbox), axis=0)  # (1, 4)
                image = draw_bboxes(image, bboxes=bbox)
                image = draw_landmarks(image, landmarks=landmarks)
                new_img_name = f"aflw_test_{file_id.split('.')[0]}_{j}.jpg"
                out_path = os.path.join(show_dir, new_img_name)
                cv2.imwrite(out_path, image)

                print(f"saved show original img to: {out_path} !")

    def _get_annotation(
            self,
            file_id: str,
            bbox: np.ndarray,
            anno: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray, List[float]],
               Tuple[None, None, None]]:
        image_path = os.path.join(self.source_aflw_image_dir, file_id)
        image = cv2.imread(image_path)
        if image is None:
            return None, None, None
        image_height, image_width, _ = image.shape
        if self.keep_pipnet_style:
            landmarks = np.zeros((19, 2))
            landmarks[:, 0] = anno[:19]
            landmarks[:, 1] = anno[19:]
        else:
            landmarks = np.zeros((21, 2))
            landmarks[:, 0] = anno[:21]
            landmarks[:, 1] = anno[21:]
        landmarks[:, 0] = np.minimum(np.maximum(0, landmarks[:, 0]), image_width)
        landmarks[:, 1] = np.minimum(np.maximum(0, landmarks[:, 1]), image_height)
        xmin, ymin, xmax, ymax = bbox
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, image_width - 1)
        ymax = min(ymax, image_height - 1)
        bbox = [xmin, ymin, xmax, ymax]

        return image, landmarks, bbox

    def _process_annotation(
            self,
            file_id: str,
            bbox: np.ndarray,
            anno: np.ndarray
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:

        image, landmarks, bbox = self._get_annotation(
            file_id=file_id, bbox=bbox, anno=anno
        )

        if image is None:
            return None, None

        image_height, image_width, _ = image.shape

        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        # padding
        xmin -= width * (self.scale - 1.) / 2.
        ymin -= height * (self.scale - 1.) / 2.
        xmax += width * (self.scale - 1.) / 2.
        ymax += height * (self.scale - 1.) / 2.

        xmin = int(max(xmin, 0))
        ymin = int(max(ymin, 0))
        xmax = int(min(xmax, image_width - 1))
        ymax = int(min(ymax, image_height - 1))
        # update h and w
        width = xmax - xmin
        height = ymax - ymin
        # crop padded face
        crop = image[int(ymin):int(ymax), int(xmin):int(xmax), :]
        # adjust according to left-top corner
        landmarks[:, 0] -= float(xmin)
        landmarks[:, 1] -= float(ymin)

        if self.target_size is not None and self.resize_op is not None:
            crop, landmarks = self.resize_op(crop, landmarks)
            # update h and w of resized crop
            height, width, _ = crop.shape

        if self.force_normalize:
            landmarks[:, 0] /= float(width)
            landmarks[:, 1] /= float(height)

        return crop, landmarks

    def _fetch_annotations(self) -> dict:
        import sqlite3
        import pandas as pd
        print("Fetching annotations ...")
        train_test_annotations = {}

        conn = sqlite3.connect(self.source_aflw_anno_sqlite)
        # noinspection SqlDialectInspection
        faces_df = pd.read_sql_query("SELECT face_id,file_id,db_id FROM Faces", conn)
        # noinspection SqlDialectInspection
        feature_coords_df = pd.read_sql_query(
            "SELECT face_id,feature_id,x,y,annot_type_id FROM FeatureCoords",
            conn
        )  # face_id feature_id x y annot_type_id
        # noinspection SqlDialectInspection
        face_rects_df = pd.read_sql_query("SELECT face_id,x,y,w,h,annot_type_id FROM FaceRect", conn)
        face_ids = np.unique(faces_df["face_id"].to_numpy())  # 去重
        print(f"face_ids: {len(face_ids)}")
        face_ids = np.sort(face_ids).tolist()

        valid_ids = []
        # fetch annotations and image names
        for face_id in tqdm.tqdm(
                face_ids,
                colour="YELLOW",
                desc="Fetching AFLW SOLite"
        ):
            try:
                annotation = {
                    "file_id": faces_df.loc[faces_df["face_id"] == face_id, "file_id"].values[0],
                    "db_id": faces_df.loc[faces_df["face_id"] == face_id, "db_id"].values[0]
                }
                feature_coords = \
                    feature_coords_df.loc[feature_coords_df["face_id"] == face_id, :].sort_values(by="feature_id")
                annotation["feature_id"] = feature_coords["feature_id"].values
                anno = []
                if self.keep_pipnet_style:
                    anno.extend(feature_coords["x"].values[1:-1])  # 19 landmarks
                    anno.extend(feature_coords["y"].values[1:-1])
                    if len(anno) != 38:
                        continue
                else:
                    anno.extend(feature_coords["x"].values)  # 21 landmarks
                    anno.extend(feature_coords["y"].values)
                    if len(anno) != 42:
                        continue

                valid_ids.append(face_id)
                annotation["anno"] = np.array(anno)
                x1, y1, w, h = face_rects_df.loc[face_rects_df["face_id"] == face_id, :].values[0][1:-1]
                x2 = x1 + w
                y2 = y1 + h
                annotation["bbox"] = np.array([x1, y1, x2, y2])
                train_test_annotations[face_id] = annotation
            except:
                continue

        print(f"valid_ids: {len(valid_ids)}")
        if self.train_count > len(valid_ids):
            self.train_count = int(0.75 * len(valid_ids))

        train_test_annotations["face_ids"] = valid_ids
        train_test_annotations["train_ids"] = valid_ids[:self.train_count]  # e. 20000
        train_test_annotations["test_ids"] = valid_ids[self.train_count:]

        conn.close()
        return train_test_annotations
