import os
import tqdm
import numpy as np
from typing import Tuple, List
from ..utils import draw_landmarks

def fetch_annotations(annotation_path: str) -> List[str]:
    """fetch annotation strings from a specific file, this file should formatted as:
       "img0_path x0 y0 x1 y1 ... xn-1,yn-1"
       "img1_path x0 y0 x1 y1 ... xn-1,yn-1"
       "img2_path x0 y0 x1 y1 ... xn-1,yn-1"
       "img3_path x0 y0 x1 y1 ... xn-1,yn-1"
       ...
    :param annotation_path:
    :return: annotations list
    """
    annotations = []
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as fin:
            for line in fin:
                line = line.strip("\n")
                img_path = line.split(" ")[0]
                if os.path.exists(img_path):
                    annotations.append(line)
    return annotations


def format_annotation(img_path: str, lms_gt: np.ndarray) -> str:
    """format img_path and lms_gt to a long annotation string.
    :param img_path: absolute path to a image, e.g xxx/xxx/xxx.jpg
    :param lms_gt: np.ndarray with shape (N,2), N denotes the number of landmarks.
    :return: A formatted annotation string, 'img_path x0 y0 x1 y1 ... xn-1,yn-1'
    """
    annotation_string = [img_path]
    annotation_string.extend(list(lms_gt.flatten()))
    annotation_string = [str(x) for x in annotation_string]
    annotation_string = " ".join(annotation_string)
    return annotation_string


def decode_annotation(annotation_string: str) -> Tuple[str, np.ndarray]:
    """decode img_path and lms_gt from a long annotation string.
    :param annotation_string:
    :return: img_path and lms_gt.
    """
    annotation = annotation_string.strip("\n").split(" ")
    img_path = annotation[0]
    lms_gt = [float(x) for x in annotation[1:]]
    lms_gt = np.array(lms_gt).reshape((-1, 2))
    return img_path, lms_gt

def generate_meanface(
        annotation_path: str,
) -> Tuple[np.ndarray, str]:
    annotations_info = fetch_annotations(annotation_path=annotation_path)
    landmarks = []
    for annotation_string in tqdm.tqdm(
            annotations_info,
            desc=f"generating meanface from: {annotation_path}"
    ):
        _, lms_gt = decode_annotation(
            annotation_string=annotation_string)  # (n,2)
        landmarks.append(np.expand_dims(lms_gt, axis=0))  # (1, n,2)
    landmarks = np.concatenate(landmarks, axis=0)  # (m,n,2)
    meanface: np.ndarray = np.mean(landmarks, axis=0, keepdims=False)  # (n,2)
    meanface_string = meanface.flatten().tolist()
    # noinspection PyTypeChecker
    meanface_string = " ".join([str(x) for x in meanface_string])

    return meanface, meanface_string

def draw_meanface(meanface: np.ndarray) -> np.ndarray:
    # noinspection PyTypeChecker, PyArgumentList
    if meanface.max() <= 1.:
        meanface *= 256
    w = int(np.max(meanface[:, 0]).item() + 1)
    h = int(np.max(meanface[:, 1]).item() + 1)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    return draw_landmarks(canvas, landmarks=meanface)



