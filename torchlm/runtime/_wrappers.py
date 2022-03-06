import numpy as np

from ..tools import FaceDetTool, LandmarksDetTool
from typing import Tuple, Any, Union

_Landmarks = np.ndarray
_BBoxes = np.ndarray

__all__ = ["set_faces", "set_landmarks", "forward", "bind"]


class RuntimeWrapper(object):
    face_det: FaceDetTool
    landmarks_det: LandmarksDetTool

    @classmethod
    def set_faces(cls, face_det: FaceDetTool):
        cls.face_det = face_det

    @classmethod
    def set_landmarks(cls, landmarks_det: LandmarksDetTool):
        cls.landmarks_det = landmarks_det

    @classmethod
    def forward(
            cls,
            image: np.ndarray,
            extend: float = 0.2,
            swapRB_before_face: bool = False,
            swapRB_before_landmarks: bool = True,
            **kwargs: Any  # params for face_det & landmarks_det
    ) -> Tuple[_Landmarks, _BBoxes]:
        """
        :param image: original input image, HWC, BGR/RGB
        :param extend: extend ratio for face cropping (1.+extend) before landmarks detection.
        :param swapRB_before_face: swap RB channel before face detection.
        :param swapRB_before_landmarks: swap RB channel before landmarks detection.
        :param kwargs: params for intern face_det and landmarks_det.
        :return: landmarks (n,m,2) -> x,y; bboxes (n,5) -> x1,y1,x2,y2,score
        """
        if cls.face_det is None or cls.landmarks_det is None:
            raise RuntimeError("Please setup face_det and landmarks_det"
                               " before run landmarks detection!")

        height, width, _ = image.shape
        if swapRB_before_face:
            image_swapRB = image[:, :, ::-1].copy()
            bboxes = cls.face_det.detect(image_swapRB, **kwargs)  # (n,5) x1,y1,x2,y2,score
        else:
            bboxes = cls.face_det.detect(image, **kwargs)  # (n,5) x1,y1,x2,y2,score

        det_num = bboxes.shape[0]
        landmarks = []
        for i in range(det_num):
            x1 = int(bboxes[i][0])
            y1 = int(bboxes[i][1])
            x2 = int(bboxes[i][2])
            y2 = int(bboxes[i][3])

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            x1 -= int(w * (1. + extend - 1) / 2)
            # remove a part of top area for alignment, see paper for details
            y1 += int(h * (1. + extend - 1) / 2)
            x2 += int(w * (1. + extend - 1) / 2)
            y2 += int(h * (1. + extend - 1) / 2)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, width - 1)
            y2 = min(y2, height - 1)
            if swapRB_before_landmarks:
                crop = image[y1:y2, x1:x2, :][:, :, ::-1]  # e.g RGB
            else:
                crop = image[y1:y2, x1:x2, :]  # e.g BGR
            lms_pred = cls.landmarks_det.detect(crop, **kwargs)  # (m,2)
            lms_pred[:, 0] += x1
            lms_pred[:, 1] += y1
            landmarks.append(lms_pred)

        landmarks = np.array(landmarks).reshape((det_num, -1, 2))

        return landmarks, bboxes  # (n,m,2) (n,5)


def set_faces(face_det: FaceDetTool):
    RuntimeWrapper.set_faces(face_det=face_det)


def set_landmarks(landmarks_det: LandmarksDetTool):
    RuntimeWrapper.set_landmarks(landmarks_det=landmarks_det)

def bind(det: Union[FaceDetTool, LandmarksDetTool]):
    if isinstance(det, FaceDetTool):
        RuntimeWrapper.set_faces(face_det=det)
    elif isinstance(det, LandmarksDetTool):
        RuntimeWrapper.set_landmarks(landmarks_det=det)
    else:
        raise ValueError("Can only bind instance of "
                         "(FaceDetTool,LandmarksDetTool)")

def forward(
        image: np.ndarray,
        extend: float = 0.2,
        swapRB_before_face: bool = False,
        swapRB_before_landmarks: bool = True,
        **kwargs: Any  # params for face_det & landmarks_det
) -> Tuple[_Landmarks, _BBoxes]:
    """
    :param image: original input image, HWC, BGR/RGB
    :param extend: extend ratio for face cropping (1.+extend) before landmarks detection.
    :param swapRB_before_face: swap RB channel before face detection.
    :param swapRB_before_landmarks: swap RB channel before landmarks detection.
    :param kwargs: params for intern face_det and landmarks_det.
    :return: landmarks (n,m,2) -> x,y; bboxes (n,5) -> x1,y1,x2,y2,score
    """
    return RuntimeWrapper.forward(
        image=image,
        extend=extend,
        swapRB_before_face=swapRB_before_face,
        swapRB_before_landmarks=swapRB_before_landmarks,
        **kwargs
    )
