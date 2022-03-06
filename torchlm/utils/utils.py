import cv2
import numpy as np
from typing import Tuple, Optional

__all__ = [
    "draw_bboxes",
    "draw_landmarks"
]


def draw_bboxes(img: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
    im = img[:, :, :].copy()

    bboxes = bboxes[:, :4]
    bboxes = bboxes.reshape(-1, 4)

    # draw bbox
    for box in bboxes:
        im = cv2.rectangle(im, (int(box[0]), int(box[1])),
                           (int(box[2]), int(box[3])), (0, 255, 0), 2)

    return im.astype(np.uint8)


def draw_landmarks(
        img: np.ndarray,
        landmarks: np.ndarray,
        font: float = 0.25,
        circle: int = 2,
        text: bool = False,
        color: Optional[Tuple[int, int, int]] = (0, 255, 0),
        offset: int = 5,
        thickness: int = 1
) -> np.ndarray:
    im = img.astype(np.uint8).copy()
    if landmarks.ndim == 2:
        landmarks = np.expand_dims(landmarks, axis=0)

    for i in range(landmarks.shape[0]):
        for j in range(landmarks[i].shape[0]):
            x, y = landmarks[i, j, :].astype(int).tolist()
            cv2.circle(im, (x, y), circle, color, -1)
            if text:
                b = np.random.randint(0, 255)
                g = np.random.randint(0, 255)
                r = np.random.randint(0, 255)
                cv2.putText(im, '{}'.format(i), (x, y - offset),
                            cv2.FONT_ITALIC, font, (b, g, r), thickness)

    return im.astype(np.uint8)
