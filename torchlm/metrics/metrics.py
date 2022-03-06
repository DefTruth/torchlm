import numpy as np
from scipy.integrate import simps
from typing import List, Tuple

__all__ = ["nme", "fr_and_auc"]


def nme(lms_pred: np.ndarray, lms_gt: np.ndarray, norm: float) -> float:
    """
    :param lms_pred: (n,2) predicted landmarks.
    :param lms_gt: (n,2) ground truth landmarks.
    :param norm: normalize value, the distance between two eyeballs.
    :return: nme value.
    """
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme_ = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm
    return nme_


def fr_and_auc(nmes: List[float], thres: float = 0.1, step: float = 0.0001) -> Tuple[float, float]:
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    return fr, auc
