import numpy as np
from typing import Tuple, List


def _get_meanface(
        meanface_string: str,
        num_nb: int = 10
) -> Tuple[List[int], List[int], List[int], int, int]:
    """
    :param meanface_string: a long string contains normalized or un-normalized
     meanface coords, the format is "x0,y0,x1,y1,x2,y2,...,xn-1,yn-1".
    :param num_nb: the number of Nearest-neighbor landmarks for NRM, default 10
    :return: meanface_indices, reverse_index1, reverse_index2, max_len
    """
    meanface = meanface_string.strip("\n").strip(" ").split(" ")
    meanface = [float(x) for x in meanface]
    meanface = np.array(meanface).reshape(-1, 2)
    meanface_lms = meanface.shape[0]
    # each landmark predicts num_nb neighbors
    meanface_indices = []
    for i in range(meanface.shape[0]):
        pt = meanface[i, :]
        dists = np.sum(np.power(pt - meanface, 2), axis=1)
        indices = np.argsort(dists)
        meanface_indices.append(indices[1:1 + num_nb])

    # each landmark predicted by X neighbors, X varies
    meanface_indices_reversed = {}
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i] = [[], []]
    for i in range(meanface.shape[0]):
        for j in range(num_nb):
            # meanface_indices[i][0,1,2,...,9] -> [[i,i,...,i],[0,1,2,...,9]]
            meanface_indices_reversed[meanface_indices[i][j]][0].append(i)
            meanface_indices_reversed[meanface_indices[i][j]][1].append(j)

    max_len = 0
    for i in range(meanface.shape[0]):
        tmp_len = len(meanface_indices_reversed[i][0])
        if tmp_len > max_len:
            max_len = tmp_len

    # tricks, make them have equal length for efficient computation
    for i in range(meanface.shape[0]):
        meanface_indices_reversed[i][0] += meanface_indices_reversed[i][0] * 10
        meanface_indices_reversed[i][1] += meanface_indices_reversed[i][1] * 10
        meanface_indices_reversed[i][0] = meanface_indices_reversed[i][0][:max_len]
        meanface_indices_reversed[i][1] = meanface_indices_reversed[i][1][:max_len]

    # make the indices 1-dim
    # [...,max_len,...,max_len*2,...]
    reverse_index1 = []
    reverse_index2 = []
    for i in range(meanface.shape[0]):
        reverse_index1 += meanface_indices_reversed[i][0]
        reverse_index2 += meanface_indices_reversed[i][1]
    return meanface_indices, reverse_index1, reverse_index2, max_len, meanface_lms


def _normalize(
        img: np.ndarray
) -> np.ndarray:
    """
    :param img: source image, RGB with HWC and range [0,255]
    :return: normalized image CHW Tensor for PIPNet
    """
    img = img.astype(np.float32)
    img /= 255.
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225
    img = img.transpose((2, 0, 1))  # HWC->CHW
    return img.astype(np.float32)

