"""
Note: The helper codes in this script are heavy based on(many thanks~):
https://github.com/Paperspace/DataAugmentationForObjectDetection/blob/master/data_aug/bbox_util.py
"""
import os
import cv2
import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Union, Optional, List


def to_tensor(x: Union[np.ndarray, Tensor]) -> Tensor:
    """without post process, such as normalize"""
    assert isinstance(x, (np.ndarray, Tensor))

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x


def to_numpy(x: Union[np.ndarray, Tensor]) -> np.ndarray:
    """without post process, such as transpose"""
    assert isinstance(x, (np.ndarray, Tensor))

    if isinstance(x, np.ndarray):
        return x
    try:
        return x.cpu().numpy()
    except:
        return x.detach().cpu().numpy()


def bbox_area(bbox: np.ndarray) -> np.ndarray:
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def clip_box(bbox: np.ndarray, img_box: Union[np.ndarray, List[int]], alpha: float) -> np.ndarray:
    """Clip the bounding boxes to the borders of an image

    Parameters
    ----------

    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    img_box: numpy.ndarray/list
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.

    Returns
    -------

    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], img_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], img_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], img_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], img_box[3]).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    delta_area = ((ar_ - bbox_area(bbox)) / ar_)

    mask = (delta_area < (1 - alpha)).astype(int)

    bbox = bbox[mask == 1, :]

    return bbox


def rotate_im(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
    return image.astype(np.uint8)


def apply_mask(img: np.ndarray, mask_w: int, mask_h: int) -> Tuple[np.ndarray, List[int]]:
    h, w, c = img.shape
    mask_w = min(max(0., mask_w), w)
    mask_h = min(max(0., mask_h), h)
    x0 = np.random.randint(0, w - mask_w + 1)
    y0 = np.random.randint(0, h - mask_h + 1)
    x1, y1 = int(x0 + mask_w), int(y0 + mask_h)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)

    pixels = list(range(0, 240, 10))
    mask_value = np.random.choice(pixels, size=c)

    if np.random.uniform(0., 1.0) < 0.35:
        mask_value = [0] * c

    img[y0:y1, x0:x1, :] = mask_value

    mask_corner = [x0, y0, x1, y1]

    return img.astype(np.uint8), mask_corner


def apply_mask_with_alpha(img: np.ndarray, mask_w: int, mask_h: int, alpha: float) -> Tuple[np.ndarray, List[int]]:
    h, w, c = img.shape
    mask_w = min(max(0., mask_w), w)
    mask_h = min(max(0., mask_h), h)
    x0 = np.random.randint(0, w - mask_w + 1)
    y0 = np.random.randint(0, h - mask_h + 1)
    x1, y1 = int(x0 + mask_w), int(y0 + mask_h)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)

    pixels = list(range(0, 240, 10))
    mask_value = np.random.choice(pixels, size=c)

    if np.random.uniform(0., 1.0) < 0.35:
        mask_value = 0
    img_patch = img[y0:y1, x0:x1, :].copy()
    mask = np.zeros_like(img_patch)
    mask[:, :, :] = mask_value
    fuse_mask = cv2.addWeighted(mask, alpha, img_patch, 1. - alpha, 0)

    img[y0:y1, x0:x1, :] = fuse_mask[:, :, :]

    mask_corner = [x0, y0, x1, y1]

    return img.astype(np.uint8), mask_corner


def apply_patch(img: np.ndarray, patch: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    h, w, c = img.shape
    patch_h, patch_w, _ = patch.shape
    patch_w = min(max(0., patch_w), w)
    patch_h = min(max(0., patch_h), h)
    x0 = np.random.randint(0, w - patch_w + 1)
    y0 = np.random.randint(0, h - patch_h + 1)
    x1, y1 = int(x0 + patch_w), int(y0 + patch_h)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    img[y0:y1, x0:x1, :] = patch[:, :, :]
    patch_corner = [x0, y0, x1, y1]

    return img.astype(np.uint8), patch_corner


def apply_patch_with_alpha(img: np.ndarray, patch: np.ndarray, alpha: float = 0.5) -> Tuple[np.ndarray, List[int]]:
    h, w, c = img.shape
    patch_h, patch_w, _ = patch.shape
    patch_w = min(max(0., patch_w), w)
    patch_h = min(max(0., patch_h), h)
    x0 = np.random.randint(0, w - patch_w + 1)
    y0 = np.random.randint(0, h - patch_h + 1)
    x1, y1 = int(x0 + patch_w), int(y0 + patch_h)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, w), min(y1, h)
    img_patch = img[y0:y1, x0:x1, :].copy()
    fuse_patch = cv2.addWeighted(patch, alpha, img_patch, 1. - alpha, 0)
    img[y0:y1, x0:x1, :] = fuse_patch[:, :, :]
    patch_corner = [x0, y0, x1, y1]

    return img.astype(np.uint8), patch_corner


def apply_background(img: np.ndarray, background: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w, c = img.shape
    b_h, b_w, _ = background.shape
    if b_h <= h or b_w <= w:
        background = cv2.resize(background, (int(w * 1.2), int(h * 1.2)))
    b_h, b_w, _ = background.shape

    im_w = min(max(0., w), b_w)
    im_h = min(max(0., h), b_h)
    x0 = np.random.randint(0, b_w - im_w + 1)
    y0 = np.random.randint(0, b_h - im_h + 1)
    x1, y1 = int(x0 + im_w), int(y0 + im_h)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, b_w), min(y1, b_h)
    background[y0:y1, x0:x1, :] = img[:, :, :]
    # need adjust landmarks
    landmarks += np.array([x0, y0])

    return background.astype(np.uint8), landmarks.astype(np.float32)


def apply_background_with_alpha(img: np.ndarray, background: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h, w, c = img.shape
    b_h, b_w, _ = background.shape
    if b_h != h or b_w != w:
        background = cv2.resize(background, (w, h))

    img = cv2.addWeighted(background, alpha, img, 1. - alpha, 0)

    return img.astype(np.uint8)


def select_patch(patch_h: int = 32, patch_w: int = 32, patches_paths: List[str] = ()) -> Union[np.ndarray, None]:
    patch_path = np.random.choice(patches_paths)
    patch_img = cv2.imread(patch_path)
    if patch_img is None:
        return None
    h, w, _ = patch_img.shape
    if h <= patch_h or w <= patch_w:
        patch = cv2.resize(patch_img, (patch_w, patch_h))
        return patch
    x1 = np.random.randint(0, w - patch_w + 1)
    y1 = np.random.randint(0, h - patch_h + 1)
    x2 = x1 + patch_w
    y2 = y1 + patch_h
    patch = patch_img[y1:y2, x1:x2, :]

    return patch


def select_background(img_h: int = 128, img_w: int = 128, background_paths: List[str] = ()) -> Union[np.ndarray, None]:
    background_path = np.random.choice(background_paths)
    background_img = cv2.imread(background_path)
    if background_img is None:
        return None
    h, w, _ = background_img.shape
    if h <= img_h or w <= img_w:
        background = cv2.resize(background_img, (img_w, img_h))
        return background
    # random start
    x1 = np.random.randint(0, w - img_w + 1)
    y1 = np.random.randint(0, h - img_h + 1)
    # random end
    nw = np.random.randint(img_w // 2, img_w)
    nh = np.random.randint(img_h // 2, img_h)
    x2 = x1 + nw
    y2 = y1 + nh
    background = background_img[y1:y2, x1:x2, :]

    if nh != img_h or nw != img_w:
        background = cv2.resize(background, (img_w, img_h))
        return background

    return background


def read_image_files(image_dirs: List[str]) -> List[str]:
    image_paths = []

    for d in image_dirs:
        if os.path.exists(d):

            files = [
                x for x in os.listdir(d) if
                any((x.lower().endswith("jpeg"),
                     x.lower().endswith("jpg"),
                     x.lower().endswith("png")))
            ]
            paths = [os.path.join(d, x) for x in files]
            image_paths.extend(paths)

    return image_paths


def get_corners(bboxes: np.ndarray) -> np.ndarray:
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4` which (x1,y1)/(x4,y4) indicates top-left/bottom-right

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners: np.ndarray, angle: float, cx: int, cy: int, h: int, w: int) -> np.ndarray:
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4` which (x1,y1)/(x4,y4) indicates top-left/bottom-right
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners: np.ndarray) -> np.ndarray:
    """Get an enclosing box for ratated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4` which (x1,y1)/(x4,y4) indicates top-left/bottom-right

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2` which (x1,y1)/(x2,y2) indicates top-left/bottom-right

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def letterbox_image(img: np.ndarray, inp_dim: Union[int, Tuple[int, int]]) -> np.ndarray:
    """resize image with unchanged aspect ratio using padding

    Parameters
    ----------

    img : numpy.ndarray
        Image

    inp_dim: tuple(int)
        shape of the reszied image (w,h)

    Returns
    -------

    numpy.ndarray:
        Resized image

    """

    if not isinstance(img, np.ndarray):
        raise ValueError('img must an numpy.ndarray.')

    if type(inp_dim) != tuple:
        if type(inp_dim) == int:
            inp_dim = (inp_dim, inp_dim)
        else:
            raise ValueError('inp_dim: tuple(int)')

    assert img.ndim == 3, 'insure (h,w,1) for gray and (h,w,3) for BGR, ' \
                          'for an input of cv2.resize(path, flags=1) with ' \
                          'shape (h,w), you have to expand it to (h,w,1)'

    img_h, img_w, img_c = img.shape
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))

    # for opencv-python(cv2), the input (h,w,1) of cv2.resize function will
    # return (new_h, new_h) shape with no channel dim. So, to maintain interface
    # consistency, we have made some adjust.
    resized_image = cv2.resize(img, (new_w, new_h))

    if resized_image.ndim == 2 and img_c == 1:
        resized_image = np.expand_dims(resized_image, axis=2).astype(np.uint8)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0, dtype=np.uint8)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h,
    (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas.astype(np.uint8)


def letterbox_image_v2(img: np.ndarray, inp_dim: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
     resize image with changed aspect ratio.
    :param img:
    :param inp_dim: (w, h)
    :return:
    """
    if not isinstance(img, np.ndarray):
        raise ValueError('img must an numpy.ndarray.')

    if type(inp_dim) != tuple:
        if type(inp_dim) == int:
            inp_dim = (inp_dim, inp_dim)
        else:
            raise ValueError('inp_dim: tuple(int)')

    assert img.ndim == 3, 'insure (h,w,1) for gray and (h,w,3) for BGR, ' \
                          'for an input of cv2.resize(path, flags=1) with ' \
                          'shape (h,w), you have to expand it to (h,w,1)'

    w, h = inp_dim
    img_h, img_w, img_c = img.shape

    # for opencv-python(cv2), the input (h,w,1) of cv2.resize function will
    # return (new_h, new_h) shape with no channel dim. So, to maintain interface
    # consistency, we have made some adjust.
    resized_image = cv2.resize(img, (w, h))

    if resized_image.ndim == 2 and img_c == 1:
        resized_image = np.expand_dims(resized_image, axis=2).astype(np.uint8)

    return resized_image.astype(np.uint8)


class Helper(object):
    """
        Transform or reverse a landmark key-point to bbox,
        then we can augment these landmarks using BBox-like method.
        The format of single landmark must be (x1,y1)
    """
    VISUAL_W = 3
    VISUAL_H = 3

    @classmethod
    def to_bboxes(
            cls,
            landmarks: np.ndarray = None,
            visual_w: Optional[int] = None,
            visual_h: Optional[int] = None,
            img_w: Optional[Union[float, int]] = None,
            img_h: Optional[Union[float, int]] = None
    ) -> np.ndarray:
        """transform the landmarks to bbox which top-left
        is original landmark key-point. bbox's format is (x1,y1,x2,y2,class),
        which col 5 is visual class for the purpose to reuse BBox-like method.
        The visual class keep zero values all the way and will dump after data augment.

        :param landmarks: landmarks an instance of 2-d numpy.ndarray [2-d]
        :param visual_w: visual width for bbox. [int/float]
        :param visual_h: visual height for bbox. [int/float]
        :param img_w: true width of img for boundary checking. [int/float]
        :param img_h: true height of img for boundary checking. [int/float]
        :return:
        """
        if landmarks is None:
            raise ValueError('landmarks can not be NoneTye.')
        if not isinstance(landmarks, np.ndarray):
            raise ValueError('landmarks an instance of 2-d numpy.ndarray')
        if len(landmarks.shape) != 2:
            raise ValueError('landmarks an instance of 2-d numpy.ndarray')
        if not visual_w:
            visual_w = cls.VISUAL_W
        if not visual_h:
            visual_h = cls.VISUAL_H

        if visual_w > 10 or visual_h > 10:
            raise NotImplementedError('visual h/w too big, should 2=<visual<=10')

        num_landmarks = landmarks.shape[0]
        bboxes = np.zeros(shape=(num_landmarks, 5))
        bboxes[:, :2] = landmarks[:, :]  # top-left (x1,y1)
        bboxes[:, 2] = bboxes[:, 0] + visual_w  # x1 + visual_w = x2
        bboxes[:, 3] = bboxes[:, 1] + visual_h  # y1 + visual_h = y2

        # visual w/h should be small in order to avoid OutOfRangeERROR.
        if img_w and img_h:
            for i in range(num_landmarks):
                x2, y2 = bboxes[i, [2, 3]]
                if x2 > img_w:
                    bboxes[i, 2] = img_w
                if y2 > img_h:
                    bboxes[i, 3] = img_h
        return bboxes

    @classmethod
    def to_landmarks(
            cls,
            bboxes: np.ndarray = None,
            img_w: Optional[Union[float, int]] = None,
            img_h: Optional[Union[float, int]] = None
    ) -> np.ndarray:
        """Extract landmarks from bbox, top-left.
        :param img_h:
        :param img_w:
        :param bboxes: 2-d numpy.ndarray (x1,y1,x2,y2,class)
        :return:
        """
        if not isinstance(bboxes, np.ndarray):
            raise ValueError('bboxes an instance of 2-d numpy.ndarray')
        if len(bboxes.shape) != 2:
            raise ValueError('bboxes an instance of 2-d numpy.ndarray')
        landmarks = bboxes[:, :2]
        # bounder check
        if img_h and img_w:
            landmarks[:, 0] = np.minimum(np.maximum(0, landmarks[:, 0]), img_w)
            landmarks[:, 1] = np.minimum(np.maximum(0, landmarks[:, 1]), img_h)
        else:
            landmarks[:, 0] = np.maximum(0, landmarks[:, 0])
            landmarks[:, 1] = np.maximum(0, landmarks[:, 1])

        return landmarks  # retain x1,y1 only.


class Error(Exception):
    pass


# alias
helper = Helper
