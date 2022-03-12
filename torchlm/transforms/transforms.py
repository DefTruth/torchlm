import os
import cv2
import math
import torch
import random
import warnings
import numpy as np
import torchvision
from torch import Tensor
from pathlib import Path
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, List, Optional, Callable, Any

from . import _functional as F
from ._autodtypes import (
    Image_InOutput_Type,
    Landmarks_InOutput_Type,
    AutoDtypeEnum,
    autodtype,
    set_autodtype_logging
)

_Have_Albumentations = False

try:
    import albumentations
    _Have_Albumentations = True
except:
    _Have_Albumentations = False


__all__ = [
    "LandmarksCompose",
    "LandmarksNormalize",
    "LandmarksUnNormalize",
    "LandmarksToTensor",
    "LandmarksToNumpy",
    "LandmarksResize",
    "LandmarksClip",
    "LandmarksAlign",
    "LandmarksRandomAlign",
    "LandmarksRandomCenterCrop",
    "LandmarksRandomHorizontalFlip",
    "LandmarksHorizontalFlip",
    "LandmarksRandomScale",
    "LandmarksRandomTranslate",
    "LandmarksRandomRotate",
    "LandmarksRandomShear",
    "LandmarksRandomHSV",
    "LandmarksRandomBlur",
    "LandmarksRandomBrightness",
    "LandmarksRandomMask",
    "LandmarksRandomPatches",
    "LandmarksRandomBackground",
    "LandmarksRandomMaskMixUp",
    "LandmarksRandomPatchesMixUp",
    "LandmarksRandomBackgroundMixUp",
    "BindAlbumentationsTransform",
    "BindTorchVisionTransform",
    "BindArrayCallable",
    "BindTensorCallable",
    "BindEnum",
    "bind",
    "set_transforms_logging",
    "set_transforms_debug",
    "set_autodtype_logging",
    "albumentations_is_available",
    "build_default_transform"
]

TransformLoggingMode: bool = False
TransformDebugMode: bool = False


def set_transforms_logging(logging: bool = False):
    global TransformLoggingMode
    TransformLoggingMode = logging


def set_transforms_debug(debug: bool = False):
    global TransformDebugMode
    TransformDebugMode = debug

def albumentations_is_available() -> bool:
    global _Have_Albumentations
    return _Have_Albumentations

def _transforms_api_logging(info: str):
    global TransformLoggingMode
    if TransformLoggingMode: print(info)


def _transforms_api_debug(error: Exception):
    global TransformDebugMode
    if TransformDebugMode: raise error


class LandmarksTransform(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        # affine records, for future use.
        self.rotate: float = 0.
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.trans_x: float = 0.
        self.trans_y: float = 0.
        self.flag: bool = False
        self.is_numpy: bool = True

    @abstractmethod
    def __call__(
            self,
            img: Image_InOutput_Type,
            landmarks: Landmarks_InOutput_Type,
            **kwargs
    ) -> Tuple[Image_InOutput_Type, Landmarks_InOutput_Type]:
        """
        :param img: np.ndarray | Tensor, H x W x C
        :param landmarks: np.ndarray | Tensor, shape (?, 2), the format is (x1,y1) for each row.
        :param kwargs: To be compatible with different parameters
        :return:
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__

    def apply_affine_to(
            self,
            other_landmarks: Landmarks_InOutput_Type,
            scale: Optional[bool] = True,
            translate: Optional[bool] = True,
            rotate: Optional[bool] = False,
            **kwargs
    ) -> Landmarks_InOutput_Type:
        # For future use, don't use now!
        _ = kwargs  # un-used
        if translate:
            other_landmarks[:, 0] -= self.trans_x
            other_landmarks[:, 1] -= self.trans_y
        if scale:
            other_landmarks[:, 0] *= self.scale_x
            other_landmarks[:, 1] *= self.scale_y
        if rotate:
            # TODO: add rotation
            pass
        return other_landmarks

    def clear_affine(self):
        self.rotate = 0.
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.trans_x = 0.
        self.trans_y = 0.
        self.flag = False


def _transforms_api_assert(self: LandmarksTransform, cond: bool, info: str = None):
    if cond:
        self.flag = False  # flag is a reference of some specific flag
        if info is None:
            info = f"{self}() missing landmarks"
        raise F.Error(info)


TorchVision_Transform_Type = torch.nn.Module


# Bind TorchVision Transforms
class BindTorchVisionTransform(LandmarksTransform):
    # torchvision >= 0.9.0
    _Supported_Image_Only_Transform_Set: Tuple = (
        torchvision.transforms.Normalize,
        torchvision.transforms.ColorJitter,
        torchvision.transforms.Grayscale,
        torchvision.transforms.RandomGrayscale,
        torchvision.transforms.RandomErasing,
        torchvision.transforms.GaussianBlur,
        torchvision.transforms.RandomInvert,
        torchvision.transforms.RandomPosterize,
        torchvision.transforms.RandomSolarize,
        torchvision.transforms.RandomAdjustSharpness,
        torchvision.transforms.RandomAutocontrast,
        torchvision.transforms.RandomEqualize
    )

    def __init__(
            self,
            transform: TorchVision_Transform_Type,
            prob: float = 1.0
    ):
        super(BindTorchVisionTransform, self).__init__()
        assert isinstance(
            transform,
            self._Supported_Image_Only_Transform_Set
        ), f"Only supported image only transform for" \
           f" torchvision:\n {self._Supported_Image_Only_Transform_Set}"
        self.prob = prob
        self.transform_internal = transform

    @autodtype(AutoDtypeEnum.Tensor_InOut)
    def __call__(
            self,
            img: Tensor,
            landmarks: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Image only transform from torchvision,
        # just let the landmarks unchanged. Note (3,H,W)
        # is need for torchvision
        try:
            if np.random.uniform(0., 1.0) > self.prob:
                self.clear_affine()
                return img, landmarks

            chw = img.size()[0] == 3
            if not chw:
                img = img.permute((2, 0, 1)).contiguous()

            img, landmarks = self.transform_internal(img), landmarks

            # permute back
            if not chw:
                img = img.permute((1, 2, 0)).contiguous()

            self.flag = True
            return img, landmarks
        except:
            self.flag = False
            return img, landmarks

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
               + self.transform_internal.__class__.__name__ \
               + '())'


try:
    import albumentations

    Albumentations_Transform_Hook_Type = Union[
        albumentations.ImageOnlyTransform,
        albumentations.DualTransform
    ]
    _Have_Albumentations = True
except Exception:
    Albumentations_Transform_Hook_Type = Any
    _Have_Albumentations = False


# TODO: 移除对于albumentations的硬依赖
class BindAlbumentationsTransform(LandmarksTransform):
    # albumentations >= v 1.1.0
    global _Have_Albumentations

    try:
        import albumentations

        _Have_Albumentations = True

        _Supported_Image_Only_Transform_Set: Tuple = (
            albumentations.Blur,
            albumentations.CLAHE,
            albumentations.ChannelDropout,
            albumentations.ChannelShuffle,
            albumentations.ColorJitter,
            albumentations.Downscale,
            albumentations.Emboss,
            albumentations.Equalize,
            albumentations.FDA,
            albumentations.FancyPCA,
            albumentations.FromFloat,
            albumentations.GaussNoise,
            albumentations.GaussianBlur,
            albumentations.GlassBlur,
            albumentations.HistogramMatching,
            albumentations.HueSaturationValue,
            albumentations.ISONoise,
            albumentations.ImageCompression,
            albumentations.InvertImg,
            albumentations.MedianBlur,
            albumentations.MotionBlur,
            albumentations.Normalize,
            albumentations.PixelDistributionAdaptation,
            albumentations.Posterize,
            albumentations.RGBShift,
            albumentations.RandomBrightnessContrast,
            albumentations.RandomFog,
            albumentations.RandomGamma,
            albumentations.RandomRain,
            albumentations.RandomShadow,
            albumentations.RandomSnow,
            albumentations.RandomSunFlare,
            albumentations.RandomToneCurve,
            albumentations.Sharpen,
            albumentations.Solarize,
            albumentations.Superpixels,
            albumentations.TemplateTransform,
            albumentations.ToFloat,
            albumentations.ToGray
        )

        _Supported_Dual_Transform_Set: Tuple = (
            albumentations.Affine,
            albumentations.CenterCrop,
            albumentations.CoarseDropout,
            albumentations.Crop,
            albumentations.CropAndPad,
            albumentations.CropNonEmptyMaskIfExists,
            albumentations.Flip,
            albumentations.HorizontalFlip,
            albumentations.Lambda,
            albumentations.LongestMaxSize,
            albumentations.NoOp,
            albumentations.PadIfNeeded,
            albumentations.Perspective,
            albumentations.PiecewiseAffine,
            albumentations.RandomCrop,
            albumentations.RandomCropNearBBox,
            albumentations.RandomGridShuffle,
            albumentations.RandomResizedCrop,
            albumentations.RandomRotate90,
            albumentations.RandomScale,
            albumentations.RandomSizedCrop,
            albumentations.Resize,
            albumentations.Rotate,
            albumentations.SafeRotate,
            albumentations.ShiftScaleRotate,
            albumentations.SmallestMaxSize,
            albumentations.Transpose,
            albumentations.VerticalFlip
        )

    except ImportError as e:
        warnings.warn(f"Can not found albumentations!: {e}")
        _Have_Albumentations = False
    except Exception as e1:
        _Have_Albumentations = False

    def __init__(
            self,
            transform: Albumentations_Transform_Hook_Type,
            prob: float = 1.0
    ):
        super(BindAlbumentationsTransform, self).__init__()
        # wrapper transform with a simple albumentations
        # Compose to fix KeyPoints format (xy), not support
        # label_fields now. Because we rarely use it, so,
        # I decided not to support it to keep the
        # interface simple.
        assert any((
            isinstance(transform, self._Supported_Image_Only_Transform_Set),
            isinstance(transform, self._Supported_Dual_Transform_Set)
        )), f"The transform from albumentations must be one of:" \
            f"\n{self._Supported_Image_Only_Transform_Set}, " \
            f"\n{self._Supported_Dual_Transform_Set}"
        self.prob = prob
        self.transform_internal = transform
        global _Have_Albumentations

        try:
            import albumentations
            self.compose_internal = albumentations.Compose(
                transforms=[transform],
                keypoint_params=albumentations.KeypointParams(
                    format="xy",
                    remove_invisible=True
                )  # no label_fields now.
            )
            _Have_Albumentations = True

        except Exception:
            self.compose_internal = None
            _Have_Albumentations = False
            warnings.warn(f"{self}: Can not found albumentations, "
                          "ignore this binding!")

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # The landmarks format for albumentations should be a list of lists(tuple)
        # in xy format by default. Such as:
        # keypoints = [
        #      (264, 203),
        #      (86, 88),
        #      (254, 160),
        #      (193, 103),
        #      (65, 341)
        # ]
        # So, we have to convert the np.ndarray input to list first and then wrap back
        # to np.ndarray after the albumentations transformation done.
        assert all((
            isinstance(img, np.ndarray),
            isinstance(landmarks, np.ndarray),
            landmarks.ndim >= 2
        )), "Inputs must be np.ndarray and the ndim of " \
            "landmarks should >= 2!"

        if not albumentations_is_available() or self.compose_internal is None:
            _transforms_api_logging(f"{self}: Can not found albumentations,"
                                    " skip this transform!")
            self.flag = False
            return img.astype(np.uint8), landmarks.astype(np.float32)

        keypoints = landmarks[:, :2].tolist()  # (x, y)
        kps_num = len(keypoints)

        try:
            # random skip
            if np.random.uniform(0., 1.0) > self.prob:
                self.clear_affine()
                return img.astype(np.uint8), landmarks.astype(np.float32)

            transformed = self.compose_internal(image=img, keypoints=keypoints)
            trans_img = transformed['image']
            trans_kps = transformed['keypoints']

            if len(trans_kps) == kps_num:
                # wrap back to np.ndarray
                trans_kps = np.array(trans_kps).astype(landmarks.dtype)
                trans_kps = trans_kps.reshape(kps_num, 2)
                landmarks[:, :2] = trans_kps
                img = trans_img
                self.flag = True
            else:
                self.flag = False
                _transforms_api_logging(
                    f"{self}() Missing landmarks after transform, "
                    f"expect {kps_num} but got {len(trans_kps)},"
                    f"skip this transform"
                )
            # changed nothings if any kps been outside
            return img.astype(np.uint8), landmarks.astype(np.float32)
        except:
            self.flag = False
            return img.astype(np.uint8), landmarks.astype(np.float32)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
               + self.transform_internal.__class__.__name__ \
               + '())'


Callable_Array_Func_Type = Union[
    Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]],
    Callable[[np.ndarray, np.ndarray, Any], Tuple[np.ndarray, np.ndarray]]
]

Callable_Tensor_Func_Type = Union[
    Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
    Callable[[Tensor, Tensor, Any], Tuple[Tensor, Tensor]]
]


class BindArrayCallable(LandmarksTransform):

    def __init__(
            self,
            call_func: Callable_Array_Func_Type,
            prob: float = 1.0
    ):
        super(BindArrayCallable, self).__init__()
        if not callable(call_func):
            raise TypeError(
                "Argument call_func should be callable, "
                "got {}".format(repr(type(call_func).__name__))
            )
        self.prob = prob
        self.call_func: Callable_Array_Func_Type = call_func

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray,
            **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # random skip
            if np.random.uniform(0., 1.0) > self.prob:
                self.clear_affine()
                return img.astype(np.int32), landmarks.astype(np.float32)

            kps_num = landmarks.shape[0]
            trans_img, trans_kps = self.call_func(img, landmarks, **kwargs)
            if trans_kps.shape[0] == kps_num:
                img = trans_img
                landmarks = trans_kps
                self.flag = True
            else:
                self.flag = False
                _transforms_api_logging(
                    f"{self}() Missing landmarks after transform, "
                    f"expect {kps_num} but got {trans_kps.shape[0]},"
                    f"skip this transform"
                )
            # changed nothings if any kps has been outside
            return img.astype(np.int32), landmarks.astype(np.float32)

        except:

            self.flag = False
            return img.astype(np.int32), landmarks.astype(np.float32)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
               + self.call_func.__name__ \
               + '())'


class BindTensorCallable(LandmarksTransform):

    def __init__(
            self,
            call_func: Callable_Tensor_Func_Type,
            prob: float = 1.0
    ):
        super(BindTensorCallable, self).__init__()
        if not callable(call_func):
            raise TypeError(
                "Argument call_func should be callable, "
                "got {}".format(repr(type(call_func).__name__))
            )
        self.prob = prob
        self.call_func: Callable_Tensor_Func_Type = call_func

    @autodtype(AutoDtypeEnum.Tensor_InOut)
    def __call__(
            self,
            img: Tensor,
            landmarks: Tensor,
            **kwargs
    ) -> Tuple[Tensor, Tensor]:
        try:
            # random skip
            if np.random.uniform(0., 1.0) > self.prob:
                self.clear_affine()
                return img, landmarks

            kps_num = landmarks.size()[0]
            trans_img, trans_kps = self.call_func(img, landmarks, **kwargs)
            if trans_kps.size()[0] == kps_num:
                img = trans_img
                landmarks = trans_kps
                self.flag = True
            else:
                self.flag = False
                _transforms_api_logging(
                    f"{self}() Missing landmarks after transform, "
                    f"expect {kps_num} but got {trans_kps.size()[0]},"
                    f"skip this transform"
                )
            # changed nothings if any kps has been outside
            return img, landmarks
        except:
            self.flag = False
            return img, landmarks

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
               + self.call_func.__name__ \
               + '())'


Bind_Transform_Or_Callable_Input_Type = Union[
    TorchVision_Transform_Type,
    Albumentations_Transform_Hook_Type,
    Callable_Array_Func_Type,
    Callable_Tensor_Func_Type
]

Bind_Transform_Output_Type = Union[
    BindTorchVisionTransform,
    BindAlbumentationsTransform,
    BindArrayCallable,
    BindTensorCallable
]


class BindEnum:
    Transform: int = 0
    Callable_Array: int = 1
    Callable_Tensor: int = 2


# bind method
def bind(
        transform_or_callable: Bind_Transform_Or_Callable_Input_Type,
        bind_type: int = BindEnum.Transform,
        **kwargs
) -> Bind_Transform_Output_Type:
    """
    :param transform_or_callable: some custom transform from torchvision and albumentations,
           or some custom transform callable functions defined by users.
    :param bind_type: See BindEnum.
    :param kwargs: extra args, such as prob(default 1.0) at bind level to force any transform
           or callable be a random-style.
    """
    global _Have_Albumentations
    try:
        import albumentations
        Albumentations_Transform_Hook_Type_ = (
            albumentations.ImageOnlyTransform,
            albumentations.DualTransform)
        _Have_Albumentations = True
    except:
        Albumentations_Transform_Hook_Type_ = object
        _Have_Albumentations = False

    if bind_type == BindEnum.Transform:
        # bind torchvision transform
        if isinstance(transform_or_callable, TorchVision_Transform_Type):
            return BindTorchVisionTransform(transform_or_callable, **kwargs)
        elif isinstance(transform_or_callable, Albumentations_Transform_Hook_Type_):
            # bind albumentations transform
            return BindAlbumentationsTransform(transform_or_callable, **kwargs)
        else:
            raise TypeError(f"not supported: {transform_or_callable}")
    elif bind_type == BindEnum.Callable_Tensor:
        return BindTensorCallable(transform_or_callable, **kwargs)
    elif bind_type == BindEnum.Callable_Array:
        return BindArrayCallable(transform_or_callable, **kwargs)
    else:
        raise TypeError(f"not supported: {transform_or_callable}")


# Pytorch Style Compose
class LandmarksCompose(object):

    def __init__(
            self,
            transforms: List[LandmarksTransform]
    ):
        self.flags: List[bool] = []
        self.transforms: List[LandmarksTransform] = transforms
        assert self.check, "Wrong! Need LandmarksTransform !" \
                           f"But got {self.__repr__()}"

    @property
    def check(self) -> bool:
        return all([isinstance(_, LandmarksTransform) for _ in self.transforms])

    def __call__(
            self,
            img: Image_InOutput_Type,
            landmarks: Landmarks_InOutput_Type
    ) -> Tuple[Image_InOutput_Type, Landmarks_InOutput_Type]:

        self.flags.clear()  # clear each time
        for t in self.transforms:
            try:
                img, landmarks = t(img, landmarks)
            except Exception as e:
                _transforms_api_logging(f"Error at {t}() Skip, Flag: "
                                        f"{t.flag} Error Info: {e}")
                _transforms_api_debug(e)  # after logging
                continue
            finally:
                _transforms_api_logging(f"{t}() Execution Flag: {t.flag}")
            self.flags.append(t.flag)

        return img, landmarks

    def apply_transform_to(
            self,
            other_img: Image_InOutput_Type,
            other_landmarks: Landmarks_InOutput_Type
    ) -> Tuple[Image_InOutput_Type, Landmarks_InOutput_Type]:
        for t, const_flag in zip(self.transforms, self.flags):
            try:
                if const_flag:
                    other_img, other_landmarks = t(other_img, other_landmarks)
            except Exception as e:
                _transforms_api_logging(f"Error at {t}() Skip, Flag: "
                                        f"{t.flag} Error Info: {e}")
                _transforms_api_debug(e)  # after logging
                continue
            finally:
                _transforms_api_logging(f"{t}() Execution Flag: {t.flag}")
        return other_img, other_landmarks

    def apply_affine_to(
            self,
            other_landmarks: Landmarks_InOutput_Type,
            scale: Optional[bool] = True,
            translate: Optional[bool] = True,
            rotate: Optional[bool] = False,
            **kwargs
    ) -> Landmarks_InOutput_Type:
        for t, const_flag in zip(self.transforms, self.flags):
            try:
                if const_flag:
                    other_landmarks = t.apply_affine_to(
                        other_landmarks=other_landmarks,
                        scale=scale,
                        translate=translate,
                        rotate=rotate,
                        **kwargs
                    )
            except Exception as e:
                _transforms_api_logging(f"Error at {t}() Skip, Flag: "
                                        f"{t.flag} Error Info: {e}")
                _transforms_api_debug(e)  # after logging
                continue
            finally:
                _transforms_api_logging(f"{t}() Execution Flag: {t.flag}")
        return other_landmarks

    def clear_affine(self):
        for t in self.transforms:
            t.clear_affine()
        self.flags.clear()

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class LandmarksNormalize(LandmarksTransform):
    def __init__(
            self,
            mean: Union[float, List[float]] = 127.5,
            std: Union[float, List[float]] = 128.,
            force_norm_before_mean_std: bool = False
    ):
        super(LandmarksNormalize, self).__init__()
        self._mean = mean
        self._std = std
        self._force_norm_before_mean_std = force_norm_before_mean_std
        if not ((isinstance(self._mean, float) and isinstance(self._std, float))
                or (isinstance(self._mean, list) and isinstance(self._std, list))
        ):
            raise ValueError("mean and std should be a float or List[float]")
        if isinstance(self._mean, list) and isinstance(self._std, list):
            assert len(self._mean) == 3 and len(self._std) == 3

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        img = img.astype(np.float32)
        if self._force_norm_before_mean_std:
            img /= 255.

        if isinstance(self._mean, float) and isinstance(self._std, float):
            img = (img - self._mean) / self._std
        else:
            img[:, :, 0] -= self._mean[0]
            img[:, :, 1] -= self._mean[1]
            img[:, :, 2] -= self._mean[2]
            img[:, :, 0] /= self._std[0]
            img[:, :, 1] /= self._std[1]
            img[:, :, 2] /= self._std[2]

        self.flag = True

        return img.astype(np.float32), landmarks.astype(np.float32)


class LandmarksUnNormalize(LandmarksTransform):
    def __init__(
            self,
            mean: Union[float, List[float]] = 127.5,
            std: Union[float, List[float]] = 128.,
            force_denorm_after_mean_std: bool = False
    ):
        super(LandmarksUnNormalize, self).__init__()
        self._mean = mean
        self._std = std
        self._force_denorm_after_mean_std = force_denorm_after_mean_std
        if not ((isinstance(self._mean, float) and isinstance(self._std, float))
                or (isinstance(self._mean, list) and isinstance(self._std, list))
        ):
            raise ValueError("mean and std should be a float or List[float]")
        if isinstance(self._mean, list) and isinstance(self._std, list):
            assert len(self._mean) == 3 and len(self._std) == 3

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # must be float
        img = img.astype(np.float32)
        if isinstance(self._mean, float) and isinstance(self._std, float):
            img = img * self._std + self._mean
        else:
            img[:, :, 0] = img[:, :, 0] * self._std[0] + self._mean[0]
            img[:, :, 1] = img[:, :, 1] * self._std[1] + self._mean[1]
            img[:, :, 2] = img[:, :, 2] * self._std[2] + self._mean[2]

        if self._force_denorm_after_mean_std:
            img *= 255.

        self.flag = True

        return img.astype(np.float32), landmarks.astype(np.float32)


class LandmarksToTensor(LandmarksTransform):
    def __init__(self):
        super(LandmarksToTensor, self).__init__()

    # force array input and don't wrap the output back to array.
    @autodtype(AutoDtypeEnum.Array_In)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[Tensor, Tensor]:
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = img.transpose((2, 0, 1))

        self.flag = True
        return F.to_tensor(img), F.to_tensor(landmarks)


class LandmarksToNumpy(LandmarksTransform):
    def __init__(self):
        super(LandmarksToNumpy, self).__init__()

    # force tensor input and don't wrap the output back to tensor.
    @autodtype(AutoDtypeEnum.Tensor_In)
    def __call__(
            self,
            img: Tensor,
            landmarks: Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # C X H X W -> H X W X C

        self.flag = True
        return F.to_numpy(img).transpose((1, 2, 0)), F.to_numpy(landmarks)


class LandmarksResize(LandmarksTransform):
    """Resize the image in accordance to `image_letter_box` function in darknet
    """

    def __init__(
            self,
            size: Union[Tuple[int, int], int],
            keep_aspect: bool = False
    ):
        super(LandmarksResize, self).__init__()
        if type(size) != tuple:
            if type(size) == int:
                size = (size, size)
            else:
                raise ValueError('size: tuple(int)')

        self._size = size
        self._keep_aspect = keep_aspect

        if self._keep_aspect:
            self._letterbox_image = F.letterbox_image
        else:
            self._letterbox_image = F.letterbox_image_v2

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        w, h = img.shape[1], img.shape[0]  # original shape
        new_img = self._letterbox_image(img.copy().astype(np.uint8), self._size)

        num_landmarks = len(landmarks)
        landmark_bboxes = F.helper.to_bboxes(landmarks)

        if self._keep_aspect:
            scale = min(self._size[1] / h, self._size[0] / w)
            landmark_bboxes[:, :4] *= scale

            new_w = scale * w
            new_h = scale * h
            # inp_dim = self.inp_dim
            inp_dim_w, inp_dim_h = self._size

            del_h = (inp_dim_h - new_h) // 2
            del_w = (inp_dim_w - new_w) // 2

            add_matrix = np.array([[del_w, del_h, del_w, del_h]], dtype=landmark_bboxes.dtype)

            landmark_bboxes[:, :4] += add_matrix
            # refine according to new shape
            new_landmarks = F.helper.to_landmarks(landmark_bboxes,
                                                  img_w=new_img.shape[1],
                                                  img_h=new_img.shape[0])

            self.scale_x = scale
            self.scale_y = scale
        else:
            scale_y, scale_x = self._size[1] / h, self._size[0] / w
            landmark_bboxes[:, (0, 2)] *= scale_x
            landmark_bboxes[:, (1, 3)] *= scale_y

            # refine according to new shape
            new_landmarks = F.helper.to_landmarks(landmark_bboxes,
                                                  img_w=new_img.shape[1],
                                                  img_h=new_img.shape[0])
            self.scale_x = scale_x
            self.scale_y = scale_y

        _transforms_api_assert(self, len(new_landmarks) != num_landmarks,
                               f"{self}() have {num_landmarks} input "
                               f"landmarks, but got {len(new_landmarks)} "
                               f"output landmarks!")

        self.flag = True

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksClip(LandmarksTransform):
    """Get the five sense organs clipping image and landmarks"""

    def __init__(
            self,
            width_pad: float = 0.2,
            height_pad: float = 0.2,
            target_size: Union[Tuple[int, int], int] = None,
            **kwargs
    ):
        """Clip enclosure box according the given landmarks.
        :param width_pad: the padding ration to extend the width of clipped box.
        :param height_pad: the padding ration to extend the height of clipped box.
        :param target_size: target size for resize operation.
        """
        super(LandmarksClip, self).__init__()
        self._width_pad = width_pad
        self._height_pad = height_pad
        self._target_size = target_size

        if self._target_size is not None:
            if isinstance(self._target_size, int) or isinstance(self._target_size, tuple):
                if isinstance(self._target_size, int):
                    self._target_size = (self._target_size, self._target_size)
                if isinstance(self._target_size, tuple):
                    assert len(self._target_size) == 2
            else:
                raise ValueError('wrong target size, should be (w,h)')

            self._resize_op = LandmarksResize(self._target_size, **kwargs)
        else:
            self._resize_op = None

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param img:
        :param landmarks: [num, 2]
        :return:
        """
        x_min = np.min(landmarks[:, 0])
        x_max = np.max(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        y_max = np.max(landmarks[:, 1])

        h, w = img.shape[0], img.shape[1]

        lh, lw = abs(y_max - y_min), abs(x_max - x_min)

        left = np.maximum(int(x_min) - int(lw * self._width_pad), 0)
        right = np.minimum(int(x_max) + int(lw * self._width_pad), w)
        top = np.maximum(int(y_min) - int(lh * self._height_pad), 0)
        bottom = np.minimum(int(y_max) + int(lh * self._height_pad), h)

        new_img = img[top:bottom, left:right, :].copy()
        new_landmarks = landmarks.copy()

        new_landmarks[:, 0] -= left
        new_landmarks[:, 1] -= top

        if self._resize_op is not None:
            new_img, new_landmarks, _ = self._resize_op(new_img, new_landmarks)
            self.scale_x = self._resize_op.scale_x
            self.scale_y = self._resize_op.scale_y

        self.flag = True

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksAlign(LandmarksTransform):
    """Get aligned image and landmarks"""

    def __init__(
            self,
            eyes_index: Union[Tuple[int, int], List[int]] = None
    ):
        """
        :param eyes_index: 2 indexes in landmarks, indicates left and right eye center.
        """
        super(LandmarksAlign, self).__init__()
        if eyes_index is None or len(eyes_index) != 2:
            raise ValueError("2 indexes in landmarks, "
                             "which indicates left and right eye center.")

        self._eyes_index = eyes_index

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        left_eye = landmarks[self._eyes_index[0]]
        right_eye = landmarks[self._eyes_index[1]]
        dx = (right_eye[0] - left_eye[0])
        dy = (right_eye[1] - left_eye[1])
        angle = math.atan2(dy, dx) * 180 / math.pi  # calc angle

        num_landmarks = len(landmarks)
        landmark_bboxes = F.helper.to_bboxes(landmarks)

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        new_img = F.rotate_im(img.copy(), angle)

        landmarks_corners = F.get_corners(landmark_bboxes)

        landmarks_corners = np.hstack((landmarks_corners, landmark_bboxes[:, 4:]))

        landmarks_corners[:, :8] = F.rotate_box(landmarks_corners[:, :8], angle, cx, cy, h, w)

        new_landmark_bbox = np.zeros_like(landmark_bboxes)
        new_landmark_bbox[:, [0, 1]] = landmarks_corners[:, [0, 1]]
        new_landmark_bbox[:, [2, 3]] = landmarks_corners[:, [6, 7]]

        scale_factor_x = new_img.shape[1] / w

        scale_factor_y = new_img.shape[0] / h

        new_img = cv2.resize(new_img, (w, h))

        new_landmark_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

        landmark_bboxes = new_landmark_bbox[:, :]

        landmark_bboxes = F.clip_box(landmark_bboxes, [0, 0, w, h], 0.0025)
        # refine according to new shape
        new_landmarks = F.helper.to_landmarks(landmark_bboxes,
                                              img_w=new_img.shape[1],
                                              img_h=new_img.shape[0])

        self.scale_x = (1 / scale_factor_x)
        self.scale_y = (1 / scale_factor_y)

        _transforms_api_assert(self, len(new_landmarks) != num_landmarks,
                               f"{self}() have {num_landmarks} input "
                               f"landmarks, but got {len(new_landmarks)} "
                               f"output landmarks!")

        self.flag = True

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksRandomAlign(LandmarksTransform):

    def __init__(
            self,
            eyes_index: Union[Tuple[int, int], List[int]] = None,
            prob: float = 0.5
    ):
        """
        :param eyes_index: 2 indexes in landmarks, indicates left and right eye center.
        """
        super(LandmarksRandomAlign, self).__init__()
        if eyes_index is None or len(eyes_index) != 2:
            raise ValueError("2 indexes in landmarks, "
                             "which indicates left and right eye center.")

        self._eyes_index = eyes_index
        self._prob = prob

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.0) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        left_eye = landmarks[self._eyes_index[0]]
        right_eye = landmarks[self._eyes_index[1]]
        dx = (right_eye[0] - left_eye[0])
        dy = (right_eye[1] - left_eye[1])
        angle = math.atan2(dy, dx) * 180 / math.pi  # calc angle

        num_landmarks = len(landmarks)
        landmark_bboxes = F.helper.to_bboxes(landmarks)

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        new_img = F.rotate_im(img.copy(), angle)

        landmarks_corners = F.get_corners(landmark_bboxes)

        landmarks_corners = np.hstack((landmarks_corners, landmark_bboxes[:, 4:]))

        landmarks_corners[:, :8] = F.rotate_box(landmarks_corners[:, :8], angle, cx, cy, h, w)

        new_landmark_bbox = np.zeros_like(landmark_bboxes)
        new_landmark_bbox[:, [0, 1]] = landmarks_corners[:, [0, 1]]
        new_landmark_bbox[:, [2, 3]] = landmarks_corners[:, [6, 7]]

        scale_factor_x = new_img.shape[1] / w

        scale_factor_y = new_img.shape[0] / h

        new_img = cv2.resize(new_img, (w, h))

        new_landmark_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

        landmark_bboxes = new_landmark_bbox[:, :]

        landmark_bboxes = F.clip_box(landmark_bboxes, [0, 0, w, h], 0.0025)
        # refine according to new shape
        new_landmarks = F.helper.to_landmarks(landmark_bboxes,
                                              img_w=new_img.shape[1],
                                              img_h=new_img.shape[0])

        self.scale_x = (1 / scale_factor_x)
        self.scale_y = (1 / scale_factor_y)

        _transforms_api_assert(self, len(new_landmarks) != num_landmarks,
                               f"{self}() have {num_landmarks} input "
                               f"landmarks, but got {len(new_landmarks)} "
                               f"output landmarks!")

        self.flag = True

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksRandomCenterCrop(LandmarksTransform):
    def __init__(
            self,
            width_range: Tuple[float, float] = (0.8, 1.0),
            height_range: Tuple[float, float] = (0.8, 1.0),
            prob: float = 0.5
    ):
        super(LandmarksRandomCenterCrop, self).__init__()
        self._width_range = width_range
        self._height_range = height_range
        self._prob = prob

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        height, width, _ = img.shape
        cx, cy = int(width / 2), int(height / 2)

        x_min = np.min(landmarks[:, 0])
        x_max = np.max(landmarks[:, 0])
        y_min = np.min(landmarks[:, 1])
        y_max = np.max(landmarks[:, 1])

        height_ratio = np.random.uniform(self._height_range[0], self._height_range[1], size=1)
        width_ratio = np.random.uniform(self._width_range[0], self._width_range[1], size=1)
        top_height_ratio = np.random.uniform(0.4, 0.6)
        left_width_ratio = np.random.uniform(0.4, 0.6)

        crop_height = min(int(height_ratio * height + 1), height)
        crop_width = min(int(width_ratio * width + 1), width)

        left_width_offset = crop_width * left_width_ratio
        right_width_offset = crop_width - left_width_offset
        top_height_offset = crop_height * top_height_ratio
        bottom_height_offset = crop_height - top_height_offset

        x1 = max(0, int(cx - left_width_offset + 1))
        x2 = min(width, int(cx + right_width_offset))
        y1 = max(0, int(cy - top_height_offset + 1))
        y2 = min(height, int(cy + bottom_height_offset))

        x1 = max(int(min(x1, x_min)), 0)
        x2 = min(int(max(x2, x_max + 1)), width)
        y1 = max(int(min(y1, y_min)), 0)
        y2 = min(int(max(y2, y_max + 1)), height)

        crop_width = abs(x2 - x1)
        crop_height = abs(y2 - y1)

        new_landmarks = landmarks.copy()
        new_landmarks[:, 0] -= x1
        new_landmarks[:, 1] -= y1

        # border check
        lx_min = np.min(new_landmarks[:, 0])
        lx_max = np.max(new_landmarks[:, 0])
        ly_min = np.min(new_landmarks[:, 1])
        ly_max = np.max(new_landmarks[:, 1])

        if any((lx_min < 0., lx_max > crop_width, ly_min < 0., ly_max > crop_height)):
            self.flag = False
            return img.astype(np.uint8), landmarks

        new_img = img[y1:y2, x1:x2, :].copy()

        self.flag = True

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksRandomHorizontalFlip(LandmarksTransform):

    def __init__(
            self,
            prob: float = 0.5
    ):
        super(LandmarksRandomHorizontalFlip, self).__init__()
        self._prob = prob

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param img: original img contain in 3-d numpy.ndarray. [HxWxC]
        :param landmarks: 2-d numpy.ndarray [num_landmarks, 2] (x1, y1)
        :return:
        """
        if np.random.uniform(0., 1.) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        h, w, _ = img.shape
        cx = float(w / 2)
        new_img = img[:, ::-1, :].copy()
        # TODO: swap the index according the transform
        # if x1<cx，then x1_flip=x1+2*(cx-x1), else if x1>cx，
        # then x1_flip=x1-2*(x1-cx)=x1+2*(cx-x1)
        new_landmarks = landmarks.copy()
        new_landmarks[:, 0] += 2 * (cx - new_landmarks[:, 0])

        self.flag = True

        _transforms_api_logging(
            "WARNING!!!: HorizontalFlip augmentation mirrors the input image.\n "
            "When you apply that augmentation to keypoints that mark the\n "
            "side of body parts (left or right), those keypoints will point\n "
            "to the wrong side (since left on the mirrored image becomes right).\n"
            " So when you are creating an augmentation pipeline look carefully\n"
            "which augmentations could be applied to the input data. Also see:\n "
            "https://albumentations.ai/docs/getting_started/keypoints_augmentation/"
        )

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksHorizontalFlip(LandmarksTransform):

    def __init__(self):
        super(LandmarksHorizontalFlip, self).__init__()

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w, _ = img.shape
        cx = float(w / 2)
        new_img = img[:, ::-1, :].copy()
        # TODO: swap the index according the transform
        # if x1<cx，then x1_flip=x1+2*(cx-x1), else if x1>cx，
        # then x1_flip=x1-2*(x1-cx)=x1+2*(cx-x1)
        new_landmarks = landmarks.copy()
        new_landmarks[:, 0] += 2 * (cx - new_landmarks[:, 0])

        self.flag = True

        _transforms_api_logging(
            "WARNING!!!: HorizontalFlip augmentation mirrors the input image.\n "
            "When you apply that augmentation to keypoints that mark the\n"
            "side of body parts (left or right), those keypoints will point\n "
            "to the wrong side (since left on the mirrored image becomes right).\n"
            " So when you are creating an augmentation pipeline look carefully\n"
            "which augmentations could be applied to the input data. Also see:\n "
            "https://albumentations.ai/docs/getting_started/keypoints_augmentation/"
        )

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksRandomScale(LandmarksTransform):

    def __init__(
            self,
            scale: Union[Tuple[float, float], float] = 0.4,
            prob: float = 0.5,
            diff: bool = True
    ):
        super(LandmarksRandomScale, self).__init__()
        self._scale = scale
        self._prob = prob

        if isinstance(self._scale, tuple):
            assert len(self._scale) == 2., "Invalid range"
            assert self._scale[0] > -1., "Scale factor can't be less than -1"
            assert self._scale[1] > -1., "Scale factor can't be less than -1"
        elif isinstance(self._scale, float):
            assert self._scale > 0., "Please input a positive float"
            self._scale = (max(-1., -self._scale), self._scale)

        self._diff = diff

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        num_landmarks = len(landmarks)
        new_landmarks = landmarks.copy()

        if self._diff:
            scale_x = random.uniform(*self._scale)
            scale_y = random.uniform(*self._scale)
        else:
            scale_x = random.uniform(*self._scale)
            scale_y = scale_x

        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y
        new_img = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)

        new_landmarks[:, 0] *= resize_scale_x
        new_landmarks[:, 1] *= resize_scale_y

        self.scale_x = resize_scale_x
        self.scale_y = resize_scale_y

        _transforms_api_assert(self, len(new_landmarks) != num_landmarks,
                               f"{self}() have {num_landmarks} input "
                               f"landmarks, but got {len(new_landmarks)} "
                               f"output landmarks!")

        self.flag = True

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksRandomTranslate(LandmarksTransform):

    def __init__(
            self,
            translate: Union[Tuple[float, float], float] = 0.2,
            prob: float = 0.5,
            diff: bool = False
    ):
        super(LandmarksRandomTranslate, self).__init__()
        self._translate = translate
        self._prob = prob

        if type(self._translate) == tuple:
            if len(self._translate) != 2:
                raise ValueError('len(self.translate) == 2, Invalid range')
            if self._translate[0] <= -1. or self._translate[0] >= 1.:
                raise ValueError('out of range (-1,1)')
            if self._translate[1] <= -1. or self._translate[1] >= 1.:
                raise ValueError('out of range (-1,1)')
            self._translate = (min(self._translate), max(self._translate))
        elif type(self._translate) == float:
            if self._translate <= -1. or self._translate >= 1.:
                raise ValueError('out of range (-1,1)')
            self._translate = (
                min(-self._translate, self._translate),
                max(-self._translate, self._translate)
            )
        else:
            raise ValueError('out of range (-1,1)')

        self._diff = diff

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        # Chose a random digit to scale by
        img_shape = img.shape
        num_landmarks = len(landmarks)
        landmark_bboxes = F.helper.to_bboxes(landmarks)
        # translate the image

        # percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self._translate)
        translate_factor_y = random.uniform(*self._translate)

        if not self._diff:
            translate_factor_y = translate_factor_x

        canvas = np.zeros(img_shape).astype(np.uint8)

        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])

        # change the origin to the top-left corner of the translated box
        orig_box_cords = [max(0, corner_y), max(corner_x, 0),
                          min(img_shape[0], corner_y + img.shape[0]),
                          min(img_shape[1], corner_x + img.shape[1])]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]),
               max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]

        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
        new_img = canvas

        landmark_bboxes[:, :4] += [corner_x, corner_y, corner_x, corner_y]

        landmark_bboxes = F.clip_box(landmark_bboxes, [0, 0, img_shape[1], img_shape[0]], 0.0025)
        # refine according to new shape
        new_landmarks = F.helper.to_landmarks(landmark_bboxes,
                                              img_w=new_img.shape[1],
                                              img_h=new_img.shape[0])

        _transforms_api_assert(self, len(new_landmarks) != num_landmarks,
                               f"{self}() have {num_landmarks} input "
                               f"landmarks, but got {len(new_landmarks)} "
                               f"output landmarks!")

        # TODO: add translate affine records
        self.flag = True

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksRandomRotate(LandmarksTransform):

    def __init__(
            self,
            angle: Union[Tuple[int, int], List[int], int] = 10,
            prob: float = 0.5,
            bins: Optional[int] = None
    ):
        super(LandmarksRandomRotate, self).__init__()
        self._angle = angle
        self._bins = bins
        self._prob = prob

        if type(self._angle) == tuple or type(self._angle) == list:
            assert len(self._angle) == 2, "Invalid range"
            self._angle = (min(self._angle), max(self._angle))
        else:
            self._angle = (
                min(-self._angle, self._angle),
                max(-self._angle, self._angle)
            )
        if self._bins is not None and isinstance(self._bins, int):
            interval = int(abs(self._angle[1] - self._angle[0]) / self._bins) + 1
            self.choose_angles = list(range(self._angle[0], self._angle[1], interval))
        else:
            interval = int(abs(self._angle[1] - self._angle[0]))
            self.choose_angles = np.random.uniform(*self._angle, size=interval)

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.0) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        angle = np.random.choice(self.choose_angles)
        num_landmarks = len(landmarks)
        landmark_bboxes = F.helper.to_bboxes(landmarks)

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        new_img = F.rotate_im(img.copy(), angle)

        landmarks_corners = F.get_corners(landmark_bboxes)

        landmarks_corners = np.hstack((landmarks_corners, landmark_bboxes[:, 4:]))

        landmarks_corners[:, :8] = F.rotate_box(landmarks_corners[:, :8], angle, cx, cy, h, w)

        new_landmark_bbox = np.zeros_like(landmark_bboxes)
        new_landmark_bbox[:, [0, 1]] = landmarks_corners[:, [0, 1]]
        new_landmark_bbox[:, [2, 3]] = landmarks_corners[:, [6, 7]]

        scale_factor_x = new_img.shape[1] / w

        scale_factor_y = new_img.shape[0] / h

        new_img = cv2.resize(new_img, (w, h))

        new_landmark_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

        landmark_bboxes = new_landmark_bbox[:, :]

        landmark_bboxes = F.clip_box(landmark_bboxes, [0, 0, w, h], 0.0025)
        # refine according to new shape
        new_landmarks = F.helper.to_landmarks(landmark_bboxes,
                                              img_w=new_img.shape[1],
                                              img_h=new_img.shape[0])
        self.scale_x = (1 / scale_factor_x)
        self.scale_y = (1 / scale_factor_y)

        _transforms_api_assert(self, len(new_landmarks) != num_landmarks,
                               f"{self}() have {num_landmarks} input "
                               f"landmarks, but got {len(new_landmarks)} "
                               f"output landmarks!")

        # TODO: add rotate affine records
        self.flag = True

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksRandomShear(LandmarksTransform):

    def __init__(
            self,
            shear_factor: Union[Tuple[float, float], List[float], float] = 0.2,
            prob: float = 0.5
    ):
        super(LandmarksRandomShear, self).__init__()
        self._shear_factor = shear_factor
        self._prob = prob

        if type(self._shear_factor) == tuple \
                or type(self._shear_factor) == list:
            assert len(self._shear_factor) == 2, "Invalid range for scaling factor"
        else:
            self._shear_factor = (
                min(-self._shear_factor, self._shear_factor),
                max(-self._shear_factor, self._shear_factor)
            )

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        num_landmarks = len(landmarks)
        shear_factor = random.uniform(*self._shear_factor)

        w, h = img.shape[1], img.shape[0]

        new_img = img.copy()
        new_landmarks = landmarks.copy()

        if shear_factor < 0:
            new_img, new_landmarks = LandmarksHorizontalFlip()(new_img, new_landmarks)

        landmark_bboxes = F.helper.to_bboxes(new_landmarks)

        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

        nW = new_img.shape[1] + abs(shear_factor * new_img.shape[0])

        landmark_bboxes[:, [0, 2]] += ((landmark_bboxes[:, [1, 3]]) * abs(shear_factor)).astype(int)

        new_img = cv2.warpAffine(new_img, M, (int(nW), new_img.shape[0]))
        new_landmarks = F.helper.to_landmarks(landmark_bboxes)

        if shear_factor < 0:
            new_img, new_landmarks = LandmarksHorizontalFlip()(new_img, new_landmarks)

        landmark_bboxes = F.helper.to_bboxes(new_landmarks)
        new_img = cv2.resize(new_img, (w, h))

        scale_factor_x = nW / w

        landmark_bboxes[:, :4] /= [scale_factor_x, 1, scale_factor_x, 1]
        # refine according to new shape
        new_landmarks = F.helper.to_landmarks(landmark_bboxes,
                                              img_w=new_img.shape[1],
                                              img_h=new_img.shape[0])

        self.scale_x = (1. / scale_factor_x)
        self.scale_y = 1.

        _transforms_api_assert(self, len(new_landmarks) != num_landmarks,
                               f"{self}() have {num_landmarks} input "
                               f"landmarks, but got {len(new_landmarks)} "
                               f"output landmarks!")

        self.flag = True

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


class LandmarksRandomHSV(LandmarksTransform):
    """HSV Transform to vary hue saturation and brightness

    Hue has a range of 0-179
    Saturation and Brightness have a range of 0-255.
    Chose the amount you want to change thhe above quantities accordingly.


    Parameters
    ----------
    hue : None or int or tuple (int)
        If None, the hue of the image is left unchanged. If int,
        a random int is uniformly sampled from (-hue, hue) and added to the
        hue of the image. If tuple, the int is sampled from the range
        specified by the tuple.

    saturation : None or int or tuple(int)
        If None, the saturation of the image is left unchanged. If int,
        a random int is uniformly sampled from (-saturation, saturation)
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.

    brightness : None or int or tuple(int)
        If None, the brightness of the image is left unchanged. If int,
        a random int is uniformly sampled from (-brightness, brightness)
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.
    """

    def __init__(
            self,
            hue: Union[Tuple[int, int], int] = 20,
            saturation: Union[Tuple[int, int], int] = 20,
            brightness: Union[Tuple[int, int], int] = 20,
            prob: float = 0.5
    ):
        super(LandmarksRandomHSV, self).__init__()
        self._prob = prob
        self._hue = hue if hue else 0
        self._saturation = saturation if saturation else 0
        self._brightness = brightness if brightness else 0

        if type(self._hue) != tuple:
            self._hue = (min(-self._hue, self._hue), max(-self._hue, self._hue))
        if type(self._saturation) != tuple:
            self._saturation = (min(-self._saturation, self._saturation),
                                max(-self._saturation, self._saturation))
        if type(self._brightness) != tuple:
            self._brightness = (min(-self._brightness, self._brightness),
                                max(-self._brightness, self._brightness))

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        hue = random.randint(*self._hue)
        saturation = random.randint(*self._saturation)
        brightness = random.randint(*self._brightness)

        new_img = img.copy()

        a = np.array([hue, saturation, brightness]).astype(np.uint8)
        new_img += np.reshape(a, (1, 1, 3))

        new_img = np.clip(new_img, 0, 255)
        new_img[:, :, 0] = np.clip(new_img[:, :, 0], 0, 179)

        new_img = new_img.astype(np.uint8)

        self.flag = True

        return new_img.astype(np.uint8), landmarks.astype(np.float32)


class LandmarksRandomMask(LandmarksTransform):

    def __init__(
            self,
            mask_ratio: float = 0.1,
            prob: float = 0.5,
            trans_ratio: float = 0.5
    ):
        """random select a mask ratio.
        :param mask_ratio: control the ratio of area to mask, must >= 0.1.
        :param prob: probability
        :param trans_ratio: control the random shape of masked area.
        """
        super(LandmarksRandomMask, self).__init__()
        assert 0.02 < mask_ratio < 1.
        assert 0 < trans_ratio < 1.
        self._mask_ratio = mask_ratio
        self._trans_ratio = trans_ratio
        self._prob = prob

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.uniform(0., 1.0) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        w, h = img.shape[1], img.shape[0]  # original shape
        # random ratios
        mask_ratio = np.random.uniform(0.05, self._mask_ratio, size=1)
        mask_ratio = np.sqrt(mask_ratio)
        mask_h, mask_w = int(h * mask_ratio), int(w * mask_ratio)
        delta = mask_h * mask_w
        # random rectangles
        down_w = max(2, int(mask_w * self._trans_ratio))
        up_w = min(int(mask_w * (1 + self._trans_ratio)), w - 2)
        new_mask_w = np.random.randint(min(down_w, up_w), max(down_w, up_w))
        new_mask_h = int(delta / new_mask_w)

        # random positions
        new_img, mask_corner = F.apply_mask(img, new_mask_w, new_mask_h)

        self.flag = True

        return new_img.astype(np.uint8), landmarks.astype(np.float32)


class LandmarksRandomMaskMixUp(LandmarksTransform):

    def __init__(
            self,
            mask_ratio: float = 0.15,
            prob: float = 0.5,
            trans_ratio: float = 0.5,
            alpha: float = 0.9
    ):
        """random select a mask ratio.
        :param mask_ratio: control the ratio of area to mask, must >= 0.1.
        :param prob: probability
        :param trans_ratio: control the random shape of masked area.
        :param alpha: max alpha value.
        """
        super(LandmarksRandomMaskMixUp, self).__init__()
        assert 0.10 < mask_ratio < 1.
        assert 0 < trans_ratio < 1.
        self._mask_ratio = mask_ratio
        self._trans_ratio = trans_ratio
        self._prob = prob
        self._alpha = alpha
        assert 0. <= alpha <= 1.0

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.uniform(0., 1.0) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        w, h = img.shape[1], img.shape[0]  # original shape
        # random ratios
        mask_ratio = np.random.uniform(0.05, self._mask_ratio, size=1)
        mask_ratio = np.sqrt(mask_ratio)
        mask_h, mask_w = int(h * mask_ratio), int(w * mask_ratio)
        delta = mask_h * mask_w
        # random rectangles
        down_w = max(2, int(mask_w * self._trans_ratio))
        up_w = min(int(mask_w * (1 + self._trans_ratio)), w - 2)
        new_mask_w = np.random.randint(min(down_w, up_w), max(down_w, up_w))
        new_mask_h = int(delta / new_mask_w)

        # random positions
        alpha = np.random.uniform(0.1, self._alpha)
        new_img, mask_corner = F.apply_mask_with_alpha(img, new_mask_w, new_mask_h, alpha)

        self.flag = True

        return new_img.astype(np.uint8), landmarks.astype(np.float32)


class LandmarksRandomBlur(LandmarksTransform):
    """Gaussian Blur"""

    def __init__(
            self,
            kernel_range: Tuple[int, int] = (3, 11),
            prob: float = 0.5,
            sigma_range: Tuple[int, int] = (0, 4)
    ):
        """
        :param kernel_range: kernels for cv2.blur
        :param prob: control the random shape of masked area.
        """
        super(LandmarksRandomBlur, self).__init__()
        self._prob = prob
        self._kernel_range = list(range(kernel_range[0], kernel_range[1] + 1))
        self._kernel_range = [x for x in self._kernel_range if (x % 2) != 0]  # 奇数
        self._sigma_range = sigma_range

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.uniform(0., 1.) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        kernel = np.random.choice(self._kernel_range)
        sigmaX = np.random.choice(self._sigma_range)
        sigmaY = np.random.choice(self._sigma_range)

        img_blur = cv2.GaussianBlur(img.copy(), (kernel, kernel), sigmaX=sigmaX, sigmaY=sigmaY)

        self.flag = True

        return img_blur.astype(np.uint8), landmarks.astype(np.float32)


class LandmarksRandomBrightness(LandmarksTransform):
    """Brightness Transform
        Parameters
        ----------
        brightness : None or int or tuple(int)
            If None, the brightness of the image is left unchanged. If int,
            a random int is uniformly sampled from (-brightness, brightness)
            and added to the hue of the image. If tuple, the int is sampled
            from the range  specified by the tuple.

        """

    def __init__(
            self,
            brightness: Tuple[float, float] = (-30., 30.),
            contrast: Tuple[float, float] = (0.5, 1.5),
            prob: float = 0.5
    ):
        super(LandmarksRandomBrightness, self).__init__()
        self._prob = prob
        if type(brightness) != tuple:
            raise ValueError
        if type(contrast) != tuple:
            raise ValueError

        self.contrast = np.linspace(contrast[0], contrast[1], num=30)
        self.brightness = np.linspace(brightness[0], brightness[1], num=60)

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.uniform(0., 1.) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        brightness = np.random.choice(self.brightness)
        contrast = np.random.choice(self.contrast)

        img = img.astype(np.float32)
        img = contrast * img + brightness
        img = np.clip(img, 0, 255)

        self.flag = True

        return img.astype(np.uint8), landmarks.astype(np.float32)


class LandmarksRandomPatches(LandmarksTransform):

    def __init__(
            self,
            patch_dirs: List[str] = None,
            patch_ratio: float = 0.15,
            prob: float = 0.5,
            trans_ratio: float = 0.5
    ):
        """random select a patch ratio.
        :param patch_dirs: paths to patches images dirs, ["xxx/xx", "xxx/xx"]
        :param patch_ratio: control the ratio of area to patch, must >= 0.1.
        :param prob: probability
        :param trans_ratio: control the random shape of patched area.
        """
        super(LandmarksRandomPatches, self).__init__()
        assert 0.10 < patch_ratio < 1.
        assert 0 < trans_ratio < 1.
        if patch_dirs is None:
            patch_dirs = [os.path.join(Path(__file__).parent, "assets")]
        self._patch_ratio = patch_ratio
        self._trans_ratio = trans_ratio
        self._prob = prob
        self._patches_paths = F.read_image_files(patch_dirs)

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.0) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        w, h = img.shape[1], img.shape[0]  # original shape
        # random patch ratios
        patch_ratio = np.random.uniform(0.05, self._patch_ratio, size=1)
        patch_ratio = np.sqrt(patch_ratio)
        patch_h, patch_w = int(h * patch_ratio), int(w * patch_ratio)
        delta = patch_h * patch_w
        # random patch rectangles
        down_w = max(2, int(patch_w * self._trans_ratio))
        up_w = min(int(patch_w * (1 + self._trans_ratio)), w - 2)
        new_patch_w = np.random.randint(min(down_w, up_w), max(down_w, up_w))
        new_patch_h = int(delta / new_patch_w)

        # random patch positions
        patch = F.select_patch(patch_h=new_patch_h, patch_w=new_patch_w,
                               patches_paths=self._patches_paths)
        if patch is None:
            self.flag = False
            img.astype(np.uint8), landmarks.astype(np.float32)

        new_img, patch_corner = F.apply_patch(img=img, patch=patch)
        self.flag = True

        return new_img.astype(np.uint8), landmarks.astype(np.float32)


class LandmarksRandomPatchesMixUp(LandmarksTransform):

    def __init__(
            self,
            patch_dirs: List[str] = None,
            patch_ratio: float = 0.2,
            prob: float = 0.5,
            trans_ratio: float = 0.5,
            alpha: float = 0.9
    ):
        """random select a patch ratio.
        :param patch_dirs: paths to patches images dirs, ["xxx/xx", "xxx/xx"]
        :param patch_ratio: control the ratio of area to patch, must >= 0.1.
        :param prob: probability
        :param trans_ratio: control the random shape of patched area.
        :param alpha: max alpha value.
        """
        super(LandmarksRandomPatchesMixUp, self).__init__()
        assert 0.10 < patch_ratio < 1.
        assert 0 < trans_ratio < 1.
        if patch_dirs is None:
            patch_dirs = [os.path.join(Path(__file__).parent, "assets")]
        self._patch_ratio = patch_ratio
        self._trans_ratio = trans_ratio
        self._prob = prob
        self._alpha = alpha
        assert 0. <= alpha <= 1.0
        self._patches_paths = F.read_image_files(patch_dirs)

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.0) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        w, h = img.shape[1], img.shape[0]  # original shape
        # random patch ratios
        patch_ratio = np.random.uniform(0.05, self._patch_ratio, size=1)
        patch_ratio = np.sqrt(patch_ratio)
        patch_h, patch_w = int(h * patch_ratio), int(w * patch_ratio)
        delta = patch_h * patch_w
        # random patch rectangles
        down_w = max(2, int(patch_w * self._trans_ratio))
        up_w = min(int(patch_w * (1 + self._trans_ratio)), w - 2)
        new_patch_w = np.random.randint(min(down_w, up_w), max(down_w, up_w))
        new_patch_h = int(delta / new_patch_w)

        # random patch positions
        patch = F.select_patch(patch_h=new_patch_h, patch_w=new_patch_w,
                               patches_paths=self._patches_paths)
        if patch is None:
            self.flag = False
            img.astype(np.uint8), landmarks.astype(np.float32)

        alpha = np.random.uniform(0.1, self._alpha)
        new_img, patch_corner = F.apply_patch_with_alpha(img=img, patch=patch, alpha=alpha)

        self.flag = True

        return new_img.astype(np.uint8), landmarks.astype(np.float32)


class LandmarksRandomBackgroundMixUp(LandmarksTransform):

    def __init__(
            self,
            background_dirs: List[str] = None,
            alpha: float = 0.3,
            prob: float = 0.5
    ):
        """
        :param background_dirs: paths to background images dirs, ["xxx/xx", "xxx/xx"]
        :param prob: probability
        :param alpha: max alpha value(<=0.5)
        """
        super(LandmarksRandomBackgroundMixUp, self).__init__()
        self._prob = prob
        self._alpha = alpha
        assert 0.1 < alpha <= 0.5
        if background_dirs is None:
            background_dirs = [os.path.join(Path(__file__).parent, "assets")]
        self._background_paths = F.read_image_files(background_dirs)

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.0) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        w, h = img.shape[1], img.shape[0]  # original shape

        alpha = np.random.uniform(0.1, self._alpha)
        background = F.select_background(
            img_h=h, img_w=w,
            background_paths=self._background_paths
        )

        if background is None:
            self.flag = False
            return img.astype(np.uint8), landmarks.astype(np.float32)

        new_img = F.apply_background_with_alpha(img=img, background=background, alpha=alpha)

        self.flag = True

        return new_img.astype(np.uint8), landmarks.astype(np.float32)


class LandmarksRandomBackground(LandmarksTransform):

    def __init__(
            self,
            background_dirs: List[str] = None,
            prob: float = 0.5
    ):
        """
        :param background_dirs: paths to background images dirs, ["xxx/xx", "xxx/xx"]
        :param prob: probability
        """
        super(LandmarksRandomBackground, self).__init__()
        self._prob = prob
        if background_dirs is None:
            background_dirs = [os.path.join(Path(__file__).parent, "assets")]
        self._background_paths = F.read_image_files(background_dirs)

    @autodtype(AutoDtypeEnum.Array_InOut)
    def __call__(
            self,
            img: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        if np.random.uniform(0., 1.0) > self._prob:
            self.clear_affine()
            return img.astype(np.uint8), landmarks.astype(np.float32)

        w, h = img.shape[1], img.shape[0]  # original shape

        background = F.select_background(
            img_h=h, img_w=w,
            background_paths=self._background_paths
        )

        if background is None:
            self.flag = False
            return img.astype(np.uint8), landmarks.astype(np.float32)

        num_landmarks = len(landmarks)
        new_img, new_landmarks = F.apply_background(
            img=img, background=background, landmarks=landmarks)

        _transforms_api_assert(self, len(new_landmarks) != num_landmarks,
                               f"{self}() have {num_landmarks} input "
                               f"landmarks, but got {len(new_landmarks)} "
                               f"output landmarks!")

        self.flag = True

        return new_img.astype(np.uint8), new_landmarks.astype(np.float32)


def build_default_transform(
        input_size: Tuple[int, int],
        mean: Union[float, List[float]] = 127.5,
        std: Union[float, List[float]] = 128.,
        force_norm_before_mean_std: bool = False,
        rotate: Optional[int] = 30,
        keep_aspect: Optional[bool] = False,
        to_tensor: Optional[bool] = False
) -> LandmarksCompose:
    if to_tensor:
        return LandmarksCompose([
            # use native torchlm transforms
            LandmarksRandomMaskMixUp(prob=0.25),
            LandmarksRandomBackgroundMixUp(prob=0.25),
            LandmarksRandomScale(prob=0.25),
            LandmarksRandomTranslate(prob=0.25),
            LandmarksRandomShear(prob=0.25),
            LandmarksRandomBlur(kernel_range=(5, 25), prob=0.25),
            LandmarksRandomBrightness(prob=0.25),
            LandmarksRandomRotate(rotate, prob=0.25, bins=8),
            LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.25),
            LandmarksResize(input_size, keep_aspect=keep_aspect),
            LandmarksNormalize(mean, std, force_norm_before_mean_std),
            LandmarksToTensor()
        ])
    return LandmarksCompose([
        # use native torchlm transforms
        LandmarksRandomMaskMixUp(prob=0.25),
        LandmarksRandomBackgroundMixUp(prob=0.25),
        LandmarksRandomScale(prob=0.25),
        LandmarksRandomTranslate(prob=0.25),
        LandmarksRandomShear(prob=0.25),
        LandmarksRandomBlur(kernel_range=(5, 25), prob=0.25),
        LandmarksRandomBrightness(prob=0.25),
        LandmarksRandomRotate(rotate, prob=0.25, bins=8),
        LandmarksRandomCenterCrop((0.5, 1.0), (0.5, 1.0), prob=0.25),
        LandmarksResize(input_size, keep_aspect=keep_aspect),
        LandmarksNormalize(mean, std, force_norm_before_mean_std)
    ])
