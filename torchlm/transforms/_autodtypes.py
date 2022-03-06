import numpy as np
from torch import Tensor
from typing import Tuple, Union, Callable, Any, Dict
from . import _functional as F

# base element_type
Base_Element_Type = Union[np.ndarray, Tensor]
Image_InOutput_Type = Base_Element_Type  # image
Landmarks_InOutput_Type = Base_Element_Type  # landmarks


class AutoDtypeEnum:
    # autodtype modes
    Array_In: int = 0
    Array_InOut: int = 1
    Tensor_In: int = 2
    Tensor_InOut: int = 3

AutoDtypeLoggingMode: bool = False

def set_autodtype_logging(logging: bool = False):
    global AutoDtypeLoggingMode
    AutoDtypeLoggingMode = logging

def _autodtype_api_logging(self: Any, mode: int):
    global AutoDtypeLoggingMode
    if AutoDtypeLoggingMode:
        mode_info_map: Dict[int, str] = {
            AutoDtypeEnum.Array_In: "AutoDtypeEnum.Array_In",
            AutoDtypeEnum.Array_InOut: "AutoDtypeEnum.Array_InOut",
            AutoDtypeEnum.Tensor_In: "AutoDtypeEnum.Tensor_In",
            AutoDtypeEnum.Tensor_InOut: "AutoDtypeEnum.Tensor_InOut"
        }
        print(f"{self}() AutoDtype Info: {mode_info_map[mode]}")


def autodtype(mode: int) -> Callable:
    # A Pythonic style to auto convert input dtype and let the output dtype unchanged

    assert 0 <= mode <= 3

    def wrapper(
            callable_array_or_tensor_func: Callable
    ) -> Callable:

        def apply(
                self,
                img: Image_InOutput_Type,
                landmarks: Landmarks_InOutput_Type,
                **kwargs
        ) -> Tuple[Image_InOutput_Type, Landmarks_InOutput_Type]:
            # Type checks
            assert all(
                [isinstance(_, (np.ndarray, Tensor))
                 for _ in (img, landmarks)]
            ), "Error dtype, must be np.ndarray or Tensor!"
            # force array before transform and then wrap back.
            if mode == AutoDtypeEnum.Array_InOut:
                _autodtype_api_logging(self, mode)

                if any((
                        isinstance(img, Tensor),
                        isinstance(landmarks, Tensor)
                )):
                    img = F.to_numpy(img)
                    landmarks = F.to_numpy(landmarks)
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
                    img = F.to_tensor(img)
                    landmarks = F.to_tensor(landmarks)
                else:
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
            # force array before transform and don't wrap back.
            elif mode == AutoDtypeEnum.Array_In:
                _autodtype_api_logging(self, mode)

                if any((
                        isinstance(img, Tensor),
                        isinstance(landmarks, Tensor)
                )):
                    img = F.to_numpy(img)
                    landmarks = F.to_numpy(landmarks)
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
                else:
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
            # force tensor before transform and then wrap back.
            elif mode == AutoDtypeEnum.Tensor_InOut:
                _autodtype_api_logging(self, mode)

                if any((
                        isinstance(img, np.ndarray),
                        isinstance(landmarks, np.ndarray)
                )):
                    img = F.to_tensor(img)
                    landmarks = F.to_tensor(landmarks)
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
                    img = F.to_numpy(img)
                    landmarks = F.to_numpy(landmarks)
                else:
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
            # force tensor before transform and don't wrap back.
            elif mode == AutoDtypeEnum.Tensor_In:
                _autodtype_api_logging(self, mode)

                if any((
                        isinstance(img, np.ndarray),
                        isinstance(landmarks, np.ndarray)
                )):
                    img = F.to_tensor(img)
                    landmarks = F.to_tensor(landmarks)
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
                else:
                    img, landmarks = callable_array_or_tensor_func(
                        self,
                        img,
                        landmarks,
                        **kwargs
                    )
            else:
                _autodtype_api_logging(self, mode)
                img, landmarks = callable_array_or_tensor_func(
                    self,
                    img,
                    landmarks,
                    **kwargs
                )

            return img, landmarks

        return apply

    return wrapper
