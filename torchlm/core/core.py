from abc import abstractmethod, ABCMeta
from typing import Any

__all__ = ["FaceDetBase", "LandmarksDetBase"]

class FaceDetBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply_detecting(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class LandmarksDetBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply_detecting(self, *args, **kwargs) -> Any:
        raise NotImplementedError
