from abc import abstractmethod, ABCMeta
from typing import Any


class FaceDetTool(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def detect(self, *args, **kwargs) -> Any:
        raise NotImplementedError


class LandmarksDetTool(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def detect(self, *args, **kwargs) -> Any:
        raise NotImplementedError
