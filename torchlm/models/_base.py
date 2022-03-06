from typing import Any
from abc import abstractmethod, ABCMeta

from ..tools import LandmarksDetTool


class LandmarksTrainable(LandmarksDetTool):
    __metaclass__ = ABCMeta

    @abstractmethod
    def loss(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def detect(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def training(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def evaluating(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def export(self, *args, **kwargs) -> Any:
        raise NotImplementedError
