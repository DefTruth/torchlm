from typing import Any
from abc import abstractmethod, ABCMeta

from ..core import LandmarksDetBase


class LandmarksTrainableBase(LandmarksDetBase):
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
