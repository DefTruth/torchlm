from typing import Any
from abc import abstractmethod, ABCMeta

from ..core import LandmarksDetBase


class LandmarksTrainableBase(LandmarksDetBase):
    __metaclass__ = ABCMeta

    @abstractmethod
    def apply_losses(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def apply_detecting(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def apply_training(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def apply_evaluating(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def apply_exporting(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    @abstractmethod
    def apply_freezing(self, *args, **kwargs) -> Any:
        raise NotImplementedError
