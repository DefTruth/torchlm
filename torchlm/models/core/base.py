import torch.nn as nn
from abc import abstractmethod, ABCMeta
from typing import Any


class ABCBaseModel(nn.Module):
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
    def export(self, *args, **kwargs) -> Any:
        raise NotImplementedError
