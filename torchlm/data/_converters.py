import os
import cv2
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import Tuple, Optional, List


class BaseConverter(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def convert(self, *args, **kwargs):
        raise NotImplementedError