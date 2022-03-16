import os
import cv2
import warnings
import numpy as np
import onnxruntime as ort

from .._runtime import FaceDetBase

from typing import Optional, List, Any


class FaceBoxesV2ORT(FaceDetBase):

    def apply_detecting(self, *args, **kwargs) -> Any:
        raise NotImplementedError
