from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np


class BaseScreenAdaptor(ABC):
    """屏幕功耗调节 adaptor 的统一接口。"""

    name = "base"

    def prepare(self, image_shape: Tuple[int, int, int]) -> None:
        """在首次处理图像前准备与图像尺寸相关的缓存。"""
        return None

    @abstractmethod
    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        """对输入图像执行一次 adaptor 处理。"""
