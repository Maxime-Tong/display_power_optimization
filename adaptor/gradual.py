from typing import Any

import numpy as np

from .base import BaseScreenAdaptor
from util.color_adaptation import GradualChromaticOptimizer


class GradualAdaptor(BaseScreenAdaptor):
    name = "gradual"

    def __init__(self, angle: float, velocity: float, t_max: float, delta_t_jnd: float = 5.0):
        self.optimizer = GradualChromaticOptimizer(
            angle=angle,
            velocity=velocity,
            t_max=t_max,
            delta_T_jnd=delta_t_jnd,
        )

    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        t = float(kwargs.get("t", 0.0))
        output = self.optimizer.apply_to_frame(image, t)
        return np.clip(output * 255.0, 0, 255).round().astype(np.uint8)
