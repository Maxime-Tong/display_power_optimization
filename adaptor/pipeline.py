from typing import Any, Iterable, List

import numpy as np

from .base import BaseScreenAdaptor


class ScreenAdaptorPipeline(BaseScreenAdaptor):
    name = "pipeline"

    def __init__(self, adaptors: Iterable[BaseScreenAdaptor]):
        self.adaptors: List[BaseScreenAdaptor] = list(adaptors)

    def prepare(self, image_shape):
        for adaptor in self.adaptors:
            adaptor.prepare(image_shape)

    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        output = image
        for adaptor in self.adaptors:
            output = adaptor.apply(output, **kwargs)
        return output
