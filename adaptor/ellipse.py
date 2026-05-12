from typing import Any, Optional

import numpy as np

from .base import BaseScreenAdaptor
from model.base_color_model import BaseColorModel
from util.colorspace import RGB2DKL, RGB2sRGB, sRGB2RGB
from util.power_optimizer import PowerOptimizedColorOptimizer


class EllipseAdaptor(BaseScreenAdaptor):
    name = "ellipse"

    def __init__(
        self,
        color_model: BaseColorModel,
        increase_vec: np.ndarray,
        abc_scaler: float = 1.0,
        ecc_no_compress: float = 5.0,
        foveated: bool = True,
        max_ecc: float = 18.0,
        h_fov: float = 110.0,
    ):
        self.color_model = color_model
        self.power_optimizer = PowerOptimizedColorOptimizer(increase_vec)
        self.abc_scaler = abc_scaler
        self.ecc_no_compress = ecc_no_compress
        self.foveated = foveated
        self.max_ecc = np.asarray(max_ecc, dtype=np.float32)
        self.d = np.asarray(1.0 / np.tan(h_fov * np.pi / 180.0 / 2.0), dtype=np.float32)
        self.v_h_ratio = 1.0
        self.image_height: Optional[int] = None
        self.image_width: Optional[int] = None
        self.xx = None
        self.yy = None
        self.ecc_map = None

    def prepare(self, image_shape):
        self.image_height, self.image_width = image_shape[:2]
        self.v_h_ratio = self.image_height / self.image_width if self.image_width > 0 else 1.0
        self.x = np.linspace(-1, 1, self.image_width, dtype=np.float32)
        self.y = np.linspace(-1, 1, self.image_height, dtype=np.float32)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

    def _ensure_ready(self, image: np.ndarray) -> None:
        if self.xx is None or self.yy is None:
            self.prepare(image.shape)

    def set_ecc_map(self, gaze_x: float = 0, gaze_y: float = 0):
        if self.xx is None:
            raise ValueError("图像尺寸未设置，无法计算偏心度映射")

        dist = np.sqrt((self.xx - gaze_x) ** 2 + ((self.yy - gaze_y) * self.v_h_ratio) ** 2)
        self.ecc_map = np.arctan(dist / self.d)[..., None] * 180 / np.pi
        self.ecc_map = np.clip(self.ecc_map, 0, self.max_ecc)

    def generate_ellipsoids(self, srgb_centers, ecc_map):
        rgb_centers = sRGB2RGB(srgb_centers)
        dkl_centers = np.einsum("ij,...j->...i", RGB2DKL, rgb_centers).reshape(-1, 3)
        centers_abc = self.color_model.compute_ellipses(srgb_centers, ecc_map)
        mean_abc = np.mean(centers_abc, axis=0)
        print(f"平均椭圆参数: a={mean_abc[0]:.4f}, b={mean_abc[1]:.4f}, c={mean_abc[2]:.4f}")

        centers_abc = np.maximum(centers_abc * self.abc_scaler, 1e-5)
        mask = np.tile((ecc_map < self.ecc_no_compress), (1, 1, 3)).reshape(-1, 3)
        centers_abc[mask] *= 0.5
        return dkl_centers, centers_abc

    def apply(self, image: np.ndarray, **kwargs: Any) -> np.ndarray:
        self._ensure_ready(image)
        gaze_x = float(kwargs.get("gaze_x", 0.0))
        gaze_y = float(kwargs.get("gaze_y", 0.0))

        if image.dtype == np.uint8 or image.max() > 1.5:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)

        if self.foveated:
            self.set_ecc_map(gaze_x, gaze_y)
        else:
            self.ecc_map = np.ones((self.image_height, self.image_width, 1), dtype=np.float32) * self.max_ecc

        dkl_centers, centers_abc = self.generate_ellipsoids(image_float, self.ecc_map)
        image_rgb = self.power_optimizer.optimize(dkl_centers, centers_abc)
        image_srgb = RGB2sRGB(image_rgb).reshape(image_float.shape)
        return np.clip(image_srgb * 255.0, 0, 255).round().astype(np.uint8)
