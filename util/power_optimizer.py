import numpy as np
from .constants import WEIGHTS_NORM
from util.colorspace import RGB2DKL, DKL2RGB

class PowerOptimizedColorOptimizer:
    """
    沿蓝色轴负方向移动颜色，在感知不可见前提下最小化显示能耗。
    直接处理整张图像，无需分块。
    """

    def __init__(self, b_max_vec):
        """
        参数
        ----------
        b_max_vec : (1,3) ndarray
            DKL 蓝色轴在 RGB 空间的投影向量，指向 B 值增加的方向。
            能耗优化将使用其反方向。
        """
        self.b_max_vec = b_max_vec
        self.b_decrease_vec = -b_max_vec

    def optimize(self, dkl_centers, centers_abc):
        """
        优化整张图像的颜色，降低显示能耗。

        参数
        ----------
        dkl_centers : (H, W, 3) or (N, 3) ndarray
            DKL 空间颜色中心点
        centers_abc : (H, W, 3) or (N, 3) ndarray
            JND 椭球半轴长度 (a, b, c)

        返回
        -------
        (H, W, 3) or (N, 3) ndarray
            优化后的 RGB 颜色，值域 [0, 1]
        """
        original_shape = dkl_centers.shape
        # 展平为 (N, 3)
        dkl_flat = dkl_centers.reshape(-1, 3)
        abc_flat = centers_abc.reshape(-1, 3)

        rgb_centers = (DKL2RGB @ dkl_flat.T).T
        rgb_centers = np.clip(rgb_centers, 0, 1)

        min_power_dkl = self.solve(dkl_flat, self.b_decrease_vec, abc_flat)

        min_power_rgb = (DKL2RGB @ min_power_dkl.T).T

        min_power_rgb = self._clamp_rgb(min_power_rgb, rgb_centers)

        return min_power_rgb.reshape(original_shape)

    # def solve(self, centers, direction_vec, abc):
    #     abc_sq = abc ** 2
    #     dir_sq = direction_vec ** 2
        
    #     weighted_sum = np.sum(dir_sq * abc_sq, axis=1)
    #     weighted_sum = np.maximum(weighted_sum, 1e-10)

    #     t = 1.0 / np.sqrt(weighted_sum)                # (N,)
    #     t = t[:, np.newaxis]    # (N,1)
    #     return centers + t * direction_vec * abc_sq
    
    def solve(self, centers, direction_vec, abc):
        # 计算沿给定方向的单位向量与椭球表面的交点
        inv_abc_sq = 1.0 / np.maximum(abc ** 2, 1e-10)
        dir_sq = direction_vec ** 2
        
        weighted_sum = np.sum(dir_sq * inv_abc_sq, axis=1)
        weighted_sum = np.maximum(weighted_sum, 1e-10)

        t = 1.0 / np.sqrt(weighted_sum)                # (N,)
        t = t[:, np.newaxis]    # (N,1)
        return centers + t * direction_vec

    def _clamp_rgb(self, points_rgb, centers_rgb):
        """
        将超出 [0,1] 的 RGB 点沿中心方向回退到有效范围。

        参数
        ----------
        points_rgb : (N, 3) ndarray
            待修正的点
        centers_rgb : (N, 3) ndarray
            原始中心点

        返回
        -------
        (N, 3) ndarray
            修正后的点
        """
        result = points_rgb.copy()
        direction = points_rgb - centers_rgb
        
        result = np.where(direction < 0, points_rgb, centers_rgb)
        return np.clip(result, 0, 1)