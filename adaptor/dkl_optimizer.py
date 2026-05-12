import numpy as np
from util.colorspace import RGB2DKL, DKL2RGB, sRGB2RGB, RGB2sRGB
from scipy.interpolate import RegularGridInterpolator

from .base import BaseScreenAdaptor

class AdaptiveDKLPowerOptimizer(BaseScreenAdaptor):
    name = "adaptive_dkl_optimizer"
    
    def __init__(self, k_base, w_weights, alpha=0.5, lut_res=16):
        """
        :param k_base: 基础 k 因子 (kL_base, kRG_base, kBY_base)
        :param w_weights: 功耗权重 (wR, wG, wB)
        :param alpha: 韦伯斜率，控制亮度对感知阈值的贡献，建议 0.4-0.6
        """
        self.lut_res = lut_res        
        self.k_factor = np.array(k_base)
        self.w_rgb = np.array(w_weights)
        self.alpha = alpha
        
        self.M_dkl_to_rgb = DKL2RGB
        self.M_rgb_to_dkl = RGB2DKL
        
        self.lut_delta_rgb = None
        self._build_adaptive_offline_lut()

        self.phi_map = None
        
    def _get_local_k(self, dkl_color):
        """
        根据当前点的 DKL 坐标计算局部 k 因子
        实现韦伯定律：亮度越高，容忍度（k值）越大
        """
        L = dkl_color[2]
        # 避免 L=0 时 k 因子为 0 导致计算崩溃，设置最小阈值
        L_clamped = max(L, 0.01)
        
        # 亮度因子随 L 的幂律增长
        scaling = (L_clamped ** self.alpha)
        
        # 返回局部 k 因子
        return self.k_factor * scaling

    def _build_adaptive_offline_lut(self):
        """
        离线阶段：针对 LUT 中每一个颜色点求解局部最优偏移
        """
        print(f"Generating Adaptive {self.lut_res}^3 LUT...")
        steps = np.linspace(0, 1, self.lut_res)
        # 生成 RGB 栅格
        r, g, b = np.meshgrid(steps, steps, steps, indexing='ij')
        rgb_grid = np.stack([r, g, b], axis=-1)
        
        # 功耗梯度在 DKL 空间是固定的 (因为功耗模型和转换矩阵是线性的)
        g_dkl = self.w_rgb @ self.M_dkl_to_rgb
        
        # 创建空的 LUT 存储 delta_srgb
        self.lut_delta_rgb = np.zeros_like(rgb_grid)
        
        # 对 LUT 每一个点进行局部优化
        for i in range(self.lut_res):
            for j in range(self.lut_res):
                for k in range(self.lut_res):
                    curr_rgb = rgb_grid[i, j, k]
                    
                    # 转到 DKL 获取当前点的局部 k 因子
                    curr_dkl = self.M_rgb_to_dkl @ curr_rgb
                    
                    local_k = self._get_local_k(curr_dkl)
                    # print(local_k)
                    
                    # 求解拉格朗日闭式解：delta_c_i = -(g_i * k_i^2) / sqrt(sum(g_j^2 * k_j^2))
                    num = g_dkl * (local_k**2)
                    denom = np.sqrt(np.sum((g_dkl**2) * (local_k**2)))
                    denom = max(denom, 1e-10)
                    
                    delta_c = -num / denom
                    # 转回 RGB 增量
                    self.lut_delta_rgb[i, j, k] = self.M_dkl_to_rgb @ delta_c

        # 初始化 3D 插值器
        self.interpolator = RegularGridInterpolator((steps, steps, steps), self.lut_delta_rgb)

    def apply(self, image, **kwargs):
        """
        在线应用：支持注视点离心率 phi_map
        """
        if image.dtype == np.uint8 or image.max() > 1.5:
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.astype(np.float32)

        if self.phi_map is None:
            self.prepare(image.shape)
            
        h, w, _ = image.shape
        linear_image = sRGB2RGB(image_float)
        flat_image = linear_image.reshape(-1, 3)  # 转到线性 RGB 空间
        
        # 查询 LUT 获得每个像素的最佳偏移方向和幅度
        deltas = self.interpolator(flat_image).reshape(h, w, 3)
        # print(deltas)
        
        # 最终应用：在线性 RGB 空间叠加局部最优 delta * 离心率权重
        result_linear = linear_image + deltas * self.phi_map[..., None]
        result_linear = np.clip(result_linear, 0.0, 1.0)
        result = RGB2sRGB(result_linear).reshape(image.shape)  # 转回 sRGB 空间
        return np.clip(result * 255.0, 0, 255).round().astype(np.uint8)
    
    def prepare(self, image_shape):
        h, w = image_shape[:2]
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2
        
        # 归一化距离 [0, 1]，中心为0
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        phi = dist / max_dist
        
        self.phi_map = phi ** 2
    
if __name__ == "__main__":
    # 配置参数
    k_factors = (1.0, 1.5, 4.0)  # kL, kRG, kBY
    w_weights = (0.3, 0.4, 0.8)  # 假设蓝色功耗最高
    
    optimizer = AdaptiveDKLPowerOptimizer(k_factors, w_weights)
    
    # 模拟输入图像 (1, 1080, 1920, 3)
    input_img = np.random.rand(1, 512, 512, 3).astype(np.float32)
    
    output_img = optimizer.apply(input_img)
    
    print(f"Optimization complete. Original Mean: {input_img.mean():.4f}, Optimized Mean: {output_img.mean():.4f}")