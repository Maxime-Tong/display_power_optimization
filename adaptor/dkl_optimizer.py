import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.colorspace import RGB2DKL, DKL2RGB, sRGB2RGB, RGB2sRGB
from scipy.interpolate import RegularGridInterpolator

from .base import BaseScreenAdaptor

class AdaptiveDKLPowerOptimizer(BaseScreenAdaptor, nn.Module):
    name = "adaptive_dkl_optimizer"
    
    def __init__(self, scale=None, w_weights=None, alpha=0.5, lut_res=16, device='cpu', trainable=False):
        """
        :param scale: 基础 semi-axis scale (kBY, kRG, kL)
        :param w_weights: 功耗权重 (wR, wG, wB)
        :param alpha: 韦伯斜率，控制亮度对感知阈值的贡献，建议 0.4-0.6
        :param lut_res: LUT 分辨率
        :param device: PyTorch device ('cpu' or 'cuda')
        :param trainable: 是否将参数设为可训练 (默认 False)
        """
        BaseScreenAdaptor.__init__(self)
        nn.Module.__init__(self)
        
        self.device = device
        self.lut_res = lut_res
        self.trainable = trainable

        if scale is None:
            raise ValueError("scale is required")

        log_scale = torch.log(torch.tensor(scale, dtype=torch.float32, device=device).clamp_min(1e-6))
        w_tensor = torch.tensor(w_weights, dtype=torch.float32, device=device).clamp_min(1e-6)
        alpha_tensor = torch.tensor([alpha], dtype=torch.float32, device=device).clamp(0.3001, 0.6999)
        
        # 将参数转换为 PyTorch Parameter（可选可训练）
        if trainable:
            self.log_scale = nn.Parameter(log_scale)
            self.w_weights = nn.Parameter(w_tensor)
            self.alpha = nn.Parameter(alpha_tensor)
        else:
            self.register_buffer('log_scale', log_scale)
            self.register_buffer('w_weights', w_tensor)
            self.register_buffer('alpha', alpha_tensor)
        
        self.register_buffer('M_dkl_to_rgb', torch.tensor(DKL2RGB, dtype=torch.float32, device=device))
        self.register_buffer('M_rgb_to_dkl', torch.tensor(RGB2DKL, dtype=torch.float32, device=device))
        
        self.lut_delta_rgb = None
        self.interpolator = None
        self.phi_map = None
        
        self._build_adaptive_offline_lut()

    @property
    def scale_tensor(self):
        return torch.exp(self.log_scale)

    @property
    def scale_numpy(self):
        return self.scale_tensor.detach().cpu().numpy()
    
    @property
    def w_weights_numpy(self):
        return self.w_weights.detach().cpu().numpy()
    
    @property
    def alpha_numpy(self):
        return self.alpha.detach().cpu().numpy()

    def _get_local_k(self, dkl_color):
        """
        根据当前点的 DKL 坐标计算局部 k 因子
        实现韦伯定律：亮度越高，容忍度（k值）越大
        支持 numpy 和 torch tensor
        """
        if isinstance(dkl_color, torch.Tensor):
            L = dkl_color[..., 2]
            L_clamped = torch.clamp(L, min=0.01)
            scaling = torch.pow(L_clamped, self.alpha.squeeze())
            scale_tensor = self.scale_tensor
            return scale_tensor.unsqueeze(0) * scaling.unsqueeze(-1) if L.dim() > 0 else scale_tensor * scaling
        else:
            L = dkl_color[2]
            L_clamped = max(L, 0.01)
            scaling = (L_clamped ** self.alpha_numpy.item())
            return self.scale_numpy * scaling

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
        w_np = self.w_weights.detach().cpu().numpy()
        dkl2rgb_np = self.M_dkl_to_rgb.cpu().numpy()
        g_dkl = w_np @ dkl2rgb_np
        
        # 创建空的 LUT 存储 delta_srgb
        self.lut_delta_rgb = np.zeros_like(rgb_grid)
        
        # 对 LUT 每一个点进行局部优化
        for i in range(self.lut_res):
            for j in range(self.lut_res):
                for k in range(self.lut_res):
                    curr_rgb = rgb_grid[i, j, k]
                    
                    # 转到 DKL 获取当前点的局部 k 因子
                    rgb2dkl_np = self.M_rgb_to_dkl.cpu().numpy()
                    curr_dkl = rgb2dkl_np @ curr_rgb
                    
                    # 直接使用 numpy 计算 k 因子
                    L = curr_dkl[2]
                    L_clamped = max(L, 0.01)
                    k_np = self.scale_numpy
                    alpha_np = self.alpha_numpy.item()
                    scaling = (L_clamped ** alpha_np)
                    local_k = k_np * scaling
                    
                    # 求解拉格朗日闭式解：delta_c_i = -(g_i * k_i^2) / sqrt(sum(g_j^2 * k_j^2))
                    num = g_dkl * (local_k**2)
                    denom = np.sqrt(np.sum((g_dkl**2) * (local_k**2)))
                    denom = max(denom, 1e-10)
                    
                    delta_c = -num / denom
                    # 转回 RGB 增量
                    self.lut_delta_rgb[i, j, k] = dkl2rgb_np @ delta_c

        # 初始化 3D 插值器
        self.interpolator = RegularGridInterpolator((steps, steps, steps), self.lut_delta_rgb)

    def apply(self, image, **kwargs):
        """
        在线应用：支持注视点离心率 phi_map（使用 LUT 推理，不参与梯度计算）
        """
        if self.trainable:
            self._update_lut_if_needed()
        
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
    
    # ======================== PyTorch 训练方法 ========================
    
    def forward_torch(self, image_tensor, phi_map=None):
        """
        可微分的前向传递（用于训练）
        :param image_tensor: Tensor 形状 (B, 3, H, W) 或 (H, W, 3)，值域 [0, 1]
        :param phi_map: 可选的凝视注视图
        :return: 优化后的 tensor
        """
        # 处理不同的输入形状
        if image_tensor.dim() == 3:
            if image_tensor.shape[-1] == 3:
                # (H, W, C) -> (1, C, H, W)
                image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
            else:
                # (C, H, W) -> (1, C, H, W)
                image_tensor = image_tensor.unsqueeze(0)
        
        B, C, H, W = image_tensor.shape
        
        # 转换 sRGB 到线性 RGB
        linear_image = self._srgb2rgb_torch(image_tensor)
        
        # 计算最优 delta
        flat_image = linear_image.permute(0, 2, 3, 1).reshape(B * H * W, 3)
        deltas = self._compute_optimal_deltas_torch(flat_image)
        deltas = deltas.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
        
        # 应用凝视注视加权
        if phi_map is None:
            phi_map = self._compute_phi_map_torch(H, W, device=image_tensor.device)
        
        # phi_map 形状 (H, W) -> (1, 1, H, W)
        phi_map = phi_map.unsqueeze(0).unsqueeze(0)
        
        result_linear = linear_image + deltas * phi_map
        result_linear = torch.clamp(result_linear, 0.0, 1.0)
        
        # 转换回 sRGB
        result = self._rgb2srgb_torch(result_linear)
        
        return result
    
    def _compute_optimal_deltas_torch(self, rgb_linear):
        """
        使用可微分操作计算最优颜色偏移
        :param rgb_linear: (N, 3) tensor，在线性 RGB 空间
        :return: (N, 3) tensor 的偏移
        """
        # 转换到 DKL
        dkl = torch.matmul(rgb_linear, self.M_rgb_to_dkl.t())  # (N, 3)
        
        # 基于亮度获取局部 k 因子
        L = dkl[..., 2]  # (N,)
        L_clamped = torch.clamp(L, min=0.01)
        
        # 约束 alpha 的范围 [0.3, 0.7] 以避免数值不稳定
        alpha_safe = torch.clamp(self.alpha.squeeze(), min=0.3, max=0.7)
        scaling = torch.pow(L_clamped, alpha_safe)  # (N,)
        
        # 为每个像素计算局部 k 因子
        if scaling.dim() == 0:
            local_k = self.scale_tensor * scaling  # (3,)
        else:
            local_k = self.scale_tensor.unsqueeze(0) * scaling.unsqueeze(1)  # (N, 3)
        
        # 约束 scale 为正数，避免负值
        local_k = torch.clamp(local_k, min=1e-6)
        
        # DKL 空间中的功耗梯度
        # 约束 w_weights 为正数
        g_dkl = torch.matmul(self.w_weights.unsqueeze(0), self.M_dkl_to_rgb)  # (1, 3)
        
        # 使用拉格朗日乘数计算最优偏移
        num = g_dkl * (local_k ** 2)  # (N, 3)
        denom = torch.sqrt(torch.sum((g_dkl ** 2) * (local_k ** 2), dim=1, keepdim=True) + 1e-10)  # (N, 1)
        denom = torch.clamp(denom, min=1e-10)
        
        delta_c = -num / denom  # (N, 3) 在 DKL 空间
        
        # 转换回 RGB
        delta_rgb = torch.matmul(delta_c, self.M_dkl_to_rgb.t())  # (N, 3)
        
        # 约束 delta 的幅度以防止梯度爆炸
        delta_rgb = torch.clamp(delta_rgb, min=-0.5, max=0.5)
        
        return delta_rgb
    
    def _srgb2rgb_torch(self, srgb):
        """使用 PyTorch 将 sRGB 转换为线性 RGB"""
        srgb = torch.clamp(srgb, 0, 1)
        mask = srgb <= 0.04045
        rgb = torch.where(
            mask,
            srgb / 12.92,
            torch.pow((srgb + 0.055) / 1.055, 2.4)
        )
        return rgb
    
    def _rgb2srgb_torch(self, rgb):
        """使用 PyTorch 将线性 RGB 转换为 sRGB"""
        rgb = torch.clamp(rgb, 0, 1)
        mask = rgb <= 0.0031308
        srgb = torch.where(
            mask,
            12.92 * rgb,
            1.055 * torch.pow(rgb, 1.0/2.4) - 0.055
        )
        return srgb
    
    def _compute_phi_map_torch(self, H, W, device):
        """计算凝视注视图"""
        y = torch.arange(H, dtype=torch.float32, device=device)
        x = torch.arange(W, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y - H/2, x - W/2, indexing='ij')
        
        dist = torch.sqrt(xx**2 + yy**2)
        max_dist = torch.sqrt(torch.tensor((H/2)**2 + (W/2)**2, dtype=torch.float32, device=device))
        max_dist = torch.clamp(max_dist, min=1e-10)
        phi = (dist / max_dist) ** 2
        
        return phi
    
    def _compute_ssim(self, x, y, window_size=11, sigma=1.5):
        """
        计算两个图像之间的 SSIM
        :param x, y: (B, C, H, W) tensors
        :return: SSIM 值
        """
        # 创建高斯窗口
        gauss_kernel = torch.Tensor([
            math.exp(-(ix - window_size//2)**2 / float(2*sigma**2))
            for ix in range(window_size)
        ]).to(x.device)
        
        _1D_window = gauss_kernel / gauss_kernel.sum()
        _2D_window = torch.mm(_1D_window.unsqueeze(1), _1D_window.unsqueeze(0))
        window = _2D_window.unsqueeze(0).unsqueeze(0).expand(3, 1, window_size, window_size)
        
        # 确保输入是 3 通道
        if x.shape[1] != 3:
            x = x.expand(-1, 3, -1, -1)
        if y.shape[1] != 3:
            y = y.expand(-1, 3, -1, -1)
        
        # 增加 C1, C2 以提高数值稳定性
        C1, C2 = (0.01)**2, (0.03)**2
        
        mu1 = F.conv2d(x, window, padding=window_size//2, groups=3)
        mu2 = F.conv2d(y, window, padding=window_size//2, groups=3)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(x*x, window, padding=window_size//2, groups=3) - mu1_sq
        sigma2_sq = F.conv2d(y*y, window, padding=window_size//2, groups=3) - mu2_sq
        sigma12 = F.conv2d(x*y, window, padding=window_size//2, groups=3) - mu1_mu2
        
        # 确保方差非负（数值稳定性）
        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)
        
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        
        # 防止除零
        denominator1 = torch.clamp(denominator1, min=1e-10)
        denominator2 = torch.clamp(denominator2, min=1e-10)
        
        ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
        
        # 确保 SSIM 值在 [-1, 1] 范围内
        ssim_map = torch.clamp(ssim_map, min=-1.0, max=1.0)
        
        return ssim_map.mean()
    
    def compute_loss(self, image_opt, image_ori, beta=0.8):
        """
        计算训练损失
        L = 0.5 * SSIM_loss + 50 * L1_loss
        
        :param image_opt: 优化后的图像 tensor (B, 3, H, W) 或 (H, W, 3)，值域 [0, 1]
        :param image_ori: 原始图像 tensor (B, 3, H, W) 或 (H, W, 3)，值域 [0, 1]
        :param beta: 显示优化程度（默认 0.8，即 80%）
        :return: 损失值
        """
        # 确保输入在同一设备上
        image_opt = image_opt.to(self.device)
        image_ori = image_ori.to(self.device)
        
        # 标准化形状
        if image_opt.dim() == 3 and image_opt.shape[-1] == 3:
            image_opt = image_opt.permute(2, 0, 1).unsqueeze(0)
        elif image_opt.dim() == 3:
            image_opt = image_opt.unsqueeze(0)
        
        if image_ori.dim() == 3 and image_ori.shape[-1] == 3:
            image_ori = image_ori.permute(2, 0, 1).unsqueeze(0)
        elif image_ori.dim() == 3:
            image_ori = image_ori.unsqueeze(0)
        
        # 确保图像值在 [0, 1] 范围内
        image_opt = torch.clamp(image_opt, 0.0, 1.0)
        image_ori = torch.clamp(image_ori, 0.0, 1.0)
        
        # SSIM 损失
        ssim_value = self._compute_ssim(image_opt, image_ori)
        ssim_loss = 1.0 - ssim_value
        
        # L1 损失（与 beta 缩放的原始图像比较）
        target = beta * image_ori
        l1_loss = F.l1_loss(image_opt, target)
        # 合并损失
        total_loss = 2 * l1_loss + 0.5 * ssim_loss
        
        return total_loss
    
    def train_step(self, optimizer, image_ori, beta=0.3):
        """
        单步训练
        
        :param optimizer: PyTorch 优化器
        :param image_ori: 原始图像 tensor
        :param beta: 优化程度
        :return: 损失值（标量）
        """
        optimizer.zero_grad()
        
        # 前向传递
        image_opt = self.forward_torch(image_ori)
        
        # 计算损失
        loss = self.compute_loss(image_opt, image_ori, beta)

        # 反向传递
        loss.backward()
        
        optimizer.step()
        
        print(self.scale_tensor.data, self.w_weights.data, self.alpha.data)
        
        return loss.item()
    
    def get_trainable_parameters(self):
        """获取可训练的参数"""
        if self.trainable:
            return [self.log_scale, self.w_weights, self.alpha]
        else:
            return []