import numpy as np
from util.constants import M_BRADFORD, M_BRADFORD_INV, D65_XYZ, D65_UPVP
from util.colorspace import sRGB2RGB, RGB2sRGB, RGB2XYZ, XYZ2RGB

def upvp_to_xyz(upvp, Y=1.0):
    """
    CIE u'v' → XYZ（假设 Y 亮度已知）

    参数
    ----------
    upvp : (2,) ndarray
        u', v' 坐标
    Y : float, optional
        亮度，默认为 1.0

    返回
    -------
    (3,) ndarray
        XYZ 坐标
    """
    u_prime, v_prime = upvp
    if v_prime == 0:
        return np.array([0.0, 0.0, 0.0])
    X = Y * 9 * u_prime / (4 * v_prime)
    Z = Y * (12 - 3 * u_prime - 20 * v_prime) / (4 * v_prime)
    return np.array([X, Y, Z])

def compute_cat_matrix(source_white, target_white):
    """
    计算 Bradford 色度适应变换矩阵

    参数
    ----------
    source_white : (3,) ndarray
        源白点 XYZ 坐标
    target_white : (3,) ndarray
        目标白点 XYZ 坐标

    返回
    -------
    (3,3) ndarray
        CAT 矩阵，可应用于线性 RGB 像素
    """
    # 转换到 Bradford 空间
    W_s = M_BRADFORD @ source_white
    W_t = M_BRADFORD @ target_white
    W_s = np.clip(W_s, 1e-6, None)

    # 对角缩放矩阵
    D_B = np.diag(W_t / W_s)
    return M_BRADFORD_INV @ D_B @ M_BRADFORD

class GradualChromaticOptimizer:
    """渐进色度适应优化器（GCA）"""

    def __init__(self, angle, velocity, t_max, delta_T_jnd=5.0):
        """
        参数
        ----------
        angle : float
            轨迹角度（弧度）
        velocity : float
            u'v' 空间移动速度（单位/秒）
        t_max : float
            总偏移时间（秒）
        delta_T_jnd : float, optional
            JND 阈值（默认为 5.0）
        """
        self.angle = angle
        self.velocity = velocity
        self.t_max = t_max
        self.delta_T = delta_T_jnd * 0.004   # JND → u'v' 单位

        self.terminal_upvp = self._compute_terminal_illuminant()
        print(f"GCA初始化: 角度={np.degrees(angle):.1f}°, "
              f"终端u'v'={self.terminal_upvp}")

    def _compute_terminal_illuminant(self):
        """计算终端光照在 u'v' 空间的坐标（简化模型）"""
        direction = np.array([np.cos(self.angle), np.sin(self.angle)])
        distance = self.velocity * self.t_max * 0.5   # 简化模型
        return D65_UPVP + direction * distance

    def get_illuminant_at_time(self, t):
        """
        获取时间 t 时的光照

        参数
        ----------
        t : float
            当前时间（秒）

        返回
        -------
        (2,) ndarray
            u'v' 坐标
        float
            进度（0~1）
        """
        t = np.clip(t, 0, self.t_max)
        if t < self.t_max:
            progress = t / self.t_max
            A_t_upvp = D65_UPVP + progress * (self.terminal_upvp - D65_UPVP)
        else:
            A_t_upvp = self.terminal_upvp
            progress = 1.0
        return A_t_upvp, progress

    def apply_to_frame(self, frame, t):
        """
        对单帧应用色度适应

        参数
        ----------
        frame : (H, W, 3) ndarray
            输入图像，值域 [0, 1] 或 [0, 255]
        t : float
            当前时间（秒）

        返回
        -------
        (H, W, 3) ndarray
            处理后的图像，值域 [0, 1]
        """
        if frame.max() > 1.5:
            frame = frame.astype(np.float32) / 255.0

        A_t_upvp, _ = self.get_illuminant_at_time(t)
        A_t_xyz = upvp_to_xyz(A_t_upvp, Y=1.0)

        M_CAT = compute_cat_matrix(D65_XYZ, A_t_xyz)

        linear = sRGB2RGB(frame)
        xyz = np.dot(linear, RGB2XYZ.T)
        output_xyz = np.dot(xyz, M_CAT.T)
        output_rgb = np.dot(output_xyz, XYZ2RGB.T)
        output_rgb = np.clip(output_rgb, 0, 1)
        output_srgb = RGB2sRGB(output_rgb)

        return output_srgb