import numpy as np

# Bradford 矩阵 (RGB -> sharpened LMS)
M_BRADFORD = np.array([
    [0.8951,  0.2664, -0.1614],
    [-0.7502, 1.7135,  0.0367],
    [0.0389, -0.0685,  1.0296]
])

# 逆 Bradford 矩阵
M_BRADFORD_INV = np.linalg.inv(M_BRADFORD)

# D65 在 XYZ 空间（归一化后）
D65_XYZ = np.array([0.95047, 1.0, 1.08883])

# D65 在 u'v' 空间的坐标
D65_UPVP = np.array([0.1978, 0.4683])

# 论文最优轨迹参数
OPTIMAL_ANGLE = 1.47          # 弧度
OPTIMAL_VELOCITY = 0.000467   # u'v' 空间单位/秒
T_MAX = 120                   # 2 分钟 = 120 秒

# 显示能耗权重（RGB 通道的功率权重）
WEIGHTS = np.array([231.53, 245.67, 530.75], dtype=np.float32).reshape(1, 3)
WEIGHTS_NORM = WEIGHTS / np.sum(WEIGHTS)