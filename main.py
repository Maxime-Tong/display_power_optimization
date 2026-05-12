from pathlib import Path

import torch

from adaptor import EllipseAdaptor, GradualAdaptor, AdaptiveDKLPowerOptimizer, ScreenAdaptorPipeline
from interface import ScreenPowerReductionInterface
from model import BaseColorModel
from util.constants import OPTIMAL_ANGLE, OPTIMAL_VELOCITY, T_MAX, WEIGHTS_NORM
from util.colorspace import DKL2RGB


def build_default_interface(dataset_path: str = "datasets/genshin_impact") -> ScreenPowerReductionInterface:
    root = Path(__file__).resolve().parent
    model_path = root / "model" / "model.pth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    color_model = BaseColorModel({"device": device})
    color_model.load(str(model_path))
    color_model.to_eval()

    ellipse_adaptor = EllipseAdaptor(
        color_model=color_model,
        increase_vec=WEIGHTS_NORM @ DKL2RGB,
        abc_scaler=1.0,
        ecc_no_compress=5,
        foveated=True,
        max_ecc=18,
        h_fov=110,
    )
    gradual_adaptor = GradualAdaptor(
        angle=OPTIMAL_ANGLE,
        velocity=OPTIMAL_VELOCITY,
        t_max=T_MAX,
        delta_t_jnd=5.0,
    )
    dkl_optimizer = AdaptiveDKLPowerOptimizer(k_base=(0.003, 0.003, 0.001), w_weights=WEIGHTS_NORM.flatten())
    pipeline = ScreenAdaptorPipeline([dkl_optimizer])
    # pipeline = ScreenAdaptorPipeline([ellipse_adaptor, gradual_adaptor])
    return ScreenPowerReductionInterface(
        dataset_path=dataset_path,
        screen_adaptor=pipeline,
        output_dir="output/optimized",
        save_opt_images=True,
    )


if __name__ == "__main__":
    interface = build_default_interface()
    results = interface.process_all_images(gaze_x=0, gaze_y=0, max_images=120, verbose=True)
    stats = interface.get_statistics(results)
    print("=== 统计信息 ===")
    print(f"平均能耗降低: {stats['avg_reduction']:.2f}%")
    print(f"标准差: {stats['std_reduction']:.2f}%")
    print(f"最小值: {stats['min_reduction']:.2f}%")
    print(f"最大值: {stats['max_reduction']:.2f}%")
    print(f"中位数: {stats['median_reduction']:.2f}%")
