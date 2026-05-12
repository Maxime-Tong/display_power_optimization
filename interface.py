import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from adaptor.base import BaseScreenAdaptor
from util.constants import WEIGHTS


@dataclass
class PowerReductionResult:
    image_name: str
    output_path: Optional[str]
    power_reduction: float
    stats: Dict[str, float]
    success: bool
    error: Optional[str] = None


class ScreenPowerReductionInterface:
    """最上层接口类，负责调度 adaptor、保存结果并统计能耗降低。"""

    def __init__(
        self,
        dataset_path: str,
        screen_adaptor: BaseScreenAdaptor,
        output_dir: str = "output/optimized",
        save_opt_images: bool = True,
    ):
        self.dataset_path = dataset_path
        self.screen_adaptor = screen_adaptor
        self.output_dir = output_dir
        self.save_opt_images = save_opt_images

        first_image_path = self._get_first_image_path()
        if first_image_path is None:
            raise ValueError(f"在 {dataset_path} 中未找到图像文件")

        first_image = self._read_image(first_image_path)
        self.screen_adaptor.prepare(first_image.shape)

        if self.save_opt_images:
            os.makedirs(self.output_dir, exist_ok=True)

    def _get_first_image_path(self) -> Optional[str]:
        supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        for file_name in os.listdir(self.dataset_path):
            if file_name.lower().endswith(supported_formats):
                return os.path.join(self.dataset_path, file_name)
        return None

    def _read_image(self, image_path: str, tile_size: int = 4) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        original_size = img.size

        new_width = original_size[0] - (original_size[0] % tile_size)
        new_height = original_size[1] - (original_size[1] % tile_size)

        if new_width == original_size[0] and new_height == original_size[1]:
            return np.array(img)

        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return np.array(resized_img)

    def _save_image(self, image: np.ndarray, save_path: str):
        img = Image.fromarray(image.astype(np.uint8))
        img.save(save_path)
        return img

    def _list_images(self) -> List[str]:
        supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        return [
            file_name
            for file_name in os.listdir(self.dataset_path)
            if file_name.lower().endswith(supported_formats)
        ]

    def compute_power_reduction(self, orig_rgb: np.ndarray, opt_rgb: np.ndarray) -> Dict[str, float]:
        weights = WEIGHTS.reshape(1, 1, 3)
        orig = orig_rgb.astype(np.float32) / 255.0
        opt = opt_rgb.astype(np.float32) / 255.0

        orig_power = np.sum(weights * orig)
        opt_power = np.sum(weights * opt)

        if orig_power == 0:
            reduction_percent = 0.0
        else:
            reduction_percent = float((orig_power - opt_power) / orig_power * 100)

        from skimage.metrics import structural_similarity as ssim
        ssim_value = ssim(orig, opt, channel_axis=-1, data_range=1.0, win_size=7)
        
        return {
            "original_power": float(orig_power),
            "optimized_power": float(opt_power),
            "power_reduction_percent": reduction_percent,
            "avg_r_reduction": float(np.mean(orig[..., 0] - opt[..., 0])),
            "avg_g_reduction": float(np.mean(orig[..., 1] - opt[..., 1])),
            "avg_b_reduction": float(np.mean(orig[..., 2] - opt[..., 2])),
            "ssim": float(ssim_value),
        }

    def process_single_image(
        self,
        image_name: str,
        t: float = 0.0,
        gaze_x: int = 0,
        gaze_y: int = 0,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        image_path = os.path.join(self.dataset_path, image_name)
        original_image = self._read_image(image_path)
        optimized_image = self.screen_adaptor.apply(
            original_image,
            t=t,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
        )
        stats = self.compute_power_reduction(original_image, optimized_image)

        output_path = None
        if self.save_opt_images:
            output_path = os.path.join(self.output_dir, image_name)
            self._save_image(optimized_image, output_path)

        if verbose:
            print(f"图像: {image_name}")
            print(f"能耗降低: {stats['power_reduction_percent']:.2f}%")
            print(f"R通道平均降低: {stats['avg_r_reduction']:.2f}")
            print(f"G通道平均降低: {stats['avg_g_reduction']:.2f}")
            print(f"B通道平均降低: {stats['avg_b_reduction']:.2f}")
            print(f"SSIM: {stats['ssim']:.4f}")
            print("-" * 30)

        return {
            "image_name": image_name,
            "output_path": output_path,
            "power_reduction": stats["power_reduction_percent"],
            "stats": stats,
            "success": True,
        }

    def process_all_images(
        self,
        gaze_x: int = 0,
        gaze_y: int = 0,
        max_images: Optional[int] = None,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        images = self._list_images()
        if max_images is not None:
            images = images[:max_images]

        start_time = datetime.now()
        if verbose:
            print(f"开始处理 {len(images)} 张图像...")
            if self.save_opt_images:
                print(f"输出目录: {self.output_dir}")
            print("-" * 50)

        results: List[Dict[str, Any]] = []
        for index, image_name in enumerate(images, 1):
            if verbose and index % 10 == 0:
                print(f"进度: {index}/{len(images)}")

            result = self.process_single_image(
                image_name,
                t=index,
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                verbose=verbose,
            )
            results.append(result)

            if verbose and result["success"]:
                print(f"[{index}/{len(images)}] {image_name}: {result['power_reduction']:.2f}%")

        successful = sum(1 for item in results if item["success"])
        failed = len(results) - successful
        end_time = datetime.now()

        if verbose:
            print("-" * 50)
            print("处理完成！")
            print(f"成功: {successful} 张, 失败: {failed} 张")
            print(f"总耗时: {end_time - start_time}")

        return results

    def get_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful_results = [result for result in results if result["success"]]
        if not successful_results:
            return {"error": "没有成功处理的图像"}

        reductions = [result["power_reduction"] for result in successful_results]
        return {
            "total_images": len(results),
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
            "avg_reduction": float(np.mean(reductions)),
            "std_reduction": float(np.std(reductions)),
            "min_reduction": float(np.min(reductions)),
            "max_reduction": float(np.max(reductions)),
            "median_reduction": float(np.median(reductions)),
        }
