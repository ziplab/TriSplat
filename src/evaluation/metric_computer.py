import csv
import logging
import os
from pathlib import Path

import torch
from lightning.pytorch import LightningModule
from tabulate import tabulate

from ..misc.image_io import save_image, load_image_crop_resize
from ..misc.utils import get_overlap_tag
from ..visualization.annotation import add_label
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_lpips, compute_psnr, compute_ssim

logger = logging.getLogger(__name__)


class MetricComputer(LightningModule):
    cfg: EvaluationCfg

    def __init__(self, cfg: EvaluationCfg) -> None:
        super().__init__()
        self.cfg = cfg

        self.per_scene_results = []
        self.field_names = []

    def test_step(self, batch, batch_idx):
        scene = batch["scene"][0]
        b, cv, _, _, _ = batch["context"]["image"].shape
        assert b == 1
        _, v, _, _, _ = batch["target"]["image"].shape

        # Skip scenes.
        for method in self.cfg.methods:
            if not (method.path / scene).exists():
                logger.warning(f'Method "{method.name}" not found for scene "{scene}". Skipping "{scene}".')
                return

        # Load the images.
        all_images = {}
        try:
            for method in self.cfg.methods:
                base_path = method.path / scene
                images = []
                for i, index in enumerate(batch["target"]["index"][0]):
                    img_path = base_path / f"color/{index.item():0>6}.png"
                    if "InstantSplat" in method.key:
                        img_path = base_path / f"frame_{(index.item()+1):0>5}.png"
                    if "AnySplat" in method.key:
                        img_path = base_path / f"pred/{i:0>6}.jpg"
                    if not img_path.exists():
                        logger.warning(f"Image not found: {img_path}")
                    images.append(load_image_crop_resize(img_path))

                all_images[method.key] = torch.stack(images).to(self.device)
        except FileNotFoundError:
            logger.warning(f'Skipping "{scene}".')
            return

        # Compute metrics.
        overlap = batch["context"]["overlap"][0, 0]
        overlap_tag = get_overlap_tag(overlap)
        if overlap_tag == "ignore":
            return

        all_metrics = {}
        rgb_gt = batch["target"]["image"][0]
        for key, images in all_images.items():
            all_metrics = {
                **all_metrics,
                f"lpips_{key}": compute_lpips(rgb_gt, images).mean().item(),
                f"ssim_{key}": compute_ssim(rgb_gt, images).mean().item(),
                f"psnr_{key}": compute_psnr(rgb_gt, images).mean().item(),
            }
        self.log_dict(all_metrics)
        self.print_preview_metrics(all_metrics, overlap_tag)

        # save per-scene results
        self.per_scene_results.append({key: all_metrics[f"psnr_{key}"] for key in all_images.keys()})
        self.per_scene_results[-1]["scene"] = scene
        if not self.field_names:
            self.field_names = ["scene"] + list(all_images.keys())

        # Skip the rest if no side-by-side is needed.
        if self.cfg.side_by_side_path is None:
            return

        # Create side-by-side.
        scene_key = f"{batch_idx:0>6}_{scene}"
        for i in range(v):
            true_index = batch["target"]["index"][0, i]
            row = [add_label(vcat(*batch["context"]["image"][0]), "Context"),
                   add_label(vcat(batch["target"]["image"][0, i], torch.zeros_like(batch["target"]["image"][0, i])), "Ground Truth")]
            for method in self.cfg.methods:
                image = all_images[method.key][i]
                error_map = torch.abs(batch["target"]["image"][0, i] - image)
                error_map = error_map.mean(dim=0)
                error_map = apply_color_map_to_image(error_map, "jet")
                image = add_label(vcat(image, error_map), method.key + f" ({all_metrics[f'psnr_{method.key}']:.3f})")
                row.append(image)

            start_frame = batch["target"]["index"][0, 0]
            end_frame = batch["target"]["index"][0, -1]
            label = f"Scene {batch['scene'][0]} (frames {start_frame} to {end_frame}, overlap {overlap:.2f})"
            row = add_border(add_label(hcat(*row), label, font_size=16))

            psnr_diff = all_metrics[f"psnr_{self.cfg.methods[0].key}"] - all_metrics[f"psnr_{self.cfg.methods[1].key}"]
            save_image(
                row,
                self.cfg.side_by_side_path / f"{overlap_tag}_{psnr_diff:.3f}_{scene_key}" / f"{true_index:0>6}.png",
            )

            # save gt and per-method results
            for method in self.cfg.methods:
                image = all_images[method.key][i]
                save_image(
                    image,
                    self.cfg.side_by_side_path / f"{overlap_tag}_{psnr_diff:.3f}_{scene_key}" / f"{method.key}_{true_index:0>6}.png",
                )

            # save gt
            save_image(
                batch["target"]["image"][0, i],
                self.cfg.side_by_side_path / f"{overlap_tag}_{psnr_diff:.3f}_{scene_key}" / f"gt_{true_index:0>6}.png",
            )

            # save context views
            for j in range(cv):
                context_index = batch["context"]["index"][0, j]
                save_image(
                    batch["context"]["image"][0, j],
                    self.cfg.side_by_side_path / f"{overlap_tag}_{psnr_diff:.3f}_{scene_key}" / f"context_{context_index:0>6}.png",
                )

        # Create an animation.
        if self.cfg.animate_side_by_side:
            (self.cfg.side_by_side_path / "videos").mkdir(exist_ok=True, parents=True)
            command = (
                'ffmpeg -y -framerate 30 -pattern_type glob -i "*.png"  -c:v libx264 '
                '-pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"'
            )
            os.system(
                f"cd {self.cfg.side_by_side_path / scene_key} && {command} "
                f"{Path.cwd()}/{self.cfg.side_by_side_path}/videos/{scene_key}.mp4"
            )

    def on_test_end(self) -> None:
        with open(self.cfg.side_by_side_path / 'compare.csv', 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.field_names)
            writer.writeheader()
            writer.writerows(self.per_scene_results)

    def print_preview_metrics(self, metrics: dict[str, float], overlap_tag: str | None = None) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        if overlap_tag is not None:
            if getattr(self, "running_metrics_sub", None) is None:
                self.running_metrics_sub = {overlap_tag: metrics}
                self.running_metric_steps_sub = {overlap_tag: 1}
            elif overlap_tag not in self.running_metrics_sub:
                self.running_metrics_sub[overlap_tag] = metrics
                self.running_metric_steps_sub[overlap_tag] = 1
            else:
                s = self.running_metric_steps_sub[overlap_tag]
                self.running_metrics_sub[overlap_tag] = {k: ((s * v) + metrics[k]) / (s + 1)
                                                         for k, v in self.running_metrics_sub[overlap_tag].items()}
                self.running_metric_steps_sub[overlap_tag] += 1

        def print_metrics(runing_metric):
            table = []
            for method in self.cfg.methods:
                row = [
                    f"{runing_metric[f'{metric}_{method.key}']:.3f}"
                    for metric in ("psnr", "lpips", "ssim")
                ]
                table.append((method.key, *row))

            table = tabulate(table, ["Method", "PSNR (dB)", "LPIPS", "SSIM"])
            logger.info(table)

        logger.info("All Pairs:")
        print_metrics(self.running_metrics)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                logger.info(f"Overlap: {k}")
                print_metrics(v)
