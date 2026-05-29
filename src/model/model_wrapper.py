import csv
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities import rank_zero_only
from tabulate import tabulate
from torch import Tensor, nn
from tqdm import tqdm

from .ply_export import export_ply
from .types import Gaussians, Triangles
from ..dataset import DatasetCfgWrapper
from ..dataset.data_module import get_data_shim
from ..geometry.projection import transform_normal_cam2world
from ..dataset.types import BatchedExample
from ..evaluation.mesh.evaluate_mesh import eval_mesh
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..mesh.mesh_exporter import GSMeshExporter
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.cam_utils import update_pose
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.step_tracker import StepTracker
from ..misc.utils import inverse_normalize, vis_depth_map, get_overlap_tag, subsample_point_cloud_views, \
    clone_batch
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.smooth_context import apply_smooth_context_target
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from ..visualization.vis_normal import normal_to_rgb
from .decoder.decoder import Decoder, DecoderOutput, DepthRenderingMode
from .encoder import Encoder
from .encoder.mono_estimator import MonoNormalEstimator
from .encoder.visualization.encoder_visualizer import EncoderVisualizer

logger = logging.getLogger(__name__)


def _scalar_to_float(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        return value.detach().item()
    return float(value)


def _make_zero_scalar(
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    return torch.zeros((), device=device, dtype=dtype)


_MESH_PUBLIC_METRIC_NAMES = (
    "dist1",
    "dist2",
    "pred_to_gt",
    "gt_to_pred",
    "cd",
    "prec",
    "recal",
    "fscore",
)

_MESH_PUBLIC_COUNT_NAMES = (
    "gt_points_before",
    "gt_points_after_bbox",
    "gt_points_after_density",
    "gt_points_after_component",
    "gt_points_after",
    "pred_points_before",
    "pred_points_after_bbox",
    "pred_points_after",
)

_MESH_PUBLIC_FIELD_NAMES = (
    "scene",
    "pred_mesh_path",
    "mesh_space",
    "index_path",
    "num_context_views",
    "pose_norm_method",
    "relative_pose",
    "use_train_view",
    "use_val_view",
    "gt_filter_quantile_lo",
    "gt_filter_quantile_hi",
    "bbox_margin_ratio",
    "bbox_margin",
    "gt_density_radius_ratio",
    "gt_density_radius",
    "gt_density_min_neighbors",
    "gt_diag_before",
    "gt_diag_after",
    "gt_filter_fallback_reason",
    "pred_keep_radius_ratio",
    "pred_keep_radius_min_factor",
    "pred_keep_radius_min",
    "pred_keep_radius",
    "pred_filter_fallback_reason",
    *_MESH_PUBLIC_COUNT_NAMES,
    *_MESH_PUBLIC_METRIC_NAMES,
)


def _public_mesh_metric_row(row: dict[str, object]) -> dict[str, object]:
    return {field: row.get(field) for field in _MESH_PUBLIC_FIELD_NAMES if field in row}


def _summarize_tensor_finiteness(
    name: str,
    tensor: Tensor | None,
) -> str:
    if tensor is None:
        return f"{name}: unavailable"

    flat = tensor.detach().reshape(-1).float()
    finite_mask = torch.isfinite(flat)
    return (
        f"{name}: shape={tuple(tensor.shape)} "
        f"finite={int(finite_mask.sum().item())}/{flat.numel()}"
    )


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    backbone_lr_multiplier: float


@dataclass
class TestCfg:
    output_path: Path
    align_pose: bool
    pose_align_steps: int
    rot_opt_lr: float
    trans_opt_lr: float
    compute_scores: bool
    save_image: bool
    save_gt_image: bool
    save_video: bool
    save_compare: bool
    save_context: bool
    save_debug_info: bool
    save_scene_ranking: bool
    render_chunk_size: int
    render_interpolated_target: bool
    interpolated_target_frames: int
    export_mesh: bool
    mesh_gt_path: Optional[Path]

    post_opt_gs: bool
    post_opt_gs_iter: int
    render_smooth_context_target: bool = False
    smooth_context_frames: int = 300
    smooth_context_chaikin_refinements: int = 5


@dataclass
class NormalBootstrapCfg:
    enabled: bool = True
    teacher_source: str = "mono"
    teacher_takeover_end_step: int = 4000
    teacher_blend_end_step: int = 12000
    blend_curve: str = "cosine"
    apply_to_triangle_rotation: bool = True


@dataclass
class PrimitiveMaskCfg:
    enabled: bool = False
    source: str = "valid_mask"
    invalid_opacity: float = 0.0


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_time_skip_steps: int

    train_ignore_large_loss: float  # ignore training samples with loss larger than this value, <=0 to disable
    train_ignore_large_loss_after_steps: int  # only start filtering after this many steps
    train_ignore_large_loss_mse: float  # mse loss threshold for filtering, <=0 to disable
    train_ignore_large_loss_pose: float  # pose loss threshold for filtering, <=0 to disable
    use_mono_normal_teacher: bool = False
    mono_normal_weights_path: Optional[str] = None
    normal_bootstrap: NormalBootstrapCfg = field(default_factory=NormalBootstrapCfg)
    primitive_mask: PrimitiveMaskCfg = field(default_factory=PrimitiveMaskCfg)


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, "t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


def box(
    image: Float[Tensor, "3 height width"],
) -> Float[Tensor, "3 new_height new_width"]:
    return add_border(add_border(image), 1, 0)


def transform_rendered_normals_to_world(
    normals: Float[Tensor, "batch view 3 height width"],
    extrinsics: Float[Tensor, "batch view 4 4"],
) -> Float[Tensor, "batch view 3 height width"]:
    normals = rearrange(normals, "b v c h w -> b v h w c")
    extrinsics = rearrange(extrinsics, "b v i j -> b v () () i j")
    normals = transform_normal_cam2world(normals, extrinsics)
    return rearrange(normals, "b v h w c -> b v c h w")


def prepare_normal_panel(
    normals: Float[Tensor, "view 3 height width"],
    mask: Float[Tensor, "view 1 height width"],
    label: str,
) -> Float[Tensor, "3 panel_height panel_width"]:
    mask_bool = mask.bool()
    normals = torch.where(mask_bool, normals, 0.0)
    return add_label(vcat(*normal_to_rgb(normals, mask=mask)), label)


def prepare_rgb_panel(
    images: Float[Tensor, "view 3 height width"],
    label: str,
) -> Float[Tensor, "3 panel_height panel_width"]:
    return add_label(vcat(*images), label)


def prepare_mask_panel(
    mask: Float[Tensor, "view 1 height width"],
    label: str,
) -> Float[Tensor, "3 panel_height panel_width"]:
    mask_rgb = mask.expand(-1, 3, -1, -1).float()
    return add_label(vcat(*mask_rgb), label)


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[list[DatasetCfgWrapper] | None]
    mesh: GSMeshExporter

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        mesh: GSMeshExporter,
        eval_data_cfg: Optional[list[DatasetCfgWrapper] | None] = None,
        gaussian_downsample_ratio=1.,
        gaussians_per_axis=14,
    ) -> None:
        super().__init__()
        # Allow partial checkpoint restore when the model structure changes.
        self.strict_loading = False
        self.automatic_optimization = False
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.gradient_clip_val: float | int | None = None
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg
        self.eval_cnt = 0
        self.scene_test_results_local: list[dict[str, object]] = []
        self.mesh_test_results_local: list[dict[str, object]] = []
        self.mesh_timing_results_local: list[dict[str, object]] = []
        self.restored_global_step: int | None = None

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.mesh = mesh
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()

        self.gaussian_downsample_ratio = gaussian_downsample_ratio
        self.gaussians_per_axis = gaussians_per_axis

        self.low_opa_ratio = []
        self.normal_estimator: MonoNormalEstimator | None = None
        self.has_normal_teacher_loss = any(loss.name == "normal_teacher" for loss in self.losses)
        self.has_normal_consistency_loss = any(loss.name == "normal_consistency" for loss in self.losses)
        teacher_requested = self._uses_triangle_encoder() and (
            self.train_cfg.use_mono_normal_teacher
            or self.has_normal_teacher_loss
            or self.train_cfg.normal_bootstrap.enabled
        )
        if teacher_requested and self.train_cfg.mono_normal_weights_path is None:
            raise ValueError(
                "Mono normal teacher is enabled but mono normal weights are unavailable. "
                "Set train.mono_normal_weights_path or export MONO_NORMAL_WEIGHTS."
            )
        if (
            teacher_requested
            and self.train_cfg.mono_normal_weights_path is not None
        ):
            self.normal_estimator = MonoNormalEstimator(
                self.train_cfg.mono_normal_weights_path
            )

    def _log_nonfinite_tensor(self, name: str, tensor: Tensor | None) -> None:
        logger.warning(
            "Detected non-finite values at train step %s: %s",
            self.global_step,
            _summarize_tensor_finiteness(name, tensor),
        )

    def _has_nonfinite_tensor(self, name: str, tensor: Tensor | None) -> bool:
        if tensor is None:
            return False
        if torch.isfinite(tensor).all():
            return False
        self._log_nonfinite_tensor(name, tensor)
        return True

    def _validate_primitives(self, primitives: Gaussians | Triangles) -> bool:
        if isinstance(primitives, Triangles):
            tensors = {
                "triangle.vertices": primitives.vertices,
                "triangle.sigma": primitives.sigma,
                "triangle.opacity": primitives.opacity,
                "triangle.features": primitives.features,
                "triangle.centers": primitives.centers,
                "triangle.normals": primitives.normals,
                "triangle.scales": primitives.scales,
                "triangle.mapped_scales": primitives.mapped_scales,
            }
        else:
            tensors = {
                "gaussian.means": primitives.means,
                "gaussian.covariances": primitives.covariances,
                "gaussian.harmonics": primitives.harmonics,
                "gaussian.opacities": primitives.opacities,
                "gaussian.rotations": primitives.rotations,
                "gaussian.scales": primitives.scales,
                "gaussian.mapped_scales": primitives.mapped_scales,
            }

        valid = True
        for name, tensor in tensors.items():
            if self._has_nonfinite_tensor(name, tensor):
                valid = False
        return valid

    def _validate_decoder_output(self, output: DecoderOutput, prefix: str) -> bool:
        tensors = {
            f"{prefix}.color": output.color,
            f"{prefix}.depth": output.depth,
            f"{prefix}.opacity": output.opacity,
            f"{prefix}.rend_normal": output.rend_normal,
            f"{prefix}.surf_normal": output.surf_normal,
            f"{prefix}.projection_matrix": output.projection_matrix,
        }

        valid = True
        for name, tensor in tensors.items():
            if self._has_nonfinite_tensor(name, tensor):
                valid = False
        return valid

    def _apply_training_optimization(
        self,
        total_loss: Tensor,
        *,
        skip_optimizer_step: bool,
    ) -> None:
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            if len(optimizer) != 1:
                raise NotImplementedError("Manual optimization expects a single optimizer.")
            optimizer = optimizer[0]

        optimizer.zero_grad()
        if skip_optimizer_step:
            return

        self.manual_backward(total_loss)

        clip_val = self.gradient_clip_val
        if clip_val is not None and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm="norm",
            )

        optimizer.step()

        lr_scheduler = self.lr_schedulers()
        if isinstance(lr_scheduler, list):
            for scheduler in lr_scheduler:
                scheduler.step()
        elif lr_scheduler is not None:
            lr_scheduler.step()

    def get_decoder_debug_kwargs(self) -> dict[str, int]:
        return {
            "global_step": self.get_schedule_step(),
            "debug_log_interval": self.train_cfg.print_log_every_n_steps,
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        global_step = checkpoint.get("global_step")
        self.restored_global_step = None if global_step is None else int(global_step)

    def get_schedule_step(self) -> int:
        trainer = getattr(self, "_trainer", None)
        trainer_fn = None if trainer is None else trainer.state.fn
        if (
            trainer_fn in (TrainerFn.TESTING, TrainerFn.VALIDATING, TrainerFn.PREDICTING)
            and self.restored_global_step is not None
        ):
            return self.restored_global_step
        return self.global_step

    def _uses_triangle_encoder(self) -> bool:
        return bool(getattr(self.encoder, "use_triangle", False))

    def _build_primitive_valid_mask(
        self,
        context: dict,
        primitives: Triangles,
    ) -> Tensor | None:
        cfg = self.train_cfg.primitive_mask
        if not cfg.enabled or not isinstance(primitives, Triangles):
            return None
        if cfg.source != "valid_mask":
            raise ValueError(f"Unsupported train.primitive_mask.source: {cfg.source}")
        if "valid_mask" not in context:
            return None

        valid_mask = context["valid_mask"].to(
            device=primitives.opacity.device,
            dtype=torch.float32,
        )
        if valid_mask.dim() != 5:
            raise ValueError(
                "context.valid_mask must have shape [B,V,1,H,W], "
                f"got {tuple(valid_mask.shape)}"
            )

        b, v, _, _, _ = valid_mask.shape
        _, num_primitives, _ = primitives.opacity.shape
        if v <= 0 or num_primitives % v != 0:
            raise ValueError(
                "Triangle count is not divisible by context views: "
                f"num_primitives={num_primitives}, context_views={v}"
            )

        tokens_per_view = num_primitives // v
        image_h, image_w = context["image"].shape[-2:]
        if image_h * image_w == tokens_per_view:
            out_h, out_w = image_h, image_w
        else:
            target_ratio = image_h / max(image_w, 1)
            estimated_h = int(round(math.sqrt(tokens_per_view * target_ratio)))
            candidates = range(max(1, estimated_h - 8), estimated_h + 9)
            divisors = [
                (h, tokens_per_view // h)
                for h in candidates
                if h > 0 and tokens_per_view % h == 0
            ]
            if divisors:
                out_h, out_w = min(
                    divisors,
                    key=lambda hw: abs((hw[0] / max(hw[1], 1)) - target_ratio),
                )
            else:
                out_h = int(round(math.sqrt(tokens_per_view)))
                out_w = tokens_per_view // max(out_h, 1)
        if out_h * out_w != tokens_per_view:
            raise ValueError(
                "Cannot infer primitive mask resolution from triangle count: "
                f"tokens_per_view={tokens_per_view}"
            )

        flat_valid = valid_mask.reshape(b * v, 1, *valid_mask.shape[-2:])
        flat_valid = F.interpolate(
            flat_valid,
            size=(out_h, out_w),
            mode="nearest",
        )
        return rearrange(flat_valid > 0.5, "(b v) 1 h w -> b (v h w)", b=b, v=v)

    def _apply_primitive_mask(
        self,
        context: dict,
        primitives: Gaussians | Triangles,
        visualization_dump: dict | None = None,
    ) -> Gaussians | Triangles:
        if not isinstance(primitives, Triangles):
            return primitives

        primitive_valid_mask = self._build_primitive_valid_mask(context, primitives)
        if primitive_valid_mask is None:
            return primitives

        gate = primitive_valid_mask.unsqueeze(-1).to(
            dtype=primitives.opacity.dtype,
            device=primitives.opacity.device,
        )
        invalid_opacity = torch.as_tensor(
            self.train_cfg.primitive_mask.invalid_opacity,
            dtype=primitives.opacity.dtype,
            device=primitives.opacity.device,
        )
        opacity = torch.where(
            primitive_valid_mask.unsqueeze(-1),
            primitives.opacity,
            torch.full_like(primitives.opacity, invalid_opacity),
        )

        if visualization_dump is not None:
            invalid_ratio = 1.0 - gate.detach().float().mean()
            visualization_dump["primitive_mask_invalid_ratio"] = invalid_ratio

        return Triangles(
            vertices=primitives.vertices,
            sigma=primitives.sigma,
            opacity=opacity,
            features=primitives.features,
            centers=primitives.centers,
            normals=primitives.normals,
            scales=primitives.scales,
            mapped_scales=primitives.mapped_scales,
            primitive_valid_mask=primitive_valid_mask,
        )

    def _compute_teacher_blend_alpha(self, global_step: int) -> float:
        cfg = self.train_cfg.normal_bootstrap
        if (
            not cfg.enabled
            or not cfg.apply_to_triangle_rotation
            or cfg.teacher_source != "mono"
            or self.normal_estimator is None
            or not self._uses_triangle_encoder()
        ):
            return 0.0

        takeover_end = max(int(cfg.teacher_takeover_end_step), 0)
        blend_end = max(int(cfg.teacher_blend_end_step), takeover_end)

        if global_step <= takeover_end:
            return 1.0
        if global_step >= blend_end:
            return 0.0

        progress = (global_step - takeover_end) / max(blend_end - takeover_end, 1)
        progress = min(max(progress, 0.0), 1.0)
        if cfg.blend_curve == "linear":
            return 1.0 - progress
        if cfg.blend_curve == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        raise ValueError(f"Unsupported normal bootstrap blend curve: {cfg.blend_curve}")

    def _prepare_encoder_context(
        self,
        context: dict,
        global_step: int,
        visualization_dump: dict | None = None,
        include_teacher_for_logging: bool = False,
    ) -> dict:
        context_inputs = dict(context)
        teacher_blend_alpha = self._compute_teacher_blend_alpha(global_step)
        context_inputs["mono_normal_blend_alpha"] = teacher_blend_alpha

        needs_teacher = (
            self.normal_estimator is not None
            and self._uses_triangle_encoder()
            and (
                teacher_blend_alpha > 0.0
                or (self.training and self.has_normal_teacher_loss)
                or include_teacher_for_logging
            )
        )

        if needs_teacher:
            with torch.no_grad():
                context_mono_normal = self.normal_estimator(context["image"])
            context_inputs["mono_normal"] = context_mono_normal
            if visualization_dump is not None:
                visualization_dump["context_mono_normal"] = context_mono_normal

        if visualization_dump is not None:
            visualization_dump["teacher_blend_alpha"] = teacher_blend_alpha

        return context_inputs

    def training_step(self, batch, batch_idx):
        # combine batch from different dataloaders
        if isinstance(batch, list):
            batch_combined = None
            for batch_per_dl in batch:
                if batch_combined is None:
                    batch_combined = batch_per_dl
                else:
                    for k in batch_combined.keys():
                        if isinstance(batch_combined[k], list):
                            batch_combined[k] += batch_per_dl[k]
                        elif isinstance(batch_combined[k], dict):
                            for kk in batch_combined[k].keys():
                                batch_combined[k][kk] = torch.cat([batch_combined[k][kk], batch_per_dl[k][kk]], dim=0)
                        else:
                            raise NotImplementedError
            batch = batch_combined
        batch: BatchedExample = self.data_shim(batch)
        _, v_tgt, _, h, w = batch["target"]["image"].shape
        loss_device = batch["target"]["image"].device
        zero_loss = _make_zero_scalar(loss_device)
        ignore_after = self.train_cfg.train_ignore_large_loss_after_steps

        # Run the model.
        visualization_dump = {}
        target_gt = batch["target"]["image"]
        primitives: Gaussians | Triangles | None = None
        output: DecoderOutput | None = None
        is_triangle = False
        skip_sample = False
        skip_reasons: list[str] = []
        raw_total_loss = zero_loss
        total_loss = zero_loss
        loss_components = {}
        try:
            context_inputs = self._prepare_encoder_context(
                batch["context"],
                self.global_step,
                visualization_dump=visualization_dump,
            )
            primitives = self.encoder(
                context_inputs,
                self.global_step,
                visualization_dump=visualization_dump,
            )
            primitives = self._apply_primitive_mask(
                batch["context"],
                primitives,
                visualization_dump,
            )
            is_triangle = isinstance(primitives, Triangles)

            if not self._validate_primitives(primitives):
                skip_sample = True
                skip_reasons.append("encoder produced non-finite primitives")
            else:
                output = self.decoder.forward(
                    primitives,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=self.train_cfg.depth_mode,
                    **self.get_decoder_debug_kwargs(),
                )

                if not self._validate_decoder_output(output, prefix="train_output"):
                    skip_sample = True
                    skip_reasons.append("decoder produced non-finite target output")
                elif is_triangle and self.has_normal_consistency_loss:
                    _, v_ctx, _, ctx_h, ctx_w = batch["context"]["image"].shape
                    context_output = self.decoder.forward(
                        primitives,
                        batch["context"]["extrinsics"],
                        batch["context"]["intrinsics"],
                        batch["context"]["near"],
                        batch["context"]["far"],
                        (ctx_h, ctx_w),
                        depth_mode=self.train_cfg.depth_mode,
                        **self.get_decoder_debug_kwargs(),
                    )
                    if not self._validate_decoder_output(
                        context_output,
                        prefix="context_output",
                    ):
                        skip_sample = True
                        skip_reasons.append("decoder produced non-finite context output")
                    else:
                        visualization_dump["context_output"] = context_output
        except FloatingPointError as exc:
            skip_sample = True
            skip_reasons.append(str(exc))
            logger.warning(
                "Skipping training step %s due to numerical error: %s",
                self.global_step,
                exc,
            )

        if not skip_sample and output is not None and primitives is not None:
            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(output.color, "b v c h w -> (b v) c h w"),
            )
            if torch.isfinite(psnr_probabilistic).all():
                self.log("train/psnr_probabilistic", psnr_probabilistic.mean())
            else:
                skip_sample = True
                skip_reasons.append("non-finite train/psnr_probabilistic")
                logger.warning(
                    "Skipping training step %s due to non-finite train/psnr_probabilistic.",
                    self.global_step,
                )

        # Compute and log loss.
        if not skip_sample and output is not None and primitives is not None:
            for loss_fn in self.losses:
                raw_loss = loss_fn.forward(
                    output,
                    batch,
                    primitives,
                    self.global_step,
                    extra_info=visualization_dump,
                )
                if not isinstance(raw_loss, Tensor):
                    raw_loss = torch.as_tensor(
                        raw_loss,
                        device=loss_device,
                        dtype=torch.float32,
                    )

                raw_total_loss = raw_total_loss + raw_loss
                safe_loss = raw_loss

                if not torch.isfinite(raw_loss).all():
                    skip_sample = True
                    skip_reasons.append(f"non-finite {loss_fn.name} loss")
                    logger.warning(
                        "Skipping training step %s due to non-finite %s loss: %s",
                        self.global_step,
                        loss_fn.name,
                        _scalar_to_float(raw_loss),
                    )
                    safe_loss = _make_zero_scalar(
                        raw_loss.device,
                        raw_loss.dtype if raw_loss.is_floating_point() else torch.float32,
                    )

                if (
                    self.global_step > ignore_after
                    and loss_fn.name == "mse"
                    and self.train_cfg.train_ignore_large_loss_mse > 0
                    and torch.isfinite(raw_loss).all()
                ):
                    raw_mse = _scalar_to_float(raw_loss)
                    if raw_mse > self.train_cfg.train_ignore_large_loss_mse:
                        skip_sample = True
                        skip_reasons.append("large mse loss")
                        logger.warning(
                            "Skipping training step %s due to large mse loss: %.6f > %.6f",
                            self.global_step,
                            raw_mse,
                            self.train_cfg.train_ignore_large_loss_mse,
                        )
                        safe_loss = _make_zero_scalar(raw_loss.device, raw_loss.dtype)

                if (
                    self.global_step > ignore_after
                    and loss_fn.name == "pose"
                    and self.train_cfg.train_ignore_large_loss_pose > 0
                ):
                    pose_loss = getattr(loss_fn, "last_unweighted_loss", None)
                    if pose_loss is not None:
                        pose_loss_value = _scalar_to_float(pose_loss)
                        if not torch.isfinite(torch.as_tensor(pose_loss)).all():
                            skip_sample = True
                            skip_reasons.append("non-finite pose loss")
                            logger.warning(
                                "Skipping training step %s due to non-finite pose loss: %s",
                                self.global_step,
                                pose_loss_value,
                            )
                            safe_loss = _make_zero_scalar(raw_loss.device, raw_loss.dtype)
                        elif pose_loss_value > self.train_cfg.train_ignore_large_loss_pose:
                            skip_sample = True
                            skip_reasons.append("large pose loss")
                            logger.warning(
                                "Skipping training step %s due to large pose loss: %.6f > %.6f",
                                self.global_step,
                                pose_loss_value,
                                self.train_cfg.train_ignore_large_loss_pose,
                            )
                            safe_loss = _make_zero_scalar(raw_loss.device, raw_loss.dtype)

                self.log(f"loss/{loss_fn.name}", safe_loss)
                loss_components[loss_fn.name] = _scalar_to_float(safe_loss)
                total_loss = total_loss + safe_loss

                # Log sub-metrics (e.g., trans_loss and rot_loss from pose loss)
                if hasattr(loss_fn, "last_metrics") and loss_fn.last_metrics:
                    for k, v in loss_fn.last_metrics.items():
                        metric_value = v
                        if isinstance(v, Tensor) and not torch.isfinite(v).all():
                            skip_sample = True
                            skip_reasons.append(f"non-finite {loss_fn.name}.{k}")
                            logger.warning(
                                "Detected non-finite auxiliary metric %s for %s at step %s: %s",
                                k,
                                loss_fn.name,
                                self.global_step,
                                _scalar_to_float(v),
                            )
                            metric_value = _make_zero_scalar(v.device, v.dtype)
                        self.log(f"loss/{k}", metric_value)
                        loss_components[f"{loss_fn.name}.{k}"] = _scalar_to_float(metric_value)

            if (
                self.train_cfg.train_ignore_large_loss > 0
                and self.global_step > ignore_after
                and torch.isfinite(raw_total_loss).all()
            ):
                raw_total_loss_value = _scalar_to_float(raw_total_loss)
                if raw_total_loss_value > self.train_cfg.train_ignore_large_loss:
                    skip_sample = True
                    skip_reasons.append("large total loss")
                    logger.warning(
                        "Skipping training step %s due to large total loss: %.6f > %.6f",
                        self.global_step,
                        raw_total_loss_value,
                        self.train_cfg.train_ignore_large_loss,
                    )

            if is_triangle:
                opcities = primitives.opacity.flatten()
                ratio_opacity = (opcities < 0.01).float().mean()
                self.log("info/ratio_opacity<0.01", ratio_opacity)
            else:
                opcities = primitives.opacities.flatten()
                ratio_opacity = (opcities < 0.01).float().mean()
                self.log("info/ratio_opacity<0.01", ratio_opacity)
                self.log_gaussian_status(batch["context"]["image"], primitives, visualization_dump)

        logged_total_loss = total_loss if not skip_sample else zero_loss
        self.log("loss/total", logged_total_loss)
        self.log("info/skipped_batch", float(skip_sample))

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            loss_breakdown = ", ".join(
                f"{name}={value:.6f}" for name, value in loss_components.items()
            )
            logger.info(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"loss = {_scalar_to_float(logged_total_loss):.6f}"
                + (f"; loss_breakdown = {loss_breakdown}" if loss_breakdown else "")
                + (
                    f"; skip_reasons = {', '.join(dict.fromkeys(skip_reasons))}"
                    if skip_reasons
                    else ""
                )
            )
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if skip_sample:
            logger.warning(
                "Skipping optimizer update at train step %s. Reasons: %s",
                self.global_step,
                "; ".join(dict.fromkeys(skip_reasons)),
            )

        self._apply_training_optimization(
            total_loss,
            skip_optimizer_step=skip_sample,
        )

        return logged_total_loss.detach()

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        schedule_step = self.get_schedule_step()

        b, v_tgt, _, h, w = batch["target"]["image"].shape
        assert b == 1
        smooth_context_metadata: dict[str, object] | None = None
        if self.test_cfg.render_smooth_context_target:
            batch = clone_batch(batch)
            batch, smooth_context_metadata = apply_smooth_context_target(
                batch,
                num_frames=int(self.test_cfg.smooth_context_frames),
                chaikin_refinements=int(self.test_cfg.smooth_context_chaikin_refinements),
            )
            b, v_tgt, _, h, w = batch["target"]["image"].shape

        if self.test_cfg.render_interpolated_target:
            num_frames = int(self.test_cfg.interpolated_target_frames)
            if num_frames < 2:
                raise ValueError("test.interpolated_target_frames must be at least 2.")
            if v_tgt < 2:
                raise ValueError("Interpolated target rendering needs at least two target poses.")

            batch = clone_batch(batch)
            source_extrinsics = batch["target"]["extrinsics"][0]
            source_intrinsics = batch["target"]["intrinsics"][0]
            source_near = batch["target"]["near"][:, :1]
            source_far = batch["target"]["far"][:, :1]
            source_image = batch["target"]["image"][:, :1]

            positions = source_extrinsics[:, :3, 3]
            segment_lengths = (positions[1:] - positions[:-1]).norm(dim=-1)
            cumulative = torch.cat(
                [torch.zeros(1, dtype=segment_lengths.dtype, device=segment_lengths.device), segment_lengths.cumsum(dim=0)],
                dim=0,
            )
            total_length = cumulative[-1]
            if total_length <= 1e-8:
                sample_positions = torch.linspace(0, v_tgt - 1, num_frames, device=self.device)
                segment_hi = sample_positions.ceil().long().clamp(1, v_tgt - 1)
                segment_lo = segment_hi - 1
                local_t = (sample_positions - segment_lo).clamp(0, 1)
            else:
                sample_lengths = torch.linspace(0, total_length, num_frames, device=self.device)
                segment_hi = torch.searchsorted(cumulative, sample_lengths).clamp(1, v_tgt - 1)
                segment_lo = segment_hi - 1
                denom = (cumulative[segment_hi] - cumulative[segment_lo]).clamp_min(1e-8)
                local_t = ((sample_lengths - cumulative[segment_lo]) / denom).clamp(0, 1)

            interpolated_extrinsics = []
            interpolated_intrinsics = []
            for lo, hi, t_value in zip(segment_lo.tolist(), segment_hi.tolist(), local_t):
                interpolated_extrinsics.append(
                    interpolate_extrinsics(
                        source_extrinsics[lo],
                        source_extrinsics[hi],
                        t_value[None],
                    )[0]
                )
                interpolated_intrinsics.append(
                    interpolate_intrinsics(
                        source_intrinsics[lo],
                        source_intrinsics[hi],
                        t_value[None],
                    )[0]
                )

            batch["target"]["extrinsics"] = torch.stack(interpolated_extrinsics, dim=0)[None]
            batch["target"]["intrinsics"] = torch.stack(interpolated_intrinsics, dim=0)[None]
            batch["target"]["near"] = source_near.expand(b, num_frames).clone()
            batch["target"]["far"] = source_far.expand(b, num_frames).clone()
            batch["target"]["image"] = source_image.expand(b, num_frames, -1, -1, -1).clone()
            batch["target"]["index"] = torch.arange(num_frames, device=self.device, dtype=torch.long)[None]
            v_tgt = num_frames

        if batch_idx % 100 == 0:
            logger.info(f"Test step {batch_idx:0>6}.")
        if batch_idx == 0 and self.global_rank == 0:
            logger.info(
                "Test schedule step = %d (trainer.global_step=%d, restored_global_step=%s).",
                schedule_step,
                self.global_step,
                self.restored_global_step,
            )

        # Render primitives.
        visualization_dump = {"collect_geom_normal": True}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_enc_start = time.perf_counter()
        with self.benchmarker.time("encoder"):
            context_inputs = self._prepare_encoder_context(
                batch["context"],
                schedule_step,
                visualization_dump=visualization_dump,
            )
            primitives = self.encoder(
                context_inputs,
                schedule_step,
                visualization_dump=visualization_dump,
            )
            primitives = self._apply_primitive_mask(
                batch["context"],
                primitives,
                visualization_dump,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _encoder_time = time.perf_counter() - _t_enc_start
        is_triangle = isinstance(primitives, Triangles)
        direct_source_strategy = getattr(self.mesh.cfg, "direct_source_view_strategy", "")
        direct_budget = int(getattr(self.mesh.cfg, "direct_triangle_budget", 0) or 0)
        collect_triangle_visibility = (
            is_triangle
            and self.test_cfg.export_mesh
            and self.mesh.cfg.export_mode == "direct"
            and (
                bool(getattr(self.mesh.cfg, "direct_target_visibility_cull", False))
                or bool(getattr(self.mesh.cfg, "direct_write_frame_meshes", False))
                or direct_source_strategy == "target_visibility_topk"
                or (
                    direct_budget > 0
                    and bool(getattr(self.mesh.cfg, "direct_visibility_rank_budget", True))
                )
            )
            and not bool(getattr(self.mesh.cfg, "direct_surface_proxy", False))
        )

        if not is_triangle:
            low_opa_ratio = (primitives.opacities < self.decoder.prune_opacity_threshold).float().mean()
            self.low_opa_ratio.append(low_opa_ratio.item())
            print(f'All: low opacity ratio: {np.mean(self.low_opa_ratio):.4f}')
            print(f'Current: low opacity ratio: {low_opa_ratio.item():.4f}')

        if self.test_cfg.post_opt_gs and not is_triangle:
            extrinsic = visualization_dump['c2w'].clone()
            primitives_opt, extrinsic_opt = self.opt_gaussian_pose(batch, primitives, extrinsic, visualization_dump['scales'], visualization_dump['rotations'])
            primitives = primitives_opt
        elif self.test_cfg.post_opt_gs and is_triangle and batch_idx == 0 and self.global_rank == 0:
            logger.info("Skipping test.post_opt_gs for triangle primitives; pose align is the supported test-time pose optimization path.")

        # align the target pose
        if self.test_cfg.align_pose:
            if v_tgt < self.test_cfg.render_chunk_size:
                output_align = self.test_step_align(batch, primitives, visualization_dump["c2w"])
            else:
                output_align_img = []
                output_align_depth = []
                batch_chunk = clone_batch(batch)
                for frames_start_idx in range(0, v_tgt, self.test_cfg.render_chunk_size):
                    frames_end_idx = min(frames_start_idx + self.test_cfg.render_chunk_size, v_tgt)
                    batch_chunk["target"]["image"] = batch["target"]["image"][:, frames_start_idx:frames_end_idx]
                    batch_chunk["target"]["extrinsics"] = batch["target"]["extrinsics"][:, frames_start_idx:frames_end_idx]
                    batch_chunk["target"]["intrinsics"] = batch["target"]["intrinsics"][:, frames_start_idx:frames_end_idx]
                    batch_chunk["target"]["near"] = batch["target"]["near"][:, frames_start_idx:frames_end_idx]
                    batch_chunk["target"]["far"] = batch["target"]["far"][:, frames_start_idx:frames_end_idx]
                    batch_chunk["target"]["index"] = batch["target"]["index"][:, frames_start_idx:frames_end_idx]

                    output_align_chunk = self.test_step_align(batch_chunk, primitives, visualization_dump["c2w"])
                    output_align_img.append(output_align_chunk.color)
                    output_align_depth.append(output_align_chunk.depth)

                    # Clear memory
                    torch.cuda.empty_cache()
                output_align = type(output_align_chunk)  # DecoderOutput
                output_align.color = torch.cat(output_align_img, dim=1)
                output_align.depth = torch.cat(output_align_depth, dim=1)

            output = output_align
        else:
            # chunk inferencing
            output_img = []
            output_depth = []
            output_visibility = []
            for frames_start_idx in range(0, v_tgt, self.test_cfg.render_chunk_size):
                frames_end_idx = min(frames_start_idx + self.test_cfg.render_chunk_size, v_tgt)
                num_calls = frames_end_idx - frames_start_idx

                decoder_kwargs = self.get_decoder_debug_kwargs()
                if collect_triangle_visibility:
                    decoder_kwargs["return_triangle_visibility_mask"] = True
                with self.benchmarker.time("decoder", num_calls=num_calls):
                    output = self.decoder.forward(
                        primitives,
                        batch["target"]["extrinsics"][:, frames_start_idx:frames_end_idx],
                        batch["target"]["intrinsics"][:, frames_start_idx:frames_end_idx],
                        batch["target"]["near"][:, frames_start_idx:frames_end_idx],
                        batch["target"]["far"][:, frames_start_idx:frames_end_idx],
                        (h, w),
                        **decoder_kwargs,
                    )
                output_img.append(output.color)
                output_depth.append(output.depth)
                if collect_triangle_visibility:
                    output_visibility.append(output.triangle_visibility_mask)

                # Clear memory
                torch.cuda.empty_cache()

            output.color = torch.cat(output_img, dim=1)
            output.depth = torch.cat(output_depth, dim=1)
            if collect_triangle_visibility and all(item is not None for item in output_visibility):
                output.triangle_visibility_mask = torch.cat(output_visibility, dim=1)

        rgb_pred = output.color[0]
        rgb_gt = batch["target"]["image"][0]
        scene_metric_values: dict[str, Tensor] | None = None

        # compute scores
        if self.test_cfg.compute_scores or self.test_cfg.save_scene_ranking:
            # overlap = batch["context"]["overlap"][0]
            # overlap_tag = get_overlap_tag(overlap)
            overlap_tag = None  # disable overlap tag

            scene_metric_values = {
                "lpips": compute_lpips(rgb_gt, rgb_pred),
                "ssim": compute_ssim(rgb_gt, rgb_pred),
                "psnr": compute_psnr(rgb_gt, rgb_pred),
            }
            all_metrics = {
                "lpips_ours": scene_metric_values["lpips"].mean(),
                "ssim_ours": scene_metric_values["ssim"].mean(),
                "psnr_ours": scene_metric_values["psnr"].mean(),
            }
            methods = ['ours']

            # if self.test_cfg.align_pose:
            #     rgb_pred_align = output_align.color[0]
            #     all_metrics[f"lpips_align"] = compute_lpips(rgb_gt, rgb_pred_align).mean()
            #     all_metrics[f"ssim_align"] = compute_ssim(rgb_gt, rgb_pred_align).mean()
            #     all_metrics[f"psnr_align"] = compute_psnr(rgb_gt, rgb_pred_align).mean()
            #     methods.append('align')

            if self.test_cfg.compute_scores:
                self.log_dict(all_metrics)
                self.print_preview_metrics(all_metrics, methods, overlap_tag=overlap_tag)

        # # if align pose, save the aligned output
        # if self.test_cfg.align_pose:
        #     output = output_align

        # Save images.
        (scene,) = batch["scene"]
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name
        scene_path = path / scene

        output_train = None
        if self.test_cfg.export_mesh:
            if self.mesh.cfg.export_mode == "direct" and not isinstance(primitives, Triangles):
                raise TypeError(
                    "test.export_mesh with mesh.tsdf_gs2d.export_mode=direct requires "
                    "triangle primitives; use a triangle experiment (e.g. trisplat_re10k_triangle) "
                    "and a triangle checkpoint."
                )
            need_decoder_for_mesh = self.mesh.cfg.export_mode != "direct"
            if need_decoder_for_mesh:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _t_dec_start = time.perf_counter()
                try:
                    output_train = self.decoder.forward(
                        primitives,
                        batch["context"]["extrinsics"],
                        batch["context"]["intrinsics"],
                        batch["context"]["near"],
                        batch["context"]["far"],
                        (h, w),
                        **self.get_decoder_debug_kwargs(),
                    )
                except TypeError:
                    output_train = self.decoder.forward(
                        primitives,
                        batch["context"]["extrinsics"],
                        batch["context"]["intrinsics"],
                        batch["context"]["near"],
                        batch["context"]["far"],
                        (h, w),
                        **self.get_decoder_debug_kwargs(),
                    )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                _decoder_time = time.perf_counter() - _t_dec_start
            else:
                output_train = None
                _decoder_time = None
            opacity_kwargs = {}
            if isinstance(primitives, Triangles) and hasattr(self.decoder, "cfg"):
                dcfg = self.decoder.cfg
                opacity_kwargs = dict(
                    global_step=schedule_step,
                    opacity_temp_initial=getattr(dcfg, "opacity_temp_initial", 1.0),
                    opacity_temp_final=getattr(dcfg, "opacity_temp_final", 1.0),
                    opacity_temp_warmup_steps=getattr(dcfg, "opacity_temp_warmup_steps", 5000),
                )
            mesh_export = self.mesh.main(
                output_train,
                output,
                batch,
                primitives,
                str(path / scene / "mesh/"),
                gt_pointcloud_root=self.test_cfg.mesh_gt_path,
                encoder_time=_encoder_time,
                decoder_time=_decoder_time,
                **opacity_kwargs,
            )
            pred_mesh_path = mesh_export.output_path
            if mesh_export.direct_timing is not None:
                direct_mesh_path = mesh_export.direct_output_path or pred_mesh_path
                self.mesh_timing_results_local.append(
                    {
                        "scene": scene,
                        "pred_mesh_path": str(Path(direct_mesh_path)),
                        "export_mode": "direct",
                        **mesh_export.direct_timing,
                    }
                )
            if mesh_export.tsdf_timing is not None:
                tsdf_mesh_path = mesh_export.tsdf_output_path or pred_mesh_path
                self.mesh_timing_results_local.append(
                    {
                        "scene": scene,
                        "pred_mesh_path": str(Path(tsdf_mesh_path)),
                        "export_mode": "tsdf",
                        **mesh_export.tsdf_timing,
                    }
                )
            if self.test_cfg.mesh_gt_path is not None:
                mesh_metrics_path = path / scene / "mesh_metrics.json"
                mesh_result_base = {
                    "scene": scene,
                    "pred_mesh_path": str(Path(pred_mesh_path)),
                }
                try:
                    mesh_metrics = eval_mesh(
                        pred_mesh_path,
                        self.test_cfg.mesh_gt_path,
                        scene,
                        mesh_space_path=mesh_export.space_metadata_path,
                    )
                except ValueError as exc:
                    logger.warning(
                        "Skipping mesh metrics for scene %s because mesh evaluation failed: %s",
                        scene,
                        exc,
                    )
                    with open(mesh_metrics_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                **mesh_result_base,
                                "error": str(exc),
                            },
                            f,
                            indent=2,
                        )
                else:
                    mesh_result = {
                        **mesh_result_base,
                        **_public_mesh_metric_row(mesh_metrics["gt_trimmed"]),
                    }
                    self.mesh_test_results_local.append(mesh_result)
                    with open(mesh_metrics_path, "w", encoding="utf-8") as f:
                        json.dump(mesh_result, f, indent=2)

        if self.test_cfg.save_image:
            for index, color in zip(batch["target"]["index"][0], output.color[0]):
                save_image(color, scene_path / f"color/{index:0>6}.png")

        if smooth_context_metadata is not None and self.global_rank == 0:
            with (scene_path / "smooth_context_trajectory.json").open("w", encoding="utf-8") as f:
                json.dump(smooth_context_metadata, f, indent=2)

        if self.test_cfg.save_gt_image:
            for index, color in zip(batch["target"]["index"][0], batch["target"]["image"][0]):
                save_image(color, scene_path / f"gt/{index:0>6}.png")

        if self.test_cfg.save_context:
            for index, color in zip(batch["context"]["index"][0], batch["context"]["image"][0]):
                save_image(color, scene_path / f"context/{index:0>6}.png")

        if self.test_cfg.save_scene_ranking:
            if scene_metric_values is None:
                raise RuntimeError(
                    "Scene ranking requires per-scene metrics, but no scene metrics were computed."
                )

            context_indices = [int(index) for index in batch["context"]["index"][0].detach().cpu().tolist()]
            target_indices = [int(index) for index in batch["target"]["index"][0].detach().cpu().tolist()]
            scene_output_dir = Path(scene)
            metrics_rel_path = scene_output_dir / "metrics.json"
            frame_results = []
            for index, lpips_value, ssim_value, psnr_value in zip(
                target_indices,
                scene_metric_values["lpips"].detach().cpu().tolist(),
                scene_metric_values["ssim"].detach().cpu().tolist(),
                scene_metric_values["psnr"].detach().cpu().tolist(),
            ):
                pred_rel_path = scene_output_dir / "color" / f"{index:0>6}.png" if self.test_cfg.save_image else None
                gt_rel_path = scene_output_dir / "gt" / f"{index:0>6}.png" if self.test_cfg.save_gt_image else None
                frame_results.append(
                    {
                        "index": index,
                        "lpips": float(lpips_value),
                        "ssim": float(ssim_value),
                        "psnr": float(psnr_value),
                        "pred_path": None if pred_rel_path is None else str(pred_rel_path),
                        "gt_path": None if gt_rel_path is None else str(gt_rel_path),
                    }
                )

            scene_result = {
                "scene": scene,
                "context_indices": context_indices,
                "target_indices": target_indices,
                "num_context_views": len(context_indices),
                "num_target_views": len(target_indices),
                "lpips_mean": _scalar_to_float(scene_metric_values["lpips"].mean()),
                "ssim_mean": _scalar_to_float(scene_metric_values["ssim"].mean()),
                "psnr_mean": _scalar_to_float(scene_metric_values["psnr"].mean()),
                "scene_dir": str(scene_output_dir),
                "metrics_path": str(metrics_rel_path),
                "frames": frame_results,
            }
            self.scene_test_results_local.append(scene_result)
            scene_path.mkdir(parents=True, exist_ok=True)
            with (scene_path / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(scene_result, f, indent=2)

        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            frame_str = frame_str[:80]  # avoid too long file name
            save_video(
                [a for a in output.color[0]],
                path / "video" / f"{scene}_frame_{frame_str}.mp4",
            )

        if self.test_cfg.save_compare:
            # Construct comparison image.
            context_img = inverse_normalize(batch["context"]["image"][0])
            comparison = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
            )
            save_image(comparison, path / f"{scene}.png")

        if self.test_cfg.save_debug_info:
            if output_train is None and output.depth is not None:
                try:
                    output_train = self.decoder.forward(
                        primitives,
                        batch["context"]["extrinsics"],
                        batch["context"]["intrinsics"],
                        batch["context"]["near"],
                        batch["context"]["far"],
                        (h, w),
                        **self.get_decoder_debug_kwargs(),
                    )
                except TypeError:
                    output_train = self.decoder.forward(
                        primitives,
                        batch["context"]["extrinsics"],
                        batch["context"]["intrinsics"],
                        batch["context"]["near"],
                        batch["context"]["far"],
                        (h, w),
                    )
            rgb_gt = batch["target"]["image"][0]
            rgb_pred = output.color[0]

            save_path = path / scene / f"debug_info"
            raw_depth_path = path / scene / "raw_depth"
            raw_alpha_path = path / scene / "raw_alpha"

            # direct depth from predicted points (used for visualization only)
            global_depth = visualization_dump["depth"][0].squeeze()
            global_depth = vis_depth_map(global_depth.contiguous())
            local_depth = visualization_dump["local_pts"][0][..., -1].squeeze()
            local_depth = vis_depth_map(local_depth.contiguous())

            context_img = batch["context"]["image"][0]

            context_vis = hcat(
                add_label(vcat(*context_img), "Context"),
                add_label(vcat(*global_depth), "Global Depth"),
                add_label(vcat(*local_depth), "Local Depth"),
            )

            target_depth = vis_depth_map(output.depth[0])
            target_vis = hcat(
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_pred), "Target (Prediction)"),
                add_label(vcat(*target_depth), "Target (Depth)"),
            )

            save_image(context_vis, save_path / f"{scene}_context.png")
            save_image(target_vis, save_path / f"{scene}_target.png")

            for index, depth in zip(batch["target"]["index"][0], target_depth):
                save_image(depth, save_path / f"depth/{index:0>6}.png")
            if output.depth is not None:
                for index, depth in zip(batch["target"]["index"][0], output.depth[0]):
                    depth_path = raw_depth_path / "target" / f"{int(index):0>6}.npy"
                    depth_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(depth_path, depth.detach().cpu().numpy())
            if output.opacity is not None:
                for index, alpha in zip(batch["target"]["index"][0], output.opacity[0]):
                    alpha_path = raw_alpha_path / "target" / f"{int(index):0>6}.npy"
                    alpha_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(alpha_path, alpha.detach().cpu().numpy())
            if output_train is not None and output_train.depth is not None:
                for index, depth in zip(batch["context"]["index"][0], output_train.depth[0]):
                    depth_path = raw_depth_path / "context" / f"{int(index):0>6}.npy"
                    depth_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(depth_path, depth.detach().cpu().numpy())
            if output_train is not None and output_train.opacity is not None:
                for index, alpha in zip(batch["context"]["index"][0], output_train.opacity[0]):
                    alpha_path = raw_alpha_path / "context" / f"{int(index):0>6}.npy"
                    alpha_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(alpha_path, alpha.detach().cpu().numpy())

            if not is_triangle:
                means = rearrange(
                    primitives.means, "() (v h w spp) xyz -> h w spp v xyz", spp=1, h=h, w=w
                )
                opacities = rearrange(
                    primitives.opacities, "() (v h w spp) -> h w spp v", spp=1, h=h, w=w
                )
                GAUSSIAN_TRIM = 8
                mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
                mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1
                mask = mask & (opacities > 0.01)

                def trim(element):
                    element = rearrange(
                        element, "() (v h w spp) ... -> h w spp v ...", spp=1, h=h, w=w
                    )
                    return element[mask][None]

                output_Gaussian_path = save_path / f"{scene}_gaussians.ply"
                export_ply(
                    trim(primitives.means)[0],
                    trim(visualization_dump["scales"])[0],
                    trim(visualization_dump["rotations"])[0],
                    trim(primitives.harmonics)[0],
                    trim(primitives.opacities)[0],
                    output_Gaussian_path,
                    save_sh_dc_only=True,
                )

                output_pc_path = save_path / f"{scene}_point_cloud.ply"
                gaussian_cts = visualization_dump["means"].squeeze(-2).cpu().numpy().reshape(-1, 3)
                colors = rearrange(primitives.harmonics, "b (v h w) d3 d_sh -> b v h w d3 d_sh", h=h, w=w)[..., 0]
                colors = (colors + 1) / 2.
                colors = (colors * 255).cpu().numpy().astype(np.uint8).reshape(-1, 3)
                colors = np.concatenate([colors.reshape(-1, 3), (torch.ones_like(primitives.opacities)).cpu().numpy().astype(np.uint8).reshape(-1, 1)], axis=-1)

                try:
                    import trimesh
                except ImportError:
                    trimesh = None
                if trimesh is not None:
                    pc = trimesh.PointCloud(vertices=gaussian_cts)
                    pc.colors = colors
                    result = pc.export(file_type='ply')
                    with open(output_pc_path, 'wb') as f:
                        f.write(result)

    def test_step_align(self, batch, primitives, _pred_camera_poses):
        self.encoder.eval()
        schedule_step = self.get_schedule_step()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["target"]["image"].shape
        is_triangle = isinstance(primitives, Triangles)
        with torch.set_grad_enabled(True):
            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            pose_optimizer = torch.optim.Adam(opt_params)

            extrinsics = batch["target"]["extrinsics"].clone()

            with self.benchmarker.time("optimize"):
                for i in range(self.test_cfg.pose_align_steps):
                    pose_optimizer.zero_grad()

                    if is_triangle:
                        render_extrinsics = update_pose(
                            cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                            cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                            extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
                        )
                        render_extrinsics = rearrange(
                            render_extrinsics,
                            "(b v) i j -> b v i j",
                            b=b,
                            v=v,
                        )
                        output = self.render_triangle_pose_aligned_step(
                            primitives,
                            extrinsics,
                            render_extrinsics,
                            batch["target"]["intrinsics"],
                            batch["target"]["near"],
                            batch["target"]["far"],
                            (h, w),
                        )
                    else:
                        output = self.decoder.forward(
                            primitives,
                            extrinsics,
                            batch["target"]["intrinsics"],
                            batch["target"]["near"],
                            batch["target"]["far"],
                            (h, w),
                            cam_rot_delta=cam_rot_delta,
                            cam_trans_delta=cam_trans_delta,
                            **self.get_decoder_debug_kwargs(),
                        )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, primitives, schedule_step)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    with torch.no_grad():
                        pose_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

        # Render primitives with the optimized target poses.
        output = self.decoder.forward(
            primitives,
            extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            **self.get_decoder_debug_kwargs(),
        )

        return output

    def render_triangle_pose_aligned_step(
        self,
        triangles: Triangles,
        base_extrinsics: Float[Tensor, "batch view 4 4"],
        render_extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
    ) -> DecoderOutput:
        batch_outputs: list[list[DecoderOutput]] = []
        for batch_idx in range(base_extrinsics.shape[0]):
            triangle_batch = self.slice_triangle_batch(triangles, batch_idx)
            view_outputs = []
            for view_idx in range(base_extrinsics.shape[1]):
                scene_transform = base_extrinsics[batch_idx, view_idx] @ torch.linalg.inv(
                    render_extrinsics[batch_idx, view_idx]
                )
                transformed_vertices = self.transform_triangle_vertices(
                    triangle_batch.vertices,
                    scene_transform,
                )
                view_triangles = Triangles(
                    vertices=transformed_vertices,
                    sigma=triangle_batch.sigma,
                    opacity=triangle_batch.opacity,
                    features=triangle_batch.features,
                    centers=triangle_batch.centers,
                    normals=triangle_batch.normals,
                    scales=triangle_batch.scales,
                    mapped_scales=triangle_batch.mapped_scales,
                    primitive_valid_mask=triangle_batch.primitive_valid_mask,
                )
                view_outputs.append(
                    self.decoder.forward(
                        view_triangles,
                        base_extrinsics[batch_idx : batch_idx + 1, view_idx : view_idx + 1],
                        intrinsics[batch_idx : batch_idx + 1, view_idx : view_idx + 1],
                        near[batch_idx : batch_idx + 1, view_idx : view_idx + 1],
                        far[batch_idx : batch_idx + 1, view_idx : view_idx + 1],
                        image_shape,
                        **self.get_decoder_debug_kwargs(),
                    )
                )
            batch_outputs.append(view_outputs)
        return self.combine_decoder_outputs(batch_outputs)

    @staticmethod
    def transform_triangle_vertices(
        vertices: Float[Tensor, "batch num_triangles 3 3"],
        scene_transform: Float[Tensor, "4 4"],
    ) -> Float[Tensor, "batch num_triangles 3 3"]:
        ones = torch.ones_like(vertices[..., :1])
        vertices_h = torch.cat((vertices, ones), dim=-1)
        transformed_vertices_h = torch.matmul(vertices_h, scene_transform.transpose(0, 1))
        return transformed_vertices_h[..., :3]

    @staticmethod
    def slice_triangle_batch(triangles: Triangles, batch_idx: int) -> Triangles:
        def maybe_slice(value):
            return None if value is None else value[batch_idx : batch_idx + 1]

        return Triangles(
            vertices=triangles.vertices[batch_idx : batch_idx + 1],
            sigma=triangles.sigma[batch_idx : batch_idx + 1],
            opacity=triangles.opacity[batch_idx : batch_idx + 1],
            features=triangles.features[batch_idx : batch_idx + 1],
            centers=maybe_slice(triangles.centers),
            normals=maybe_slice(triangles.normals),
            scales=maybe_slice(triangles.scales),
            mapped_scales=maybe_slice(triangles.mapped_scales),
            primitive_valid_mask=maybe_slice(triangles.primitive_valid_mask),
        )

    @staticmethod
    def combine_decoder_outputs(batch_outputs: list[list[DecoderOutput]]) -> DecoderOutput:
        first_output = batch_outputs[0][0]

        def combine_attr(name: str):
            first_value = getattr(first_output, name)
            if first_value is None:
                return None
            per_batch = [
                torch.cat([getattr(view_output, name) for view_output in view_outputs], dim=1)
                for view_outputs in batch_outputs
            ]
            return torch.cat(per_batch, dim=0)

        return DecoderOutput(
            color=combine_attr("color"),
            depth=combine_attr("depth"),
            opacity=combine_attr("opacity"),
            rend_normal=combine_attr("rend_normal"),
            surf_normal=combine_attr("surf_normal"),
            projection_matrix=combine_attr("projection_matrix"),
            triangle_visibility_mask=combine_attr("triangle_visibility_mask"),
        )

    def opt_gaussian_pose(self, batch, gaussians, pred_poses, scales, rotations):
        self.encoder.eval()
        schedule_step = self.get_schedule_step()
        # freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        b, v, _, h, w = batch["context"]["image"].shape
        with torch.set_grad_enabled(True):
            gaussians_opt = Gaussians(
                nn.Parameter(gaussians.means.clone(), requires_grad=True),
                nn.Parameter(gaussians.covariances.clone(), requires_grad=True),
                nn.Parameter(gaussians.harmonics.clone(), requires_grad=True),
                nn.Parameter(gaussians.opacities.clone(), requires_grad=True),
                gaussians.rotations,
                gaussians.scales,
                gaussians.mapped_scales,
            )

            cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))
            cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], requires_grad=True, device=self.device))

            opt_params = []
            # pose parameters
            opt_params.append(
                {
                    "params": [cam_rot_delta],
                    "lr": self.test_cfg.rot_opt_lr,
                }
            )
            opt_params.append(
                {
                    "params": [cam_trans_delta],
                    "lr": self.test_cfg.trans_opt_lr,
                }
            )
            # gaussian parameters
            opt_params.append(
                {
                    "params": [gaussians_opt.means],
                    "lr": 0.0016,
                }
            )
            opt_params.append(
                {
                    "params": [gaussians_opt.harmonics],
                    "lr": 0.0025,
                }
            )

            post_optimizer = torch.optim.Adam(opt_params)

            extrinsics = pred_poses
            with self.benchmarker.time("optimize_gs"):
                for i in range(self.test_cfg.post_opt_gs_iter):
                    post_optimizer.zero_grad(set_to_none=True)

                    output = self.decoder.forward(
                        gaussians_opt,
                        extrinsics,
                        batch["context"]["intrinsics"],
                        batch["context"]["near"],
                        batch["context"]["far"],
                        (h, w),
                        cam_rot_delta=cam_rot_delta,
                        cam_trans_delta=cam_trans_delta,
                        **self.get_decoder_debug_kwargs(),
                    )

                    # Compute and log loss.
                    total_loss = 0
                    for loss_fn in self.losses:
                        loss = loss_fn.forward(output, batch, gaussians, schedule_step, use_context=True)
                        total_loss = total_loss + loss

                    total_loss.backward()
                    with torch.no_grad():
                        post_optimizer.step()
                        new_extrinsic = update_pose(cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                                                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                                                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j")
                                                    )
                        cam_rot_delta.data.fill_(0)
                        cam_trans_delta.data.fill_(0)

                        extrinsics = rearrange(new_extrinsic, "(b v) i j -> b v i j", b=b, v=v)

        return gaussians_opt, extrinsics

    def on_test_end(self) -> None:
        output_root = self.get_test_output_root()
        self.benchmarker.dump(output_root / "benchmark.json")
        self.benchmarker.dump_memory(
            output_root / "peak_memory.json"
        )
        self.benchmarker.summarize()

        if self.test_cfg.export_mesh:
            shard_dir = output_root / "mesh_timing_shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / f"rank_{self.global_rank:04d}.json"
            with shard_path.open("w", encoding="utf-8") as f:
                json.dump(self.mesh_timing_results_local, f, indent=2)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if not dist.is_available() or not dist.is_initialized() or self.global_rank == 0:
                timing_results = []
                if dist.is_available() and dist.is_initialized():
                    for rank in range(dist.get_world_size()):
                        rank_shard_path = shard_dir / f"rank_{rank:04d}.json"
                        if not rank_shard_path.exists():
                            raise FileNotFoundError(f"Missing mesh timing shard: {rank_shard_path}")
                        with rank_shard_path.open("r", encoding="utf-8") as f:
                            timing_results.extend(json.load(f))
                else:
                    timing_results = list(self.mesh_timing_results_local)
                if timing_results:
                    self.write_mesh_timing_summary(output_root, timing_results)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if self.test_cfg.export_mesh and self.test_cfg.mesh_gt_path is not None:
            shard_dir = output_root / "mesh_metrics_shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / f"rank_{self.global_rank:04d}.json"
            with shard_path.open("w", encoding="utf-8") as f:
                json.dump(self.mesh_test_results_local, f, indent=2)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if not dist.is_available() or not dist.is_initialized() or self.global_rank == 0:
                mesh_results = []
                if dist.is_available() and dist.is_initialized():
                    for rank in range(dist.get_world_size()):
                        rank_shard_path = shard_dir / f"rank_{rank:04d}.json"
                        if not rank_shard_path.exists():
                            raise FileNotFoundError(f"Missing mesh metric shard: {rank_shard_path}")
                        with rank_shard_path.open("r", encoding="utf-8") as f:
                            mesh_results.extend(json.load(f))
                else:
                    mesh_results = list(self.mesh_test_results_local)
                self.write_mesh_metrics_summary(
                    output_root,
                    mesh_results,
                    summary_stem="mesh_metrics_summary",
                )

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

        if self.test_cfg.save_scene_ranking:
            shard_dir = output_root / "scene_ranking_shards"
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard_path = shard_dir / f"rank_{self.global_rank:04d}.json"
            with shard_path.open("w", encoding="utf-8") as f:
                json.dump(self.scene_test_results_local, f, indent=2)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            if not dist.is_available() or not dist.is_initialized() or self.global_rank == 0:
                scene_results = []
                if dist.is_available() and dist.is_initialized():
                    for rank in range(dist.get_world_size()):
                        rank_shard_path = shard_dir / f"rank_{rank:04d}.json"
                        if not rank_shard_path.exists():
                            raise FileNotFoundError(f"Missing scene ranking shard: {rank_shard_path}")
                        with rank_shard_path.open("r", encoding="utf-8") as f:
                            scene_results.extend(json.load(f))
                else:
                    scene_results = list(self.scene_test_results_local)
                self.write_scene_ranking(output_root, scene_results)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

    def get_test_output_root(self) -> Path:
        return self.test_cfg.output_path / get_cfg()["wandb"]["name"]

    def write_scene_ranking(
        self,
        output_root: Path,
        scene_results: list[dict[str, object]],
    ) -> None:
        ranking_rows = []
        sorted_results = sorted(
            scene_results,
            key=lambda item: (-float(item["psnr_mean"]), str(item["scene"])),
        )
        for rank, item in enumerate(sorted_results, start=1):
            ranking_rows.append(
                {
                    "rank": rank,
                    "scene": item["scene"],
                    "psnr_mean": float(item["psnr_mean"]),
                    "ssim_mean": float(item["ssim_mean"]),
                    "lpips_mean": float(item["lpips_mean"]),
                    "num_target_views": int(item["num_target_views"]),
                    "context_indices": item["context_indices"],
                    "target_indices": item["target_indices"],
                    "scene_dir": item["scene_dir"],
                    "metrics_path": item["metrics_path"],
                }
            )

        with (output_root / "scene_psnr_ranking.json").open("w", encoding="utf-8") as f:
            json.dump(ranking_rows, f, indent=2)

        field_names = [
            "rank",
            "scene",
            "psnr_mean",
            "ssim_mean",
            "lpips_mean",
            "num_target_views",
            "metrics_path",
        ]
        with (output_root / "scene_psnr_ranking.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for row in ranking_rows:
                writer.writerow({field: row[field] for field in field_names})

    def write_mesh_metrics_summary(
        self,
        output_root: Path,
        scene_results: list[dict[str, object]],
        summary_stem: str = "mesh_metrics_summary",
    ) -> None:
        metric_names = _MESH_PUBLIC_METRIC_NAMES
        count_metric_names = _MESH_PUBLIC_COUNT_NAMES
        sorted_results = sorted(
            (_public_mesh_metric_row(row) for row in scene_results),
            key=lambda item: str(item["scene"]),
        )

        overall = {
            f"{metric}_mean": float(np.mean([float(item[metric]) for item in sorted_results]))
            for metric in metric_names
        } if sorted_results else {}
        first_result = sorted_results[0] if sorted_results else {}
        summary = {
            "num_scenes": len(sorted_results),
            "mesh_space": first_result.get("mesh_space"),
            "index_path": first_result.get("index_path"),
            "num_context_views": first_result.get("num_context_views"),
            "pose_norm_method": first_result.get("pose_norm_method"),
            "relative_pose": first_result.get("relative_pose"),
            "use_train_view": first_result.get("use_train_view"),
            "use_val_view": first_result.get("use_val_view"),
            "gt_filter_quantile_lo": first_result.get("gt_filter_quantile_lo"),
            "gt_filter_quantile_hi": first_result.get("gt_filter_quantile_hi"),
            "bbox_margin_ratio": first_result.get("bbox_margin_ratio"),
            "bbox_margin": first_result.get("bbox_margin"),
            "gt_density_radius_ratio": first_result.get("gt_density_radius_ratio"),
            "gt_density_radius": first_result.get("gt_density_radius"),
            "gt_density_min_neighbors": first_result.get("gt_density_min_neighbors"),
            "pred_keep_radius_ratio": first_result.get("pred_keep_radius_ratio"),
            "pred_keep_radius_min_factor": first_result.get("pred_keep_radius_min_factor"),
            "pred_keep_radius_min": first_result.get("pred_keep_radius_min"),
            "pred_keep_radius": first_result.get("pred_keep_radius"),
            "overall": overall,
            "scenes": sorted_results,
        }
        for metric in count_metric_names:
            values = [float(item[metric]) for item in sorted_results if item.get(metric) is not None]
            if values:
                summary["overall"][f"{metric}_mean"] = float(np.mean(values))
        with (output_root / f"{summary_stem}.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        field_names = list(_MESH_PUBLIC_FIELD_NAMES)
        with (output_root / f"{summary_stem}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for row in sorted_results:
                writer.writerow({field: row.get(field) for field in field_names})
            if overall:
                writer.writerow(
                    {
                        "scene": "__mean__",
                        "pred_mesh_path": "",
                        "mesh_space": first_result.get("mesh_space"),
                        "index_path": first_result.get("index_path"),
                        "num_context_views": first_result.get("num_context_views"),
                        "pose_norm_method": first_result.get("pose_norm_method"),
                        "relative_pose": first_result.get("relative_pose"),
                        "use_train_view": first_result.get("use_train_view"),
                        "use_val_view": first_result.get("use_val_view"),
                        "gt_filter_quantile_lo": first_result.get("gt_filter_quantile_lo"),
                        "gt_filter_quantile_hi": first_result.get("gt_filter_quantile_hi"),
                        "bbox_margin_ratio": first_result.get("bbox_margin_ratio"),
                        "bbox_margin": first_result.get("bbox_margin"),
                        "gt_density_radius_ratio": first_result.get("gt_density_radius_ratio"),
                        "gt_density_radius": first_result.get("gt_density_radius"),
                        "gt_density_min_neighbors": first_result.get("gt_density_min_neighbors"),
                        "pred_keep_radius_ratio": first_result.get("pred_keep_radius_ratio"),
                        "pred_keep_radius_min_factor": first_result.get("pred_keep_radius_min_factor"),
                        "pred_keep_radius_min": first_result.get("pred_keep_radius_min"),
                        "pred_keep_radius": first_result.get("pred_keep_radius"),
                        **{metric: overall[f"{metric}_mean"] for metric in metric_names},
                        **{
                            metric: summary["overall"].get(f"{metric}_mean")
                            for metric in count_metric_names
                        },
                    }
                )

    def write_mesh_timing_summary(
        self,
        output_root: Path,
        scene_results: list[dict[str, object]],
    ) -> None:
        sorted_results = sorted(scene_results, key=lambda item: str(item["scene"]))
        timing_names_by_mode = {
            "direct": (
                "trim",
                "normal",
                "dedup",
                "cpu_copy",
                "build",
                "pack",
                "disk",
                "mesh_total",
                "encoder",
                "end2end",
            ),
            "tsdf": (
                "tsdf_fuse",
                "save_raw",
                "post_process",
                "save_post",
                "mesh_total",
                "encoder",
                "decoder",
                "end2end",
            ),
        }
        overall_by_mode: dict[str, dict[str, object]] = {}
        for mode, timing_names in timing_names_by_mode.items():
            mode_results = [item for item in sorted_results if item.get("export_mode") == mode]
            if not mode_results:
                continue
            overall_by_mode[mode] = {
                "num_records": len(mode_results),
                "overall": {
                    f"{name}_mean": float(np.mean([float(item[name]) for item in mode_results]))
                    for name in timing_names
                },
            }
        summary = {
            "num_scenes": len({str(item["scene"]) for item in sorted_results}),
            "num_records": len(sorted_results),
            "overall_by_mode": overall_by_mode,
            "scenes": sorted_results,
        }
        if len(overall_by_mode) == 1:
            summary["overall"] = next(iter(overall_by_mode.values()))["overall"]
        with (output_root / "mesh_timing_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    @rank_zero_only
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch: BatchedExample = self.data_shim(batch)
        schedule_step = self.get_schedule_step()

        if self.global_rank == 0:
            logger.info(
                f"validation step {schedule_step}; "
                f"scene = {batch['scene']}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render primitives.
        b, v_tgt, _, h, w = batch["target"]["image"].shape
        assert b == 1
        visualization_dump = {"collect_geom_normal": True}
        context_inputs = self._prepare_encoder_context(
            batch["context"],
            schedule_step,
            visualization_dump=visualization_dump,
            include_teacher_for_logging=True,
        )
        primitives = self.encoder(
            context_inputs,
            schedule_step,
            visualization_dump=visualization_dump,
        )
        primitives = self._apply_primitive_mask(
            batch["context"],
            primitives,
            visualization_dump,
        )
        is_triangle = isinstance(primitives, Triangles)

        tgt_extrinsics= batch["target"]["extrinsics"]

        output = self.decoder.forward(
            primitives,
            tgt_extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            "depth",
            **self.get_decoder_debug_kwargs(),
        )
        rgb_pred = output.color[0]
        depth_pred = vis_depth_map(output.depth[0])

        # direct depth from predicted points (used for visualization only)
        pred_depth_vis = visualization_dump["depth"][0].squeeze()
        if pred_depth_vis.shape[-1] == 3:
            pred_depth_vis = pred_depth_vis.mean(dim=-1)

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        psnr = compute_psnr(rgb_gt, rgb_pred).mean()
        self.log(f"val/psnr", psnr)
        lpips = compute_lpips(rgb_gt, rgb_pred).mean()
        self.log(f"val/lpips", lpips)
        ssim = compute_ssim(rgb_gt, rgb_pred).mean()
        self.log(f"val/ssim", ssim)

        # Construct comparison image.
        context_img = batch["context"]["image"][0]
        context_img_depth = vis_depth_map(pred_depth_vis)
        context = []
        for i in range(context_img.shape[0]):
            context.append(context_img[i])
            context.append(context_img_depth[i])
        comparison = hcat(
            add_label(vcat(*context), "Context"),
            add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
            add_label(vcat(*rgb_pred), "Target (Prediction)"),
            add_label(vcat(*depth_pred), "Depth (Prediction)"),
        )

        self.logger.log_image(
            "comparison",
            [prep_image(add_border(comparison))],
            step=schedule_step,
            caption=batch["scene"],
        )

        if self.logger is not None and is_triangle:
            normal_specs = (
                (output.rend_normal, "Render Normal (World)"),
                (output.surf_normal, "Surface Normal (World)"),
            )
            reference_normal = next((normals for normals, _ in normal_specs if normals is not None), None)
            if reference_normal is not None and reference_normal.shape[1] > 0:
                num_views = min(rgb_gt.shape[0], reference_normal.shape[1])
                depth_mask = torch.ones_like(reference_normal[:, :, :1])
                if output.depth is not None:
                    depth_mask = (output.depth > 0).unsqueeze(2).to(output.color.dtype)

                normal_panels = [prepare_rgb_panel(rgb_gt[:num_views], "GT RGB")]
                for normals, label in normal_specs:
                    if normals is None:
                        continue
                    normals_world = transform_rendered_normals_to_world(normals, tgt_extrinsics)
                    normals_world = torch.nan_to_num(normals_world, 0.0, 0.0, 0.0)
                    normal_panels.append(
                        prepare_normal_panel(
                            normals_world[0, :num_views],
                            depth_mask[0, :num_views],
                            label,
                        )
                    )

                normal_comparison = hcat(*normal_panels)
                self.logger.log_image(
                    "normal/comparison_world",
                    [prep_image(add_border(normal_comparison))],
                    step=schedule_step,
                    caption=batch["scene"],
                )

        if self.logger is not None:
            context_geom_normal_raw = visualization_dump.get("geom_normal_cam_raw")
            context_geom_normal_base = visualization_dump.get("geom_normal_cam_base")
            context_geom_normal_pred = visualization_dump.get("geom_normal_cam_pred")
            context_geom_normal = visualization_dump.get("geom_normal_cam_forward")
            if context_geom_normal is None:
                context_geom_normal = visualization_dump.get("geom_normal_cam")
            context_geom_mask = visualization_dump.get("geom_normal_mask")
            context_teacher_normal = visualization_dump.get("context_mono_normal")

            if context_geom_normal is not None:
                context_rgb = batch["context"]["image"][0]
                context_mask = context_geom_mask
                if context_mask is None:
                    context_mask = torch.ones_like(context_geom_normal[:, :, :1])
                context_mask = context_mask[0].to(context_rgb.dtype)

                context_panels = [prepare_rgb_panel(context_rgb, "Context RGB")]
                if context_geom_normal_raw is not None:
                    context_panels.append(
                        prepare_normal_panel(
                            context_geom_normal_raw[0],
                            context_mask,
                            "PT Geometry Normal Raw (Cam)",
                        )
                    )
                if context_geom_normal_base is not None:
                    context_panels.append(
                        prepare_normal_panel(
                            context_geom_normal_base[0],
                            context_mask,
                            "PT Geometry Normal Base (Cam)",
                        )
                    )
                if context_geom_normal_pred is not None:
                    context_panels.append(
                        prepare_normal_panel(
                            context_geom_normal_pred[0],
                            context_mask,
                            "PT Geometry Normal Pred (Cam)",
                        )
                    )
                context_panels.append(
                    prepare_normal_panel(
                        context_geom_normal[0],
                        context_mask,
                        "PT Geometry Normal Forward (Cam)",
                    )
                )
                context_panels.append(
                    prepare_mask_panel(
                        context_mask,
                        "PT Geometry Mask",
                    )
                )
                if context_teacher_normal is not None:
                    context_panels.append(
                        prepare_normal_panel(
                            context_teacher_normal[0],
                            torch.ones_like(context_teacher_normal[0, :, :1]),
                            "MonoNormal Teacher (Cam)",
                        )
                    )

                context_normal_comparison = hcat(*context_panels)
                self.logger.log_image(
                    "normal/context_comparison_cam",
                    [prep_image(add_border(context_normal_comparison))],
                    step=schedule_step,
                    caption=batch["scene"],
                )

        if not is_triangle:
            projections = hcat(
                    *render_projections(
                        primitives,
                        256,
                        extra_label="",
                    )[0]
                )
            self.logger.log_image(
                "projection",
                [prep_image(add_border(projections))],
                step=schedule_step,
            )

        # Draw cameras.
        cameras = hcat(*render_cameras(batch, 256))
        self.logger.log_image(
            "cameras", [prep_image(add_border(cameras))], step=schedule_step
        )

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                batch["context"], schedule_step
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=schedule_step)

        # Run video validation step.
        self.render_video_interpolation(batch)
        self.render_video_wobble(batch)
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(batch)

    def _log_pts3d(self, pts3d, name):
        # log the 3D points [v, h, w, d] to wandb
        pts3d = pts3d.cpu().numpy()
        pts3d_subsampled = subsample_point_cloud_views(pts3d)
        pts3d_subsampled = rearrange(pts3d_subsampled, "v h w d -> (v h w) d")
        try:
            wandb.log({f"point_cloud/{name}": wandb.Object3D(pts3d_subsampled)})
        except:
            pass

    def on_validation_epoch_end(self) -> None:
        """hack to run the full validation"""
        if self.trainer.sanity_checking and self.global_rank == 0:
            logger.debug(self.encoder)  # log the model to wandb log files

        if self.eval_data_cfg is not None:
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                self.run_full_test_sets_eval()

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        start_t = time.time()
        schedule_step = self.get_schedule_step()

        test_datasets = self.trainer.datamodule.test_dataloader(
            dataset_cfg=self.eval_data_cfg
        )

        test_datasets = [test_datasets] if not isinstance(test_datasets, list) else test_datasets

        for test_dataset in test_datasets:
            self.benchmarker.clear_history()
            scores_dict = {}

            for score_tag in ("psnr", "ssim", "lpips"):
                scores_dict[score_tag] = {}
                for method_tag in ("no_opt",):
                    scores_dict[score_tag][method_tag] = []

            dataset_name = test_dataset.dataset.name
            time_skip_first_n_steps = min(
                self.train_cfg.eval_time_skip_steps, test_dataset.dataset.test_len()
            )
            time_skip_steps_dict = {"encoder": 0, "decoder": 0}
            for batch_idx, batch in tqdm(
                enumerate(test_dataset),
                total=min(test_dataset.dataset.test_len(), self.train_cfg.eval_data_length),
            ):
                if batch_idx >= self.train_cfg.eval_data_length:
                    break

                batch = self.data_shim(batch)
                batch = self.transfer_batch_to_device(batch, self.device, dataloader_idx=0)

                # Render Gaussians.
                b, v, _, h, w = batch["target"]["image"].shape
                assert b == 1
                if batch_idx < time_skip_first_n_steps:
                    time_skip_steps_dict["encoder"] += 1
                    time_skip_steps_dict["decoder"] += v

                # Render primitives.
                with self.benchmarker.time("encoder"):
                    context_inputs = self._prepare_encoder_context(
                        batch["context"],
                        schedule_step,
                    )
                    primitives = self.encoder(
                        context_inputs,
                        schedule_step,
                    )
                    primitives = self._apply_primitive_mask(
                        batch["context"],
                        primitives,
                    )

                with self.benchmarker.time("decoder", num_calls=v):
                    output = self.decoder.forward(
                        primitives,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        **self.get_decoder_debug_kwargs(),
                    )
                rgbs = [output.color[0]]
                tags = ["no_opt"]

                # Compute validation metrics.
                rgb_gt = batch["target"]["image"][0]
                for tag, rgb in zip(tags, rgbs):
                    scores_dict["psnr"][tag].append(
                        compute_psnr(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["lpips"][tag].append(
                        compute_lpips(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["ssim"][tag].append(
                        compute_ssim(rgb_gt, rgb).mean().item()
                    )

            # summarise scores and log to logger
            for score_tag, methods in scores_dict.items():
                for method_tag, cur_scores in methods.items():
                    if len(cur_scores) > 0:
                        cur_mean = sum(cur_scores) / len(cur_scores)
                        self.log(f"test/{dataset_name}_{method_tag}_{score_tag}", cur_mean)
            # summarise run time
            logger.info(f"Evaluation Dataset: {dataset_name}")
            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(time_skip_steps_dict[tag]) :]
                logger.info(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
                self.log(f"test/{dataset_name}_runtime_avg_{tag}", np.mean(times))
            self.benchmarker.clear_history()

            overall_eval_time = time.time() - start_t
            logger.info(f"Eval total time cost: {overall_eval_time:.3f}s")
            self.log("test/runtime_all", overall_eval_time)

    def visualize_gaussians(
        self,
        context_images: Float[Tensor, "view 3 height width"],
        opacities: Float[Tensor, "batch vrspp"],
        covariances: Float[Tensor, "batch vrspp 3 3"],
        colors: Float[Tensor, "batch vrspp 3"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        v, _, h, w = context_images.shape
        h, w = h // 14 * self.gaussians_per_axis, w // 14 * self.gaussians_per_axis
        rb = 0
        opacities = repeat(
            opacities[rb], "(v h w spp) -> spp v c h w", v=v, c=3, h=h, w=w
        )
        colors = rearrange(colors[rb], "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)
        colors = colors * 0.5 + 0.5

        # Color-map Gaussian covariawnces.
        det = covariances[rb].det()
        det = apply_color_map(det / det.max(), "inferno")
        det = rearrange(det, "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)

        return add_border(
            hcat(
                add_label(box(hcat(*context_images)), "Context"),
                add_label(box(vcat(*[hcat(*x) for x in opacities])), "Opacities"),
                add_label(
                    box(vcat(*[hcat(*x) for x in (colors * opacities)])), "Colors"
                ),
                add_label(box(vcat(*[hcat(*x) for x in colors])), "Colors (Raw)"),
                add_label(box(vcat(*[hcat(*x) for x in det])), "Determinant"),
            )
        )

    def log_gaussian_status(self, context_images, gaussians, visualization_dump):
        def log_gaussian_params(params, name):
            # params: (n, 3) or (n, 1)
            if name == "depth" and params.shape[-1] == 3:
                params = params[..., 0].unsqueeze(-1)
            if name == 'opacities' and params.shape[-1] == 3:
                params = params[..., 0].unsqueeze(-1)

            max_val = params.max(dim=0)[0]
            min_val = params.min(dim=0)[0]
            median_val = params.median(dim=0)[0]

            if params.shape[-1] == 1:
                self.log(f"gaussian/max_{name}", max_val)
                self.log(f"gaussian/min_{name}", min_val)
                self.log(f"gaussian/median_{name}", median_val)
            else:
                self.log(f"gaussian/max_x_{name}", max_val[0])
                self.log(f"gaussian/max_y_{name}", max_val[1])
                self.log(f"gaussian/max_z_{name}", max_val[2])
                self.log(f"gaussian/min_x_{name}", min_val[0])
                self.log(f"gaussian/min_y_{name}", min_val[1])
                self.log(f"gaussian/min_z_{name}", min_val[2])
                self.log(f"gaussian/median_x_{name}", median_val[0])
                self.log(f"gaussian/median_y_{name}", median_val[1])
                self.log(f"gaussian/median_z_{name}", median_val[2])

        b, v, _, h, w = context_images.shape
        gaussian_ctrs = gaussians.means
        log_gaussian_params(gaussian_ctrs.flatten(end_dim=-2), "ctrs_all")
        log_gaussian_params(gaussian_ctrs.flatten(end_dim=-2).norm(dim=-1, keepdim=True), "ctrs_all_norm")

        gaussian_ctrs_per_view = rearrange(gaussian_ctrs, "b (v hw) xyz -> b v hw xyz", v=v)
        for i in range(v):
            log_gaussian_params(gaussian_ctrs_per_view[:, i].flatten(end_dim=-2), f"ctrs_view{i}")
            log_gaussian_params(gaussian_ctrs_per_view[:, i].flatten(end_dim=-2).norm(dim=-1, keepdim=True), f"ctrs_view{i}_norm")

        log_gaussian_params(visualization_dump["scales"].flatten(end_dim=-2), "scales")
        log_gaussian_params(visualization_dump["opacities"].flatten(end_dim=-2), "opacities")

        del visualization_dump
        torch.cuda.empty_cache()

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        schedule_step = self.get_schedule_step()
        context_inputs = self._prepare_encoder_context(
            batch["context"],
            schedule_step,
        )
        primitives = self.encoder(context_inputs, schedule_step)
        primitives = self._apply_primitive_mask(
            batch["context"],
            primitives,
        )

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output = self.decoder.forward(
            primitives, extrinsics, intrinsics, near, far, (h, w), "depth",
            **self.get_decoder_debug_kwargs(),
        )
        images = [
            vcat(rgb, depth)
            for rgb, depth in zip(output.color[0], vis_depth_map(output.depth[0]))
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            pass

    def print_preview_metrics(self, metrics: dict[str, float | Tensor], methods: list[str] | None = None, overlap_tag: str | None = None) -> None:
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

        metric_list = ["psnr", "lpips", "ssim"]

        def print_metrics(runing_metric, methods=None):
            table = []
            if methods is None:
                methods = ['ours']

            for method in methods:
                row = [
                    f"{runing_metric[f'{metric}_{method}']:.3f}"
                    for metric in metric_list
                ]
                table.append((method, *row))

            headers = ["Method"] + metric_list
            table = tabulate(table, headers)
            print(table)

        print("All Pairs:")
        print_metrics(self.running_metrics, methods)
        if overlap_tag is not None:
            for k, v in self.running_metrics_sub.items():
                print(f"Overlap: {k}")
                print_metrics(v, methods)

    def configure_optimizers(self):
        new_params, new_param_names = [], []
        pretrained_params, pretrained_param_names = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if (
                "gaussian" in name
                or "triangle" in name
                or "rgb_embed" in name
                or "intrinsics_embed" in name
                or "normal_refiner" in name
            ):
                new_params.append(param)
                new_param_names.append(name)
            else:
                pretrained_params.append(param)
                pretrained_param_names.append(name)

        param_dicts = [
            {
                "params": new_params,
                "lr": self.optimizer_cfg.lr,
             },
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95))
        warm_up_steps = self.optimizer_cfg.warm_up_steps
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            1 / warm_up_steps,
            1,
            total_iters=warm_up_steps,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=get_cfg()["trainer"]["max_steps"], eta_min=self.optimizer_cfg.lr * 0.1)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warm_up, lr_scheduler], milestones=[warm_up_steps])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
