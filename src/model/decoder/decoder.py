import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Literal, Optional, TypeVar, Union

import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..types import Gaussians, Triangles

DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]


@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]
    depth: Float[Tensor, "batch view height width"] | None = None
    opacity: Optional[Float[Tensor, "batch view height width"]] = None
    rend_normal: Optional[Float[Tensor, "batch view 3 height width"]] = None
    surf_normal: Optional[Float[Tensor, "batch view 3 height width"]] = None
    projection_matrix: Optional[Float[Tensor, "batch view 4 4"]] = None
    triangle_visibility_mask: Optional[Tensor] = None


T = TypeVar("T")
logger = logging.getLogger(__name__)


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg
        self.prune_opacity_threshold = getattr(cfg, 'prune_opacity_threshold', 0.0)
        self._last_debug_log_key: tuple[bool, int] | None = None

    def should_log_debug_stats(
        self,
        global_step: int | None,
        debug_log_interval: int | None,
    ) -> bool:
        if global_step is None or debug_log_interval is None or debug_log_interval <= 0:
            return False
        if global_step % debug_log_interval != 0:
            return False
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_rank() != 0:
                return False

        log_key = (self.training, global_step)
        if self._last_debug_log_key == log_key:
            return False

        self._last_debug_log_key = log_key
        return True

    def log_debug_tensor_stats(
        self,
        name: str,
        tensor: Tensor | None,
        global_step: int | None,
    ) -> None:
        if tensor is None:
            logger.info(f"[decoder debug] step={global_step} {name}: unavailable")
            return

        tensor = tensor.detach()
        if tensor.numel() == 0:
            logger.info(
                f"[decoder debug] step={global_step} {name}: empty shape={tuple(tensor.shape)}"
            )
            return

        flat = tensor.reshape(-1).float()
        finite_mask = torch.isfinite(flat)
        finite = flat[finite_mask]

        if finite.numel() == 0:
            logger.info(
                "[decoder debug] step=%s %s: shape=%s finite=0/%d",
                global_step,
                name,
                tuple(tensor.shape),
                flat.numel(),
            )
            return

        logger.info(
            "[decoder debug] step=%s %s: shape=%s finite=%d/%d min=%.6g max=%.6g mean=%.6g median=%.6g std=%.6g",
            global_step,
            name,
            tuple(tensor.shape),
            finite.numel(),
            flat.numel(),
            finite.min().item(),
            finite.max().item(),
            finite.mean().item(),
            finite.median().item(),
            finite.std(unbiased=False).item(),
        )

        if tensor.ndim > 0 and tensor.shape[-1] > 1 and tensor.shape[-1] <= 4:
            for i in range(tensor.shape[-1]):
                self.log_debug_tensor_stats(
                    f"{name}[..., {i}]",
                    tensor[..., i],
                    global_step,
                )

    @abstractmethod
    def forward(
        self,
        primitives: Union[Gaussians, Triangles],
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        **kwargs,
    ) -> DecoderOutput:
        pass
