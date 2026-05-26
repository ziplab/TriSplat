from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Primitives
from .loss import Loss


@dataclass
class LossIntrinsicCfg:
    weight: float


@dataclass
class LossIntrinsicCfgWrapper:
    intrinsic: LossIntrinsicCfg


class LossIntrinsic(Loss[LossIntrinsicCfg, LossIntrinsicCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Primitives,
        global_step: int,
        use_context: bool = False,
        extra_info: dict | None = None,
    ) -> Float[Tensor, ""]:
        device = gaussians.means.device if hasattr(gaussians, 'means') else gaussians.vertices.device
        if extra_info is None or "intrinsic_pred" not in extra_info:
            return torch.tensor(0, dtype=torch.float32, device=device)

        intrinsic_pred = extra_info["intrinsic_pred"]
        if intrinsic_pred is None:
            return torch.tensor(0, dtype=torch.float32, device=device)

        intrinsic_gt = batch["context"]["intrinsics"]
        gt_fx, gt_fy = intrinsic_gt[:, :, 0, 0], intrinsic_gt[:, :, 1, 1]
        gt_focal = torch.stack([gt_fx, gt_fy], dim=-1)
        gt_focal = rearrange(gt_focal, "b v c -> (b v) c")

        intrinsic_loss = ((gt_focal - intrinsic_pred) ** 2).mean()
        return self.cfg.weight * intrinsic_loss
