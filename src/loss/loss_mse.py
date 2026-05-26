from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Primitives
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float
    use_valid_mask: bool = False


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Primitives,
        global_step: int,
        use_context: bool = False,
        extra_info: dict | None = None,
    ) -> Float[Tensor, ""]:
        views = batch["context"] if use_context else batch["target"]
        delta = prediction.color - views["image"]
        if not self.cfg.use_valid_mask or "valid_mask" not in views:
            return self.cfg.weight * (delta**2).mean()

        mask = views["valid_mask"].to(dtype=delta.dtype, device=delta.device)
        if mask.ndim == delta.ndim - 1:
            mask = mask.unsqueeze(2)
        weighted = (delta**2) * mask
        denom = mask.sum() * delta.shape[2]
        return self.cfg.weight * weighted.sum() / denom.clamp_min(1.0)
