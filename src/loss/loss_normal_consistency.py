from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Primitives
from .loss import Loss


@dataclass
class LossNormalConsistencyCfg:
    weight: float = 0.05
    start_step: int = 4000
    use_cosine: bool = True


@dataclass
class LossNormalConsistencyCfgWrapper:
    normal_consistency: LossNormalConsistencyCfg


class LossNormalConsistency(
    Loss[LossNormalConsistencyCfg, LossNormalConsistencyCfgWrapper]
):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Primitives,
        global_step: int,
        use_context: bool = False,
        extra_info: dict | None = None,
    ) -> Float[Tensor, ""]:
        device = prediction.color.device
        if global_step < self.cfg.start_step or extra_info is None:
            return torch.tensor(0.0, device=device)

        context_output = extra_info.get("context_output")
        geom_normal = extra_info.get("geom_normal_cam_forward")
        if geom_normal is None:
            geom_normal = extra_info.get("geom_normal_cam")
        geom_mask = extra_info.get("geom_normal_mask")
        if context_output is None or context_output.rend_normal is None:
            return torch.tensor(0.0, device=device)
        if geom_normal is None or geom_mask is None:
            return torch.tensor(0.0, device=device)

        render_normal = context_output.rend_normal
        b, v, _, h_render, w_render = render_normal.shape

        geom_flat = geom_normal.reshape(b * v, 3, *geom_normal.shape[-2:])
        geom_flat = F.interpolate(
            geom_flat,
            size=(h_render, w_render),
            mode="bilinear",
            align_corners=True,
        )
        geom_normal = geom_flat.reshape(b, v, 3, h_render, w_render)

        geom_mask = geom_mask.reshape(b * v, 1, *geom_mask.shape[-2:]).float()
        geom_mask = F.interpolate(
            geom_mask,
            size=(h_render, w_render),
            mode="nearest",
        )
        geom_mask = geom_mask.reshape(b, v, 1, h_render, w_render) > 0.5

        geom_normal = F.normalize(geom_normal, dim=2, eps=1e-6)
        render_normal = F.normalize(render_normal, dim=2, eps=1e-6)

        valid_mask = geom_mask
        if context_output.depth is not None:
            valid_mask = valid_mask & (context_output.depth.unsqueeze(2) > 0)
        if "mask" in batch["context"]:
            ctx_mask = batch["context"]["mask"]
            if ctx_mask.shape[-2:] != (h_render, w_render):
                ctx_mask = F.interpolate(
                    ctx_mask.reshape(b * v, 1, *ctx_mask.shape[-2:]),
                    size=(h_render, w_render),
                    mode="nearest",
                ).reshape(b, v, 1, h_render, w_render)
            valid_mask = valid_mask & (ctx_mask > 0.5)
        valid_mask = valid_mask.expand_as(geom_normal)

        finite_mask = torch.isfinite(geom_normal) & torch.isfinite(render_normal)
        valid_mask = valid_mask & finite_mask
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        if self.cfg.use_cosine:
            cosine = (geom_normal * render_normal).sum(dim=2, keepdim=True).clamp(-1.0, 1.0)
            loss = (1.0 - cosine)[valid_mask[:, :, :1]].mean()
        else:
            loss = (geom_normal - render_normal).abs()[valid_mask].mean()
        return self.cfg.weight * loss
