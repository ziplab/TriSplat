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
class LossNormalTeacherCfg:
    weight: float = 0.01
    start_step: int = 2000
    use_cosine: bool = True
    use_l1: bool = False


@dataclass
class LossNormalTeacherCfgWrapper:
    normal_teacher: LossNormalTeacherCfg


class LossNormalTeacher(Loss[LossNormalTeacherCfg, LossNormalTeacherCfgWrapper]):
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

        geom_normal = extra_info.get("geom_normal_cam_pred")
        if geom_normal is None:
            geom_normal = extra_info.get("geom_normal_cam")
        teacher_normal = extra_info.get("context_mono_normal")
        geom_mask = extra_info.get("geom_normal_mask")
        if geom_normal is None or teacher_normal is None or geom_mask is None:
            return torch.tensor(0.0, device=device)

        b, v, _, h_teacher, w_teacher = teacher_normal.shape
        geom_flat = geom_normal.reshape(b * v, 3, *geom_normal.shape[-2:])
        geom_flat = F.interpolate(
            geom_flat,
            size=(h_teacher, w_teacher),
            mode="bilinear",
            align_corners=True,
        )
        geom_normal = geom_flat.reshape(b, v, 3, h_teacher, w_teacher)

        geom_mask = geom_mask.reshape(b * v, 1, *geom_mask.shape[-2:]).float()
        geom_mask = F.interpolate(
            geom_mask,
            size=(h_teacher, w_teacher),
            mode="nearest",
        )
        geom_mask = geom_mask.reshape(b, v, 1, h_teacher, w_teacher) > 0.5

        teacher_normal = F.normalize(teacher_normal, dim=2, eps=1e-6)
        geom_normal = F.normalize(geom_normal, dim=2, eps=1e-6)

        valid_mask = geom_mask
        if "mask" in batch["context"]:
            valid_mask = valid_mask & (batch["context"]["mask"] > 0.5)
        valid_mask = valid_mask.expand_as(geom_normal)

        finite_mask = torch.isfinite(geom_normal) & torch.isfinite(teacher_normal)
        valid_mask = valid_mask & finite_mask
        if not valid_mask.any():
            return torch.tensor(0.0, device=device)

        loss = torch.tensor(0.0, device=device)
        if self.cfg.use_cosine:
            cosine = (geom_normal * teacher_normal).sum(dim=2, keepdim=True).clamp(-1.0, 1.0)
            cosine_loss = 1.0 - cosine
            cosine_mask = valid_mask[:, :, :1]
            loss = loss + cosine_loss[cosine_mask].mean()
        if self.cfg.use_l1:
            l1_loss = (geom_normal - teacher_normal).abs()
            loss = loss + l1_loss[valid_mask].mean()
        return self.cfg.weight * loss
