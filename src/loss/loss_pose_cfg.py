from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Primitives
from .loss import Loss
from .loss_pose import CameraLoss


@dataclass
class LossPoseCfg:
    weight: float
    alpha: float


@dataclass
class LossPoseCfgWrapper:
    pose: LossPoseCfg


class LossPose(Loss[LossPoseCfg, LossPoseCfgWrapper]):
    def __init__(self, cfg: LossPoseCfgWrapper) -> None:
        super().__init__(cfg)
        self.camera_loss = CameraLoss(alpha=self.cfg.alpha)
        self.last_metrics = {}
        self.last_unweighted_loss = None

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Primitives,
        global_step: int,
        use_context: bool = False,
        extra_info: dict | None = None,
    ) -> Float[Tensor, ""]:
        self.last_metrics = {}
        self.last_unweighted_loss = None

        device = gaussians.means.device if hasattr(gaussians, 'means') else gaussians.vertices.device
        if extra_info is None or "pred_camera_poses" not in extra_info:
            return torch.tensor(0, dtype=torch.float32, device=device)

        pred_camera_poses = extra_info["pred_camera_poses"]
        target_camera_poses = batch["context"]["extrinsics"]
        loss_pose, loss_pose_dict = self.camera_loss(pred_camera_poses, target_camera_poses)

        self.last_unweighted_loss = loss_pose.detach()
        self.last_metrics = loss_pose_dict

        return self.cfg.weight * loss_pose
