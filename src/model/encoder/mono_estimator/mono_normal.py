from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from src.third_party.omnidata.modules.midas.dpt_depth import DPTDepthModel


class MonoNormalEstimator(nn.Module):
    def __init__(self, pretrained_weights_path: str) -> None:
        super().__init__()
        weights_path = Path(pretrained_weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Mono normal weights not found: {weights_path}"
            )

        self.model = DPTDepthModel(backbone="vitb_rn50_384", num_channels=3)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "state_dict" in checkpoint:
            state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items()}
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._device_ready = False

    def forward(
        self,
        imgs: Float[Tensor, "batch view 3 height width"],
    ) -> Float[Tensor, "batch view 3 height width"]:
        b, v, _, _, _ = imgs.shape
        image = rearrange(imgs, "b v c h w -> (b v) c h w")
        if not self._device_ready:
            self.model.to(image.device)
            self._device_ready = True
        with torch.no_grad():
            normal = self.model(image).clamp(min=0, max=1)
            normal = torch.nn.functional.normalize(normal * 2 - 1, dim=1)
        return rearrange(normal, "(b v) c h w -> b v c h w", b=b, v=v).detach()
