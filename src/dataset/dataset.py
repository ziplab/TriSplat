from dataclasses import dataclass
from typing import Literal

from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    original_image_shape: list[int]
    input_image_shape: list[int]
    crop_mode: Literal["default", "gs"]
    background_color: list[float]
    load_valid_mask: bool
    apply_valid_mask: bool
    cameras_are_circular: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg
