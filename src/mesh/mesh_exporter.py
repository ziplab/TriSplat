from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Generic, TypeVar

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


@dataclass(frozen=True)
class MeshExportResult:
    output_path: str
    space_metadata_path: str | None = None
    direct_output_path: str | None = None
    direct_timing: dict[str, float | int] | None = None
    tsdf_output_path: str | None = None
    tsdf_timing: dict[str, float | int] | None = None


class GSMeshExporter(ABC, Generic[T_cfg, T_wrapper]):
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    @abstractmethod
    def main(
        self,
        prediction_train: DecoderOutput,
        prediction_val: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        output: str,
        gt_pointcloud_root: Path | None = None,
    ) -> MeshExportResult:
        pass
