from typing import Union

from .mesh_exporter import GSMeshExporter
from .tsdf_gs2d import TsdfGs2d, TsdfGs2dCfgWrapper

MESHES = {
    TsdfGs2dCfgWrapper: TsdfGs2d,
}

MeshCfgWrapper = Union[TsdfGs2dCfgWrapper]


def get_mesh(cfg: MeshCfgWrapper) -> GSMeshExporter:
    return MESHES[type(cfg)](cfg)
