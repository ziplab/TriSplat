from dataclasses import dataclass
from typing import Optional, Union

from jaxtyping import Float
from jaxtyping import Bool
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]
    rotations: Optional[Float[Tensor, "batch gaussian 4"]]
    scales: Optional[Float[Tensor, "batch gaussian 3"]]
    mapped_scales: Optional[Float[Tensor, "batch gaussian 3"]] = None


@dataclass
class Triangles:
    vertices: Float[Tensor, "batch num_points 3 3"]
    sigma: Float[Tensor, "batch num_points 1"]
    opacity: Float[Tensor, "batch num_points 1"]
    features: Float[Tensor, "batch num_points c"]
    centers: Optional[Float[Tensor, "batch num_points 3"]] = None
    normals: Optional[Float[Tensor, "batch num_points 3"]] = None
    scales: Optional[Float[Tensor, "batch num_points 3"]] = None
    mapped_scales: Optional[Float[Tensor, "batch num_points 3"]] = None
    primitive_valid_mask: Optional[Bool[Tensor, "batch num_points"]] = None


Primitives = Union[Gaussians, Triangles]
