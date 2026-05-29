import json
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import open3d as o3d
import torch

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians, Triangles

from .mesh_exporter import GSMeshExporter, MeshExportResult


@dataclass
class TsdfGs2dCfg:
    voxel_size: float = 0.005
    """Mesh: voxel size for TSDF"""
    sdf_truc: float = 0.1
    """Mesh: truncation value for TSDF"""
    depth_truc: float = 5.0
    """Mesh: Max depth range for TSDF"""
    num_cluster: int = 200
    """Mesh: number of connected clusters to export"""
    bounded: bool = True
    """Mesh: Using bounded or unbounded mode for meshing"""
    mesh_res: int = 1024
    """Mesh: resolution for unbounded mesh extraction"""
    use_train_view: bool = True
    """Use Training Views to Sample Points"""
    use_val_view: bool = True
    """Use Validation Views to Sample Points"""
    export_mode: Literal["tsdf", "direct", "both"] = "both"
    """Mesh: export mode, tsdf/direct/both"""
    export_format: Literal["ply", "off", "both"] = "both"
    """Mesh: output format, ply/off/both"""
    tsdf_alpha_threshold: float = 0.9
    """Mask pixels with rendered alpha below this before TSDF integration"""
    tsdf_depth_edge_threshold: float = 0.05
    """Zero depth at pixels with gradient > threshold * depth (0 disables)"""
    tsdf_context_color_source: Literal["prediction", "input"] = "prediction"
    """RGB source for context-view TSDF integration."""
    tsdf_target_color_source: Literal["prediction", "input"] = "prediction"
    """RGB source for target-view TSDF integration. Keep 'prediction' for fair novel-view evaluation."""
    gs_depth_backend: Literal["raw", "surface_approx"] = "raw"
    """Depth source for GS TSDF export."""
    gs_surface_eps: float = 1e-6
    """Minimum alpha used when constructing GS surface-like depth."""
    tsdf_auto_depth_mode: Literal["depth_quantile", "depth_max"] = "depth_quantile"
    """Automatic depth truncation strategy when depth_truc < 0."""
    tsdf_auto_depth_quantile: float = 0.999
    """Depth quantile used by the depth_quantile auto strategy."""
    tsdf_auto_depth_margin: float = 1.10
    """Safety margin applied to the chosen automatic depth truncation estimate."""
    tsdf_auto_depth_max_scale: float = 10.0
    """Clamp automatic depth truncation to at most this multiple of median valid depth."""
    tsdf_auto_depth_floor_ratio: float = 2.0
    """Clamp automatic depth truncation to at least this multiple of median valid depth."""
    tsdf_auto_voxel_min: float = 0.003
    """Minimum auto voxel size when voxel_size < 0."""
    tsdf_auto_voxel_max: float = 0.020
    """Maximum auto voxel size when voxel_size < 0."""
    tsdf_auto_sdf_multiplier: float = 12.0
    """Multiplier used to derive automatic sdf_truc from voxel size when sdf_truc < 0."""
    tsdf_auto_sdf_max: float = 0.1
    """Maximum automatic sdf_truc when sdf_truc < 0."""
    tsdf_post_mode: Literal["legacy", "adaptive_legacy_guard"] = "adaptive_legacy_guard"
    """Connected-component filtering strategy for TSDF post-processing."""
    tsdf_post_min_keep_triangle_ratio: float = 0.75
    """Require at least this retained triangle ratio before legacy top-K is accepted."""
    tsdf_post_min_keep_area_ratio: float = 0.97
    """Require at least this retained area ratio before legacy top-K is accepted."""
    tsdf_post_min_cluster_triangles: int = 10
    """Always drop connected components smaller than this triangle count."""
    tsdf_post_write_stats: bool = True
    """Write TSDF post-process diagnostics next to the exported mesh."""
    direct_opacity_threshold: float = 0.5
    """Mesh: filter direct triangles with opacity threshold (<0 disables filter)"""
    direct_opacity_threshosld: Optional[float] = None
    """Deprecated typo alias for direct_opacity_threshold."""
    direct_post_process: bool = False
    """Mesh: apply connected-component post-process for direct mesh"""
    direct_skip_dedup: bool = False
    """Skip vertex deduplication for maximum speed (larger file, no shared vertices)"""
    direct_bench_10: bool = False
    """Run direct mesh export in bench x10 mode and report averaged timings"""
    direct_write_frame_meshes: bool = False
    """Write target-frame-specific direct meshes using per-frame triangle visibility."""
    direct_frame_color_mode: Literal["features", "target_render"] = "features"
    """Color mode for direct target-frame meshes."""
    direct_source_view_filter_enabled: bool = False
    """Keep direct triangles only from selected context/source views."""
    direct_source_view_strategy: Literal["uniform", "target_nearest", "target_visibility_topk"] = "uniform"
    """Strategy used when direct_source_view_slots is empty."""
    direct_source_view_keep_count: int = 6
    """Number of source views to keep for uniform direct mesh filtering."""
    direct_source_view_slots: list[int] = field(default_factory=list)
    """Explicit context/source view slots to keep. Overrides keep_count when non-empty."""
    direct_scale_cull_enabled: bool = False
    """Cull unusually large direct triangles by 3D max-edge length."""
    direct_scale_cull_quantile: float = 0.98
    """Per-group max-edge quantile above which direct triangles are culled."""
    direct_scale_cull_per_source_view: bool = True
    """Compute scale-cull thresholds independently for each retained source view."""
    direct_scale_cull_max_removed_ratio: float = 0.50
    """Fallback when scale culling would remove more than this fraction of active triangles."""
    direct_target_visibility_cull: bool = False
    """Cull direct triangles that are not visible in target-view triangle rendering."""
    direct_visibility_min_target_views: int = 1
    """Minimum number of target views a triangle must be visible in to survive visibility culling."""
    direct_visibility_max_removed_ratio: float = 0.80
    """Fallback when target-visibility culling would remove more than this fraction."""
    direct_triangle_budget: int = 0
    """Maximum number of active direct triangles to keep after opacity/visibility (<=0 disables)."""
    direct_visibility_rank_budget: bool = True
    """Rank budgeted triangles by opacity weighted with target visibility count when available."""
    direct_triangle_budget_score_mode: Literal[
        "auto",
        "opacity",
        "opacity_area_penalty",
        "visibility",
        "visibility_opacity",
        "visibility_opacity_area_penalty",
    ] = "auto"
    """Score used when selecting direct triangles under direct_triangle_budget."""
    direct_triangle_budget_area_penalty_views: Literal["target", "context", "all"] = "target"
    """Camera views used by area-penalty budget scoring."""
    direct_triangle_budget_area_penalty_quantile: float = 0.98
    """Projected-area quantile used as the area-penalty reference threshold."""
    direct_triangle_budget_area_penalty_min_area_frac: float = 0.05
    """Minimum projected bbox area fraction for area-penalty budget scoring."""
    direct_triangle_budget_area_penalty_strength: float = 0.50
    """Penalty strength for oversized projected footprints in budget scoring."""
    direct_triangle_budget_area_penalty_chunk_size: int = 200000
    """Chunk size for projected-area budget scoring."""
    direct_projected_outlier_cull_enabled: bool = False
    """Cull low-opacity triangles with abnormal projected target-camera footprint."""
    direct_projected_outlier_low_opacity_threshold: float = 0.20
    """Only projected outlier triangles below this opacity are removed."""
    direct_projected_outlier_edge_quantile: float = 0.995
    """Per-view projected-edge quantile used as an adaptive outlier threshold."""
    direct_projected_outlier_area_quantile: float = 0.995
    """Per-view projected-bbox-area quantile used as an adaptive outlier threshold."""
    direct_projected_outlier_min_edge_px: float = 256.0
    """Minimum projected-edge threshold for projected outlier culling."""
    direct_projected_outlier_min_area_frac: float = 0.50
    """Minimum projected-bbox area threshold as a fraction of image area."""
    direct_projected_outlier_image_margin: float = 0.25
    """Image-size margin used when deciding whether projected outliers overlap a view."""
    direct_projected_outlier_max_removed_ratio: float = 0.30
    """Fallback when projected outlier culling would remove more than this fraction."""
    direct_projected_outlier_views: Literal["target", "context", "all"] = "target"
    """Camera views used by projected outlier culling."""
    direct_projected_outlier_chunk_size: int = 200000
    """Chunk size for projected outlier direct mesh culling."""
    direct_projective_conflict_cull_enabled: bool = False
    """Cull repeated projective conflicts using only camera geometry and triangle opacity."""
    direct_projective_conflict_views: Literal["target", "context", "all"] = "target"
    """Camera views used by projective conflict culling."""
    direct_projective_conflict_tile_size: int = 16
    """Coarse image tile size, in pixels, for projective conflict grouping."""
    direct_projective_conflict_depth_abs_tolerance: float = 0.015
    """Absolute depth tolerance used to quantize projective conflicts."""
    direct_projective_conflict_depth_rel_tolerance: float = 0.015
    """Relative depth tolerance used to quantize projective conflicts."""
    direct_projective_conflict_min_views: int = 2
    """Minimum number of camera views where a triangle must lose a conflict before removal."""
    direct_projective_conflict_max_removed_ratio: float = 0.35
    """Fallback when projective conflict culling would remove more than this fraction."""
    direct_projective_conflict_max_per_cell: int = 8
    """Number of highest-scoring triangles kept in each tile/depth cell per view."""
    direct_projective_conflict_min_group_size: int = 4
    """Minimum tile/depth group size before projective conflict culling can remove triangles."""
    direct_projective_conflict_area_weight: float = 8.0
    """Projected area penalty in projective conflict scores."""
    direct_projective_conflict_chunk_size: int = 200000
    """Chunk size for projective conflict culling."""
    direct_camera_cull_enabled: bool = False
    """Cull direct triangles that project as target-camera near-plane outliers."""
    direct_camera_cull_views: Literal["target", "context", "all"] = "target"
    """Camera views used by direct camera-aware culling."""
    direct_camera_cull_near_multiplier: float = 1.5
    """Multiplier on per-view near bound for camera-aware near-plane culling."""
    direct_camera_cull_near_min: float = 1e-4
    """Minimum normalized near-plane threshold for camera-aware culling."""
    direct_camera_cull_image_margin: float = 1.0
    """Image-size margin used when deciding whether an outlier triangle overlaps the view."""
    direct_camera_cull_max_projected_edge_px: float = 256.0
    """Cull triangles whose near-clamped projected edge exceeds this many pixels."""
    direct_camera_cull_max_projected_area_frac: float = 1.0
    """Cull triangles whose near-clamped projected bbox area exceeds this image-area fraction."""
    direct_camera_cull_max_removed_ratio: float = 0.60
    """Fallback when camera-aware culling would remove more than this fraction."""
    direct_camera_cull_chunk_size: int = 200000
    """Chunk size for camera-aware direct mesh culling."""
    direct_target_depth_cull: bool = False
    """Cull direct triangles that sit in front of the target rendered depth surface."""
    direct_depth_cull_abs_tolerance: float = 0.02
    """Absolute target-depth tolerance for direct foreground occluder culling."""
    direct_depth_cull_rel_tolerance: float = 0.03
    """Relative target-depth tolerance for direct foreground occluder culling."""
    direct_depth_cull_min_opacity: float = 0.15
    """Minimum rendered target alpha required before using a depth sample for culling."""
    direct_depth_cull_use_vertices: bool = True
    """Also test triangle vertices, in addition to centroids, against the target depth map."""
    direct_depth_cull_max_removed_ratio: float = 0.90
    """Fallback when target-depth culling would remove more than this fraction."""
    direct_depth_cull_chunk_size: int = 200000
    """Chunk size for target-depth direct mesh culling."""
    direct_occluder_cull_enabled: bool = False
    """Cull large projected direct triangles that sit in front of target rendered depth."""
    direct_occluder_views: Literal["target", "context", "all"] = "target"
    """Camera views used by direct occluder culling."""
    direct_occluder_min_projected_edge_px: float = 160.0
    """Minimum projected edge length for a triangle to be tested as an occluder."""
    direct_occluder_min_projected_area_frac: float = 0.08
    """Minimum projected bbox area fraction for a triangle to be tested as an occluder."""
    direct_occluder_force_area_frac: float = 0.35
    """Projected bbox area fraction above which candidates are removed without a depth conflict."""
    direct_occluder_force_remove_near_cross: bool = True
    """Remove occluder candidates that cross the target near plane."""
    direct_occluder_image_margin: float = 0.25
    """Image-size margin used when deciding whether occluder candidates overlap a view."""
    direct_occluder_depth_abs_tolerance: float = 0.02
    """Absolute depth tolerance for direct occluder foreground tests."""
    direct_occluder_depth_rel_tolerance: float = 0.03
    """Relative depth tolerance for direct occluder foreground tests."""
    direct_occluder_min_opacity: float = 0.15
    """Minimum rendered target alpha required before using a depth sample for occluder culling."""
    direct_occluder_min_valid_samples: int = 2
    """Minimum foreground-conflict samples needed to remove an occluder candidate."""
    direct_occluder_max_removed_ratio: float = 0.50
    """Fallback when occluder culling would remove more than this fraction."""
    direct_occluder_chunk_size: int = 200000
    """Chunk size for direct occluder culling."""
    direct_surface_proxy: bool = False
    """Export a direct mesh from target rendered depth/color instead of raw triangle primitives."""
    direct_surface_proxy_stride: int = 1
    """Pixel stride for direct target-depth surface proxy export."""
    direct_surface_proxy_alpha_threshold: float = 0.0
    """Drop proxy vertices below this rendered target alpha threshold."""
    direct_surface_proxy_depth_edge_rel: float = 0.0
    """Drop proxy faces whose depth range exceeds this relative threshold (<=0 disables)."""
    direct_surface_proxy_depth_mode: Literal["render", "near_plane"] = "render"
    """Depth source for direct surface proxy export."""
    direct_surface_proxy_depth_scale: float = 0.995
    """Scale target depths toward the camera to keep each proxy surface frontmost in its own view."""
    direct_surface_proxy_near_multiplier: float = 0.5
    """Near value multiplier used by direct_surface_proxy_depth_mode=near_plane."""
    direct_surface_proxy_write_frame_meshes: bool = False
    """Also write one proxy mesh per target frame for frame-specific rendering."""
    direct_surface_proxy_frame_pixel_quads: bool = False
    """Use independent per-pixel quads for frame proxy meshes to avoid interpolation/border loss."""
    direct_surface_proxy_unified_pixel_quads: bool = False
    """Use independent per-pixel quads in the single unified surface proxy mesh."""
    direct_trim_enabled: bool = True
    """Trim direct-export triangles using geometry-density heuristics before mesh assembly."""
    direct_trim_mode: Literal["geometry"] = "geometry"
    """Active direct trim mode. Only geometry-density trimming is supported."""
    direct_trim_voxel_scale: float = 2.0
    """Voxel size multiplier relative to the median triangle max-edge length."""
    direct_trim_min_voxel_occupancy: int = 2
    """Triangles in denser voxels are preserved by default."""
    direct_trim_area_quantile_hi: float = 0.99
    """High quantile used to flag unusually large triangles."""
    direct_trim_edge_quantile_hi: float = 0.99
    """High quantile used to flag unusually long triangle edges."""
    direct_trim_score_threshold: float = 5.0
    """Threshold for the geometry-density outlier score."""
    direct_trim_keep_sparse_small: bool = True
    """Keep sparse triangles when both area and edge length stay within normal range."""
    direct_trim_voxel_component_min_triangles: int = 64
    """Drop sparse voxel clusters smaller than this triangle count when also low-area."""
    direct_trim_voxel_component_min_area_ratio: float = 0.01
    """Drop sparse voxel clusters smaller than this area ratio when also low-count."""
    direct_trim_max_removed_ratio: float = 0.90
    """Fallback when geometry trimming would remove more than this fraction of triangles."""
    direct_trim_core_quantile: float = 0.70
    """Keep only the dominant high-density voxel core, defined by this occupancy quantile."""
    direct_trim_core_min_occupancy: int = 3
    """Minimum occupancy floor for dense-core voxels."""
    direct_trim_core_halo_layers: int = 1
    """Number of voxel-neighborhood expansion layers kept around the dense core."""
    direct_trim_core_max_removed_ratio: float = 0.70
    """Fallback when dense-core filtering would remove more than this fraction of triangles."""
    direct_trim_gt_quantile: float = 0.99
    """Deprecated legacy GT trim option; ignored by geometry trim."""
    direct_trim_gt_margin: float = 1.10
    """Deprecated legacy GT trim option; ignored by geometry trim."""
    direct_trim_camera_margin: float = 1.10
    """Deprecated legacy camera trim option; ignored by geometry trim."""
    debug_camera_dump: bool = False
    """Write per-view TSDF camera/depth diagnostics under the mesh output directory"""


@dataclass
class TsdfGs2dCfgWrapper:
    tsdf_gs2d: TsdfGs2dCfg


@dataclass(frozen=True)
class DirectMeshBuildStats:
    triangles_before_opacity: int
    triangles_after_opacity: int
    triangles_after_trim: int
    trim_source: str
    triangles_after_source_view: int | None = None
    triangles_after_visibility: int | None = None
    triangles_after_primitive_mask: int | None = None
    triangles_after_budget: int | None = None
    triangles_after_scale_cull: int | None = None
    triangles_after_projected_outlier_cull: int | None = None
    triangles_after_projective_conflict_cull: int | None = None
    triangles_after_camera_cull: int | None = None
    triangles_after_depth_cull: int | None = None
    triangles_after_occluder_cull: int | None = None
    voxel_size: float | None = None
    median_area: float | None = None
    median_edge: float | None = None
    area_threshold_hi: float | None = None
    edge_threshold_hi: float | None = None
    score_threshold: float | None = None
    sparse_triangle_fraction: float | None = None
    occupancy_quantiles: tuple[float, float, float] | None = None
    score_quantiles: tuple[float, float, float, float] | None = None
    removed_by_single_triangle: int = 0
    removed_by_source_view: int = 0
    removed_by_target_visibility: int = 0
    removed_by_primitive_mask: int = 0
    removed_by_triangle_budget: int = 0
    removed_by_scale_cull: int = 0
    removed_by_projected_outlier: int = 0
    removed_by_projective_conflict: int = 0
    removed_by_camera_cull: int = 0
    removed_by_target_depth: int = 0
    removed_by_occluder: int = 0
    removed_by_voxel_component: int = 0
    removed_by_core_cluster: int = 0
    voxel_components: int = 0
    removed_voxel_components: int = 0
    core_components: int = 0
    core_voxels: int = 0
    core_threshold: float | None = None
    core_halo_layers: int = 0
    removed_ratio: float = 0.0
    fallback_reason: str | None = None
    source_view_selected_slots: tuple[int, ...] | None = None
    source_view_selection_strategy: str | None = None
    source_view_score_quantiles: tuple[float, float, float] | None = None
    source_view_filter_removed_ratio: float | None = None
    source_view_filter_fallback_reason: str | None = None
    visibility_cull_removed_ratio: float | None = None
    visibility_cull_fallback_reason: str | None = None
    triangle_budget_removed_ratio: float | None = None
    triangle_budget_score_quantiles: tuple[float, float, float] | None = None
    triangle_budget_score_mode: str | None = None
    triangle_budget_area_quantiles: tuple[float, float, float] | None = None
    triangle_budget_area_threshold: float | None = None
    triangle_budget_fallback_reason: str | None = None
    scale_cull_removed_ratio: float | None = None
    scale_cull_fallback_reason: str | None = None
    scale_cull_threshold_quantiles: tuple[float, float, float] | None = None
    projected_outlier_removed_ratio: float | None = None
    projected_outlier_fallback_reason: str | None = None
    projected_outlier_threshold_quantiles: tuple[float, float, float] | None = None
    projective_conflict_removed_ratio: float | None = None
    projective_conflict_fallback_reason: str | None = None
    projective_conflict_count_quantiles: tuple[float, float, float] | None = None
    projective_conflict_group_size_quantiles: tuple[float, float, float] | None = None
    projective_conflict_views: str | None = None
    camera_cull_removed_ratio: float | None = None
    camera_cull_fallback_reason: str | None = None
    camera_cull_views: str | None = None
    camera_cull_near_clip: float | None = None
    depth_cull_removed_ratio: float | None = None
    depth_cull_fallback_reason: str | None = None
    occluder_removed_ratio: float | None = None
    occluder_fallback_reason: str | None = None
    occluder_projected_edge_quantiles: tuple[float, float, float] | None = None
    occluder_projected_area_quantiles: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class TsdfPostProcessStats:
    mode: str
    strategy_used: str
    strategy_reason: str | None
    cluster_to_keep: int
    num_clusters: int
    raw_num_vertices: int
    raw_num_triangles: int
    raw_surface_area: float
    min_cluster_triangles: int
    min_keep_triangle_ratio: float
    min_keep_area_ratio: float
    legacy_triangle_threshold: int
    legacy_kept_clusters: int
    legacy_retained_triangle_ratio: float
    legacy_retained_area_ratio: float
    final_kept_clusters: int
    final_retained_triangle_ratio: float
    final_retained_area_ratio: float
    final_num_vertices: int
    final_num_triangles: int


class TsdfGs2d(GSMeshExporter[TsdfGs2dCfg, TsdfGs2dCfgWrapper]):
    """
    TSDF fusion (2DGS ver)
    """

    @staticmethod
    def _sync_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _empty_direct_phase_times() -> dict[str, float]:
        return {
            "trim": 0.0,
            "normal": 0.0,
            "dedup": 0.0,
            "cpu_copy": 0.0,
        }

    @staticmethod
    def _accumulate_direct_phase_times(
        totals: dict[str, float],
        current: dict[str, float],
    ) -> None:
        for key in totals:
            totals[key] += float(current[key])

    @staticmethod
    def _average_direct_phase_times(
        totals: dict[str, float],
        num_runs: int,
    ) -> dict[str, float]:
        return {key: value / num_runs for key, value in totals.items()}

    @staticmethod
    def _build_direct_timing_record(
        phase_times: dict[str, float],
        pack_time: float,
        disk_time: float,
        encoder_time: float | None,
        num_triangles: int,
        num_vertices: int,
        num_faces: int,
    ) -> dict[str, float | int]:
        build_time = sum(float(phase_times[key]) for key in ("trim", "normal", "dedup", "cpu_copy"))
        mesh_total = build_time + pack_time + disk_time
        encoder_value = float(encoder_time) if encoder_time is not None else 0.0
        return {
            "trim": float(phase_times["trim"]),
            "normal": float(phase_times["normal"]),
            "dedup": float(phase_times["dedup"]),
            "cpu_copy": float(phase_times["cpu_copy"]),
            "build": float(build_time),
            "pack": float(pack_time),
            "disk": float(disk_time),
            "mesh_total": float(mesh_total),
            "encoder": encoder_value,
            "end2end": float(encoder_value + mesh_total),
            "num_triangles": int(num_triangles),
            "num_vertices": int(num_vertices),
            "num_faces": int(num_faces),
        }

    @staticmethod
    def _build_tsdf_timing_record(
        *,
        fuse_time: float,
        save_raw_time: float,
        post_process_time: float,
        save_post_time: float,
        encoder_time: float | None,
        decoder_time: float | None,
        raw_vertices: int,
        raw_faces: int,
        post_vertices: int,
        post_faces: int,
    ) -> dict[str, float | int]:
        mesh_total = fuse_time + save_raw_time + post_process_time + save_post_time
        encoder_value = float(encoder_time) if encoder_time is not None else 0.0
        decoder_value = float(decoder_time) if decoder_time is not None else 0.0
        return {
            "tsdf_fuse": float(fuse_time),
            "save_raw": float(save_raw_time),
            "post_process": float(post_process_time),
            "save_post": float(save_post_time),
            "mesh_total": float(mesh_total),
            "encoder": encoder_value,
            "decoder": decoder_value,
            "end2end": float(encoder_value + decoder_value + mesh_total),
            "raw_num_vertices": int(raw_vertices),
            "raw_num_faces": int(raw_faces),
            "post_num_vertices": int(post_vertices),
            "post_num_faces": int(post_faces),
        }

    @staticmethod
    def _mesh_has_geometry(mesh: o3d.geometry.TriangleMesh) -> bool:
        return len(mesh.vertices) > 0 and len(mesh.triangles) > 0

    def _validate_tsdf_post_cfg(self) -> None:
        if self.cfg.tsdf_post_mode not in ("legacy", "adaptive_legacy_guard"):
            raise ValueError(
                "mesh.tsdf_gs2d.tsdf_post_mode only supports 'legacy' or "
                f"'adaptive_legacy_guard', got {self.cfg.tsdf_post_mode!r}."
            )
        for name in ("tsdf_post_min_keep_triangle_ratio", "tsdf_post_min_keep_area_ratio"):
            value = float(getattr(self.cfg, name))
            if not (0.0 < value <= 1.0):
                raise ValueError(f"mesh.tsdf_gs2d.{name} must be in (0, 1], got {value}.")
        if int(self.cfg.tsdf_post_min_cluster_triangles) < 0:
            raise ValueError("mesh.tsdf_gs2d.tsdf_post_min_cluster_triangles must be >= 0.")

    @staticmethod
    def _compute_retained_ratio(kept_values: np.ndarray, total_values: np.ndarray) -> float:
        total_sum = float(np.asarray(total_values, dtype=np.float64).sum())
        if total_sum <= 0:
            return 1.0
        kept_sum = float(np.asarray(kept_values, dtype=np.float64).sum())
        return kept_sum / total_sum

    @staticmethod
    def _build_cluster_keep_mask_from_indices(
        num_clusters: int,
        keep_indices: np.ndarray,
    ) -> np.ndarray:
        keep_mask = np.zeros((num_clusters,), dtype=bool)
        if keep_indices.size > 0:
            keep_mask[np.asarray(keep_indices, dtype=np.int64)] = True
        return keep_mask

    def _build_tsdf_legacy_keep_mask(
        self,
        cluster_n_triangles: np.ndarray,
        cluster_to_keep: int,
    ) -> tuple[np.ndarray, int]:
        num_clusters = int(cluster_n_triangles.shape[0])
        if num_clusters == 0:
            threshold = max(int(self.cfg.tsdf_post_min_cluster_triangles), 0)
            return np.zeros((0,), dtype=bool), threshold

        cluster_to_keep = min(max(int(cluster_to_keep), 1), num_clusters)
        nth_index = num_clusters - cluster_to_keep
        threshold = int(np.partition(cluster_n_triangles, nth_index)[nth_index])
        threshold = max(threshold, int(self.cfg.tsdf_post_min_cluster_triangles))
        keep_mask = cluster_n_triangles >= threshold
        return keep_mask, threshold

    def _build_tsdf_adaptive_keep_mask(
        self,
        cluster_n_triangles: np.ndarray,
        cluster_area: np.ndarray,
    ) -> np.ndarray:
        num_clusters = int(cluster_n_triangles.shape[0])
        if num_clusters == 0:
            return np.zeros((0,), dtype=bool)

        min_cluster_triangles = int(self.cfg.tsdf_post_min_cluster_triangles)
        valid_indices = np.flatnonzero(cluster_n_triangles >= min_cluster_triangles)
        if valid_indices.size == 0:
            return np.zeros((num_clusters,), dtype=bool)

        sort_order = np.lexsort(
            (
                -cluster_area[valid_indices].astype(np.float64, copy=False),
                -cluster_n_triangles[valid_indices].astype(np.int64, copy=False),
            )
        )
        ordered_indices = valid_indices[sort_order]
        ordered_triangles = cluster_n_triangles[ordered_indices]
        ordered_area = cluster_area[ordered_indices]

        triangle_ratios = np.cumsum(ordered_triangles, dtype=np.float64) / max(
            float(cluster_n_triangles.sum()),
            1.0,
        )
        total_area = float(cluster_area.sum())
        if total_area > 0:
            area_ratios = np.cumsum(ordered_area, dtype=np.float64) / total_area
        else:
            area_ratios = np.ones_like(triangle_ratios)

        satisfied = np.flatnonzero(
            (triangle_ratios >= float(self.cfg.tsdf_post_min_keep_triangle_ratio))
            & (area_ratios >= float(self.cfg.tsdf_post_min_keep_area_ratio))
        )
        if satisfied.size > 0:
            ordered_indices = ordered_indices[: satisfied[0] + 1]

        return self._build_cluster_keep_mask_from_indices(num_clusters, ordered_indices)

    def _collect_valid_depths(self) -> torch.Tensor | None:
        valid_depths: list[torch.Tensor] = []
        for depth in self.depthmaps:
            valid = depth[torch.isfinite(depth) & (depth > 0)].float().flatten()
            if valid.numel() > 0:
                valid_depths.append(valid)
        if not valid_depths:
            return None
        return torch.cat(valid_depths)

    def _resolve_auto_depth_trunc(self, valid_depths: torch.Tensor | None) -> tuple[float, str]:
        if valid_depths is None:
            fallback = max(float(self.radius * 2.0), 1e-3)
            return fallback, "camera_radius_fallback"
        depth_min = float(valid_depths.min().item())
        depth_median = float(valid_depths.median().item())
        depth_max = float(valid_depths.max().item())
        depth_quantile = float(
            torch.quantile(
                valid_depths,
                torch.tensor(
                    self.cfg.tsdf_auto_depth_quantile,
                    device=valid_depths.device,
                    dtype=valid_depths.dtype,
                ),
            ).item()
        )
        if self.cfg.tsdf_auto_depth_mode == "depth_quantile":
            candidate = depth_quantile
        else:
            candidate = depth_max
        candidate *= float(self.cfg.tsdf_auto_depth_margin)
        floor = depth_median * float(self.cfg.tsdf_auto_depth_floor_ratio)
        if self.cfg.tsdf_auto_depth_max_scale > 0:
            cap = depth_median * float(self.cfg.tsdf_auto_depth_max_scale)
            candidate = min(candidate, cap)
        else:
            cap = candidate
        resolved = max(candidate, floor, depth_min * 1.05, 1e-3)
        debug = (
            f"{self.cfg.tsdf_auto_depth_mode}"
            f"(min={depth_min:.4f},median={depth_median:.4f},"
            f"q={depth_quantile:.4f},max={depth_max:.4f},floor={floor:.4f},cap={cap:.4f},resolved={resolved:.4f})"
        )
        return resolved, debug

    def _resolve_auto_voxel_size(self, depth_trunc: float) -> tuple[float, str]:
        unclamped = depth_trunc / self.cfg.mesh_res
        resolved = float(
            min(
                max(unclamped, self.cfg.tsdf_auto_voxel_min),
                self.cfg.tsdf_auto_voxel_max,
            )
        )
        debug = (
            "auto_voxel"
            f"(unclamped={unclamped:.6f},min={self.cfg.tsdf_auto_voxel_min:.6f},"
            f"max={self.cfg.tsdf_auto_voxel_max:.6f},resolved={resolved:.6f})"
        )
        return resolved, debug

    def _resolve_auto_sdf_trunc(self, voxel_size: float) -> tuple[float, str]:
        candidate = voxel_size * float(self.cfg.tsdf_auto_sdf_multiplier)
        resolved = float(min(candidate, float(self.cfg.tsdf_auto_sdf_max)))
        debug = (
            "auto_sdf"
            f"(candidate={candidate:.6f},max={self.cfg.tsdf_auto_sdf_max:.6f},resolved={resolved:.6f})"
        )
        return resolved, debug

    @staticmethod
    def _print_direct_timing(
        timing: dict[str, float | int],
        *,
        ply_num_bytes: int,
        show_encoder: bool,
        bench_tag: str = "",
    ) -> None:
        enc_str = f"encoder={float(timing['encoder']):.4f}s  " if show_encoder else ""
        ply_mb = ply_num_bytes / (1024 * 1024)
        print(
            f"[Mesh][direct]{bench_tag} "
            f"{enc_str}"
            f"trim={float(timing['trim']):.4f}s  normal={float(timing['normal']):.4f}s  "
            f"dedup={float(timing['dedup']):.4f}s  cpu_copy={float(timing['cpu_copy']):.4f}s  "
            f"build={float(timing['build']):.4f}s  pack={float(timing['pack']):.4f}s  "
            f"disk={float(timing['disk']):.4f}s  mesh_total={float(timing['mesh_total']):.4f}s  "
            f"end2end={float(timing['end2end']):.4f}s  "
            f"(verts={int(timing['num_vertices'])}, faces={int(timing['num_faces'])}, ply={ply_mb:.1f}MB)"
        )

    @staticmethod
    def _print_tsdf_timing(timing: dict[str, float | int], *, show_encoder: bool, show_decoder: bool) -> None:
        enc_str = f"encoder={float(timing['encoder']):.4f}s  " if show_encoder else ""
        dec_str = f"decoder={float(timing['decoder']):.4f}s  " if show_decoder else ""
        print(
            f"[Mesh][tsdf][bench] "
            f"{enc_str}{dec_str}"
            f"tsdf_fuse={float(timing['tsdf_fuse']):.4f}s  save_raw={float(timing['save_raw']):.4f}s  "
            f"post_process={float(timing['post_process']):.4f}s  save_post={float(timing['save_post']):.4f}s  "
            f"mesh_total={float(timing['mesh_total']):.4f}s  end2end={float(timing['end2end']):.4f}s  "
            f"(raw_verts={int(timing['raw_num_vertices'])}, raw_faces={int(timing['raw_num_faces'])}, "
            f"post_verts={int(timing['post_num_vertices'])}, post_faces={int(timing['post_num_faces'])})"
        )

    @staticmethod
    def build_o3d_intrinsic_from_normalized_k(
        intrinsics: np.ndarray,
        w: int,
        h: int,
    ) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Convert the dataset's normalized intrinsics into Open3D pixel intrinsics.
        The normalized K has fx, cx scaled by image width and fy, cy scaled by image height.
        """
        fx = float(intrinsics[0, 0]) * w
        fy = float(intrinsics[1, 1]) * h
        cx = float(intrinsics[0, 2]) * w
        cy = float(intrinsics[1, 2]) * h
        return o3d.camera.PinholeCameraIntrinsic(width=w, height=h, cx=cx, cy=cy, fx=fx, fy=fy)

    @staticmethod
    def build_o3d_intrinsic_from_projection(
        projection: np.ndarray,
        w: int,
        h: int,
    ) -> o3d.camera.PinholeCameraIntrinsic:
        """
        Legacy helper that infers pixel intrinsics from the decoder projection matrix.
        This is kept only for debugging comparisons; TSDF fusion should use normalized K directly.
        """
        ndc2pix = np.array(
            [
                [w / 2, 0, 0, (w - 1) / 2],
                [0, h / 2, 0, (h - 1) / 2],
                [0, 0, 0, 1],
            ]
        ).T
        intrins = (projection @ ndc2pix)[:3, :3].T
        return o3d.camera.PinholeCameraIntrinsic(
            width=w,
            height=h,
            cx=float(intrins[0, 2]),
            cy=float(intrins[1, 2]),
            fx=float(intrins[0, 0]),
            fy=float(intrins[1, 1]),
        )

    @staticmethod
    def _intrinsic_debug_dict(intrinsic: o3d.camera.PinholeCameraIntrinsic) -> dict[str, float]:
        matrix = intrinsic.intrinsic_matrix
        return {
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
        }

    @staticmethod
    def _tensor_debug_dict(tensor: torch.Tensor) -> dict[str, float | int]:
        valid = tensor[torch.isfinite(tensor) & (tensor > 0)].float()
        if valid.numel() == 0:
            return {
                "num_valid": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
            }
        return {
            "num_valid": int(valid.numel()),
            "min": float(valid.min().item()),
            "max": float(valid.max().item()),
            "mean": float(valid.mean().item()),
            "median": float(valid.median().item()),
        }

    def _build_surface_like_depth_from_gs(
        self,
        prediction: DecoderOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if prediction.depth is None or prediction.opacity is None:
            raise ValueError("GS surface_approx backend requires both depth and opacity from the decoder.")

        raw_depth = prediction.depth[0].clone().detach().float()
        alpha = prediction.opacity[0].clone().detach().float()
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)

        valid = torch.isfinite(raw_depth) & (raw_depth > 0) & (alpha > self.cfg.gs_surface_eps)
        surface_depth = torch.zeros_like(raw_depth)
        surface_depth[valid] = raw_depth[valid] / alpha[valid].clamp_min(self.cfg.gs_surface_eps)
        surface_depth = torch.nan_to_num(surface_depth, nan=0.0, posinf=0.0, neginf=0.0)
        surface_depth = torch.where(alpha >= self.cfg.tsdf_alpha_threshold, surface_depth, 0.0)
        surface_depth = torch.where(surface_depth > 0, surface_depth, 0.0)
        return surface_depth, alpha

    def _write_camera_debug_dump(self, output_dir: Path) -> None:
        if not self.cfg.debug_camera_dump:
            return
        payload = {"views": self.camera_debug_records}
        with (output_dir / "tsdf_camera_debug.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _get_direct_opacity_threshold(self) -> float:
        """
        Resolve the active direct-export opacity threshold.
        Prefer the correctly spelled config key, but keep the old typo as a
        backwards-compatible fallback for existing overrides.
        """
        deprecated_threshold = getattr(self.cfg, "direct_opacity_threshosld", None)
        threshold = self.cfg.direct_opacity_threshold
        if (
            deprecated_threshold is not None
            and threshold == TsdfGs2dCfg.direct_opacity_threshold
        ):
            return deprecated_threshold
        return threshold

    def _validate_direct_source_view_filter_cfg(self) -> None:
        if self.cfg.direct_source_view_strategy not in ("uniform", "target_nearest", "target_visibility_topk"):
            raise ValueError(
                "mesh.tsdf_gs2d.direct_source_view_strategy only supports 'uniform' or "
                "'target_nearest', or 'target_visibility_topk', "
                f"got {self.cfg.direct_source_view_strategy!r}."
            )
        if int(self.cfg.direct_source_view_keep_count) <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_source_view_keep_count must be > 0.")
        slots = [int(slot) for slot in (self.cfg.direct_source_view_slots or [])]
        if any(slot < 0 for slot in slots):
            raise ValueError("mesh.tsdf_gs2d.direct_source_view_slots must be non-negative.")
        if len(set(slots)) != len(slots):
            raise ValueError("mesh.tsdf_gs2d.direct_source_view_slots must not contain duplicates.")

    def _validate_direct_scale_cull_cfg(self) -> None:
        quantile = float(self.cfg.direct_scale_cull_quantile)
        if not (0.0 < quantile <= 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_scale_cull_quantile must be in (0, 1].")
        if not (0.0 <= float(self.cfg.direct_scale_cull_max_removed_ratio) < 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_scale_cull_max_removed_ratio must be in [0, 1).")

    def _validate_direct_budget_cfg(self) -> None:
        if int(self.cfg.direct_triangle_budget) < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_triangle_budget must be >= 0.")
        if self.cfg.direct_triangle_budget_score_mode not in (
            "auto",
            "opacity",
            "opacity_area_penalty",
            "visibility",
            "visibility_opacity",
            "visibility_opacity_area_penalty",
        ):
            raise ValueError(
                "mesh.tsdf_gs2d.direct_triangle_budget_score_mode only supports 'auto', "
                "'opacity', 'opacity_area_penalty', 'visibility', 'visibility_opacity', "
                "or 'visibility_opacity_area_penalty', "
                f"got {self.cfg.direct_triangle_budget_score_mode!r}."
            )
        if self.cfg.direct_triangle_budget_area_penalty_views not in ("target", "context", "all"):
            raise ValueError(
                "mesh.tsdf_gs2d.direct_triangle_budget_area_penalty_views only supports "
                "'target', 'context', or 'all', "
                f"got {self.cfg.direct_triangle_budget_area_penalty_views!r}."
            )
        if not (0.0 < float(self.cfg.direct_triangle_budget_area_penalty_quantile) <= 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_triangle_budget_area_penalty_quantile must be in (0, 1].")
        if float(self.cfg.direct_triangle_budget_area_penalty_min_area_frac) < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_triangle_budget_area_penalty_min_area_frac must be >= 0.")
        if float(self.cfg.direct_triangle_budget_area_penalty_strength) < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_triangle_budget_area_penalty_strength must be >= 0.")
        if int(self.cfg.direct_triangle_budget_area_penalty_chunk_size) <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_triangle_budget_area_penalty_chunk_size must be > 0.")

    def _validate_direct_projected_outlier_cfg(self) -> None:
        if not (0.0 <= float(self.cfg.direct_projected_outlier_low_opacity_threshold) <= 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_projected_outlier_low_opacity_threshold must be in [0, 1].")
        for name in ("direct_projected_outlier_edge_quantile", "direct_projected_outlier_area_quantile"):
            value = float(getattr(self.cfg, name))
            if not (0.0 < value <= 1.0):
                raise ValueError(f"mesh.tsdf_gs2d.{name} must be in (0, 1].")
        if float(self.cfg.direct_projected_outlier_min_edge_px) <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projected_outlier_min_edge_px must be > 0.")
        if float(self.cfg.direct_projected_outlier_min_area_frac) <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projected_outlier_min_area_frac must be > 0.")
        if float(self.cfg.direct_projected_outlier_image_margin) < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projected_outlier_image_margin must be >= 0.")
        if not (0.0 <= float(self.cfg.direct_projected_outlier_max_removed_ratio) < 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_projected_outlier_max_removed_ratio must be in [0, 1).")
        if self.cfg.direct_projected_outlier_views not in ("target", "context", "all"):
            raise ValueError(
                "mesh.tsdf_gs2d.direct_projected_outlier_views only supports 'target', 'context', or 'all', "
                f"got {self.cfg.direct_projected_outlier_views!r}."
            )
        if int(self.cfg.direct_projected_outlier_chunk_size) <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projected_outlier_chunk_size must be > 0.")

    def _validate_direct_projective_conflict_cfg(self) -> None:
        if self.cfg.direct_projective_conflict_views not in ("target", "context", "all"):
            raise ValueError(
                "mesh.tsdf_gs2d.direct_projective_conflict_views only supports "
                "'target', 'context', or 'all', "
                f"got {self.cfg.direct_projective_conflict_views!r}."
            )
        if int(self.cfg.direct_projective_conflict_tile_size) <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projective_conflict_tile_size must be > 0.")
        if float(self.cfg.direct_projective_conflict_depth_abs_tolerance) < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projective_conflict_depth_abs_tolerance must be >= 0.")
        if float(self.cfg.direct_projective_conflict_depth_rel_tolerance) < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projective_conflict_depth_rel_tolerance must be >= 0.")
        if (
            float(self.cfg.direct_projective_conflict_depth_abs_tolerance) == 0
            and float(self.cfg.direct_projective_conflict_depth_rel_tolerance) == 0
        ):
            raise ValueError(
                "At least one of mesh.tsdf_gs2d.direct_projective_conflict_depth_abs_tolerance "
                "or direct_projective_conflict_depth_rel_tolerance must be > 0."
            )
        if int(self.cfg.direct_projective_conflict_min_views) <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projective_conflict_min_views must be > 0.")
        if not (0.0 <= float(self.cfg.direct_projective_conflict_max_removed_ratio) < 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_projective_conflict_max_removed_ratio must be in [0, 1).")
        if int(self.cfg.direct_projective_conflict_max_per_cell) <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projective_conflict_max_per_cell must be > 0.")
        if int(self.cfg.direct_projective_conflict_min_group_size) <= 1:
            raise ValueError("mesh.tsdf_gs2d.direct_projective_conflict_min_group_size must be > 1.")
        if float(self.cfg.direct_projective_conflict_area_weight) < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projective_conflict_area_weight must be >= 0.")
        if int(self.cfg.direct_projective_conflict_chunk_size) <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_projective_conflict_chunk_size must be > 0.")

    def _infer_direct_source_view_slots(
        self,
        num_triangles: int,
        batch: BatchedExample | None,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if batch is None or "context" not in batch:
            return None, {"fallback_reason": "missing_batch"}

        context = batch["context"]
        if "image" not in context:
            return None, {"fallback_reason": "missing_context_image"}

        context_image = context["image"]
        if context_image.dim() != 5:
            return None, {
                "fallback_reason": f"invalid_context_image_shape_{tuple(context_image.shape)}",
            }

        _, num_views, _, image_h, image_w = context_image.shape
        triangles_per_view = int(image_h) * int(image_w)
        expected = int(num_views) * triangles_per_view
        if int(num_triangles) != expected:
            return None, {
                "fallback_reason": (
                    f"triangle_count_mismatch_expected_{expected}_got_{int(num_triangles)}"
                ),
                "num_source_views": int(num_views),
                "source_image_shape": (int(image_h), int(image_w)),
            }

        source_slots = torch.arange(num_triangles, device=device, dtype=torch.long) // triangles_per_view
        context_indices = None
        target_indices = None
        if "index" in context:
            context_index_tensor = context["index"]
            if context_index_tensor.dim() >= 2:
                context_index_tensor = context_index_tensor[0]
            context_indices = [int(value) for value in context_index_tensor.detach().cpu().tolist()]
        if "target" in batch and "index" in batch["target"]:
            target_index_tensor = batch["target"]["index"]
            if target_index_tensor.dim() >= 2:
                target_index_tensor = target_index_tensor[0]
            target_indices = [int(value) for value in target_index_tensor.detach().cpu().tolist()]

        return source_slots, {
            "fallback_reason": None,
            "num_source_views": int(num_views),
            "source_image_shape": (int(image_h), int(image_w)),
            "context_indices": context_indices,
            "target_indices": target_indices,
        }

    def _resolve_direct_source_view_slots(
        self,
        num_source_views: int,
        source_info: dict[str, Any],
        source_slots: torch.Tensor | None = None,
        visibility_counts: torch.Tensor | None = None,
        tri_opacity: torch.Tensor | None = None,
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        self._validate_direct_source_view_filter_cfg()
        explicit_slots = tuple(int(slot) for slot in (self.cfg.direct_source_view_slots or []))
        if explicit_slots:
            invalid = [slot for slot in explicit_slots if slot >= int(num_source_views)]
            if invalid:
                raise ValueError(
                    "mesh.tsdf_gs2d.direct_source_view_slots contains slots outside the "
                    f"available source-view range [0, {int(num_source_views) - 1}]: {invalid}"
            )
            return tuple(sorted(explicit_slots)), {
                "strategy": "explicit",
                "fallback_reason": None,
                "score_quantiles": None,
            }

        keep_count = min(int(self.cfg.direct_source_view_keep_count), int(num_source_views))
        if keep_count == int(num_source_views):
            return tuple(range(int(num_source_views))), {
                "strategy": "all",
                "fallback_reason": None,
                "score_quantiles": None,
            }

        if self.cfg.direct_source_view_strategy == "target_visibility_topk":
            if source_slots is not None and visibility_counts is not None:
                weights = visibility_counts.float().clamp_min(0.0)
                if tri_opacity is not None:
                    weights = weights * torch.nan_to_num(
                        tri_opacity.float(),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).clamp_min(0.0)
                else:
                    weights = weights * (weights > 0).float()

                scores = torch.zeros((int(num_source_views),), dtype=torch.float32, device=source_slots.device)
                scores.scatter_add_(0, source_slots.long().clamp(0, int(num_source_views) - 1), weights)
                if torch.isfinite(scores).any() and float(scores.max().item()) > 0:
                    ranked_slots = sorted(
                        range(int(num_source_views)),
                        key=lambda slot: (-float(scores[slot].item()), slot),
                    )
                    score_values = scores.detach().cpu().numpy().astype(np.float64)
                    return tuple(sorted(ranked_slots[:keep_count])), {
                        "strategy": "target_visibility_topk",
                        "fallback_reason": None,
                        "score_quantiles": (
                            float(np.min(score_values)),
                            float(np.median(score_values)),
                            float(np.max(score_values)),
                        ),
                    }
                visibility_fallback = "target_visibility_topk_zero_scores"
            else:
                visibility_fallback = "missing_visibility_counts_for_target_visibility_topk"
        else:
            visibility_fallback = None

        if self.cfg.direct_source_view_strategy == "target_nearest":
            context_indices = source_info.get("context_indices")
            target_indices = source_info.get("target_indices")
            if (
                isinstance(context_indices, list)
                and isinstance(target_indices, list)
                and len(context_indices) == int(num_source_views)
                and len(target_indices) > 0
            ):
                ranked_slots = sorted(
                    range(int(num_source_views)),
                    key=lambda slot: (
                        min(abs(int(context_indices[slot]) - int(target)) for target in target_indices),
                        slot,
                    ),
                )
                return tuple(sorted(ranked_slots[:keep_count])), {
                    "strategy": "target_nearest",
                    "fallback_reason": visibility_fallback,
                    "score_quantiles": None,
                }

        selected = [int(round(float(value))) for value in np.linspace(0, int(num_source_views) - 1, keep_count)]
        selected = sorted(set(selected))
        if len(selected) < keep_count:
            for slot in range(int(num_source_views)):
                if slot not in selected:
                    selected.append(slot)
                    if len(selected) == keep_count:
                        break
            selected = sorted(selected)
        return tuple(selected), {
            "strategy": "uniform",
            "fallback_reason": visibility_fallback,
            "score_quantiles": None,
        }

    def _build_direct_source_view_keep_mask(
        self,
        num_triangles: int,
        source_slots: torch.Tensor | None,
        source_info: dict[str, Any],
        device: torch.device,
        visibility_counts: torch.Tensor | None = None,
        tri_opacity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if not self.cfg.direct_source_view_filter_enabled:
            return None, {
                "selected_slots": None,
                "selection_strategy": None,
                "score_quantiles": None,
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": None,
            }

        if source_slots is None:
            return None, {
                "selected_slots": None,
                "selection_strategy": None,
                "score_quantiles": None,
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": source_info.get("fallback_reason") or "missing_source_view_layout",
            }

        num_source_views = int(source_info["num_source_views"])
        selected_slots, selection_info = self._resolve_direct_source_view_slots(
            num_source_views,
            source_info,
            source_slots=source_slots,
            visibility_counts=visibility_counts,
            tri_opacity=tri_opacity,
        )
        keep_mask = torch.zeros((num_triangles,), dtype=torch.bool, device=device)
        for slot in selected_slots:
            keep_mask |= source_slots == int(slot)

        kept = int(keep_mask.sum().item())
        removed = int(num_triangles) - kept
        return keep_mask, {
            "selected_slots": selected_slots,
            "selection_strategy": selection_info.get("strategy"),
            "score_quantiles": selection_info.get("score_quantiles"),
            "removed": removed,
            "removed_ratio": removed / max(int(num_triangles), 1),
            "fallback_reason": selection_info.get("fallback_reason"),
        }

    @staticmethod
    def _direct_triangle_max_edges(tri_vertices: torch.Tensor) -> torch.Tensor:
        edge_01 = torch.linalg.norm(tri_vertices[:, 1] - tri_vertices[:, 0], dim=-1)
        edge_12 = torch.linalg.norm(tri_vertices[:, 2] - tri_vertices[:, 1], dim=-1)
        edge_20 = torch.linalg.norm(tri_vertices[:, 0] - tri_vertices[:, 2], dim=-1)
        return torch.maximum(edge_01, torch.maximum(edge_12, edge_20))

    @staticmethod
    def _build_direct_visibility_counts(
        target_visibility_mask: torch.Tensor | None,
        num_triangles: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if target_visibility_mask is None:
            return None, {"fallback_reason": "missing_visibility_mask"}

        visibility = target_visibility_mask.detach()
        if visibility.dim() == 3:
            visibility = visibility[0]
        if visibility.dim() != 2 or visibility.shape[-1] != int(num_triangles):
            return None, {
                "fallback_reason": f"invalid_visibility_shape_{tuple(visibility.shape)}",
            }
        visibility = visibility.to(device=device, dtype=torch.bool)
        return visibility.to(torch.int16).sum(dim=0), {"fallback_reason": None}

    def _build_direct_triangle_budget_keep_mask(
        self,
        active_mask: torch.Tensor,
        tri_vertices: torch.Tensor,
        tri_opacity: torch.Tensor,
        visibility_counts: torch.Tensor | None,
        batch: BatchedExample | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        self._validate_direct_budget_cfg()
        budget = int(self.cfg.direct_triangle_budget)
        if budget <= 0:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": None,
                "score_quantiles": None,
                "score_mode": self._resolve_direct_triangle_budget_score_mode(),
                "area_quantiles": None,
                "area_threshold": None,
            }

        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        active_count = int(active_indices.numel())
        if active_count == 0:
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "empty_active_input",
                "score_quantiles": None,
                "score_mode": self._resolve_direct_triangle_budget_score_mode(),
                "area_quantiles": None,
                "area_threshold": None,
            }
        if active_count <= budget:
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "already_under_budget",
                "score_quantiles": None,
                "score_mode": self._resolve_direct_triangle_budget_score_mode(),
                "area_quantiles": None,
                "area_threshold": None,
            }

        opacity_score = torch.nan_to_num(
            tri_opacity.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)
        visibility_score = None
        if visibility_counts is not None:
            visibility_score = visibility_counts.float().clamp_min(0.0)

        score_mode = self._resolve_direct_triangle_budget_score_mode()
        if score_mode == "opacity" or visibility_score is None:
            scores = opacity_score
        elif score_mode == "visibility":
            scores = visibility_score
        else:
            scores = opacity_score * (1.0 + visibility_score)

        area_quantiles = None
        area_threshold = None
        if score_mode in ("opacity_area_penalty", "visibility_opacity_area_penalty"):
            area_penalty, area_info = self._build_direct_triangle_budget_area_penalty(
                tri_vertices,
                active_indices,
                batch,
            )
            area_quantiles = area_info.get("area_quantiles")
            area_threshold = area_info.get("area_threshold")
            if area_penalty is not None:
                scores = scores.clone()
                scores[active_indices] = scores[active_indices] * area_penalty
            elif area_info.get("fallback_reason") is not None:
                score_mode = f"{score_mode}:{area_info['fallback_reason']}"

        active_scores = scores[active_indices]
        finite = torch.isfinite(active_scores)
        if not finite.any():
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "budget_scores_non_finite",
                "score_quantiles": None,
                "score_mode": score_mode,
                "area_quantiles": area_quantiles,
                "area_threshold": area_threshold,
            }
        active_scores = torch.where(finite, active_scores, torch.zeros_like(active_scores))
        keep_local = torch.topk(active_scores, k=budget, largest=True, sorted=False).indices
        keep_mask = torch.zeros_like(active_mask)
        keep_mask[active_indices[keep_local]] = True

        removed = active_count - int(keep_mask.sum().item())
        score_quantiles = self._tensor_quantiles(active_scores, (0.0, 0.5, 1.0))
        return keep_mask, {
            "removed": removed,
            "removed_ratio": removed / max(active_count, 1),
            "fallback_reason": None,
            "score_quantiles": score_quantiles,
            "score_mode": score_mode,
            "area_quantiles": area_quantiles,
            "area_threshold": area_threshold,
        }

    def _resolve_direct_triangle_budget_score_mode(self) -> str:
        mode = str(self.cfg.direct_triangle_budget_score_mode)
        if mode != "auto":
            return mode
        if bool(self.cfg.direct_visibility_rank_budget):
            return "visibility_opacity"
        return "opacity"

    def _build_direct_triangle_budget_area_penalty(
        self,
        tri_vertices: torch.Tensor,
        active_indices: torch.Tensor,
        batch: BatchedExample | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        strength = float(self.cfg.direct_triangle_budget_area_penalty_strength)
        if strength <= 0:
            return None, {
                "fallback_reason": "area_penalty_disabled",
                "area_quantiles": None,
                "area_threshold": None,
            }

        selected = self._select_direct_camera_cull_views(
            batch,
            tri_vertices.device,
            mode=self.cfg.direct_triangle_budget_area_penalty_views,
        )
        if selected is None:
            return None, {
                "fallback_reason": "missing_batch",
                "area_quantiles": None,
                "area_threshold": None,
            }

        extrinsics, intrinsics, near, image_h, image_w = selected
        if tri_vertices.numel() == 0 or int(active_indices.numel()) == 0 or extrinsics.numel() == 0:
            return None, {
                "fallback_reason": "empty_input",
                "area_quantiles": None,
                "area_threshold": None,
            }

        w2c = torch.linalg.inv(extrinsics)
        area_frac = torch.zeros((active_indices.shape[0],), dtype=torch.float32, device=tri_vertices.device)
        ones = torch.ones((0, 3, 1), dtype=tri_vertices.dtype, device=tri_vertices.device)
        chunk_size = int(self.cfg.direct_triangle_budget_area_penalty_chunk_size)
        eps = float(self.cfg.direct_camera_cull_near_min)
        image_area = float(max(image_h * image_w, 1))
        margin_x = float(image_w) * float(self.cfg.direct_projected_outlier_image_margin)
        margin_y = float(image_h) * float(self.cfg.direct_projected_outlier_image_margin)

        for view_idx in range(w2c.shape[0]):
            view_w2c = w2c[view_idx].to(dtype=tri_vertices.dtype)
            view_k = intrinsics[view_idx].to(dtype=tri_vertices.dtype)
            view_near = torch.clamp(near[view_idx].to(dtype=tri_vertices.dtype), min=eps)
            fx = view_k[0, 0] * float(image_w)
            fy = view_k[1, 1] * float(image_h)
            cx = view_k[0, 2] * float(image_w)
            cy = view_k[1, 2] * float(image_h)

            for start in range(0, int(active_indices.numel()), chunk_size):
                end = min(start + chunk_size, int(active_indices.numel()))
                indices = active_indices[start:end]
                verts = tri_vertices[indices]
                if ones.shape[0] != verts.shape[0]:
                    ones = torch.ones(
                        (verts.shape[0], 3, 1),
                        dtype=tri_vertices.dtype,
                        device=tri_vertices.device,
                    )
                verts_h = torch.cat([verts, ones[: verts.shape[0]]], dim=-1)
                cam = torch.matmul(verts_h, view_w2c.transpose(0, 1))[..., :3]
                z = cam[..., 2]
                has_front = z.amax(dim=1) > eps
                if not has_front.any():
                    continue

                z_clamped = z.clamp_min(view_near)
                px = fx * (cam[..., 0] / z_clamped) + cx
                py = fy * (cam[..., 1] / z_clamped) + cy
                min_x = px.amin(dim=1)
                max_x = px.amax(dim=1)
                min_y = py.amin(dim=1)
                max_y = py.amax(dim=1)
                intersects = (
                    (max_x >= -margin_x)
                    & (min_x < float(image_w) + margin_x)
                    & (max_y >= -margin_y)
                    & (min_y < float(image_h) + margin_y)
                    & has_front
                )
                bbox_area_frac = ((max_x - min_x).clamp_min(0) * (max_y - min_y).clamp_min(0)) / image_area
                bbox_area_frac = torch.where(
                    intersects & torch.isfinite(bbox_area_frac),
                    bbox_area_frac.float(),
                    torch.zeros_like(bbox_area_frac, dtype=torch.float32),
                )
                area_frac[start:end] = torch.maximum(area_frac[start:end], bbox_area_frac)

        finite = torch.isfinite(area_frac)
        if not finite.any():
            return None, {
                "fallback_reason": "area_scores_non_finite",
                "area_quantiles": None,
                "area_threshold": None,
            }
        area_frac = torch.where(finite, area_frac, torch.zeros_like(area_frac))
        area_threshold = max(
            float(self.cfg.direct_triangle_budget_area_penalty_min_area_frac),
            float(torch.quantile(area_frac, float(self.cfg.direct_triangle_budget_area_penalty_quantile)).item()),
        )
        area_threshold = max(area_threshold, 1e-8)
        excess = (area_frac / area_threshold - 1.0).clamp_min(0.0)
        penalty = 1.0 / (1.0 + strength * excess)
        return penalty, {
            "fallback_reason": None,
            "area_quantiles": self._tensor_quantiles(area_frac, (0.0, 0.5, 1.0)),
            "area_threshold": area_threshold,
        }

    def _build_direct_scale_remove_mask(
        self,
        tri_vertices: torch.Tensor,
        active_mask: torch.Tensor,
        source_slots: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if not self.cfg.direct_scale_cull_enabled:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": None,
                "threshold_quantiles": None,
            }

        self._validate_direct_scale_cull_cfg()
        active_count = int(active_mask.sum().item())
        if active_count == 0:
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "empty_active_input",
                "threshold_quantiles": None,
            }

        quantile = float(self.cfg.direct_scale_cull_quantile)
        max_edges = self._direct_triangle_max_edges(tri_vertices)
        remove_mask = torch.zeros_like(active_mask)
        thresholds: list[float] = []

        use_per_source_view = bool(self.cfg.direct_scale_cull_per_source_view) and source_slots is not None
        fallback_reason = None if use_per_source_view or not self.cfg.direct_scale_cull_per_source_view else "missing_source_view_layout_global_threshold"

        if use_per_source_view:
            active_slots = sorted(int(slot) for slot in torch.unique(source_slots[active_mask]).detach().cpu().tolist())
            for slot in active_slots:
                group_mask = active_mask & (source_slots == slot)
                group_indices = torch.nonzero(group_mask, as_tuple=False).flatten()
                if group_indices.numel() == 0:
                    continue
                group_edges = max_edges[group_indices]
                valid = torch.isfinite(group_edges) & (group_edges > 0)
                group_remove = ~valid
                if valid.any():
                    threshold = torch.quantile(group_edges[valid].float(), quantile)
                    thresholds.append(float(threshold.item()))
                    group_remove |= group_edges > threshold.to(dtype=group_edges.dtype)
                else:
                    fallback_reason = "scale_cull_group_has_no_valid_edges"
                    group_remove |= torch.ones_like(group_remove)
                remove_mask[group_indices[group_remove]] = True
        else:
            active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
            active_edges = max_edges[active_indices]
            valid = torch.isfinite(active_edges) & (active_edges > 0)
            active_remove = ~valid
            if valid.any():
                threshold = torch.quantile(active_edges[valid].float(), quantile)
                thresholds.append(float(threshold.item()))
                active_remove |= active_edges > threshold.to(dtype=active_edges.dtype)
            else:
                fallback_reason = "scale_cull_has_no_valid_edges"
                active_remove |= torch.ones_like(active_remove)
            remove_mask[active_indices[active_remove]] = True

        removed = int(remove_mask.sum().item())
        removed_ratio = removed / max(active_count, 1)
        if removed == active_count or removed_ratio > float(self.cfg.direct_scale_cull_max_removed_ratio):
            remove_mask.zero_()
            removed = 0
            fallback_reason = "scale_cull_overtrim"

        threshold_quantiles = None
        if thresholds:
            threshold_array = np.asarray(thresholds, dtype=np.float64)
            threshold_quantiles = (
                float(np.min(threshold_array)),
                float(np.median(threshold_array)),
                float(np.max(threshold_array)),
            )

        return remove_mask, {
            "removed": removed,
            "removed_ratio": removed_ratio,
            "fallback_reason": fallback_reason,
            "threshold_quantiles": threshold_quantiles,
        }

    def _validate_direct_trim_cfg(self) -> None:
        if self.cfg.direct_trim_mode != "geometry":
            raise ValueError(
                "mesh.tsdf_gs2d.direct_trim_mode only supports 'geometry', "
                f"got {self.cfg.direct_trim_mode!r}."
            )
        for name in ("direct_trim_area_quantile_hi", "direct_trim_edge_quantile_hi"):
            value = float(getattr(self.cfg, name))
            if not (0.0 < value <= 1.0):
                raise ValueError(f"mesh.tsdf_gs2d.{name} must be in (0, 1], got {value}.")
        if self.cfg.direct_trim_voxel_scale <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_trim_voxel_scale must be > 0.")
        if self.cfg.direct_trim_min_voxel_occupancy < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_trim_min_voxel_occupancy must be >= 0.")
        if self.cfg.direct_trim_score_threshold <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_trim_score_threshold must be > 0.")
        if self.cfg.direct_trim_voxel_component_min_triangles < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_trim_voxel_component_min_triangles must be >= 0.")
        if self.cfg.direct_trim_voxel_component_min_area_ratio < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_trim_voxel_component_min_area_ratio must be >= 0.")
        if not (0.0 <= self.cfg.direct_trim_max_removed_ratio < 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_trim_max_removed_ratio must be in [0, 1).")
        if not (0.0 < self.cfg.direct_trim_core_quantile <= 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_trim_core_quantile must be in (0, 1].")
        if self.cfg.direct_trim_core_min_occupancy < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_trim_core_min_occupancy must be >= 0.")
        if self.cfg.direct_trim_core_halo_layers < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_trim_core_halo_layers must be >= 0.")
        if not (0.0 <= self.cfg.direct_trim_core_max_removed_ratio < 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_trim_core_max_removed_ratio must be in [0, 1).")

    def _validate_direct_camera_cull_cfg(self) -> None:
        if self.cfg.direct_visibility_min_target_views < 1:
            raise ValueError("mesh.tsdf_gs2d.direct_visibility_min_target_views must be >= 1.")
        if not (0.0 <= self.cfg.direct_visibility_max_removed_ratio < 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_visibility_max_removed_ratio must be in [0, 1).")
        if self.cfg.direct_camera_cull_near_multiplier <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_camera_cull_near_multiplier must be > 0.")
        if self.cfg.direct_camera_cull_near_min <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_camera_cull_near_min must be > 0.")
        if self.cfg.direct_camera_cull_image_margin < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_camera_cull_image_margin must be >= 0.")
        if self.cfg.direct_camera_cull_max_projected_edge_px <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_camera_cull_max_projected_edge_px must be > 0.")
        if self.cfg.direct_camera_cull_max_projected_area_frac <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_camera_cull_max_projected_area_frac must be > 0.")
        if not (0.0 <= self.cfg.direct_camera_cull_max_removed_ratio < 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_camera_cull_max_removed_ratio must be in [0, 1).")
        if self.cfg.direct_camera_cull_chunk_size <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_camera_cull_chunk_size must be > 0.")
        if self.cfg.direct_depth_cull_abs_tolerance < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_depth_cull_abs_tolerance must be >= 0.")
        if self.cfg.direct_depth_cull_rel_tolerance < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_depth_cull_rel_tolerance must be >= 0.")
        if not (0.0 <= self.cfg.direct_depth_cull_min_opacity <= 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_depth_cull_min_opacity must be in [0, 1].")
        if not (0.0 <= self.cfg.direct_depth_cull_max_removed_ratio < 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_depth_cull_max_removed_ratio must be in [0, 1).")
        if self.cfg.direct_depth_cull_chunk_size <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_depth_cull_chunk_size must be > 0.")

    def _validate_direct_occluder_cull_cfg(self) -> None:
        if self.cfg.direct_occluder_views != "target":
            raise ValueError(
                "mesh.tsdf_gs2d.direct_occluder_views currently supports only 'target', "
                f"got {self.cfg.direct_occluder_views!r}."
            )
        if self.cfg.direct_occluder_min_projected_edge_px <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_occluder_min_projected_edge_px must be > 0.")
        if self.cfg.direct_occluder_min_projected_area_frac <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_occluder_min_projected_area_frac must be > 0.")
        if self.cfg.direct_occluder_force_area_frac <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_occluder_force_area_frac must be > 0.")
        if self.cfg.direct_occluder_image_margin < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_occluder_image_margin must be >= 0.")
        if self.cfg.direct_occluder_depth_abs_tolerance < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_occluder_depth_abs_tolerance must be >= 0.")
        if self.cfg.direct_occluder_depth_rel_tolerance < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_occluder_depth_rel_tolerance must be >= 0.")
        if not (0.0 <= self.cfg.direct_occluder_min_opacity <= 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_occluder_min_opacity must be in [0, 1].")
        if self.cfg.direct_occluder_min_valid_samples < 1:
            raise ValueError("mesh.tsdf_gs2d.direct_occluder_min_valid_samples must be >= 1.")
        if not (0.0 <= self.cfg.direct_occluder_max_removed_ratio < 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_occluder_max_removed_ratio must be in [0, 1).")
        if self.cfg.direct_occluder_chunk_size <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_occluder_chunk_size must be > 0.")

    def _select_direct_camera_cull_views(
        self,
        batch: BatchedExample | None,
        device: torch.device,
        mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
        if batch is None:
            return None

        mode = self.cfg.direct_camera_cull_views if mode is None else mode
        if mode == "target":
            view_batches = [batch["target"]]
        elif mode == "context":
            view_batches = [batch["context"]]
        elif mode == "all":
            view_batches = [batch["context"], batch["target"]]
        else:
            raise ValueError(f"Unknown direct_camera_cull_views={mode!r}.")

        extrinsics = torch.cat([item["extrinsics"] for item in view_batches], dim=1)[0].to(
            device=device,
            dtype=torch.float32,
        )
        intrinsics = torch.cat([item["intrinsics"] for item in view_batches], dim=1)[0].to(
            device=device,
            dtype=torch.float32,
        )
        near = torch.cat([item["near"] for item in view_batches], dim=1)[0].to(
            device=device,
            dtype=torch.float32,
        )
        _, _, _, image_h, image_w = batch["target"]["image"].shape
        return extrinsics, intrinsics, near, int(image_h), int(image_w)

    def _build_direct_camera_remove_mask(
        self,
        tri_vertices: torch.Tensor,
        batch: BatchedExample | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if not self.cfg.direct_camera_cull_enabled:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": None,
                "near_clip": None,
                "views": None,
            }

        self._validate_direct_camera_cull_cfg()
        selected = self._select_direct_camera_cull_views(batch, tri_vertices.device)
        if selected is None:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": "missing_batch",
                "near_clip": None,
                "views": self.cfg.direct_camera_cull_views,
            }

        extrinsics, intrinsics, near, image_h, image_w = selected
        if tri_vertices.numel() == 0 or extrinsics.numel() == 0:
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "empty_input",
                "near_clip": None,
                "views": self.cfg.direct_camera_cull_views,
            }

        w2c = torch.linalg.inv(extrinsics)
        near_clip = torch.clamp(
            near * float(self.cfg.direct_camera_cull_near_multiplier),
            min=float(self.cfg.direct_camera_cull_near_min),
        )
        remove_mask = torch.zeros((tri_vertices.shape[0],), dtype=torch.bool, device=tri_vertices.device)
        ones = torch.ones((0, 3, 1), dtype=tri_vertices.dtype, device=tri_vertices.device)
        chunk_size = int(self.cfg.direct_camera_cull_chunk_size)
        eps = float(self.cfg.direct_camera_cull_near_min)
        edge_threshold = float(self.cfg.direct_camera_cull_max_projected_edge_px)
        area_threshold = float(self.cfg.direct_camera_cull_max_projected_area_frac) * float(image_h * image_w)
        margin_x = float(image_w) * float(self.cfg.direct_camera_cull_image_margin)
        margin_y = float(image_h) * float(self.cfg.direct_camera_cull_image_margin)

        for view_idx in range(w2c.shape[0]):
            view_w2c = w2c[view_idx].to(dtype=tri_vertices.dtype)
            view_k = intrinsics[view_idx].to(dtype=tri_vertices.dtype)
            view_near = near_clip[view_idx].to(dtype=tri_vertices.dtype)
            fx = view_k[0, 0] * float(image_w)
            fy = view_k[1, 1] * float(image_h)
            cx = view_k[0, 2] * float(image_w)
            cy = view_k[1, 2] * float(image_h)

            for start in range(0, tri_vertices.shape[0], chunk_size):
                end = min(start + chunk_size, tri_vertices.shape[0])
                verts = tri_vertices[start:end]
                if ones.shape[0] != verts.shape[0]:
                    ones = torch.ones(
                        (verts.shape[0], 3, 1),
                        dtype=tri_vertices.dtype,
                        device=tri_vertices.device,
                    )
                verts_h = torch.cat([verts, ones[: verts.shape[0]]], dim=-1)
                cam = torch.matmul(verts_h, view_w2c.transpose(0, 1))[..., :3]
                z = cam[..., 2]
                has_front = z.amax(dim=1) > eps
                near_cross = has_front & (z.amin(dim=1) < view_near)
                if not has_front.any():
                    continue

                z_clamped = z.clamp_min(view_near)
                px = fx * (cam[..., 0] / z_clamped) + cx
                py = fy * (cam[..., 1] / z_clamped) + cy
                min_x = px.amin(dim=1)
                max_x = px.amax(dim=1)
                min_y = py.amin(dim=1)
                max_y = py.amax(dim=1)
                intersects = (
                    (max_x >= -margin_x)
                    & (min_x < float(image_w) + margin_x)
                    & (max_y >= -margin_y)
                    & (min_y < float(image_h) + margin_y)
                    & has_front
                )

                p0 = torch.stack([px[:, 0], py[:, 0]], dim=-1)
                p1 = torch.stack([px[:, 1], py[:, 1]], dim=-1)
                p2 = torch.stack([px[:, 2], py[:, 2]], dim=-1)
                edge_01 = torch.linalg.norm(p0 - p1, dim=-1)
                edge_12 = torch.linalg.norm(p1 - p2, dim=-1)
                edge_20 = torch.linalg.norm(p2 - p0, dim=-1)
                max_edge = torch.maximum(edge_01, torch.maximum(edge_12, edge_20))
                bbox_area = (max_x - min_x).clamp_min(0) * (max_y - min_y).clamp_min(0)
                oversized = (max_edge > edge_threshold) | (bbox_area > area_threshold)
                remove_mask[start:end] |= intersects & (near_cross | oversized)

        removed = int(remove_mask.sum().item())
        removed_ratio = removed / max(int(tri_vertices.shape[0]), 1)
        fallback_reason = None
        if removed == 0:
            return remove_mask, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": None,
                "near_clip": float(near_clip.median().item()),
                "views": self.cfg.direct_camera_cull_views,
            }
        if removed == int(tri_vertices.shape[0]) or removed_ratio > float(self.cfg.direct_camera_cull_max_removed_ratio):
            fallback_reason = "camera_cull_overtrim"
            remove_mask.zero_()
            removed = 0

        return remove_mask, {
            "removed": removed,
            "removed_ratio": removed_ratio,
            "fallback_reason": fallback_reason,
            "near_clip": float(near_clip.median().item()),
            "views": self.cfg.direct_camera_cull_views,
        }

    def _build_direct_projected_outlier_remove_mask(
        self,
        tri_vertices: torch.Tensor,
        tri_opacity: torch.Tensor,
        batch: BatchedExample | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if not self.cfg.direct_projected_outlier_cull_enabled:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": None,
                "threshold_quantiles": None,
            }

        self._validate_direct_projected_outlier_cfg()
        selected = self._select_direct_camera_cull_views(
            batch,
            tri_vertices.device,
            mode=self.cfg.direct_projected_outlier_views,
        )
        if selected is None:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": "missing_batch",
                "threshold_quantiles": None,
            }

        extrinsics, intrinsics, near, image_h, image_w = selected
        if tri_vertices.numel() == 0 or extrinsics.numel() == 0:
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "empty_input",
                "threshold_quantiles": None,
            }

        low_opacity = torch.nan_to_num(
            tri_opacity.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ) < float(self.cfg.direct_projected_outlier_low_opacity_threshold)
        if not low_opacity.any():
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "no_low_opacity_candidates",
                "threshold_quantiles": None,
            }

        w2c = torch.linalg.inv(extrinsics)
        remove_mask = torch.zeros((tri_vertices.shape[0],), dtype=torch.bool, device=tri_vertices.device)
        ones = torch.ones((0, 3, 1), dtype=tri_vertices.dtype, device=tri_vertices.device)
        chunk_size = int(self.cfg.direct_projected_outlier_chunk_size)
        eps = float(self.cfg.direct_camera_cull_near_min)
        margin_x = float(image_w) * float(self.cfg.direct_projected_outlier_image_margin)
        margin_y = float(image_h) * float(self.cfg.direct_projected_outlier_image_margin)
        min_edge_threshold = float(self.cfg.direct_projected_outlier_min_edge_px)
        min_area_threshold = float(self.cfg.direct_projected_outlier_min_area_frac) * float(image_h * image_w)
        edge_quantile = float(self.cfg.direct_projected_outlier_edge_quantile)
        area_quantile = float(self.cfg.direct_projected_outlier_area_quantile)
        thresholds: list[float] = []

        for view_idx in range(w2c.shape[0]):
            view_w2c = w2c[view_idx].to(dtype=tri_vertices.dtype)
            view_k = intrinsics[view_idx].to(dtype=tri_vertices.dtype)
            view_near = torch.clamp(near[view_idx].to(dtype=tri_vertices.dtype), min=eps)
            fx = view_k[0, 0] * float(image_w)
            fy = view_k[1, 1] * float(image_h)
            cx = view_k[0, 2] * float(image_w)
            cy = view_k[1, 2] * float(image_h)

            view_edges: list[torch.Tensor] = []
            view_areas: list[torch.Tensor] = []
            view_chunks: list[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []

            for start in range(0, tri_vertices.shape[0], chunk_size):
                end = min(start + chunk_size, tri_vertices.shape[0])
                verts = tri_vertices[start:end]
                if ones.shape[0] != verts.shape[0]:
                    ones = torch.ones(
                        (verts.shape[0], 3, 1),
                        dtype=tri_vertices.dtype,
                        device=tri_vertices.device,
                    )
                verts_h = torch.cat([verts, ones[: verts.shape[0]]], dim=-1)
                cam = torch.matmul(verts_h, view_w2c.transpose(0, 1))[..., :3]
                z = cam[..., 2]
                has_front = z.amax(dim=1) > eps
                if not has_front.any():
                    continue

                z_clamped = z.clamp_min(view_near)
                px = fx * (cam[..., 0] / z_clamped) + cx
                py = fy * (cam[..., 1] / z_clamped) + cy
                min_x = px.amin(dim=1)
                max_x = px.amax(dim=1)
                min_y = py.amin(dim=1)
                max_y = py.amax(dim=1)
                intersects = (
                    (max_x >= -margin_x)
                    & (min_x < float(image_w) + margin_x)
                    & (max_y >= -margin_y)
                    & (min_y < float(image_h) + margin_y)
                    & has_front
                )
                if not intersects.any():
                    continue

                p0 = torch.stack([px[:, 0], py[:, 0]], dim=-1)
                p1 = torch.stack([px[:, 1], py[:, 1]], dim=-1)
                p2 = torch.stack([px[:, 2], py[:, 2]], dim=-1)
                edge_01 = torch.linalg.norm(p0 - p1, dim=-1)
                edge_12 = torch.linalg.norm(p1 - p2, dim=-1)
                edge_20 = torch.linalg.norm(p2 - p0, dim=-1)
                max_edge = torch.maximum(edge_01, torch.maximum(edge_12, edge_20))
                bbox_area = (max_x - min_x).clamp_min(0) * (max_y - min_y).clamp_min(0)
                valid = intersects & torch.isfinite(max_edge) & torch.isfinite(bbox_area)
                if not valid.any():
                    continue
                view_edges.append(max_edge[valid].float())
                view_areas.append(bbox_area[valid].float())
                view_chunks.append((start, end, max_edge, bbox_area, valid))

            if not view_chunks or not view_edges or not view_areas:
                continue

            all_edges = torch.cat(view_edges)
            all_areas = torch.cat(view_areas)
            edge_threshold = max(
                min_edge_threshold,
                float(torch.quantile(all_edges, edge_quantile).item()),
            )
            area_threshold = max(
                min_area_threshold,
                float(torch.quantile(all_areas, area_quantile).item()),
            )
            thresholds.append(edge_threshold)
            thresholds.append(area_threshold)

            for start, end, max_edge, bbox_area, valid in view_chunks:
                oversized = valid & (
                    (max_edge > edge_threshold)
                    | (bbox_area > area_threshold)
                )
                remove_mask[start:end] |= low_opacity[start:end] & oversized

        removed = int(remove_mask.sum().item())
        removed_ratio = removed / max(int(tri_vertices.shape[0]), 1)
        fallback_reason = None
        if removed == 0:
            return remove_mask, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": None,
                "threshold_quantiles": None,
            }
        if removed == int(tri_vertices.shape[0]) or removed_ratio > float(self.cfg.direct_projected_outlier_max_removed_ratio):
            fallback_reason = "projected_outlier_overtrim"
            remove_mask.zero_()
            removed = 0

        threshold_quantiles = None
        if thresholds:
            threshold_values = np.asarray(thresholds, dtype=np.float64)
            threshold_quantiles = (
                float(np.min(threshold_values)),
                float(np.median(threshold_values)),
                float(np.max(threshold_values)),
            )

        return remove_mask, {
            "removed": removed,
            "removed_ratio": removed_ratio,
            "fallback_reason": fallback_reason,
            "threshold_quantiles": threshold_quantiles,
        }

    def _build_direct_projective_conflict_remove_mask(
        self,
        tri_vertices: torch.Tensor,
        tri_opacity: torch.Tensor,
        active_mask: torch.Tensor,
        batch: BatchedExample | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if not self.cfg.direct_projective_conflict_cull_enabled:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": None,
                "count_quantiles": None,
                "group_size_quantiles": None,
                "views": None,
            }

        self._validate_direct_projective_conflict_cfg()
        selected = self._select_direct_camera_cull_views(
            batch,
            tri_vertices.device,
            mode=self.cfg.direct_projective_conflict_views,
        )
        if selected is None:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": "missing_batch",
                "count_quantiles": None,
                "group_size_quantiles": None,
                "views": self.cfg.direct_projective_conflict_views,
            }

        active_indices = torch.nonzero(active_mask, as_tuple=False).flatten()
        active_count = int(active_indices.numel())
        if active_count == 0:
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "empty_active_input",
                "count_quantiles": None,
                "group_size_quantiles": None,
                "views": self.cfg.direct_projective_conflict_views,
            }

        extrinsics, intrinsics, near, image_h, image_w = selected
        if tri_vertices.numel() == 0 or extrinsics.numel() == 0:
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "empty_input",
                "count_quantiles": None,
                "group_size_quantiles": None,
                "views": self.cfg.direct_projective_conflict_views,
            }

        w2c = torch.linalg.inv(extrinsics)
        conflict_counts = torch.zeros((tri_vertices.shape[0],), dtype=torch.int16, device=tri_vertices.device)
        ones = torch.ones((0, 3, 1), dtype=tri_vertices.dtype, device=tri_vertices.device)
        tile_size = int(self.cfg.direct_projective_conflict_tile_size)
        tiles_x = max(int(np.ceil(float(image_w) / float(tile_size))), 1)
        tiles_y = max(int(np.ceil(float(image_h) / float(tile_size))), 1)
        depth_abs_tol = float(self.cfg.direct_projective_conflict_depth_abs_tolerance)
        depth_rel_tol = float(self.cfg.direct_projective_conflict_depth_rel_tolerance)
        max_per_cell = int(self.cfg.direct_projective_conflict_max_per_cell)
        min_group_size = int(self.cfg.direct_projective_conflict_min_group_size)
        area_weight = float(self.cfg.direct_projective_conflict_area_weight)
        chunk_size = int(self.cfg.direct_projective_conflict_chunk_size)
        eps = float(self.cfg.direct_camera_cull_near_min)
        image_area = float(max(image_h * image_w, 1))
        group_sizes: list[np.ndarray] = []

        opacity_score_all = torch.nan_to_num(
            tri_opacity.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).clamp_min(0.0)

        for view_idx in range(w2c.shape[0]):
            view_w2c = w2c[view_idx].to(dtype=tri_vertices.dtype)
            view_k = intrinsics[view_idx].to(dtype=tri_vertices.dtype)
            view_near = torch.clamp(near[view_idx].to(dtype=tri_vertices.dtype), min=eps)
            fx = view_k[0, 0] * float(image_w)
            fy = view_k[1, 1] * float(image_h)
            cx = view_k[0, 2] * float(image_w)
            cy = view_k[1, 2] * float(image_h)

            key_chunks: list[np.ndarray] = []
            score_chunks: list[np.ndarray] = []
            index_chunks: list[np.ndarray] = []

            for start in range(0, active_count, chunk_size):
                end = min(start + chunk_size, active_count)
                indices = active_indices[start:end]
                verts = tri_vertices[indices]
                if ones.shape[0] != verts.shape[0]:
                    ones = torch.ones(
                        (verts.shape[0], 3, 1),
                        dtype=tri_vertices.dtype,
                        device=tri_vertices.device,
                    )
                verts_h = torch.cat([verts, ones[: verts.shape[0]]], dim=-1)
                cam = torch.matmul(verts_h, view_w2c.transpose(0, 1))[..., :3]
                z = cam[..., 2]
                z_centroid = z.mean(dim=1)
                has_front = z.amax(dim=1) > eps
                valid_z = has_front & torch.isfinite(z_centroid) & (z_centroid > view_near)
                if not valid_z.any():
                    continue

                z_clamped = z.clamp_min(view_near)
                px = fx * (cam[..., 0] / z_clamped) + cx
                py = fy * (cam[..., 1] / z_clamped) + cy
                min_x = px.amin(dim=1)
                max_x = px.amax(dim=1)
                min_y = py.amin(dim=1)
                max_y = py.amax(dim=1)
                intersects = (
                    (max_x >= 0.0)
                    & (min_x < float(image_w))
                    & (max_y >= 0.0)
                    & (min_y < float(image_h))
                    & valid_z
                )
                bbox_area_frac = ((max_x - min_x).clamp_min(0) * (max_y - min_y).clamp_min(0)) / image_area
                valid = (
                    intersects
                    & torch.isfinite(min_x)
                    & torch.isfinite(max_x)
                    & torch.isfinite(min_y)
                    & torch.isfinite(max_y)
                    & torch.isfinite(bbox_area_frac)
                )
                if not valid.any():
                    continue

                center_x = ((min_x + max_x) * 0.5).clamp(0, float(image_w - 1))
                center_y = ((min_y + max_y) * 0.5).clamp(0, float(image_h - 1))
                tile_x = torch.floor(center_x / float(tile_size)).long().clamp(0, tiles_x - 1)
                tile_y = torch.floor(center_y / float(tile_size)).long().clamp(0, tiles_y - 1)
                tile_id = tile_y * tiles_x + tile_x

                depth_unit = depth_abs_tol + depth_rel_tol * z_centroid.abs().float()
                depth_unit = depth_unit.clamp_min(1.0e-6)
                depth_bin = torch.floor(z_centroid.float() / depth_unit).long().clamp(0, 1_000_000)
                key = tile_id.long() * 1_000_003 + depth_bin
                area_frac = bbox_area_frac.float().clamp_min(0.0)
                score = opacity_score_all[indices] / (1.0 + area_weight * area_frac)

                valid_indices = indices[valid]
                key_chunks.append(key[valid].detach().cpu().numpy().astype(np.int64, copy=False))
                score_chunks.append(score[valid].detach().cpu().numpy().astype(np.float32, copy=False))
                index_chunks.append(valid_indices.detach().cpu().numpy().astype(np.int64, copy=False))

            if not key_chunks:
                continue

            keys = np.concatenate(key_chunks)
            if keys.size <= max_per_cell:
                continue
            scores = np.concatenate(score_chunks)
            indices_np = np.concatenate(index_chunks)
            order = np.lexsort((-scores, keys))
            sorted_keys = keys[order]
            sorted_indices = indices_np[order]
            boundaries = np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [sorted_keys.size]))
            lengths = ends - starts
            if lengths.size == 0:
                continue

            large_groups = lengths[lengths >= min_group_size]
            if large_groups.size > 0:
                group_sizes.append(large_groups.astype(np.float32, copy=False))

            group_ids = np.repeat(np.arange(lengths.size, dtype=np.int64), lengths)
            ranks = np.arange(sorted_keys.size, dtype=np.int64) - starts[group_ids]
            remove_sorted = (lengths[group_ids] >= min_group_size) & (ranks >= max_per_cell)
            if not np.any(remove_sorted):
                continue
            remove_indices = sorted_indices[remove_sorted]
            remove_indices_t = torch.as_tensor(remove_indices, dtype=torch.long, device=tri_vertices.device)
            conflict_counts[remove_indices_t] += 1

        active_conflict_counts = conflict_counts[active_indices].float()
        min_views = int(self.cfg.direct_projective_conflict_min_views)
        remove_mask = active_mask & (conflict_counts >= min_views)
        removed = int(remove_mask.sum().item())
        removed_ratio = removed / max(active_count, 1)
        fallback_reason = None
        if removed == 0:
            return remove_mask, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": None,
                "count_quantiles": self._tensor_quantiles(active_conflict_counts, (0.0, 0.5, 1.0)),
                "group_size_quantiles": None if not group_sizes else tuple(
                    float(value)
                    for value in np.quantile(np.concatenate(group_sizes), [0.0, 0.5, 1.0])
                ),
                "views": self.cfg.direct_projective_conflict_views,
            }
        if removed == active_count or removed_ratio > float(self.cfg.direct_projective_conflict_max_removed_ratio):
            fallback_reason = "projective_conflict_overtrim"
            remove_mask.zero_()
            removed = 0

        return remove_mask, {
            "removed": removed,
            "removed_ratio": removed_ratio,
            "fallback_reason": fallback_reason,
            "count_quantiles": self._tensor_quantiles(active_conflict_counts, (0.0, 0.5, 1.0)),
            "group_size_quantiles": None if not group_sizes else tuple(
                float(value)
                for value in np.quantile(np.concatenate(group_sizes), [0.0, 0.5, 1.0])
            ),
            "views": self.cfg.direct_projective_conflict_views,
        }

    def _build_direct_target_depth_remove_mask(
        self,
        tri_vertices: torch.Tensor,
        batch: BatchedExample | None,
        target_depth: torch.Tensor | None,
        target_opacity: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if not self.cfg.direct_target_depth_cull:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": None,
            }

        self._validate_direct_camera_cull_cfg()
        if batch is None:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": "missing_batch",
            }
        if target_depth is None:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": "missing_target_depth",
            }

        depth = target_depth.detach()
        if depth.dim() == 4:
            depth = depth[0]
        if depth.dim() != 3:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": f"invalid_target_depth_shape_{tuple(depth.shape)}",
            }

        opacity = None
        if target_opacity is not None:
            opacity = target_opacity.detach()
            if opacity.dim() == 4:
                opacity = opacity[0]
            if opacity.dim() != 3 or opacity.shape != depth.shape:
                opacity = None

        extrinsics = batch["target"]["extrinsics"][0].to(device=tri_vertices.device, dtype=torch.float32)
        intrinsics = batch["target"]["intrinsics"][0].to(device=tri_vertices.device, dtype=torch.float32)
        near = batch["target"]["near"][0].to(device=tri_vertices.device, dtype=torch.float32)
        depth = depth.to(device=tri_vertices.device, dtype=torch.float32)
        opacity = None if opacity is None else opacity.to(device=tri_vertices.device, dtype=torch.float32)

        num_views, image_h, image_w = depth.shape
        if tri_vertices.numel() == 0 or num_views == 0:
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "empty_input",
            }

        if extrinsics.shape[0] != num_views or intrinsics.shape[0] != num_views:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": "target_view_count_mismatch",
            }

        w2c = torch.linalg.inv(extrinsics)
        remove_mask = torch.zeros((tri_vertices.shape[0],), dtype=torch.bool, device=tri_vertices.device)
        ones = torch.ones((0, 3, 1), dtype=tri_vertices.dtype, device=tri_vertices.device)
        chunk_size = int(self.cfg.direct_depth_cull_chunk_size)
        eps = float(self.cfg.direct_camera_cull_near_min)
        min_alpha = float(self.cfg.direct_depth_cull_min_opacity)
        abs_tol = float(self.cfg.direct_depth_cull_abs_tolerance)
        rel_tol = float(self.cfg.direct_depth_cull_rel_tolerance)
        use_vertices = bool(self.cfg.direct_depth_cull_use_vertices)

        for view_idx in range(num_views):
            view_w2c = w2c[view_idx].to(dtype=tri_vertices.dtype)
            view_k = intrinsics[view_idx].to(dtype=tri_vertices.dtype)
            depth_map = depth[view_idx]
            opacity_map = None if opacity is None else opacity[view_idx]
            view_near = near[view_idx].to(dtype=tri_vertices.dtype)
            fx = view_k[0, 0] * float(image_w)
            fy = view_k[1, 1] * float(image_h)
            cx = view_k[0, 2] * float(image_w)
            cy = view_k[1, 2] * float(image_h)

            for start in range(0, tri_vertices.shape[0], chunk_size):
                end = min(start + chunk_size, tri_vertices.shape[0])
                verts = tri_vertices[start:end]
                if ones.shape[0] != verts.shape[0]:
                    ones = torch.ones(
                        (verts.shape[0], 3, 1),
                        dtype=tri_vertices.dtype,
                        device=tri_vertices.device,
                    )
                verts_h = torch.cat([verts, ones[: verts.shape[0]]], dim=-1)
                cam_verts = torch.matmul(verts_h, view_w2c.transpose(0, 1))[..., :3]
                samples = cam_verts
                centroid = cam_verts.mean(dim=1, keepdim=True)
                if use_vertices:
                    samples = torch.cat([cam_verts, centroid], dim=1)
                else:
                    samples = centroid

                z = samples[..., 2]
                in_front = z > torch.maximum(
                    view_near,
                    torch.as_tensor(eps, dtype=z.dtype, device=z.device),
                )
                z_safe = z.clamp_min(eps)
                px = fx * (samples[..., 0] / z_safe) + cx
                py = fy * (samples[..., 1] / z_safe) + cy
                inside = (
                    (px >= 0)
                    & (px < float(image_w))
                    & (py >= 0)
                    & (py < float(image_h))
                    & in_front
                )
                if not inside.any():
                    continue

                x_idx = px.round().to(torch.long).clamp_(0, image_w - 1)
                y_idx = py.round().to(torch.long).clamp_(0, image_h - 1)
                ref_depth = depth_map[y_idx, x_idx]
                valid_ref = torch.isfinite(ref_depth) & (ref_depth > 0)
                if opacity_map is not None and min_alpha > 0:
                    ref_opacity = opacity_map[y_idx, x_idx]
                    valid_ref &= torch.isfinite(ref_opacity) & (ref_opacity >= min_alpha)

                tolerance = torch.maximum(
                    torch.full_like(ref_depth, abs_tol),
                    ref_depth * rel_tol,
                )
                foreground = inside & valid_ref & (z < ref_depth - tolerance)
                remove_mask[start:end] |= foreground.any(dim=1)

        removed = int(remove_mask.sum().item())
        removed_ratio = removed / max(int(tri_vertices.shape[0]), 1)
        fallback_reason = None
        if removed == 0:
            return remove_mask, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": None,
            }
        if removed == int(tri_vertices.shape[0]) or removed_ratio > float(self.cfg.direct_depth_cull_max_removed_ratio):
            fallback_reason = "depth_cull_overtrim"
            remove_mask.zero_()
            removed = 0

        return remove_mask, {
            "removed": removed,
            "removed_ratio": removed_ratio,
            "fallback_reason": fallback_reason,
        }

    def _build_direct_occluder_remove_mask(
        self,
        tri_vertices: torch.Tensor,
        batch: BatchedExample | None,
        target_depth: torch.Tensor | None,
        target_opacity: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        if not self.cfg.direct_occluder_cull_enabled:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": None,
                "edge_quantiles": None,
                "area_quantiles": None,
            }

        self._validate_direct_occluder_cull_cfg()
        if batch is None:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": "missing_batch",
                "edge_quantiles": None,
                "area_quantiles": None,
            }
        if target_depth is None:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": "missing_target_depth",
                "edge_quantiles": None,
                "area_quantiles": None,
            }

        depth = target_depth.detach()
        if depth.dim() == 4:
            depth = depth[0]
        if depth.dim() != 3:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": f"invalid_target_depth_shape_{tuple(depth.shape)}",
                "edge_quantiles": None,
                "area_quantiles": None,
            }

        opacity = None
        if target_opacity is not None:
            opacity = target_opacity.detach()
            if opacity.dim() == 4:
                opacity = opacity[0]
            if opacity.dim() != 3 or opacity.shape != depth.shape:
                opacity = None

        extrinsics = batch["target"]["extrinsics"][0].to(device=tri_vertices.device, dtype=torch.float32)
        intrinsics = batch["target"]["intrinsics"][0].to(device=tri_vertices.device, dtype=torch.float32)
        near = batch["target"]["near"][0].to(device=tri_vertices.device, dtype=torch.float32)
        depth = depth.to(device=tri_vertices.device, dtype=torch.float32)
        opacity = None if opacity is None else opacity.to(device=tri_vertices.device, dtype=torch.float32)

        num_views, image_h, image_w = depth.shape
        if tri_vertices.numel() == 0 or num_views == 0:
            return None, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": "empty_input",
                "edge_quantiles": None,
                "area_quantiles": None,
            }
        if extrinsics.shape[0] != num_views or intrinsics.shape[0] != num_views:
            return None, {
                "removed": 0,
                "removed_ratio": None,
                "fallback_reason": "target_view_count_mismatch",
                "edge_quantiles": None,
                "area_quantiles": None,
            }

        w2c = torch.linalg.inv(extrinsics)
        remove_mask = torch.zeros((tri_vertices.shape[0],), dtype=torch.bool, device=tri_vertices.device)
        ones = torch.ones((0, 7, 1), dtype=tri_vertices.dtype, device=tri_vertices.device)
        chunk_size = int(self.cfg.direct_occluder_chunk_size)
        eps = float(self.cfg.direct_camera_cull_near_min)
        edge_threshold = float(self.cfg.direct_occluder_min_projected_edge_px)
        area_threshold = float(self.cfg.direct_occluder_min_projected_area_frac) * float(image_h * image_w)
        force_area_threshold = float(self.cfg.direct_occluder_force_area_frac) * float(image_h * image_w)
        force_near_cross = bool(self.cfg.direct_occluder_force_remove_near_cross)
        margin_x = float(image_w) * float(self.cfg.direct_occluder_image_margin)
        margin_y = float(image_h) * float(self.cfg.direct_occluder_image_margin)
        abs_tol = float(self.cfg.direct_occluder_depth_abs_tolerance)
        rel_tol = float(self.cfg.direct_occluder_depth_rel_tolerance)
        min_alpha = float(self.cfg.direct_occluder_min_opacity)
        min_samples = int(self.cfg.direct_occluder_min_valid_samples)
        edge_samples: list[torch.Tensor] = []
        area_samples: list[torch.Tensor] = []

        for view_idx in range(num_views):
            view_w2c = w2c[view_idx].to(dtype=tri_vertices.dtype)
            view_k = intrinsics[view_idx].to(dtype=tri_vertices.dtype)
            depth_map = depth[view_idx]
            opacity_map = None if opacity is None else opacity[view_idx]
            view_near = torch.maximum(
                near[view_idx].to(dtype=tri_vertices.dtype),
                torch.as_tensor(eps, dtype=tri_vertices.dtype, device=tri_vertices.device),
            )
            fx = view_k[0, 0] * float(image_w)
            fy = view_k[1, 1] * float(image_h)
            cx = view_k[0, 2] * float(image_w)
            cy = view_k[1, 2] * float(image_h)

            for start in range(0, tri_vertices.shape[0], chunk_size):
                end = min(start + chunk_size, tri_vertices.shape[0])
                verts = tri_vertices[start:end]
                num_chunk = int(verts.shape[0])

                verts_h = torch.cat(
                    [
                        verts,
                        torch.ones((num_chunk, 3, 1), dtype=tri_vertices.dtype, device=tri_vertices.device),
                    ],
                    dim=-1,
                )
                cam_verts = torch.matmul(verts_h, view_w2c.transpose(0, 1))[..., :3]
                z = cam_verts[..., 2]
                has_front = z.amax(dim=1) > view_near
                near_cross = has_front & (z.amin(dim=1) < view_near)
                if not has_front.any():
                    continue

                z_clamped = z.clamp_min(view_near)
                px = fx * (cam_verts[..., 0] / z_clamped) + cx
                py = fy * (cam_verts[..., 1] / z_clamped) + cy
                min_x = px.amin(dim=1)
                max_x = px.amax(dim=1)
                min_y = py.amin(dim=1)
                max_y = py.amax(dim=1)
                intersects = (
                    (max_x >= -margin_x)
                    & (min_x < float(image_w) + margin_x)
                    & (max_y >= -margin_y)
                    & (min_y < float(image_h) + margin_y)
                    & has_front
                )
                if not intersects.any():
                    continue

                p0 = torch.stack([px[:, 0], py[:, 0]], dim=-1)
                p1 = torch.stack([px[:, 1], py[:, 1]], dim=-1)
                p2 = torch.stack([px[:, 2], py[:, 2]], dim=-1)
                edge_01 = torch.linalg.norm(p0 - p1, dim=-1)
                edge_12 = torch.linalg.norm(p1 - p2, dim=-1)
                edge_20 = torch.linalg.norm(p2 - p0, dim=-1)
                max_edge = torch.maximum(edge_01, torch.maximum(edge_12, edge_20))
                bbox_area = (max_x - min_x).clamp_min(0) * (max_y - min_y).clamp_min(0)
                candidate = intersects & (
                    near_cross
                    | (max_edge > edge_threshold)
                    | (bbox_area > area_threshold)
                )
                if not candidate.any():
                    continue

                edge_samples.append(max_edge[candidate].detach().float().cpu())
                area_samples.append(bbox_area[candidate].detach().float().cpu())
                forced = candidate & ((bbox_area > force_area_threshold) | (force_near_cross & near_cross))
                if forced.any():
                    remove_mask[start:end] |= forced

                v0 = verts[:, 0]
                v1 = verts[:, 1]
                v2 = verts[:, 2]
                samples = torch.stack(
                    [
                        v0,
                        v1,
                        v2,
                        0.5 * (v0 + v1),
                        0.5 * (v1 + v2),
                        0.5 * (v2 + v0),
                        (v0 + v1 + v2) / 3.0,
                    ],
                    dim=1,
                )
                if ones.shape[0] != num_chunk:
                    ones = torch.ones(
                        (num_chunk, 7, 1),
                        dtype=tri_vertices.dtype,
                        device=tri_vertices.device,
                    )
                samples_h = torch.cat([samples, ones[:num_chunk]], dim=-1)
                cam_samples = torch.matmul(samples_h, view_w2c.transpose(0, 1))[..., :3]
                sample_z = cam_samples[..., 2]
                in_front = sample_z > view_near
                z_safe = sample_z.clamp_min(torch.as_tensor(eps, dtype=sample_z.dtype, device=sample_z.device))
                sample_px = fx * (cam_samples[..., 0] / z_safe) + cx
                sample_py = fy * (cam_samples[..., 1] / z_safe) + cy
                inside = (
                    (sample_px >= 0)
                    & (sample_px < float(image_w))
                    & (sample_py >= 0)
                    & (sample_py < float(image_h))
                    & in_front
                )
                if not inside.any():
                    continue

                x_idx = sample_px.round().to(torch.long).clamp_(0, image_w - 1)
                y_idx = sample_py.round().to(torch.long).clamp_(0, image_h - 1)
                ref_depth = depth_map[y_idx, x_idx]
                valid_ref = inside & torch.isfinite(ref_depth) & (ref_depth > 0)
                if opacity_map is not None and min_alpha > 0:
                    ref_opacity = opacity_map[y_idx, x_idx]
                    valid_ref &= torch.isfinite(ref_opacity) & (ref_opacity >= min_alpha)

                tolerance = torch.maximum(
                    torch.full_like(ref_depth, abs_tol),
                    ref_depth * rel_tol,
                )
                foreground = valid_ref & (sample_z < ref_depth - tolerance)
                remove_mask[start:end] |= candidate & (foreground.sum(dim=1) >= min_samples)

        removed = int(remove_mask.sum().item())
        removed_ratio = removed / max(int(tri_vertices.shape[0]), 1)
        edge_quantiles = None
        area_quantiles = None
        if edge_samples:
            edge_values = torch.cat(edge_samples).numpy().astype(np.float64)
            edge_quantiles = (
                float(np.min(edge_values)),
                float(np.median(edge_values)),
                float(np.max(edge_values)),
            )
        if area_samples:
            area_values = torch.cat(area_samples).numpy().astype(np.float64)
            area_quantiles = (
                float(np.min(area_values)),
                float(np.median(area_values)),
                float(np.max(area_values)),
            )

        fallback_reason = None
        if removed == 0:
            return remove_mask, {
                "removed": 0,
                "removed_ratio": 0.0,
                "fallback_reason": None,
                "edge_quantiles": edge_quantiles,
                "area_quantiles": area_quantiles,
            }
        if removed == int(tri_vertices.shape[0]) or removed_ratio > float(self.cfg.direct_occluder_max_removed_ratio):
            fallback_reason = "occluder_cull_overtrim"
            remove_mask.zero_()
            removed = 0

        return remove_mask, {
            "removed": removed,
            "removed_ratio": removed_ratio,
            "fallback_reason": fallback_reason,
            "edge_quantiles": edge_quantiles,
            "area_quantiles": area_quantiles,
        }

    def _build_direct_surface_proxy_mesh(
        self,
        prediction_val: DecoderOutput | None,
        batch: BatchedExample,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DirectMeshBuildStats, dict[str, float]]:
        if prediction_val is None or prediction_val.depth is None or prediction_val.color is None:
            raise ValueError(
                "mesh.tsdf_gs2d.direct_surface_proxy=true requires target decoder color and depth outputs."
            )

        stride = int(self.cfg.direct_surface_proxy_stride)
        if stride < 1:
            raise ValueError("mesh.tsdf_gs2d.direct_surface_proxy_stride must be >= 1.")
        alpha_threshold = float(self.cfg.direct_surface_proxy_alpha_threshold)
        if not (0.0 <= alpha_threshold <= 1.0):
            raise ValueError("mesh.tsdf_gs2d.direct_surface_proxy_alpha_threshold must be in [0, 1].")
        edge_rel = float(self.cfg.direct_surface_proxy_depth_edge_rel)
        if edge_rel < 0:
            raise ValueError("mesh.tsdf_gs2d.direct_surface_proxy_depth_edge_rel must be >= 0.")
        depth_mode = self.cfg.direct_surface_proxy_depth_mode
        if depth_mode not in ("render", "near_plane"):
            raise ValueError(
                "mesh.tsdf_gs2d.direct_surface_proxy_depth_mode must be 'render' or 'near_plane', "
                f"got {depth_mode!r}."
            )
        depth_scale = float(self.cfg.direct_surface_proxy_depth_scale)
        if depth_scale <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_surface_proxy_depth_scale must be > 0.")
        near_multiplier = float(self.cfg.direct_surface_proxy_near_multiplier)
        if near_multiplier <= 0:
            raise ValueError("mesh.tsdf_gs2d.direct_surface_proxy_near_multiplier must be > 0.")
        frame_pixel_quads = bool(self.cfg.direct_surface_proxy_frame_pixel_quads)
        unified_pixel_quads = bool(self.cfg.direct_surface_proxy_unified_pixel_quads)
        if unified_pixel_quads and stride != 1:
            raise ValueError(
                "mesh.tsdf_gs2d.direct_surface_proxy_unified_pixel_quads=true requires "
                "mesh.tsdf_gs2d.direct_surface_proxy_stride=1."
            )

        phase_times = self._empty_direct_phase_times()
        self._direct_surface_proxy_frame_meshes = []
        self._sync_cuda()
        t_start = time.perf_counter()

        depth = prediction_val.depth.detach()
        color = prediction_val.color.detach()
        opacity = None if prediction_val.opacity is None else prediction_val.opacity.detach()
        if depth.dim() == 4:
            depth = depth[0]
        if color.dim() == 5:
            color = color[0]
        if opacity is not None and opacity.dim() == 4:
            opacity = opacity[0]

        if depth.dim() != 3:
            raise ValueError(f"Unexpected target depth shape for surface proxy: {tuple(depth.shape)}")
        if color.dim() != 4 or color.shape[0] != depth.shape[0] or color.shape[-2:] != depth.shape[-2:]:
            raise ValueError(
                "Unexpected target color shape for surface proxy: "
                f"color={tuple(color.shape)} depth={tuple(depth.shape)}"
            )
        if opacity is not None and (opacity.dim() != 3 or opacity.shape != depth.shape):
            opacity = None

        device = depth.device
        depth = depth.float()
        color = color.float()
        opacity = None if opacity is None else opacity.float()
        extrinsics = batch["target"]["extrinsics"][0].to(device=device, dtype=torch.float32)
        intrinsics = batch["target"]["intrinsics"][0].to(device=device, dtype=torch.float32)
        near = batch["target"]["near"][0].to(device=device, dtype=torch.float32)

        num_views, image_h, image_w = depth.shape
        if extrinsics.shape[0] != num_views or intrinsics.shape[0] != num_views:
            raise ValueError(
                "Surface proxy target view count mismatch: "
                f"depth={num_views}, extrinsics={extrinsics.shape[0]}, intrinsics={intrinsics.shape[0]}"
            )

        rows = torch.arange(0, image_h, stride, device=device)
        cols = torch.arange(0, image_w, stride, device=device)
        if rows.numel() < 2 or cols.numel() < 2:
            raise ValueError("Surface proxy stride leaves fewer than 2 pixels along an image dimension.")
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        hh, ww = int(rows.numel()), int(cols.numel())
        coords = torch.stack(
            [
                (xx.float() + 0.5) / float(image_w),
                (yy.float() + 0.5) / float(image_h),
                torch.ones_like(xx, dtype=torch.float32),
            ],
            dim=-1,
        )
        grid_indices = torch.arange(hh * ww, device=device, dtype=torch.int64).reshape(hh, ww)

        vertices_list: list[torch.Tensor] = []
        colors_list: list[torch.Tensor] = []
        normals_list: list[torch.Tensor] = []
        faces_list: list[torch.Tensor] = []
        frame_mesh_tensors: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        frame_indices = batch["target"]["index"][0].detach().cpu().tolist()
        vertex_offset = 0
        raw_faces = 0

        for view_idx in range(num_views):
            if unified_pixel_quads:
                full_render_depth = depth[view_idx]
                full_color = color[view_idx].permute(1, 2, 0).clamp(0.0, 1.0)
                if depth_mode == "near_plane":
                    full_depth = torch.full_like(full_render_depth, near[view_idx] * near_multiplier * depth_scale)
                else:
                    full_depth = full_render_depth * depth_scale
                full_valid = torch.isfinite(full_render_depth) & (full_render_depth > near[view_idx])
                if opacity is not None and alpha_threshold > 0:
                    full_opacity = opacity[view_idx]
                    full_valid &= torch.isfinite(full_opacity) & (full_opacity >= alpha_threshold)

                vertices_flat, faces, colors_flat, normals_flat = self._build_surface_proxy_pixel_quad_frame(
                    full_color,
                    full_valid,
                    full_depth,
                    extrinsics[view_idx],
                    intrinsics[view_idx],
                )
                raw_faces += int(image_h * image_w * 2)
                vertices_list.append(vertices_flat)
                colors_list.append(colors_flat)
                normals_list.append(normals_flat)
                faces_list.append(faces + vertex_offset)
                if self.cfg.direct_surface_proxy_write_frame_meshes:
                    frame_mesh_tensors.append(
                        (int(frame_indices[view_idx]), vertices_flat, faces, colors_flat, normals_flat)
                    )
                vertex_offset += int(vertices_flat.shape[0])
                continue

            render_depth_view = depth[view_idx][rows][:, cols]
            if depth_mode == "near_plane":
                depth_view = torch.full_like(render_depth_view, near[view_idx] * near_multiplier)
            else:
                depth_view = render_depth_view
            color_view = color[view_idx][:, rows][:, :, cols].permute(1, 2, 0).clamp(0.0, 1.0)
            opacity_view = None if opacity is None else opacity[view_idx][rows][:, cols]

            inv_k = torch.linalg.inv(intrinsics[view_idx])
            dirs_cam = torch.matmul(coords, inv_k.transpose(0, 1))
            dirs_cam = torch.nn.functional.normalize(dirs_cam, dim=-1, eps=1e-8)
            rotation = extrinsics[view_idx, :3, :3]
            dirs_world = torch.matmul(dirs_cam, rotation.transpose(0, 1))
            dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1, eps=1e-8)
            origin = extrinsics[view_idx, :3, 3].view(1, 1, 3)

            if depth_mode == "near_plane":
                valid = torch.isfinite(render_depth_view) & (render_depth_view > near[view_idx])
            else:
                valid = torch.isfinite(depth_view) & (depth_view > near[view_idx])
            if opacity_view is not None and alpha_threshold > 0:
                valid &= torch.isfinite(opacity_view) & (opacity_view >= alpha_threshold)

            vertices = origin + dirs_world * (depth_view * depth_scale).unsqueeze(-1)
            normals = -dirs_world
            local_indices = grid_indices + vertex_offset

            f1 = torch.stack(
                [
                    local_indices[:-1, :-1],
                    local_indices[1:, :-1],
                    local_indices[:-1, 1:],
                ],
                dim=-1,
            )
            f2 = torch.stack(
                [
                    local_indices[1:, :-1],
                    local_indices[1:, 1:],
                    local_indices[:-1, 1:],
                ],
                dim=-1,
            )
            valid_00 = valid[:-1, :-1]
            valid_10 = valid[1:, :-1]
            valid_01 = valid[:-1, 1:]
            valid_11 = valid[1:, 1:]
            f1_valid = valid_00 & valid_10 & valid_01
            f2_valid = valid_10 & valid_11 & valid_01

            if edge_rel > 0:
                d00 = depth_view[:-1, :-1]
                d10 = depth_view[1:, :-1]
                d01 = depth_view[:-1, 1:]
                d11 = depth_view[1:, 1:]
                f1_depths = torch.stack([d00, d10, d01], dim=-1)
                f2_depths = torch.stack([d10, d11, d01], dim=-1)
                f1_range = f1_depths.amax(dim=-1) - f1_depths.amin(dim=-1)
                f2_range = f2_depths.amax(dim=-1) - f2_depths.amin(dim=-1)
                f1_ref = f1_depths.mean(dim=-1).clamp_min(1e-6)
                f2_ref = f2_depths.mean(dim=-1).clamp_min(1e-6)
                f1_valid &= f1_range <= edge_rel * f1_ref
                f2_valid &= f2_range <= edge_rel * f2_ref

            raw_faces += int(f1.numel() // 3 + f2.numel() // 3)
            faces = torch.cat([f1[f1_valid], f2[f2_valid]], dim=0)
            vertices_flat = vertices.reshape(-1, 3)
            colors_flat = color_view.reshape(-1, 3)
            normals_flat = normals.reshape(-1, 3)
            vertices_list.append(vertices_flat)
            colors_list.append(colors_flat)
            normals_list.append(normals_flat)
            faces_list.append(faces)
            if self.cfg.direct_surface_proxy_write_frame_meshes:
                frame_index = int(frame_indices[view_idx])
                if frame_pixel_quads:
                    full_render_depth = depth[view_idx]
                    full_color = color[view_idx].permute(1, 2, 0).clamp(0.0, 1.0)
                    if depth_mode == "near_plane":
                        full_depth = torch.full_like(full_render_depth, near[view_idx] * near_multiplier)
                    else:
                        full_depth = full_render_depth * depth_scale
                    full_valid = torch.isfinite(full_render_depth) & (full_render_depth > near[view_idx])
                    if opacity is not None and alpha_threshold > 0:
                        full_opacity = opacity[view_idx]
                        full_valid &= torch.isfinite(full_opacity) & (full_opacity >= alpha_threshold)
                    frame_vertices, frame_faces, frame_colors, frame_normals = self._build_surface_proxy_pixel_quad_frame(
                        full_color,
                        full_valid,
                        full_depth,
                        extrinsics[view_idx],
                        intrinsics[view_idx],
                    )
                    frame_mesh_tensors.append(
                        (frame_index, frame_vertices, frame_faces, frame_colors, frame_normals)
                    )
                else:
                    frame_mesh_tensors.append(
                        (
                            frame_index,
                            vertices_flat,
                            faces - vertex_offset,
                            colors_flat,
                            normals_flat,
                        )
                    )
            vertex_offset += hh * ww

        if not faces_list:
            raise ValueError("Surface proxy did not produce any faces.")

        vertices_all = torch.cat(vertices_list, dim=0)
        colors_all = torch.cat(colors_list, dim=0)
        normals_all = torch.cat(normals_list, dim=0)
        faces_all = torch.cat(faces_list, dim=0)
        if faces_all.numel() == 0:
            raise ValueError("Surface proxy did not produce any valid faces.")

        self._sync_cuda()
        t_cpu_start = time.perf_counter()
        verts_np = vertices_all.detach().cpu().numpy().astype(np.float32)
        colors_np = (colors_all.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        normals_np = normals_all.detach().cpu().numpy().astype(np.float32)
        faces_np = faces_all.detach().cpu().numpy().astype(np.int32)
        if frame_mesh_tensors:
            self._direct_surface_proxy_frame_meshes = [
                (
                    frame_index,
                    frame_vertices.detach().cpu().numpy().astype(np.float32),
                    frame_faces.detach().cpu().numpy().astype(np.int32),
                    (frame_colors.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8),
                    frame_normals.detach().cpu().numpy().astype(np.float32),
                )
                for frame_index, frame_vertices, frame_faces, frame_colors, frame_normals in frame_mesh_tensors
            ]
        t_end = time.perf_counter()

        phase_times["trim"] = t_cpu_start - t_start
        phase_times["cpu_copy"] = t_end - t_cpu_start
        stats = DirectMeshBuildStats(
            triangles_before_opacity=raw_faces,
            triangles_after_opacity=int(faces_all.shape[0]),
            triangles_after_trim=int(faces_all.shape[0]),
            trim_source="surface_proxy",
            removed_ratio=(raw_faces - int(faces_all.shape[0])) / max(raw_faces, 1),
        )
        return verts_np, faces_np, colors_np, normals_np, stats, phase_times

    @staticmethod
    def _build_surface_proxy_pixel_quad_frame(
        color_view: torch.Tensor,
        valid: torch.Tensor,
        depth_view: torch.Tensor,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_h, image_w, _ = color_view.shape
        device = color_view.device
        yy, xx = torch.meshgrid(
            torch.arange(image_h, device=device),
            torch.arange(image_w, device=device),
            indexing="ij",
        )
        xx = xx.float()
        yy = yy.float()
        corner_x = torch.stack([xx, xx, xx + 1.0, xx + 1.0], dim=-1) / float(image_w)
        corner_y = torch.stack([yy, yy + 1.0, yy, yy + 1.0], dim=-1) / float(image_h)
        coords = torch.stack(
            [
                corner_x,
                corner_y,
                torch.ones_like(corner_x),
            ],
            dim=-1,
        )

        inv_k = torch.linalg.inv(intrinsic)
        dirs_cam = torch.matmul(coords, inv_k.transpose(0, 1))
        dirs_cam = torch.nn.functional.normalize(dirs_cam, dim=-1, eps=1e-8)
        rotation = extrinsic[:3, :3]
        dirs_world = torch.matmul(dirs_cam, rotation.transpose(0, 1))
        dirs_world = torch.nn.functional.normalize(dirs_world, dim=-1, eps=1e-8)
        vertices = extrinsic[:3, 3].view(1, 1, 1, 3) + dirs_world * depth_view.unsqueeze(-1).unsqueeze(-1)
        colors = color_view.unsqueeze(2).expand(-1, -1, 4, -1)
        normals = -dirs_world

        base = torch.arange(image_h * image_w, device=device, dtype=torch.int64).reshape(image_h, image_w) * 4
        f1 = torch.stack([base, base + 1, base + 2], dim=-1)
        f2 = torch.stack([base + 1, base + 3, base + 2], dim=-1)
        faces = torch.cat([f1[valid], f2[valid]], dim=0)
        return (
            vertices.reshape(-1, 3),
            faces,
            colors.reshape(-1, 3),
            normals.reshape(-1, 3),
        )

    @staticmethod
    def _tensor_quantiles(
        values: torch.Tensor,
        quantiles: tuple[float, ...],
    ) -> tuple[float, ...]:
        if values.numel() == 0:
            return tuple(0.0 for _ in quantiles)
        q_tensor = torch.tensor(quantiles, device=values.device, dtype=values.dtype)
        return tuple(float(v) for v in torch.quantile(values, q_tensor).detach().cpu().tolist())

    @staticmethod
    def _pack_integer_keys(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = coords - coords.min(dim=0).values
        span = shifted.max(dim=0).values + 1
        keys = (
            shifted[:, 0] * (span[1] * span[2])
            + shifted[:, 1] * span[2]
            + shifted[:, 2]
        )
        return keys, shifted

    @staticmethod
    def _cluster_voxel_components(voxel_coords: np.ndarray) -> tuple[np.ndarray, int]:
        if voxel_coords.size == 0:
            return np.empty((0,), dtype=np.int32), 0

        coords_list = voxel_coords.tolist()
        coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coords_list)}
        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        component_ids = np.full(len(coords_list), -1, dtype=np.int32)
        component_index = 0
        for start in range(len(coords_list)):
            if component_ids[start] != -1:
                continue
            stack = [start]
            component_ids[start] = component_index
            while stack:
                current = stack.pop()
                x, y, z = coords_list[current]
                for dx, dy, dz in offsets:
                    neighbor = coord_to_idx.get((x + dx, y + dy, z + dz))
                    if neighbor is None or component_ids[neighbor] != -1:
                        continue
                    component_ids[neighbor] = component_index
                    stack.append(neighbor)
            component_index += 1
        return component_ids, component_index

    @staticmethod
    def _expand_voxel_set(
        seed_voxels: set[tuple[int, int, int]],
        active_voxels: set[tuple[int, int, int]],
        halo_layers: int,
    ) -> set[tuple[int, int, int]]:
        keep = set(seed_voxels)
        frontier = set(seed_voxels)
        if halo_layers <= 0:
            return keep

        for _ in range(halo_layers):
            next_frontier: set[tuple[int, int, int]] = set()
            for x, y, z in frontier:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if dx == 0 and dy == 0 and dz == 0:
                                continue
                            neighbor = (x + dx, y + dy, z + dz)
                            if neighbor in active_voxels and neighbor not in keep:
                                next_frontier.add(neighbor)
            keep.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break
        return keep

    def _log_direct_build_stats(self, stats: DirectMeshBuildStats) -> None:
        occupancy_quantiles = (
            "n/a"
            if stats.occupancy_quantiles is None
            else ",".join(f"{value:.1f}" for value in stats.occupancy_quantiles)
        )
        score_quantiles = (
            "n/a"
            if stats.score_quantiles is None
            else ",".join(f"{value:.2f}" for value in stats.score_quantiles)
        )
        extras = []
        if stats.voxel_size is not None:
            extras.append(f"voxel={stats.voxel_size:.4f}")
        if stats.median_edge is not None:
            extras.append(f"median_edge={stats.median_edge:.4f}")
        if stats.sparse_triangle_fraction is not None:
            extras.append(f"sparse_frac={stats.sparse_triangle_fraction:.4f}")
        if stats.triangles_after_source_view is not None:
            extras.append(f"after_source={stats.triangles_after_source_view}")
        if stats.triangles_after_visibility is not None:
            extras.append(f"after_visibility={stats.triangles_after_visibility}")
        if stats.triangles_after_primitive_mask is not None:
            extras.append(f"after_primitive_mask={stats.triangles_after_primitive_mask}")
        if stats.triangles_after_budget is not None:
            extras.append(f"after_budget={stats.triangles_after_budget}")
        if stats.triangles_after_scale_cull is not None:
            extras.append(f"after_scale={stats.triangles_after_scale_cull}")
        if stats.triangles_after_projected_outlier_cull is not None:
            extras.append(f"after_projected={stats.triangles_after_projected_outlier_cull}")
        if stats.triangles_after_projective_conflict_cull is not None:
            extras.append(f"after_projective_conflict={stats.triangles_after_projective_conflict_cull}")
        if stats.triangles_after_camera_cull is not None:
            extras.append(f"after_camera={stats.triangles_after_camera_cull}")
        if stats.triangles_after_occluder_cull is not None:
            extras.append(f"after_occluder={stats.triangles_after_occluder_cull}")
        if stats.triangles_after_depth_cull is not None:
            extras.append(f"after_depth={stats.triangles_after_depth_cull}")
        extras.append(f"removed_source={stats.removed_by_source_view}")
        extras.append(f"removed_visibility={stats.removed_by_target_visibility}")
        extras.append(f"removed_primitive_mask={stats.removed_by_primitive_mask}")
        extras.append(f"removed_budget={stats.removed_by_triangle_budget}")
        extras.append(f"removed_scale={stats.removed_by_scale_cull}")
        extras.append(f"removed_projected={stats.removed_by_projected_outlier}")
        extras.append(f"removed_projective_conflict={stats.removed_by_projective_conflict}")
        extras.append(f"removed_camera={stats.removed_by_camera_cull}")
        extras.append(f"removed_occluder={stats.removed_by_occluder}")
        extras.append(f"removed_depth={stats.removed_by_target_depth}")
        extras.append(f"removed_single={stats.removed_by_single_triangle}")
        extras.append(f"removed_cluster={stats.removed_by_voxel_component}")
        extras.append(f"removed_core={stats.removed_by_core_cluster}")
        extras.append(f"components={stats.voxel_components}")
        extras.append(f"removed_components={stats.removed_voxel_components}")
        extras.append(f"core_components={stats.core_components}")
        extras.append(f"core_voxels={stats.core_voxels}")
        if stats.core_threshold is not None:
            extras.append(f"core_thr={stats.core_threshold:.1f}")
        if stats.source_view_selected_slots is not None:
            extras.append(f"source_slots={','.join(str(slot) for slot in stats.source_view_selected_slots)}")
        if stats.source_view_selection_strategy is not None:
            extras.append(f"source_strategy={stats.source_view_selection_strategy}")
        if stats.source_view_score_quantiles is not None:
            source_q = ",".join(f"{value:.2f}" for value in stats.source_view_score_quantiles)
            extras.append(f"source_score_q={source_q}")
        if stats.triangle_budget_score_quantiles is not None:
            budget_q = ",".join(f"{value:.2f}" for value in stats.triangle_budget_score_quantiles)
            extras.append(f"budget_score_q={budget_q}")
        if stats.triangle_budget_score_mode is not None:
            extras.append(f"budget_score_mode={stats.triangle_budget_score_mode}")
        if stats.triangle_budget_area_quantiles is not None:
            area_q = ",".join(f"{value:.4f}" for value in stats.triangle_budget_area_quantiles)
            extras.append(f"budget_area_q={area_q}")
        if stats.triangle_budget_area_threshold is not None:
            extras.append(f"budget_area_thr={stats.triangle_budget_area_threshold:.4f}")
        if stats.scale_cull_threshold_quantiles is not None:
            scale_q = ",".join(f"{value:.4f}" for value in stats.scale_cull_threshold_quantiles)
            extras.append(f"scale_thr_q={scale_q}")
        if stats.projected_outlier_threshold_quantiles is not None:
            projected_q = ",".join(f"{value:.2f}" for value in stats.projected_outlier_threshold_quantiles)
            extras.append(f"projected_thr_q={projected_q}")
        if stats.projective_conflict_count_quantiles is not None:
            conflict_q = ",".join(f"{value:.1f}" for value in stats.projective_conflict_count_quantiles)
            extras.append(f"projective_conflict_q={conflict_q}")
        if stats.projective_conflict_group_size_quantiles is not None:
            group_q = ",".join(f"{value:.1f}" for value in stats.projective_conflict_group_size_quantiles)
            extras.append(f"projective_group_q={group_q}")
        if stats.projective_conflict_views is not None:
            extras.append(f"projective_views={stats.projective_conflict_views}")
        if stats.occluder_projected_edge_quantiles is not None:
            occluder_edge_q = ",".join(f"{value:.2f}" for value in stats.occluder_projected_edge_quantiles)
            extras.append(f"occluder_edge_q={occluder_edge_q}")
        if stats.occluder_projected_area_quantiles is not None:
            occluder_area_q = ",".join(f"{value:.1f}" for value in stats.occluder_projected_area_quantiles)
            extras.append(f"occluder_area_q={occluder_area_q}")
        extras.append(f"core_halo={stats.core_halo_layers}")
        extras.append(f"removed_ratio={stats.removed_ratio:.4f}")
        extras.append(f"occ_q={occupancy_quantiles}")
        extras.append(f"score_q={score_quantiles}")
        if stats.fallback_reason is not None:
            extras.append(f"fallback={stats.fallback_reason}")
        if stats.source_view_filter_fallback_reason is not None:
            extras.append(f"source_fallback={stats.source_view_filter_fallback_reason}")
        if stats.visibility_cull_fallback_reason is not None:
            extras.append(f"visibility_fallback={stats.visibility_cull_fallback_reason}")
        if stats.triangle_budget_fallback_reason is not None:
            extras.append(f"budget_fallback={stats.triangle_budget_fallback_reason}")
        if stats.scale_cull_fallback_reason is not None:
            extras.append(f"scale_fallback={stats.scale_cull_fallback_reason}")
        if stats.projected_outlier_fallback_reason is not None:
            extras.append(f"projected_fallback={stats.projected_outlier_fallback_reason}")
        if stats.projective_conflict_fallback_reason is not None:
            extras.append(f"projective_conflict_fallback={stats.projective_conflict_fallback_reason}")
        if stats.camera_cull_fallback_reason is not None:
            extras.append(f"camera_fallback={stats.camera_cull_fallback_reason}")
        if stats.occluder_fallback_reason is not None:
            extras.append(f"occluder_fallback={stats.occluder_fallback_reason}")
        if stats.depth_cull_fallback_reason is not None:
            extras.append(f"depth_fallback={stats.depth_cull_fallback_reason}")
        print(
            f"[Mesh][direct][trim] source={stats.trim_source} "
            f"triangles={stats.triangles_before_opacity}->{stats.triangles_after_opacity}->{stats.triangles_after_trim} "
            + " ".join(extras)
        )

    def _write_direct_trim_stats(
        self,
        output_dir: Path,
        basename: str,
        stats: DirectMeshBuildStats,
    ) -> None:
        payload = asdict(stats)
        path = output_dir / f"{basename}_trim_stats.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _write_tsdf_post_stats(
        self,
        output_dir: Path,
        basename: str,
        stats: TsdfPostProcessStats,
    ) -> None:
        payload = asdict(stats)
        path = output_dir / f"{basename}_post_stats.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @torch.no_grad()
    def extract_mesh_bounded(self, mask_background=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """

        self.estimate_bounding_sphere()

        valid_depths = self._collect_valid_depths()
        voxel_size = self.cfg.voxel_size
        sdf_trunc_cfg = self.cfg.sdf_truc
        depth_trunc_cfg = self.cfg.depth_truc
        mesh_res = self.cfg.mesh_res

        if depth_trunc_cfg < 0:
            depth_trunc, depth_trunc_source = self._resolve_auto_depth_trunc(valid_depths)
        else:
            depth_trunc = depth_trunc_cfg
            depth_trunc_source = "config"
        if voxel_size < 0:
            voxel_size, voxel_size_source = self._resolve_auto_voxel_size(depth_trunc)
        else:
            voxel_size_source = "config"
        if sdf_trunc_cfg < 0:
            sdf_trunc, sdf_trunc_source = self._resolve_auto_sdf_trunc(voxel_size)
        else:
            sdf_trunc = sdf_trunc_cfg
            sdf_trunc_source = "config"

        print(
            "[Mesh][tsdf] "
            f"depth_trunc={depth_trunc:.4f} ({depth_trunc_source}) "
            f"voxel_size={voxel_size:.6f} ({voxel_size_source}) "
            f"sdf_trunc={sdf_trunc:.6f} ({sdf_trunc_source}) mesh_res={mesh_res}"
        )
        print("Running tsdf volume integration ...")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        alpha_thr = self.cfg.tsdf_alpha_threshold
        edge_thr = self.cfg.tsdf_depth_edge_threshold

        for i in range(len(self.extrinsics)):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i].clone()

            if mask_background:
                depth[(self.gt_alpha_masks[i] < alpha_thr)] = 0

            if edge_thr > 0:
                d = depth
                valid = d > 0
                valid_pair_x = valid[:, 1:] & valid[:, :-1]
                valid_pair_y = valid[1:, :] & valid[:-1, :]

                grad_x = torch.abs(d[:, 1:] - d[:, :-1])
                grad_y = torch.abs(d[1:, :] - d[:-1, :])
                ref_x = (d[:, 1:] + d[:, :-1]) * 0.5
                ref_y = (d[1:, :] + d[:-1, :]) * 0.5

                is_edge_x = valid_pair_x & (grad_x > edge_thr * ref_x)
                is_edge_y = valid_pair_y & (grad_y > edge_thr * ref_y)

                edge_mask = torch.zeros_like(d, dtype=torch.bool)
                edge_mask[:, 1:] |= is_edge_x
                edge_mask[:, :-1] |= is_edge_x
                edge_mask[1:, :] |= is_edge_y
                edge_mask[:-1, :] |= is_edge_y
                depth[edge_mask] = 0

            rgb_clamped = rgb.clamp(0, 1)

            depth_1hw = depth.unsqueeze(0)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(
                    np.asarray(
                        rgb_clamped.permute(1, 2, 0).cpu().numpy() * 255,
                        order="C",
                        dtype=np.uint8,
                    )
                ),
                o3d.geometry.Image(
                    np.asarray(
                        depth_1hw.permute(1, 2, 0).cpu().numpy(),
                        order="C",
                    )
                ),
                depth_trunc=depth_trunc,
                convert_rgb_to_intensity=False,
                depth_scale=1.0,
            )

            h, w = rgb.shape[1:]
            intrinsic = self.build_o3d_intrinsic_from_normalized_k(self.intrinsics[i], w, h)
            if i > 1:
                pass
            volume.integrate(rgbd, intrinsic=intrinsic, extrinsic=self.extrinsics[i])
        mesh = volume.extract_triangle_mesh()
        return mesh

    def _apply_opacity_temperature(
        self,
        opacity: torch.Tensor,
        global_step: int,
        opacity_temp_initial: float,
        opacity_temp_final: float,
        opacity_temp_warmup_steps: int,
    ) -> torch.Tensor:
        """Apply the same opacity temperature scaling as the decoder."""
        opacity = opacity.float()
        if opacity.numel() == 0:
            return opacity

        if global_step >= opacity_temp_warmup_steps:
            temperature = opacity_temp_final
        else:
            progress = global_step / opacity_temp_warmup_steps
            temperature = opacity_temp_initial + (opacity_temp_final - opacity_temp_initial) * progress

        opacity_clamped = opacity.clamp(min=1e-6, max=1 - 1e-6)
        opacity_logits = torch.logit(opacity_clamped)
        return torch.sigmoid(opacity_logits * temperature)

    @staticmethod
    def _normalize_vectors(
        vectors: torch.Tensor,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize a batch of vectors and report which entries are valid."""
        norm = torch.linalg.norm(vectors, dim=-1, keepdim=True)
        valid = norm.squeeze(-1) > eps
        normalized = torch.where(
            valid.unsqueeze(-1),
            vectors / norm.clamp_min(eps),
            torch.zeros_like(vectors),
        )
        return normalized, valid

    def _apply_direct_geometry_trim(
        self,
        tri_vertices: torch.Tensor,
        tri_features: torch.Tensor,
        tri_normals: torch.Tensor | None,
        triangles_before_opacity: int,
        triangles_after_opacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, DirectMeshBuildStats]:
        if not self.cfg.direct_trim_enabled:
            return tri_vertices, tri_features, tri_normals, DirectMeshBuildStats(
                triangles_before_opacity=triangles_before_opacity,
                triangles_after_opacity=triangles_after_opacity,
                triangles_after_trim=triangles_after_opacity,
                trim_source="disabled",
            )

        self._validate_direct_trim_cfg()

        tri_centroids = tri_vertices.mean(dim=1)
        edge_01 = torch.linalg.norm(tri_vertices[:, 0] - tri_vertices[:, 1], dim=-1)
        edge_12 = torch.linalg.norm(tri_vertices[:, 1] - tri_vertices[:, 2], dim=-1)
        edge_20 = torch.linalg.norm(tri_vertices[:, 2] - tri_vertices[:, 0], dim=-1)
        max_edge = torch.maximum(edge_01, torch.maximum(edge_12, edge_20)).float()
        tri_area = (
            0.5
            * torch.linalg.norm(
                torch.cross(
                    tri_vertices[:, 1] - tri_vertices[:, 0],
                    tri_vertices[:, 2] - tri_vertices[:, 0],
                    dim=-1,
                ),
                dim=-1,
            )
        ).float()

        eps = 1e-6
        median_area = float(tri_area.median().item())
        median_edge = float(max_edge.median().item())
        median_area_safe = max(median_area, eps)
        median_edge_safe = max(median_edge, eps)
        voxel_size = max(median_edge_safe * float(self.cfg.direct_trim_voxel_scale), eps)

        voxel_coords = torch.floor(tri_centroids / voxel_size).to(torch.int64)
        voxel_keys, _ = self._pack_integer_keys(voxel_coords)
        unique_keys, inverse_indices, voxel_counts = torch.unique(
            voxel_keys,
            return_inverse=True,
            return_counts=True,
        )
        sort_order = torch.argsort(inverse_indices)
        sorted_inverse = inverse_indices[sort_order]
        first_mask = torch.ones_like(sorted_inverse, dtype=torch.bool)
        first_mask[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
        unique_voxel_coords = voxel_coords[sort_order[first_mask]]
        del unique_keys
        occupancy = voxel_counts[inverse_indices].float()

        area_threshold_hi = float(
            torch.quantile(
                tri_area,
                torch.tensor(self.cfg.direct_trim_area_quantile_hi, device=tri_area.device, dtype=tri_area.dtype),
            ).item()
        )
        edge_threshold_hi = float(
            torch.quantile(
                max_edge,
                torch.tensor(self.cfg.direct_trim_edge_quantile_hi, device=max_edge.device, dtype=max_edge.dtype),
            ).item()
        )
        area_ratio = tri_area / median_area_safe
        edge_ratio = max_edge / median_edge_safe
        score = area_ratio * edge_ratio / occupancy.clamp_min(1.0)

        sparse_mask = occupancy <= float(self.cfg.direct_trim_min_voxel_occupancy)
        large_mask = (tri_area >= area_threshold_hi) | (max_edge >= edge_threshold_hi)
        high_score_mask = score > float(self.cfg.direct_trim_score_threshold)
        remove_single_mask = sparse_mask & (large_mask | high_score_mask)
        if self.cfg.direct_trim_keep_sparse_small:
            remove_single_mask &= ~((tri_area < area_threshold_hi) & (max_edge < edge_threshold_hi))
        keep_after_single = ~remove_single_mask

        removed_by_single_triangle = int(remove_single_mask.sum().item())
        sparse_triangle_fraction = float(sparse_mask.float().mean().item())
        occupancy_quantiles = self._tensor_quantiles(occupancy, (0.01, 0.50, 0.95))
        score_quantiles = self._tensor_quantiles(score, (0.50, 0.90, 0.95, 0.99))

        max_removed_ratio = float(self.cfg.direct_trim_max_removed_ratio)
        fallback_reason: str | None = None
        removed_by_voxel_component = 0
        removed_by_core_cluster = 0
        voxel_components = 0
        removed_voxel_components = 0
        core_components = 0
        core_voxels = 0
        core_threshold = None
        core_halo_layers = int(self.cfg.direct_trim_core_halo_layers)
        final_keep_mask = keep_after_single

        removed_ratio_after_single = removed_by_single_triangle / max(triangles_after_opacity, 1)
        if final_keep_mask.sum().item() == 0 or removed_ratio_after_single > max_removed_ratio:
            final_keep_mask = torch.ones_like(keep_after_single)
            removed_by_single_triangle = 0
            voxel_components = 0
            removed_voxel_components = 0
            fallback_reason = "single_stage_overtrim"
        else:
            kept_inverse = inverse_indices[keep_after_single]
            kept_area = tri_area[keep_after_single]
            num_voxels = int(voxel_counts.shape[0])
            voxel_triangle_counts = torch.zeros(num_voxels, dtype=torch.int64, device=tri_vertices.device)
            voxel_triangle_counts.scatter_add_(
                0,
                kept_inverse,
                torch.ones_like(kept_inverse, dtype=torch.int64),
            )
            voxel_area_sums = torch.zeros(num_voxels, dtype=tri_area.dtype, device=tri_vertices.device)
            voxel_area_sums.scatter_add_(0, kept_inverse, kept_area)
            active_voxel_indices = torch.nonzero(voxel_triangle_counts > 0, as_tuple=False).squeeze(1)

            if active_voxel_indices.numel() > 0:
                active_voxel_coords = unique_voxel_coords[active_voxel_indices]
                component_ids_np, voxel_components = self._cluster_voxel_components(
                    active_voxel_coords.detach().cpu().numpy().astype(np.int64, copy=False)
                )
                if voxel_components > 0:
                    component_ids = torch.from_numpy(component_ids_np).to(
                        device=tri_vertices.device,
                        dtype=torch.int64,
                    )
                    component_triangle_counts = torch.zeros(
                        voxel_components,
                        dtype=torch.int64,
                        device=tri_vertices.device,
                    )
                    component_triangle_counts.scatter_add_(
                        0,
                        component_ids,
                        voxel_triangle_counts[active_voxel_indices],
                    )
                    component_area_sums = torch.zeros(
                        voxel_components,
                        dtype=tri_area.dtype,
                        device=tri_vertices.device,
                    )
                    component_area_sums.scatter_add_(
                        0,
                        component_ids,
                        voxel_area_sums[active_voxel_indices],
                    )
                    total_kept_area = float(component_area_sums.sum().item())
                    min_component_area = total_kept_area * float(self.cfg.direct_trim_voxel_component_min_area_ratio)
                    remove_component_mask = (
                        (component_triangle_counts < self.cfg.direct_trim_voxel_component_min_triangles)
                        & (component_area_sums < min_component_area)
                    )
                    removed_voxel_components = int(remove_component_mask.sum().item())

                    voxel_component_ids = torch.full(
                        (num_voxels,),
                        -1,
                        dtype=torch.int64,
                        device=tri_vertices.device,
                    )
                    voxel_component_ids[active_voxel_indices] = component_ids
                    kept_component_ids = voxel_component_ids[kept_inverse]
                    keep_after_component = keep_after_single.clone()
                    keep_after_component[keep_after_single] = ~remove_component_mask[kept_component_ids]
                    removed_by_voxel_component = int(
                        keep_after_single.sum().item() - keep_after_component.sum().item()
                    )
                    removed_ratio_after_component = (
                        (removed_by_single_triangle + removed_by_voxel_component)
                        / max(triangles_after_opacity, 1)
                    )
                    if keep_after_component.sum().item() == 0:
                        final_keep_mask = keep_after_single
                        removed_by_voxel_component = 0
                        removed_voxel_components = 0
                        fallback_reason = "component_stage_empty"
                    elif removed_ratio_after_component > max_removed_ratio:
                        final_keep_mask = keep_after_single
                        removed_by_voxel_component = 0
                        removed_voxel_components = 0
                        fallback_reason = "component_stage_overtrim"
                    else:
                        final_keep_mask = keep_after_component

        if final_keep_mask.sum().item() > 0 and fallback_reason != "single_stage_overtrim":
            core_inverse = inverse_indices[final_keep_mask]
            num_voxels = int(voxel_counts.shape[0])
            voxel_triangle_counts_core = torch.zeros(
                num_voxels,
                dtype=torch.int64,
                device=tri_vertices.device,
            )
            voxel_triangle_counts_core.scatter_add_(
                0,
                core_inverse,
                torch.ones_like(core_inverse, dtype=torch.int64),
            )
            active_voxel_indices_core = torch.nonzero(
                voxel_triangle_counts_core > 0,
                as_tuple=False,
            ).squeeze(1)
            if active_voxel_indices_core.numel() > 0:
                active_voxel_coords = unique_voxel_coords[active_voxel_indices_core]
                active_voxel_occ = voxel_triangle_counts_core[active_voxel_indices_core].float()
                core_threshold = max(
                    float(self.cfg.direct_trim_core_min_occupancy),
                    float(
                        torch.quantile(
                            active_voxel_occ,
                            torch.tensor(
                                self.cfg.direct_trim_core_quantile,
                                device=active_voxel_occ.device,
                                dtype=active_voxel_occ.dtype,
                            ),
                        ).item()
                    ),
                )
                dense_core_mask = active_voxel_occ >= core_threshold
                dense_core_coords = active_voxel_coords[dense_core_mask]
                if dense_core_coords.numel() > 0:
                    core_component_ids_np, core_components = self._cluster_voxel_components(
                        dense_core_coords.detach().cpu().numpy().astype(np.int64, copy=False)
                    )
                    if core_components > 0:
                        core_component_ids = torch.from_numpy(core_component_ids_np).to(
                            device=tri_vertices.device,
                            dtype=torch.int64,
                        )
                        core_component_triangle_counts = torch.zeros(
                            core_components,
                            dtype=torch.int64,
                            device=tri_vertices.device,
                        )
                        core_component_triangle_counts.scatter_add_(
                            0,
                            core_component_ids,
                            active_voxel_occ[dense_core_mask].to(torch.int64),
                        )
                        best_core_component = int(core_component_triangle_counts.argmax().item())
                        best_core_mask = core_component_ids == best_core_component
                        dense_core_coords_np = dense_core_coords.detach().cpu().numpy().astype(np.int64, copy=False)
                        seed_voxels = {
                            tuple(coord.tolist())
                            for coord in dense_core_coords_np[best_core_mask.detach().cpu().numpy()]
                        }
                        active_voxels = {
                            tuple(coord.tolist())
                            for coord in active_voxel_coords.detach().cpu().numpy().astype(np.int64, copy=False)
                        }
                        keep_voxels = self._expand_voxel_set(
                            seed_voxels,
                            active_voxels,
                            halo_layers=core_halo_layers,
                        )
                        core_voxels = len(seed_voxels)
                        keep_active_voxel_mask = np.array(
                            [tuple(coord.tolist()) in keep_voxels for coord in active_voxel_coords.detach().cpu().numpy()],
                            dtype=bool,
                        )
                        voxel_keep_mask = torch.zeros(
                            num_voxels,
                            dtype=torch.bool,
                            device=tri_vertices.device,
                        )
                        voxel_keep_mask[active_voxel_indices_core] = torch.from_numpy(keep_active_voxel_mask).to(
                            device=tri_vertices.device
                        )
                        keep_after_core = final_keep_mask.clone()
                        keep_after_core[final_keep_mask] = voxel_keep_mask[core_inverse]
                        removed_ratio_after_core = (
                            (final_keep_mask.sum().item() - keep_after_core.sum().item())
                            / max(triangles_after_opacity, 1)
                        )
                        if (
                            keep_after_core.sum().item() > 0
                            and removed_ratio_after_core <= float(self.cfg.direct_trim_core_max_removed_ratio)
                        ):
                            removed_by_core_cluster = int(
                                final_keep_mask.sum().item() - keep_after_core.sum().item()
                            )
                            final_keep_mask = keep_after_core
                        elif fallback_reason is None:
                            fallback_reason = "core_stage_overtrim"

        if final_keep_mask.sum().item() == 0:
            final_keep_mask = torch.ones_like(final_keep_mask)
            removed_by_single_triangle = 0
            removed_by_voxel_component = 0
            removed_by_core_cluster = 0
            removed_voxel_components = 0
            voxel_components = 0
            core_components = 0
            core_voxels = 0
            core_threshold = None
            fallback_reason = "full_fallback_to_opacity_only"

        tri_vertices = tri_vertices[final_keep_mask]
        tri_features = tri_features[final_keep_mask]
        if tri_normals is not None:
            tri_normals = tri_normals[final_keep_mask]

        stats = DirectMeshBuildStats(
            triangles_before_opacity=triangles_before_opacity,
            triangles_after_opacity=triangles_after_opacity,
            triangles_after_trim=int(final_keep_mask.sum().item()),
            trim_source="geometry_density",
            voxel_size=voxel_size,
            median_area=median_area,
            median_edge=median_edge,
            area_threshold_hi=area_threshold_hi,
            edge_threshold_hi=edge_threshold_hi,
            score_threshold=float(self.cfg.direct_trim_score_threshold),
            sparse_triangle_fraction=sparse_triangle_fraction,
            occupancy_quantiles=occupancy_quantiles,
            score_quantiles=score_quantiles,
            removed_by_single_triangle=removed_by_single_triangle,
            removed_by_voxel_component=removed_by_voxel_component,
            removed_by_core_cluster=removed_by_core_cluster,
            voxel_components=voxel_components,
            removed_voxel_components=removed_voxel_components,
            core_components=core_components,
            core_voxels=core_voxels,
            core_threshold=core_threshold,
            core_halo_layers=core_halo_layers,
            removed_ratio=(
                (triangles_after_opacity - int(final_keep_mask.sum().item()))
                / max(triangles_after_opacity, 1)
            ),
            fallback_reason=fallback_reason,
        )
        return tri_vertices, tri_features, tri_normals, stats

    def _build_direct_mesh_from_triangles(
        self,
        primitives: Union[Gaussians, Triangles],
        batch: BatchedExample | None = None,
        target_visibility_mask: torch.Tensor | None = None,
        target_depth: torch.Tensor | None = None,
        target_opacity: torch.Tensor | None = None,
        global_step: int = 0,
        opacity_temp_initial: float = 1.0,
        opacity_temp_final: float = 1.0,
        opacity_temp_warmup_steps: int = 5000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DirectMeshBuildStats, dict[str, float]]:
        """
        Convert triangle primitives to raw numpy arrays (no Open3D overhead).
        Returns (vertices_f32, faces_i32, colors_u8, normals_f32, build_stats, phase_times).
        """
        if not isinstance(primitives, Triangles):
            raise TypeError("Direct mesh export requires Triangles primitives.")

        vertices = primitives.vertices.detach()
        if vertices.dim() != 4 or vertices.shape[-2:] != (3, 3):
            raise ValueError(f"Unexpected triangle vertices shape: {tuple(vertices.shape)}")

        if vertices.shape[0] < 1:
            raise ValueError("No batch found in triangles.")
        if vertices.shape[0] > 1:
            print(f"[Mesh][direct] batch size is {vertices.shape[0]}, only using batch 0.")

        tri_vertices = vertices[0]  # [N, 3, 3]
        tri_opacity = primitives.opacity.detach()[0, :, 0]  # [N]
        tri_features = primitives.features.detach()[0]  # [N, C]
        tri_normals = (
            primitives.normals.detach()[0]
            if primitives.normals is not None
            else None
        )  # [N, 3] or None

        phase_times = self._empty_direct_phase_times()

        self._sync_cuda()
        t_trim_start = time.perf_counter()
        triangles_before_opacity = int(tri_vertices.shape[0])
        keep_mask = torch.ones((triangles_before_opacity,), dtype=torch.bool, device=tri_vertices.device)

        source_slots, source_layout_info = self._infer_direct_source_view_slots(
            triangles_before_opacity,
            batch,
            tri_vertices.device,
        )
        visibility_counts, visibility_counts_info = self._build_direct_visibility_counts(
            target_visibility_mask,
            triangles_before_opacity,
            tri_vertices.device,
        )
        source_keep_mask, source_info = self._build_direct_source_view_keep_mask(
            triangles_before_opacity,
            source_slots,
            source_layout_info,
            tri_vertices.device,
            visibility_counts=visibility_counts,
            tri_opacity=tri_opacity,
        )
        if source_keep_mask is not None:
            keep_mask &= source_keep_mask
        source_removed = int(source_info.get("removed") or 0)
        triangles_after_source_view = int(keep_mask.sum().item())

        triangles_after_primitive_mask: int | None = None
        primitive_mask_removed = 0
        primitive_valid_mask = getattr(primitives, "primitive_valid_mask", None)
        if primitive_valid_mask is not None:
            primitive_valid_mask = primitive_valid_mask.detach()
            if primitive_valid_mask.dim() != 2 or primitive_valid_mask.shape[0] < 1:
                raise ValueError(
                    "primitive_valid_mask must have shape [B,N], "
                    f"got {tuple(primitive_valid_mask.shape)}"
                )
            primitive_keep = primitive_valid_mask[0].to(
                device=keep_mask.device,
                dtype=torch.bool,
            )
            if primitive_keep.shape[0] != triangles_before_opacity:
                raise ValueError(
                    "primitive_valid_mask length does not match triangle count: "
                    f"{primitive_keep.shape[0]} vs {triangles_before_opacity}"
                )
            current_count = int(keep_mask.sum().item())
            keep_mask &= primitive_keep
            triangles_after_primitive_mask = int(keep_mask.sum().item())
            primitive_mask_removed = current_count - triangles_after_primitive_mask

        threshold = self._get_direct_opacity_threshold()
        if threshold >= 0:
            tri_opacity = self._apply_opacity_temperature(
                tri_opacity, global_step,
                opacity_temp_initial, opacity_temp_final, opacity_temp_warmup_steps,
            )
            keep_mask &= tri_opacity >= threshold

        triangles_after_opacity = int(keep_mask.sum().item())
        triangles_after_visibility: int | None = None
        visibility_removed = 0
        visibility_removed_ratio: float | None = None
        visibility_fallback_reason: str | None = None
        if self.cfg.direct_target_visibility_cull:
            self._validate_direct_camera_cull_cfg()
            if visibility_counts is None:
                visibility_fallback_reason = visibility_counts_info.get("fallback_reason") or "missing_visibility_mask"
            else:
                min_views = int(self.cfg.direct_visibility_min_target_views)
                visible_keep = visibility_counts >= min_views
                candidate_keep = keep_mask & visible_keep
                candidate_count = int(candidate_keep.sum().item())
                visibility_removed = triangles_after_opacity - candidate_count
                visibility_removed_ratio = visibility_removed / max(triangles_after_opacity, 1)
                if candidate_count == 0 or visibility_removed_ratio > float(self.cfg.direct_visibility_max_removed_ratio):
                    visibility_fallback_reason = "visibility_cull_overtrim"
                    visibility_removed = 0
                else:
                    keep_mask = candidate_keep
            triangles_after_visibility = int(keep_mask.sum().item())

        triangles_after_camera_cull: int | None = None
        camera_removed = 0
        camera_info = {
            "removed_ratio": None,
            "fallback_reason": None,
            "near_clip": None,
            "views": None,
        }
        if self.cfg.direct_camera_cull_enabled and int(keep_mask.sum().item()) > 0:
            active_indices = torch.nonzero(keep_mask, as_tuple=False).flatten()
            camera_remove_mask, camera_info = self._build_direct_camera_remove_mask(
                tri_vertices[active_indices],
                batch,
            )
            if camera_remove_mask is not None:
                camera_keep_mask = ~camera_remove_mask
                camera_removed = int(camera_remove_mask.sum().item())
                if int(camera_keep_mask.sum().item()) > 0:
                    keep_mask = torch.zeros_like(keep_mask)
                    keep_mask[active_indices[camera_keep_mask]] = True
                triangles_after_camera_cull = int(keep_mask.sum().item())

        triangles_after_projective_conflict_cull: int | None = None
        projective_conflict_removed = 0
        projective_conflict_info = {
            "removed_ratio": None,
            "fallback_reason": None,
            "count_quantiles": None,
            "group_size_quantiles": None,
            "views": None,
        }
        if self.cfg.direct_projective_conflict_cull_enabled and int(keep_mask.sum().item()) > 0:
            projective_conflict_remove_mask, projective_conflict_info = (
                self._build_direct_projective_conflict_remove_mask(
                    tri_vertices,
                    tri_opacity,
                    keep_mask,
                    batch,
                )
            )
            if projective_conflict_remove_mask is not None:
                projective_conflict_keep_mask = ~projective_conflict_remove_mask
                projective_conflict_removed = int(projective_conflict_remove_mask.sum().item())
                if int((keep_mask & projective_conflict_keep_mask).sum().item()) > 0:
                    keep_mask &= projective_conflict_keep_mask
                triangles_after_projective_conflict_cull = int(keep_mask.sum().item())

        triangles_after_budget: int | None = None
        budget_removed = 0
        budget_info = {
            "removed_ratio": None,
            "fallback_reason": None,
            "score_quantiles": None,
            "score_mode": self._resolve_direct_triangle_budget_score_mode(),
            "area_quantiles": None,
            "area_threshold": None,
        }
        budget_keep_mask, budget_info = self._build_direct_triangle_budget_keep_mask(
            keep_mask,
            tri_vertices,
            tri_opacity,
            visibility_counts,
            batch,
        )
        if budget_keep_mask is not None:
            current_count = int(keep_mask.sum().item())
            if int(budget_keep_mask.sum().item()) > 0:
                keep_mask = budget_keep_mask
                budget_removed = current_count - int(keep_mask.sum().item())
            triangles_after_budget = int(keep_mask.sum().item())

        triangles_after_scale_cull: int | None = None
        scale_removed = 0
        scale_info = {
            "removed_ratio": None,
            "fallback_reason": None,
            "threshold_quantiles": None,
        }
        scale_remove_mask, scale_info = self._build_direct_scale_remove_mask(
            tri_vertices,
            keep_mask,
            source_slots,
        )
        if scale_remove_mask is not None:
            current_count = int(keep_mask.sum().item())
            candidate_keep = keep_mask & ~scale_remove_mask
            candidate_count = int(candidate_keep.sum().item())
            if candidate_count > 0:
                keep_mask = candidate_keep
                scale_removed = current_count - candidate_count
            triangles_after_scale_cull = int(keep_mask.sum().item())

        tri_vertices = tri_vertices[keep_mask]
        tri_features = tri_features[keep_mask]
        tri_opacity = tri_opacity[keep_mask]
        if tri_normals is not None:
            tri_normals = tri_normals[keep_mask]

        if tri_vertices.shape[0] == 0:
            raise ValueError("No triangles remaining after direct mesh filtering.")

        triangles_after_occluder_cull: int | None = None
        occluder_removed = 0
        occluder_info = {
            "removed_ratio": None,
            "fallback_reason": None,
            "edge_quantiles": None,
            "area_quantiles": None,
        }
        occluder_remove_mask, occluder_info = self._build_direct_occluder_remove_mask(
            tri_vertices,
            batch,
            target_depth,
            target_opacity,
        )
        if occluder_remove_mask is not None:
            occluder_keep_mask = ~occluder_remove_mask
            occluder_removed = int(occluder_remove_mask.sum().item())
            if occluder_keep_mask.sum().item() > 0:
                tri_vertices = tri_vertices[occluder_keep_mask]
                tri_features = tri_features[occluder_keep_mask]
                tri_opacity = tri_opacity[occluder_keep_mask]
                if tri_normals is not None:
                    tri_normals = tri_normals[occluder_keep_mask]
            triangles_after_occluder_cull = int(tri_vertices.shape[0])

        triangles_after_projected_outlier_cull: int | None = None
        projected_outlier_removed = 0
        projected_outlier_info = {
            "removed_ratio": None,
            "fallback_reason": None,
            "threshold_quantiles": None,
        }
        projected_outlier_remove_mask, projected_outlier_info = self._build_direct_projected_outlier_remove_mask(
            tri_vertices,
            tri_opacity,
            batch,
        )
        if projected_outlier_remove_mask is not None:
            projected_outlier_keep_mask = ~projected_outlier_remove_mask
            projected_outlier_removed = int(projected_outlier_remove_mask.sum().item())
            if projected_outlier_keep_mask.sum().item() > 0:
                tri_vertices = tri_vertices[projected_outlier_keep_mask]
                tri_features = tri_features[projected_outlier_keep_mask]
                tri_opacity = tri_opacity[projected_outlier_keep_mask]
                if tri_normals is not None:
                    tri_normals = tri_normals[projected_outlier_keep_mask]
            triangles_after_projected_outlier_cull = int(tri_vertices.shape[0])

        triangles_after_depth_cull: int | None = None
        depth_removed = 0
        depth_info = {
            "removed_ratio": None,
            "fallback_reason": None,
        }
        depth_remove_mask, depth_info = self._build_direct_target_depth_remove_mask(
            tri_vertices,
            batch,
            target_depth,
            target_opacity,
        )
        if depth_remove_mask is not None:
            depth_keep_mask = ~depth_remove_mask
            depth_removed = int(depth_remove_mask.sum().item())
            if depth_keep_mask.sum().item() > 0:
                tri_vertices = tri_vertices[depth_keep_mask]
                tri_features = tri_features[depth_keep_mask]
                tri_opacity = tri_opacity[depth_keep_mask]
                if tri_normals is not None:
                    tri_normals = tri_normals[depth_keep_mask]
            triangles_after_depth_cull = int(tri_vertices.shape[0])

        tri_vertices, tri_features, tri_normals, stats = self._apply_direct_geometry_trim(
            tri_vertices,
            tri_features,
            tri_normals,
            triangles_before_opacity=triangles_before_opacity,
            triangles_after_opacity=int(tri_vertices.shape[0]),
        )
        stats = replace(
            stats,
            triangles_after_source_view=triangles_after_source_view,
            triangles_after_opacity=triangles_after_opacity,
            triangles_after_visibility=triangles_after_visibility,
            triangles_after_primitive_mask=triangles_after_primitive_mask,
            triangles_after_budget=triangles_after_budget,
            triangles_after_scale_cull=triangles_after_scale_cull,
            triangles_after_projected_outlier_cull=triangles_after_projected_outlier_cull,
            triangles_after_projective_conflict_cull=triangles_after_projective_conflict_cull,
            triangles_after_camera_cull=triangles_after_camera_cull,
            triangles_after_depth_cull=triangles_after_depth_cull,
            triangles_after_occluder_cull=triangles_after_occluder_cull,
            removed_by_source_view=source_removed,
            removed_by_target_visibility=visibility_removed,
            removed_by_primitive_mask=primitive_mask_removed,
            removed_by_triangle_budget=budget_removed,
            removed_by_scale_cull=scale_removed,
            removed_by_projected_outlier=projected_outlier_removed,
            removed_by_projective_conflict=projective_conflict_removed,
            removed_by_camera_cull=camera_removed,
            removed_by_target_depth=depth_removed,
            removed_by_occluder=occluder_removed,
            source_view_selected_slots=source_info.get("selected_slots"),
            source_view_selection_strategy=source_info.get("selection_strategy"),
            source_view_score_quantiles=source_info.get("score_quantiles"),
            source_view_filter_removed_ratio=source_info.get("removed_ratio"),
            source_view_filter_fallback_reason=source_info.get("fallback_reason"),
            visibility_cull_removed_ratio=visibility_removed_ratio,
            visibility_cull_fallback_reason=visibility_fallback_reason,
            triangle_budget_removed_ratio=budget_info.get("removed_ratio"),
            triangle_budget_score_quantiles=budget_info.get("score_quantiles"),
            triangle_budget_score_mode=budget_info.get("score_mode"),
            triangle_budget_area_quantiles=budget_info.get("area_quantiles"),
            triangle_budget_area_threshold=budget_info.get("area_threshold"),
            triangle_budget_fallback_reason=budget_info.get("fallback_reason"),
            scale_cull_removed_ratio=scale_info.get("removed_ratio"),
            scale_cull_fallback_reason=scale_info.get("fallback_reason"),
            scale_cull_threshold_quantiles=scale_info.get("threshold_quantiles"),
            projected_outlier_removed_ratio=projected_outlier_info.get("removed_ratio"),
            projected_outlier_fallback_reason=projected_outlier_info.get("fallback_reason"),
            projected_outlier_threshold_quantiles=projected_outlier_info.get("threshold_quantiles"),
            projective_conflict_removed_ratio=projective_conflict_info.get("removed_ratio"),
            projective_conflict_fallback_reason=projective_conflict_info.get("fallback_reason"),
            projective_conflict_count_quantiles=projective_conflict_info.get("count_quantiles"),
            projective_conflict_group_size_quantiles=projective_conflict_info.get("group_size_quantiles"),
            projective_conflict_views=projective_conflict_info.get("views"),
            camera_cull_removed_ratio=camera_info.get("removed_ratio"),
            camera_cull_fallback_reason=camera_info.get("fallback_reason"),
            camera_cull_views=camera_info.get("views"),
            camera_cull_near_clip=camera_info.get("near_clip"),
            depth_cull_removed_ratio=depth_info.get("removed_ratio"),
            depth_cull_fallback_reason=depth_info.get("fallback_reason"),
            occluder_removed_ratio=occluder_info.get("removed_ratio"),
            occluder_fallback_reason=occluder_info.get("fallback_reason"),
            occluder_projected_edge_quantiles=occluder_info.get("edge_quantiles"),
            occluder_projected_area_quantiles=occluder_info.get("area_quantiles"),
        )
        self._sync_cuda()
        phase_times["trim"] = time.perf_counter() - t_trim_start

        self._sync_cuda()
        t_normal_start = time.perf_counter()
        face_normals, face_valid = self._normalize_vectors(
            torch.cross(
                tri_vertices[:, 1] - tri_vertices[:, 0],
                tri_vertices[:, 2] - tri_vertices[:, 0],
                dim=-1,
            )
        )
        if tri_normals is not None:
            tri_normals, tri_normal_valid = self._normalize_vectors(tri_normals)
            flip_mask = face_valid & tri_normal_valid & ((face_normals * tri_normals).sum(dim=-1) < 0)
            if flip_mask.any():
                tri_vertices = tri_vertices.clone()
                tmp = tri_vertices[flip_mask, 1].clone()
                tri_vertices[flip_mask, 1] = tri_vertices[flip_mask, 2]
                tri_vertices[flip_mask, 2] = tmp
                face_normals, face_valid = self._normalize_vectors(
                    torch.cross(
                        tri_vertices[:, 1] - tri_vertices[:, 0],
                        tri_vertices[:, 2] - tri_vertices[:, 0],
                        dim=-1,
                    )
                )

        N = tri_vertices.shape[0]
        all_verts = tri_vertices.reshape(-1, 3)  # [N*3, 3]
        all_normals = face_normals.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)

        if self.cfg.direct_skip_dedup:
            SH_C0 = 0.28209479177387814
            tri_colors = (tri_features[:, :3].float() * SH_C0 + 0.5).clamp(0, 1)
            all_colors = tri_colors.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
            faces = torch.arange(N * 3, device=all_verts.device, dtype=torch.int32).reshape(-1, 3)
            vertex_colors = all_colors
            unique_verts = all_verts
            vertex_normals = all_normals
            self._sync_cuda()
            phase_times["normal"] = time.perf_counter() - t_normal_start
        else:
            SH_C0 = 0.28209479177387814
            tri_colors = (tri_features[:, :3].float() * SH_C0 + 0.5).clamp(0, 1)
            all_colors = tri_colors.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
            self._sync_cuda()
            phase_times["normal"] = time.perf_counter() - t_normal_start
            self._sync_cuda()
            t_dedup_start = time.perf_counter()
            unique_verts, faces, vertex_colors, vertex_normals = self._quantized_unique_vertices(
                all_verts, all_colors, all_normals, N
            )
            self._sync_cuda()
            phase_times["dedup"] = time.perf_counter() - t_dedup_start

        self._sync_cuda()
        t_cpu_copy_start = time.perf_counter()
        verts_np = unique_verts.cpu().numpy().astype(np.float32)
        faces_np = faces.cpu().to(torch.int32).numpy()
        colors_np = (vertex_colors.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        normals_np = vertex_normals.cpu().numpy().astype(np.float32)
        self._sync_cuda()
        phase_times["cpu_copy"] = time.perf_counter() - t_cpu_copy_start

        return verts_np, faces_np, colors_np, normals_np, stats, phase_times

    @staticmethod
    def _pack_ply_bytes(
        verts: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray,
    ) -> bytes:
        """
        Pack mesh data into a binary PLY byte buffer (no disk I/O).
        Vertex record: xyz(3xfloat32) + nxyz(3xfloat32) + rgb(3xuint8) = 27 bytes each.
        Face record:   count(uint8=3) + 3xint32 = 13 bytes each.
        """
        n_verts = len(verts)
        n_faces = len(faces)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n_verts}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            f"element face {n_faces}\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
        ).encode("ascii")

        verts_c = np.ascontiguousarray(verts, dtype="<f4")  # [N, 3]
        normals_c = np.ascontiguousarray(normals, dtype="<f4")  # [N, 3]
        colors_c = np.ascontiguousarray(colors, dtype=np.uint8)  # [N, 3]
        vert_buf = np.empty(
            n_verts,
            dtype=np.dtype([("v", "<f4", 3), ("n", "<f4", 3), ("c", "u1", 3)]),
        )
        vert_buf["v"] = verts_c
        vert_buf["n"] = normals_c
        vert_buf["c"] = colors_c

        faces_i32 = np.ascontiguousarray(faces, dtype="<i4")  # [M, 3]
        face_buf = np.empty(n_faces, dtype=np.dtype([("n", "u1"), ("f", "<i4", 3)]))
        face_buf["n"] = 3
        face_buf["f"] = faces_i32

        return header + vert_buf.tobytes() + face_buf.tobytes()

    @staticmethod
    def _write_binary_ply(
        path: Path,
        verts: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray,
    ) -> None:
        """Write binary PLY synchronously."""
        data = TsdfGs2d._pack_ply_bytes(verts, faces, colors, normals)
        with open(path, "wb") as f:
            f.write(data)

    @staticmethod
    def _write_binary_ply_async(
        path: Path,
        verts: np.ndarray,
        faces: np.ndarray,
        colors: np.ndarray,
        normals: np.ndarray,
    ) -> threading.Thread:
        """
        Pack PLY in the current thread, then flush to disk in a background
        thread so the main pipeline can continue without waiting for I/O.
        Returns the thread handle (caller may join if needed).
        """
        data = TsdfGs2d._pack_ply_bytes(verts, faces, colors, normals)
        t = threading.Thread(target=lambda: open(path, "wb").write(data), daemon=True)
        t.start()
        return t

    def _write_direct_surface_proxy_frame_meshes(self, output_dir: Path, basename: str) -> None:
        frame_meshes = getattr(self, "_direct_surface_proxy_frame_meshes", [])
        if not frame_meshes:
            return
        for frame_index, verts, faces, colors, normals in frame_meshes:
            frame_path = output_dir / f"{basename}_frame_{int(frame_index):06d}.ply"
            self._write_binary_ply(frame_path, verts, faces, colors, normals)

    def _write_direct_visibility_frame_meshes(
        self,
        output_dir: Path,
        basename: str,
        primitives: Union[Gaussians, Triangles],
        batch: BatchedExample,
        target_visibility_mask: torch.Tensor | None,
        target_color: torch.Tensor | None,
        target_depth: torch.Tensor | None,
        target_opacity: torch.Tensor | None,
        *,
        global_step: int,
        opacity_temp_initial: float,
        opacity_temp_final: float,
        opacity_temp_warmup_steps: int,
    ) -> None:
        if not bool(self.cfg.direct_write_frame_meshes):
            return
        if self.cfg.direct_surface_proxy:
            return
        if target_visibility_mask is None:
            print("[Mesh][direct][frames] skip: missing target visibility mask")
            return
        if not isinstance(primitives, Triangles):
            print("[Mesh][direct][frames] skip: direct frame meshes require triangle primitives")
            return

        visibility = target_visibility_mask.detach()
        if visibility.dim() == 2:
            visibility = visibility.unsqueeze(0)
        if visibility.dim() != 3:
            print(f"[Mesh][direct][frames] skip: invalid visibility shape {tuple(visibility.shape)}")
            return
        num_frames = int(visibility.shape[1])
        if num_frames == 0:
            return

        target_indices_value = batch["target"].get("index") if "target" in batch else None
        if torch.is_tensor(target_indices_value):
            target_indices = [
                int(value)
                for value in target_indices_value[0].detach().cpu().flatten().tolist()
            ]
        else:
            target_indices = list(range(num_frames))
        if len(target_indices) < num_frames:
            target_indices.extend(range(len(target_indices), num_frames))

        old_target_visibility_cull = bool(self.cfg.direct_target_visibility_cull)
        old_visibility_min_target_views = int(self.cfg.direct_visibility_min_target_views)
        old_visibility_max_removed_ratio = float(self.cfg.direct_visibility_max_removed_ratio)

        self.cfg.direct_target_visibility_cull = True
        self.cfg.direct_visibility_min_target_views = 1
        self.cfg.direct_visibility_max_removed_ratio = max(old_visibility_max_removed_ratio, 0.99)

        written = 0
        try:
            for frame_slot in range(num_frames):
                frame_index = int(target_indices[frame_slot])
                frame_visibility = visibility[:, frame_slot : frame_slot + 1]
                frame_depth = (
                    None
                    if target_depth is None or target_depth.dim() < 2
                    else target_depth[:, frame_slot : frame_slot + 1]
                )
                frame_opacity = (
                    None
                    if target_opacity is None or target_opacity.dim() < 2
                    else target_opacity[:, frame_slot : frame_slot + 1]
                )
                try:
                    verts, faces, colors, normals, _build_stats, _phase_times = self._build_direct_mesh_from_triangles(
                        primitives,
                        batch=batch,
                        target_visibility_mask=frame_visibility,
                        target_depth=frame_depth,
                        target_opacity=frame_opacity,
                        global_step=global_step,
                        opacity_temp_initial=opacity_temp_initial,
                        opacity_temp_final=opacity_temp_final,
                        opacity_temp_warmup_steps=opacity_temp_warmup_steps,
                    )
                except Exception as exc:
                    print(f"[Mesh][direct][frames] skip frame {frame_index:06d}: {type(exc).__name__}: {exc}")
                    continue

                if self.cfg.direct_frame_color_mode == "target_render":
                    colors = self._sample_direct_frame_vertex_colors(
                        verts,
                        colors,
                        batch,
                        target_color,
                        frame_slot,
                    )

                frame_path = output_dir / f"{basename}_frame_{frame_index:06d}.ply"
                ply_bytes = self._pack_ply_bytes(verts, faces, colors, normals)
                with open(frame_path, "wb") as f:
                    f.write(ply_bytes)
                if self.cfg.direct_post_process:
                    post_frame_path = output_dir / f"{basename}_post_frame_{frame_index:06d}.ply"
                    with open(post_frame_path, "wb") as f:
                        f.write(ply_bytes)
                written += 1
        finally:
            self.cfg.direct_target_visibility_cull = old_target_visibility_cull
            self.cfg.direct_visibility_min_target_views = old_visibility_min_target_views
            self.cfg.direct_visibility_max_removed_ratio = old_visibility_max_removed_ratio

        print(f"[Mesh][direct][frames] wrote {written}/{num_frames} target-frame meshes")

    @staticmethod
    def _sample_direct_frame_vertex_colors(
        verts: np.ndarray,
        fallback_colors: np.ndarray,
        batch: BatchedExample,
        target_color: torch.Tensor | None,
        frame_slot: int,
    ) -> np.ndarray:
        if target_color is None:
            return fallback_colors
        if target_color.dim() != 5 or target_color.shape[0] < 1:
            return fallback_colors
        if frame_slot >= int(target_color.shape[1]):
            return fallback_colors

        image = target_color[0, frame_slot].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        image_h, image_w = int(image.shape[0]), int(image.shape[1])
        if image_h <= 0 or image_w <= 0 or len(verts) == 0:
            return fallback_colors

        extrinsics = batch["target"]["extrinsics"][0, frame_slot].detach().float().cpu().numpy()
        intrinsics = batch["target"]["intrinsics"][0, frame_slot].detach().float().cpu().numpy()
        near = float(batch["target"]["near"][0, frame_slot].detach().float().cpu().item())
        eps = max(near, 1.0e-6)

        verts_f64 = np.asarray(verts, dtype=np.float64)
        ones = np.ones((verts_f64.shape[0], 1), dtype=np.float64)
        verts_h = np.concatenate([verts_f64, ones], axis=1)
        w2c = np.linalg.inv(extrinsics.astype(np.float64))
        cam = verts_h @ w2c.T
        z = cam[:, 2]
        valid_z = np.isfinite(z) & (z > eps)
        z_safe = np.maximum(z, eps)

        fx = float(intrinsics[0, 0]) * float(image_w)
        fy = float(intrinsics[1, 1]) * float(image_h)
        cx = float(intrinsics[0, 2]) * float(image_w)
        cy = float(intrinsics[1, 2]) * float(image_h)
        px = fx * (cam[:, 0] / z_safe) + cx
        py = fy * (cam[:, 1] / z_safe) + cy
        inside = (
            valid_z
            & np.isfinite(px)
            & np.isfinite(py)
            & (px >= 0.0)
            & (px < float(image_w))
            & (py >= 0.0)
            & (py < float(image_h))
        )
        if not np.any(inside):
            return fallback_colors

        sampled = np.array(fallback_colors, copy=True)
        x_idx = np.rint(px[inside]).astype(np.int64).clip(0, image_w - 1)
        y_idx = np.rint(py[inside]).astype(np.int64).clip(0, image_h - 1)
        sampled[inside] = (image[y_idx, x_idx] * 255.0).clip(0, 255).astype(np.uint8)
        return sampled

    def _quantized_unique_vertices(
        self,
        all_verts: torch.Tensor,
        all_colors: torch.Tensor,
        all_normals: torch.Tensor,
        N: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Deduplicate vertices by quantizing to int64 keys and running 1D unique
        (much faster than torch.unique on 2D float tensors).
        Normal octants are included in the key so opposite-facing triangles do
        not get merged into the same shared vertex.
        """
        PRECISION = 1e5
        vi = (all_verts * PRECISION).to(torch.int64)
        vmin = vi.min(dim=0).values
        vi = vi - vmin
        span_y = vi[:, 1].max().item() + 1
        span_z = vi[:, 2].max().item() + 1
        normal_octant = (
            (all_normals[:, 0] >= 0).to(torch.int64) * 4
            + (all_normals[:, 1] >= 0).to(torch.int64) * 2
            + (all_normals[:, 2] >= 0).to(torch.int64)
        )
        keys = (
            vi[:, 0] * (span_y * span_z) + vi[:, 1] * span_z + vi[:, 2]
        ) * 8 + normal_octant

        _, inv_idx = torch.unique(keys, return_inverse=True)
        faces = inv_idx.reshape(-1, 3)

        num_unique = inv_idx.max().item() + 1
        vertex_positions = torch.zeros(num_unique, 3, device=all_verts.device)
        vertex_colors = torch.zeros(num_unique, 3, device=all_verts.device)
        vertex_normals = torch.zeros(num_unique, 3, device=all_verts.device)
        vertex_counts = torch.zeros(num_unique, 1, device=all_verts.device)

        idx_expand3 = inv_idx.unsqueeze(1).expand(-1, 3)
        vertex_positions.scatter_add_(0, idx_expand3, all_verts)
        vertex_colors.scatter_add_(0, idx_expand3, all_colors)
        vertex_normals.scatter_add_(0, idx_expand3, all_normals)
        vertex_counts.scatter_add_(
            0,
            inv_idx.unsqueeze(1),
            torch.ones(N * 3, 1, device=all_verts.device),
        )
        counts_clamped = vertex_counts.clamp(min=1)
        vertex_positions = vertex_positions / counts_clamped
        vertex_colors = vertex_colors / counts_clamped
        vertex_normals = vertex_normals / counts_clamped
        vertex_normals, _ = self._normalize_vectors(vertex_normals)

        return vertex_positions, faces, vertex_colors, vertex_normals

    def _write_off(self, path: Path, mesh: o3d.geometry.TriangleMesh) -> None:
        """
        Write mesh as plain OFF file.
        """
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        with open(path, "w", encoding="utf-8") as f:
            f.write("OFF\n")
            f.write(f"{len(vertices)} {len(triangles)} 0\n")
            for v in vertices:
                f.write(f"{float(v[0])} {float(v[1])} {float(v[2])}\n")
            for tri in triangles:
                f.write(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}\n")

    def _save_mesh_with_format(
        self,
        mesh: o3d.geometry.TriangleMesh,
        output_dir: Path,
        basename: str,
        export_format: Literal["ply", "off", "both"],
    ) -> dict[str, Path]:
        """
        Save mesh to requested format(s). Return generated file paths.
        """
        outputs: dict[str, Path] = {}
        if not self._mesh_has_geometry(mesh):
            print(f"[Mesh][save] skip writing {basename} because mesh is empty.")
            return outputs
        if export_format in ("ply", "both"):
            ply_path = output_dir / f"{basename}.ply"
            o3d.io.write_triangle_mesh(str(ply_path), mesh)
            outputs["ply"] = ply_path
        if export_format in ("off", "both"):
            off_path = output_dir / f"{basename}.off"
            self._write_off(off_path, mesh)
            outputs["off"] = off_path
        return outputs

    @staticmethod
    def _json_ready_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.ndim > 0 and tensor.shape[0] == 1:
                tensor = tensor[0]
            if tensor.ndim == 0:
                return tensor.item()
            return tensor.tolist()
        if isinstance(value, np.ndarray):
            array = value
            if array.ndim > 0 and array.shape[0] == 1:
                array = array[0]
            return array.tolist()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {
                key: TsdfGs2d._json_ready_value(item)
                for key, item in value.items()
            }
        if isinstance(value, tuple):
            return [TsdfGs2d._json_ready_value(item) for item in value]
        if isinstance(value, list):
            if len(value) == 1 and not isinstance(value[0], (list, tuple, dict)):
                return TsdfGs2d._json_ready_value(value[0])
            return [TsdfGs2d._json_ready_value(item) for item in value]
        return value

    def _build_mesh_space_payload(self, batch: BatchedExample) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mesh_space": "unknown",
            "world_space": "raw_world",
            "use_train_view": bool(self.cfg.use_train_view),
            "use_val_view": bool(self.cfg.use_val_view),
            "export_mode": str(self.cfg.export_mode),
            "export_format": str(self.cfg.export_format),
            "tsdf_context_color_source": str(self.cfg.tsdf_context_color_source),
            "tsdf_target_color_source": str(self.cfg.tsdf_target_color_source),
        }
        scenes = batch.get("scene")
        if scenes:
            payload["scene"] = str(scenes[0])

        scene_metadata = batch.get("scene_metadata")
        if scene_metadata is not None:
            payload.update(
                {
                    key: self._json_ready_value(value)
                    for key, value in scene_metadata.items()
                }
            )
        return payload

    def _write_mesh_space_metadata(
        self,
        output_dir: Path,
        batch: BatchedExample,
    ) -> Path:
        metadata_path = output_dir / "mesh_space.json"
        payload = self._build_mesh_space_payload(batch)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return metadata_path

    def main(
        self,
        prediction_train: Optional[DecoderOutput],
        prediction_val: Optional[DecoderOutput],
        batch: BatchedExample,
        gaussians: Union[Gaussians, Triangles],
        output: str,
        gt_pointcloud_root: Path | None = None,
        encoder_time: Optional[float] = None,
        decoder_time: Optional[float] = None,
        global_step: int = 0,
        opacity_temp_initial: float = 1.0,
        opacity_temp_final: float = 1.0,
        opacity_temp_warmup_steps: int = 5000,
    ) -> MeshExportResult:
        """
        Main entry for mesh export.
        When export_mode == 'direct', takes a fast path that only needs
        `gaussians` (no decoder outputs) and writes a single binary PLY.
        """
        output_dir = Path(output)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        space_metadata_path = self._write_mesh_space_metadata(output_dir, batch)

        export_mode = self.cfg.export_mode
        export_format = self.cfg.export_format
        _ = gt_pointcloud_root
        direct_timing: dict[str, float | int] | None = None
        tsdf_timing: dict[str, float | int] | None = None

        # ---- fast path: direct only ----
        if export_mode == "direct":
            num_bench = 10 if self.cfg.direct_bench_10 else 1
            phase_time_totals = self._empty_direct_phase_times()

            self._sync_cuda()
            t0 = time.perf_counter()
            for _i in range(num_bench):
                if self.cfg.direct_surface_proxy:
                    verts, faces, colors, normals, build_stats, phase_times = self._build_direct_surface_proxy_mesh(
                        prediction_val,
                        batch,
                    )
                else:
                    verts, faces, colors, normals, build_stats, phase_times = self._build_direct_mesh_from_triangles(
                        gaussians,
                        batch=batch,
                        target_visibility_mask=(
                            None if prediction_val is None else prediction_val.triangle_visibility_mask
                        ),
                        target_depth=None if prediction_val is None else prediction_val.depth,
                        target_opacity=None if prediction_val is None else prediction_val.opacity,
                        global_step=global_step,
                        opacity_temp_initial=opacity_temp_initial,
                        opacity_temp_final=opacity_temp_final,
                        opacity_temp_warmup_steps=opacity_temp_warmup_steps,
                    )
                self._accumulate_direct_phase_times(phase_time_totals, phase_times)
            self._sync_cuda()
            t1 = time.perf_counter()
            for _i in range(num_bench):
                ply_bytes = self._pack_ply_bytes(verts, faces, colors, normals)
            t2 = time.perf_counter()
            ply_path = output_dir / "DIRECT_triangle_mesh.ply"
            for _i in range(num_bench):
                with open(ply_path, "wb") as _f:
                    _f.write(ply_bytes)
            if self.cfg.direct_surface_proxy and self.cfg.direct_surface_proxy_write_frame_meshes:
                self._write_direct_surface_proxy_frame_meshes(output_dir, "DIRECT_triangle_mesh")
            if self.cfg.direct_write_frame_meshes:
                self._write_direct_visibility_frame_meshes(
                    output_dir,
                    "DIRECT_triangle_mesh",
                    gaussians,
                    batch,
                    None if prediction_val is None else prediction_val.triangle_visibility_mask,
                    None if prediction_val is None else prediction_val.color,
                    None if prediction_val is None else prediction_val.depth,
                    None if prediction_val is None else prediction_val.opacity,
                    global_step=global_step,
                    opacity_temp_initial=opacity_temp_initial,
                    opacity_temp_final=opacity_temp_final,
                    opacity_temp_warmup_steps=opacity_temp_warmup_steps,
                )
            t3 = time.perf_counter()

            phase_times_avg = self._average_direct_phase_times(phase_time_totals, num_bench)
            pack_time = (t2 - t1) / num_bench
            disk_time = (t3 - t2) / num_bench
            direct_timing = self._build_direct_timing_record(
                phase_times_avg,
                pack_time=pack_time,
                disk_time=disk_time,
                encoder_time=encoder_time,
                num_triangles=len(faces),
                num_vertices=len(verts),
                num_faces=len(faces),
            )
            bench_tag = f"[bench x{num_bench}]" if num_bench > 1 else ""
            self._print_direct_timing(
                direct_timing,
                ply_num_bytes=len(ply_bytes),
                show_encoder=encoder_time is not None,
                bench_tag=bench_tag,
            )
            self._log_direct_build_stats(build_stats)
            self._write_direct_trim_stats(output_dir, "DIRECT_triangle_mesh", build_stats)
            print(f"[Mesh][direct] saved {ply_path}")
            output_path = ply_path
            if self.cfg.direct_post_process:
                direct_mesh = o3d.geometry.TriangleMesh()
                direct_mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
                direct_mesh.triangles = o3d.utility.Vector3iVector(faces)
                direct_mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
                direct_mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
                direct_mesh_post, direct_post_stats = self.post_process_mesh(
                    direct_mesh,
                    self.cfg.num_cluster,
                    mode="direct",
                )
                if len(direct_mesh_post.vertices) > 0 and len(direct_mesh_post.triangles) > 0:
                    direct_mesh_post.compute_triangle_normals()
                    direct_mesh_post.compute_vertex_normals()
                post_outputs = self._save_mesh_with_format(
                    direct_mesh_post,
                    output_dir,
                    "DIRECT_triangle_mesh_post",
                    export_format,
                )
                if direct_post_stats is not None and self.cfg.tsdf_post_write_stats:
                    self._write_tsdf_post_stats(output_dir, "DIRECT_triangle_mesh", direct_post_stats)
                output_path = post_outputs.get("ply") or post_outputs.get("off") or ply_path
            return MeshExportResult(
                output_path=str(output_path),
                space_metadata_path=str(space_metadata_path),
                direct_output_path=str(ply_path),
                direct_timing=direct_timing,
            )

        # ---- original path: tsdf / both ----
        need_tsdf = export_mode in ("tsdf", "both")

        self.extrinsics, self.intrinsics, self.projections = [], [], []
        self.rgbmaps, self.depthmaps, self.gt_alpha_masks = [], [], []
        self.camera_debug_records: list[dict[str, object]] = []

        if need_tsdf:
            if self.cfg.use_train_view:
                if prediction_train is None or prediction_train.depth is None:
                    print("[Mesh][tsdf] Skip train-view TSDF inputs: no depth available.")
                else:
                    if isinstance(gaussians, Gaussians) and self.cfg.gs_depth_backend == "surface_approx":
                        depths, alpha_masks = self._build_surface_like_depth_from_gs(prediction_train)
                    else:
                        depths = prediction_train.depth[0].clone().detach()
                        if getattr(prediction_train, "opacity", None) is not None:
                            alpha_masks = prediction_train.opacity[0].clone().detach()
                        elif "mask" in batch["context"]:
                            alpha_masks = batch["context"]["mask"][0, :, 0].clone().detach()
                        else:
                            alpha_masks = torch.ones_like(depths)
                    if self.cfg.tsdf_context_color_source == "prediction":
                        images = prediction_train.color[0].clone().detach()
                    elif self.cfg.tsdf_context_color_source == "input":
                        images = batch["context"]["image"][0].clone().detach()
                    else:
                        raise ValueError(
                            "mesh.tsdf_gs2d.tsdf_context_color_source must be 'prediction' or 'input', "
                            f"got {self.cfg.tsdf_context_color_source!r}."
                        )
                    if images.shape[0] != depths.shape[0] or images.shape[-2:] != depths.shape[-2:]:
                        raise ValueError(
                            "Context TSDF color/depth shape mismatch: "
                            f"color={tuple(images.shape)} depth={tuple(depths.shape)}"
                        )
                    projection_matrix = (
                        prediction_train.projection_matrix[0].clone().detach()
                        if prediction_train.projection_matrix is not None
                        else None
                    )

                    v = depths.shape[0]
                    for i in range(v):
                        intrinsics_i = batch["context"]["intrinsics"][0, i].double().cpu().numpy()
                        projection_i = (
                            projection_matrix[i].double().cpu().numpy()
                            if projection_matrix is not None
                            else None
                        )
                        self.extrinsics.append(
                            batch["context"]["extrinsics"][0, i].double().inverse().cpu().numpy()
                        )
                        self.intrinsics.append(intrinsics_i)
                        self.projections.append(projection_i)
                        self.rgbmaps.append(images[i])
                        self.depthmaps.append(depths[i])
                        self.gt_alpha_masks.append(alpha_masks[i])
                        if self.cfg.debug_camera_dump:
                            h, w = images[i].shape[1:]
                            batch_intrinsic = self.build_o3d_intrinsic_from_normalized_k(intrinsics_i, w, h)
                            record = {
                                "split": "context",
                                "view_slot": i,
                                "frame_index": int(batch["context"]["index"][0, i].item()),
                                "image_height": h,
                                "image_width": w,
                                "normalized_intrinsics": {
                                    "fx": float(intrinsics_i[0, 0]),
                                    "fy": float(intrinsics_i[1, 1]),
                                    "cx": float(intrinsics_i[0, 2]),
                                    "cy": float(intrinsics_i[1, 2]),
                                },
                                "depth_backend": self.cfg.gs_depth_backend if isinstance(gaussians, Gaussians) else "triangle",
                                "batch_intrinsic_px": self._intrinsic_debug_dict(batch_intrinsic),
                                "depth": self._tensor_debug_dict(depths[i]),
                                "alpha": self._tensor_debug_dict(alpha_masks[i]),
                            }
                            if isinstance(gaussians, Gaussians):
                                raw_depth = prediction_train.depth[0].clone().detach()
                                record["raw_depth"] = self._tensor_debug_dict(raw_depth[i])
                            if projection_i is not None:
                                record["projection_intrinsic_px"] = self._intrinsic_debug_dict(
                                    self.build_o3d_intrinsic_from_projection(projection_i, w, h)
                                )
                            self.camera_debug_records.append(record)

            if self.cfg.use_val_view:
                if prediction_val is None or prediction_val.depth is None:
                    print("[Mesh][tsdf] Skip val-view TSDF inputs: no depth available.")
                else:
                    if isinstance(gaussians, Gaussians) and self.cfg.gs_depth_backend == "surface_approx":
                        depths, alpha_masks = self._build_surface_like_depth_from_gs(prediction_val)
                    else:
                        depths = prediction_val.depth[0].clone().detach()
                        if getattr(prediction_val, "opacity", None) is not None:
                            alpha_masks = prediction_val.opacity[0].clone().detach()
                        elif "mask" in batch["target"]:
                            alpha_masks = batch["target"]["mask"][0, :, 0].clone().detach()
                        else:
                            alpha_masks = torch.ones_like(depths)
                    if self.cfg.tsdf_target_color_source == "prediction":
                        images = prediction_val.color[0].clone().detach()
                    elif self.cfg.tsdf_target_color_source == "input":
                        images = batch["target"]["image"][0].clone().detach()
                    else:
                        raise ValueError(
                            "mesh.tsdf_gs2d.tsdf_target_color_source must be 'prediction' or 'input', "
                            f"got {self.cfg.tsdf_target_color_source!r}."
                        )
                    if images.shape[0] != depths.shape[0] or images.shape[-2:] != depths.shape[-2:]:
                        raise ValueError(
                            "Target TSDF color/depth shape mismatch: "
                            f"color={tuple(images.shape)} depth={tuple(depths.shape)}"
                        )
                    projection_matrix = (
                        prediction_val.projection_matrix[0].clone().detach()
                        if prediction_val.projection_matrix is not None
                        else None
                    )

                    v = depths.shape[0]
                    for i in range(v):
                        intrinsics_i = batch["target"]["intrinsics"][0, i].double().cpu().numpy()
                        projection_i = (
                            projection_matrix[i].double().cpu().numpy()
                            if projection_matrix is not None
                            else None
                        )
                        self.extrinsics.append(
                            batch["target"]["extrinsics"][0, i].double().inverse().cpu().numpy()
                        )
                        self.intrinsics.append(intrinsics_i)
                        self.projections.append(projection_i)
                        self.rgbmaps.append(images[i])
                        self.depthmaps.append(depths[i])
                        self.gt_alpha_masks.append(alpha_masks[i])
                        if self.cfg.debug_camera_dump:
                            h, w = images[i].shape[1:]
                            batch_intrinsic = self.build_o3d_intrinsic_from_normalized_k(intrinsics_i, w, h)
                            record = {
                                "split": "target",
                                "view_slot": i,
                                "frame_index": int(batch["target"]["index"][0, i].item()),
                                "image_height": h,
                                "image_width": w,
                                "normalized_intrinsics": {
                                    "fx": float(intrinsics_i[0, 0]),
                                    "fy": float(intrinsics_i[1, 1]),
                                    "cx": float(intrinsics_i[0, 2]),
                                    "cy": float(intrinsics_i[1, 2]),
                                },
                                "depth_backend": self.cfg.gs_depth_backend if isinstance(gaussians, Gaussians) else "triangle",
                                "batch_intrinsic_px": self._intrinsic_debug_dict(batch_intrinsic),
                                "depth": self._tensor_debug_dict(depths[i]),
                                "alpha": self._tensor_debug_dict(alpha_masks[i]),
                            }
                            if isinstance(gaussians, Gaussians):
                                raw_depth = prediction_val.depth[0].clone().detach()
                                record["raw_depth"] = self._tensor_debug_dict(raw_depth[i])
                            if projection_i is not None:
                                record["projection_intrinsic_px"] = self._intrinsic_debug_dict(
                                    self.build_o3d_intrinsic_from_projection(projection_i, w, h)
                                )
                            self.camera_debug_records.append(record)

            self._write_camera_debug_dump(output_dir)

            if len(self.depthmaps) == 0:
                if export_mode == "tsdf":
                    raise ValueError(
                        "TSDF export requires depth predictions, but neither train nor val depth is available. "
                        "Use export_mode=direct or enable depth output in decoder/test config."
                    )
                print("[Mesh][tsdf] No depth available; skip TSDF export and keep direct export.")
                need_tsdf = False

        output_paths: dict[str, Path] = {}
        fallback_path: Optional[Path] = None
        direct_output_path: Optional[Path] = None
        tsdf_output_path: Optional[Path] = None

        if export_mode in ("direct", "both"):
            if self.cfg.direct_surface_proxy:
                verts, faces, colors, normals, build_stats, phase_times = self._build_direct_surface_proxy_mesh(
                    prediction_val,
                    batch,
                )
            else:
                verts, faces, colors, normals, build_stats, phase_times = self._build_direct_mesh_from_triangles(
                    gaussians,
                    batch=batch,
                    target_visibility_mask=(
                        None if prediction_val is None else prediction_val.triangle_visibility_mask
                    ),
                    target_depth=None if prediction_val is None else prediction_val.depth,
                    target_opacity=None if prediction_val is None else prediction_val.opacity,
                    global_step=global_step,
                    opacity_temp_initial=opacity_temp_initial,
                    opacity_temp_final=opacity_temp_final,
                    opacity_temp_warmup_steps=opacity_temp_warmup_steps,
                )
            direct_base = "DIRECT_triangle_mesh"
            ply_path = output_dir / f"{direct_base}.ply"

            t_pack_start = time.perf_counter()
            ply_bytes = self._pack_ply_bytes(verts, faces, colors, normals)
            t_pack_end = time.perf_counter()
            with open(ply_path, "wb") as f:
                f.write(ply_bytes)
            if self.cfg.direct_surface_proxy and self.cfg.direct_surface_proxy_write_frame_meshes:
                self._write_direct_surface_proxy_frame_meshes(output_dir, direct_base)
            if self.cfg.direct_write_frame_meshes:
                self._write_direct_visibility_frame_meshes(
                    output_dir,
                    direct_base,
                    gaussians,
                    batch,
                    None if prediction_val is None else prediction_val.triangle_visibility_mask,
                    None if prediction_val is None else prediction_val.color,
                    None if prediction_val is None else prediction_val.depth,
                    None if prediction_val is None else prediction_val.opacity,
                    global_step=global_step,
                    opacity_temp_initial=opacity_temp_initial,
                    opacity_temp_final=opacity_temp_final,
                    opacity_temp_warmup_steps=opacity_temp_warmup_steps,
                )
            t_disk_end = time.perf_counter()

            direct_timing = self._build_direct_timing_record(
                phase_times,
                pack_time=t_pack_end - t_pack_start,
                disk_time=t_disk_end - t_pack_end,
                encoder_time=encoder_time,
                num_triangles=len(faces),
                num_vertices=len(verts),
                num_faces=len(faces),
            )
            output_paths["ply"] = ply_path
            direct_output_path = ply_path
            self._print_direct_timing(
                direct_timing,
                ply_num_bytes=len(ply_bytes),
                show_encoder=encoder_time is not None,
            )
            self._log_direct_build_stats(build_stats)
            self._write_direct_trim_stats(output_dir, direct_base, build_stats)

            if self.cfg.direct_post_process:
                direct_mesh = o3d.geometry.TriangleMesh()
                direct_mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
                direct_mesh.triangles = o3d.utility.Vector3iVector(faces)
                direct_mesh.vertex_colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
                direct_mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
                direct_mesh_post, direct_post_stats = self.post_process_mesh(
                    direct_mesh,
                    self.cfg.num_cluster,
                    mode="direct",
                )
                if len(direct_mesh_post.vertices) > 0 and len(direct_mesh_post.triangles) > 0:
                    direct_mesh_post.compute_triangle_normals()
                    direct_mesh_post.compute_vertex_normals()
                output_paths.update(
                    self._save_mesh_with_format(
                        direct_mesh_post, output_dir, f"{direct_base}_post", export_format
                    )
                )
                if direct_post_stats is not None and self.cfg.tsdf_post_write_stats:
                    self._write_tsdf_post_stats(output_dir, direct_base, direct_post_stats)
            fallback_path = output_paths.get("ply") or output_paths.get("off")

        if need_tsdf:
            t_tsdf_start = time.perf_counter()
            if self.cfg.bounded:
                mesh = self.extract_mesh_bounded()
            else:
                raise NotImplementedError("Unbounded mode is not implemented")
            t_tsdf_fuse = time.perf_counter()
            output_paths.update(self._save_mesh_with_format(mesh, output_dir, "TSDF_2dgs_mesh", export_format))
            t_tsdf_save = time.perf_counter()
            mesh_post, mesh_post_stats = self.post_process_mesh(
                mesh,
                self.cfg.num_cluster,
                mode="tsdf",
            )
            t_tsdf_post = time.perf_counter()
            mesh_to_save = mesh_post
            if not self._mesh_has_geometry(mesh_post) and self._mesh_has_geometry(mesh):
                print("[Mesh][tsdf] post-process produced empty mesh; falling back to raw TSDF mesh.")
                mesh_to_save = mesh
            elif not self._mesh_has_geometry(mesh_post):
                print("[Mesh][tsdf] post-process produced empty mesh and raw TSDF mesh is also empty.")
            output_paths.update(
                self._save_mesh_with_format(mesh_to_save, output_dir, "TSDF_2dgs_post_mesh", export_format)
            )
            if mesh_post_stats is not None and self.cfg.tsdf_post_write_stats:
                self._write_tsdf_post_stats(output_dir, "TSDF_2dgs_post_mesh", mesh_post_stats)
            t_tsdf_end = time.perf_counter()

            fuse_t = t_tsdf_fuse - t_tsdf_start
            save_t = t_tsdf_save - t_tsdf_fuse
            post_t = t_tsdf_post - t_tsdf_save
            save_post_t = t_tsdf_end - t_tsdf_post
            tsdf_timing = self._build_tsdf_timing_record(
                fuse_time=fuse_t,
                save_raw_time=save_t,
                post_process_time=post_t,
                save_post_time=save_post_t,
                encoder_time=encoder_time,
                decoder_time=decoder_time,
                raw_vertices=len(mesh.vertices),
                raw_faces=len(mesh.triangles),
                post_vertices=len(mesh_to_save.vertices),
                post_faces=len(mesh_to_save.triangles),
            )
            self._print_tsdf_timing(
                tsdf_timing,
                show_encoder=encoder_time is not None,
                show_decoder=decoder_time is not None,
            )

            tsdf_output_path = (
                output_dir / "TSDF_2dgs_post_mesh.ply"
                if export_format in ("ply", "both")
                else output_dir / "TSDF_2dgs_post_mesh.off"
            )
            fallback_path = tsdf_output_path

        if fallback_path is None:
            raise ValueError(f"No mesh exported. export_mode={export_mode}, export_format={export_format}")
        return MeshExportResult(
            output_path=str(fallback_path),
            space_metadata_path=str(space_metadata_path),
            direct_output_path=None if direct_output_path is None else str(direct_output_path),
            direct_timing=direct_timing,
            tsdf_output_path=None if tsdf_output_path is None else str(tsdf_output_path),
            tsdf_timing=tsdf_timing if need_tsdf else None,
        )

    def estimate_bounding_sphere(self):
        def focus_point_fn(poses: np.ndarray) -> np.ndarray:
            """Calculate nearest point to all focal axes in poses."""
            directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
            m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
            mt_m = np.transpose(m, [0, 2, 1]) @ m
            focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
            return focus_pt

        torch.cuda.empty_cache()
        # self.extrinsics stores W2C; invert to get C2W for bounding sphere
        c2ws = np.linalg.inv(np.array(self.extrinsics))
        poses = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
        center = focus_point_fn(poses)
        self.radius = np.linalg.norm(c2ws[:, :3, 3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()

    def post_process_mesh(
        self,
        mesh,
        cluster_to_keep=1000,
        *,
        mode: Literal["tsdf", "direct"] = "tsdf",
    ) -> tuple[o3d.geometry.TriangleMesh, TsdfPostProcessStats | None]:
        """
        Post-process a mesh to filter out floaters and disconnected parts
        """
        import copy

        print("post processing the mesh to have {} clusters".format(cluster_to_keep))
        mesh_0 = copy.deepcopy(mesh)
        if not self._mesh_has_geometry(mesh_0):
            print("[Mesh][post] skip connected-component filtering because mesh is empty.")
            return mesh_0, None

        if mode == "tsdf":
            self._validate_tsdf_post_cfg()
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
            triangle_clusters, cluster_n_triangles, cluster_area = mesh_0.cluster_connected_triangles()

        triangle_clusters = np.asarray(triangle_clusters, dtype=np.int64)
        cluster_n_triangles = np.asarray(cluster_n_triangles, dtype=np.int64)
        cluster_area = np.asarray(cluster_area, dtype=np.float64)
        if len(cluster_n_triangles) == 0:
            print("[Mesh][post] skip connected-component filtering because no triangle clusters were found.")
            return mesh_0, None

        cluster_to_keep = max(int(cluster_to_keep), 1)
        legacy_keep_cluster_mask, legacy_threshold = self._build_tsdf_legacy_keep_mask(
            cluster_n_triangles,
            cluster_to_keep,
        )
        legacy_triangle_ratio = self._compute_retained_ratio(
            cluster_n_triangles[legacy_keep_cluster_mask],
            cluster_n_triangles,
        )
        legacy_area_ratio = self._compute_retained_ratio(
            cluster_area[legacy_keep_cluster_mask],
            cluster_area,
        )

        final_keep_cluster_mask = legacy_keep_cluster_mask
        strategy_used = "legacy_top_k"
        strategy_reason: str | None = None
        if mode == "tsdf" and self.cfg.tsdf_post_mode == "adaptive_legacy_guard":
            legacy_ok = (
                legacy_triangle_ratio >= float(self.cfg.tsdf_post_min_keep_triangle_ratio)
                and legacy_area_ratio >= float(self.cfg.tsdf_post_min_keep_area_ratio)
            )
            if legacy_ok:
                strategy_reason = "legacy_guard_passed"
            else:
                adaptive_keep_cluster_mask = self._build_tsdf_adaptive_keep_mask(
                    cluster_n_triangles,
                    cluster_area,
                )
                if adaptive_keep_cluster_mask.sum() > 0:
                    final_keep_cluster_mask = adaptive_keep_cluster_mask
                    strategy_used = "adaptive_expand"
                    strategy_reason = "legacy_guard_failed"
                else:
                    strategy_reason = "adaptive_empty_fallback"

        triangles_to_remove = ~final_keep_cluster_mask[triangle_clusters]
        mesh_0.remove_triangles_by_mask(triangles_to_remove)
        mesh_0.remove_unreferenced_vertices()
        mesh_0.remove_degenerate_triangles()
        final_triangle_ratio = self._compute_retained_ratio(
            cluster_n_triangles[final_keep_cluster_mask],
            cluster_n_triangles,
        )
        final_area_ratio = self._compute_retained_ratio(
            cluster_area[final_keep_cluster_mask],
            cluster_area,
        )
        if not self._mesh_has_geometry(mesh_0):
            print("[Mesh][post] filtered mesh became empty; falling back to raw mesh.")
            mesh_0 = copy.deepcopy(mesh)
            final_keep_cluster_mask = np.ones_like(cluster_n_triangles, dtype=bool)
            final_triangle_ratio = 1.0
            final_area_ratio = 1.0
            strategy_used = "raw_fallback"
            strategy_reason = "filtered_empty"

        stats = TsdfPostProcessStats(
            mode=mode,
            strategy_used=strategy_used,
            strategy_reason=strategy_reason,
            cluster_to_keep=cluster_to_keep,
            num_clusters=int(len(cluster_n_triangles)),
            raw_num_vertices=int(len(mesh.vertices)),
            raw_num_triangles=int(len(mesh.triangles)),
            raw_surface_area=float(cluster_area.sum()),
            min_cluster_triangles=int(self.cfg.tsdf_post_min_cluster_triangles),
            min_keep_triangle_ratio=float(self.cfg.tsdf_post_min_keep_triangle_ratio),
            min_keep_area_ratio=float(self.cfg.tsdf_post_min_keep_area_ratio),
            legacy_triangle_threshold=int(legacy_threshold),
            legacy_kept_clusters=int(legacy_keep_cluster_mask.sum()),
            legacy_retained_triangle_ratio=float(legacy_triangle_ratio),
            legacy_retained_area_ratio=float(legacy_area_ratio),
            final_kept_clusters=int(final_keep_cluster_mask.sum()),
            final_retained_triangle_ratio=float(final_triangle_ratio),
            final_retained_area_ratio=float(final_area_ratio),
            final_num_vertices=int(len(mesh_0.vertices)),
            final_num_triangles=int(len(mesh_0.triangles)),
        )
        print(
            f"[Mesh][post][{mode}] "
            f"strategy={strategy_used} reason={strategy_reason or 'n/a'} "
            f"legacy_keep={stats.legacy_kept_clusters}/{stats.num_clusters} "
            f"legacy_tri_ratio={stats.legacy_retained_triangle_ratio:.3f} "
            f"legacy_area_ratio={stats.legacy_retained_area_ratio:.3f} "
            f"final_keep={stats.final_kept_clusters}/{stats.num_clusters} "
            f"final_tri_ratio={stats.final_retained_triangle_ratio:.3f} "
            f"final_area_ratio={stats.final_retained_area_ratio:.3f}"
        )
        print("num vertices raw {}".format(len(mesh.vertices)))
        print("num vertices post {}".format(len(mesh_0.vertices)))
        return mesh_0, stats
