import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_EXPERIMENTS = {
    "direct": "trisplat_dl3dv_triangle_refiner_unet_10m_224x448",
    "tsdf": "trisplat_dl3dv_triangle_refiner_unet_10m_224x448",
    "both": "trisplat_dl3dv_triangle_refiner_unet_10m_224x448",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DL3DV meshes for TriSplat checkpoints.",
    )
    parser.add_argument("--export-mode", choices=("direct", "tsdf", "both"), required=True)
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument(
        "--dl3dv-root",
        default=None,
        help="Optional DL3DV dataset root override for dataset.dl3dv.roots/test_roots.",
    )
    parser.add_argument(
        "--gt-root",
        default=None,
        help="Optional GT point cloud directory containing <scene>.ply files.",
    )
    parser.add_argument("--out-dir", required=True, help="Output root directory for test results.")
    parser.add_argument("--run-name", default=None, help="Hydra/W&B run name under the output root.")
    parser.add_argument(
        "--index-path",
        default="assets/dl3dv_start_0_distance_100_ctx_12v_tgt_8v_first20.json",
        help="Scene index JSON for mesh export.",
    )
    parser.add_argument(
        "--num-context-views",
        type=int,
        default=6,
        help="Number of context views expected by the evaluation index.",
    )
    parser.add_argument(
        "--image-shape",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Optional dataset input image shape override.",
    )
    parser.add_argument(
        "--export-format",
        choices=("ply", "off", "both"),
        default="ply",
        help="Mesh file format written by export modes that support multiple formats.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Optional explicit experiment override. Defaults depend on variant/export-mode.",
    )
    parser.add_argument(
        "--compute-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to compute image metrics during model inference.",
    )
    parser.add_argument(
        "--save-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save model-predicted target RGB images.",
    )
    parser.add_argument(
        "--save-gt-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save cropped GT target RGB images.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override. Can be repeated.",
    )
    return parser.parse_args()


def resolve_path(path_str: str, repo_root: Path, kind: str, must_exist: bool = True) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")
    return path


def validate_request(export_mode: str, export_format: str) -> None:
    if export_mode == "direct" and export_format != "ply":
        raise ValueError(
            "--export-mode direct currently only writes DIRECT_triangle_mesh.ply. "
            "Use --export-format ply."
        )


def resolve_experiment(export_mode: str, experiment: str | None) -> str:
    if experiment is not None:
        return experiment
    try:
        return DEFAULT_EXPERIMENTS[export_mode]
    except KeyError as exc:
        raise ValueError(
            f"No default experiment configured for export_mode={export_mode!r}."
        ) from exc


def default_run_name(export_mode: str) -> str:
    return f"dl3dv_trisplat_{export_mode}_mesh_eval"


def build_command(
    *,
    export_mode: str,
    ckpt: str,
    dl3dv_root: str | None,
    gt_root: str | None,
    out_dir: str,
    run_name: str | None,
    index_path: str,
    num_context_views: int,
    image_shape: Sequence[int] | None,
    export_format: str,
    experiment: str | None,
    compute_scores: bool,
    save_image: bool,
    save_gt_image: bool,
    extra_overrides: Sequence[str],
) -> tuple[list[str], Path]:
    repo_root = Path(__file__).resolve().parents[2]
    validate_request(export_mode, export_format)

    ckpt_path = resolve_path(ckpt, repo_root, "Checkpoint")
    dl3dv_root_path = None
    if dl3dv_root is not None:
        dl3dv_root_path = resolve_path(dl3dv_root, repo_root, "DL3DV root")
    gt_root_path = None
    if gt_root is not None:
        gt_root_path = resolve_path(gt_root, repo_root, "GT root")
    index_path_resolved = resolve_path(index_path, repo_root, "Evaluation index")
    out_dir_path = resolve_path(out_dir, repo_root, "Output directory", must_exist=False)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    resolved_run_name = run_name or default_run_name(export_mode)
    resolved_experiment = resolve_experiment(export_mode, experiment)

    command = [
        sys.executable,
        "-m",
        "src.main",
        f"+experiment={resolved_experiment}",
        "mode=test",
        f"checkpointing.load={ckpt_path}",
        "dataset/view_sampler@dataset.dl3dv.view_sampler=evaluation",
        f"dataset.dl3dv.view_sampler.index_path={index_path_resolved}",
        f"dataset.dl3dv.view_sampler.num_context_views={num_context_views}",
        f"test.compute_scores={'true' if compute_scores else 'false'}",
        "test.align_pose=false",
        f"test.save_image={'true' if save_image else 'false'}",
        f"test.save_gt_image={'true' if save_gt_image else 'false'}",
        "test.save_video=false",
        "test.save_compare=false",
        "test.save_context=false",
        "test.save_debug_info=false",
        "test.save_scene_ranking=false",
        "test.export_mesh=true",
        f"test.output_path={out_dir_path}",
        f"hydra.run.dir={out_dir_path}",
        f"mesh.tsdf_gs2d.export_mode={export_mode}",
        f"mesh.tsdf_gs2d.export_format={export_format}",
        "wandb.mode=disabled",
        f"wandb.name={resolved_run_name}",
    ]
    if gt_root_path is not None:
        command.append(f"test.mesh_gt_path={gt_root_path}")
    if dl3dv_root_path is not None:
        command.extend(
            [
                f"dataset.dl3dv.roots=[{dl3dv_root_path}]",
                f"dataset.dl3dv.test_roots=[{dl3dv_root_path}]",
            ]
        )
    if image_shape is not None:
        height, width = (int(image_shape[0]), int(image_shape[1]))
        command.append(f"dataset.dl3dv.input_image_shape=[{height},{width}]")
    command.extend(extra_overrides)
    return command, repo_root


def run_mesh_eval(
    *,
    export_mode: str,
    ckpt: str,
    dl3dv_root: str | None,
    gt_root: str | None,
    out_dir: str,
    run_name: str | None,
    index_path: str,
    num_context_views: int,
    image_shape: Sequence[int] | None = None,
    export_format: str = "ply",
    experiment: str | None = None,
    compute_scores: bool = False,
    save_image: bool = True,
    save_gt_image: bool = True,
    extra_overrides: Sequence[str] = (),
) -> None:
    command, repo_root = build_command(
        export_mode=export_mode,
        ckpt=ckpt,
        dl3dv_root=dl3dv_root,
        gt_root=gt_root,
        out_dir=out_dir,
        run_name=run_name,
        index_path=index_path,
        num_context_views=num_context_views,
        image_shape=image_shape,
        export_format=export_format,
        experiment=experiment,
        compute_scores=compute_scores,
        save_image=save_image,
        save_gt_image=save_gt_image,
        extra_overrides=extra_overrides,
    )

    print("Running command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, cwd=repo_root, check=True)


def main() -> None:
    args = parse_args()
    run_mesh_eval(
        export_mode=args.export_mode,
        ckpt=args.ckpt,
        dl3dv_root=args.dl3dv_root,
        gt_root=args.gt_root,
        out_dir=args.out_dir,
        run_name=args.run_name,
        index_path=args.index_path,
        num_context_views=args.num_context_views,
        image_shape=args.image_shape,
        export_format=args.export_format,
        experiment=args.experiment,
        compute_scores=args.compute_scores,
        save_image=args.save_image,
        save_gt_image=args.save_gt_image,
        extra_overrides=args.extra_override,
    )


if __name__ == "__main__":
    main()
