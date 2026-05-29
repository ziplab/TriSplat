import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_EXPERIMENTS = {
    "direct": "trisplat_re10k_triangle_refiner_unet_10m_wide",
    "tsdf": "trisplat_re10k_triangle_refiner_unet_10m_wide",
    "both": "trisplat_re10k_triangle_refiner_unet_10m_wide",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Re10K meshes for TriSplat checkpoints and evaluate them against GT point clouds.",
    )
    parser.add_argument("--export-mode", choices=("direct", "tsdf", "both"), required=True)
    parser.add_argument("--ckpt", required=True, help="Checkpoint path.")
    parser.add_argument("--gt-root", required=True, help="Directory containing <scene>.ply GT point clouds.")
    parser.add_argument("--out-dir", required=True, help="Output root directory for test results.")
    parser.add_argument("--run-name", default=None, help="Hydra/W&B run name under the output root.")
    parser.add_argument(
        "--index-path",
        default="assets/evaluation_index_re10k_mesh_6ctx.json",
        help="Scene index JSON for mesh evaluation.",
    )
    parser.add_argument(
        "--num-context-views",
        type=int,
        default=2,
        help="Number of context views expected by the evaluation index.",
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
    return f"re10k_trisplat_{export_mode}_mesh_eval"


def build_command(
    *,
    export_mode: str,
    ckpt: str,
    gt_root: str,
    out_dir: str,
    run_name: str | None,
    index_path: str,
    num_context_views: int,
    export_format: str,
    experiment: str | None,
    extra_overrides: Sequence[str],
) -> tuple[list[str], Path]:
    repo_root = Path(__file__).resolve().parents[2]
    validate_request(export_mode, export_format)

    ckpt_path = resolve_path(ckpt, repo_root, "Checkpoint")
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
        "dataset/view_sampler@dataset.re10k.view_sampler=evaluation",
        f"dataset.re10k.view_sampler.index_path={index_path_resolved}",
        f"dataset.re10k.view_sampler.num_context_views={num_context_views}",
        "test.compute_scores=false",
        "test.align_pose=false",
        "test.save_image=false",
        "test.save_gt_image=false",
        "test.save_video=false",
        "test.save_compare=false",
        "test.save_context=false",
        "test.save_debug_info=false",
        "test.save_scene_ranking=false",
        "test.export_mesh=true",
        f"test.mesh_gt_path={gt_root_path}",
        f"test.output_path={out_dir_path}",
        f"hydra.run.dir={out_dir_path}",
        f"mesh.tsdf_gs2d.export_mode={export_mode}",
        f"mesh.tsdf_gs2d.export_format={export_format}",
        "wandb.mode=disabled",
        f"wandb.name={resolved_run_name}",
        *extra_overrides,
    ]
    return command, repo_root


def run_mesh_eval(
    *,
    export_mode: str,
    ckpt: str,
    gt_root: str,
    out_dir: str,
    run_name: str | None,
    index_path: str,
    num_context_views: int,
    export_format: str = "ply",
    experiment: str | None = None,
    extra_overrides: Sequence[str] = (),
) -> None:
    command, repo_root = build_command(
        export_mode=export_mode,
        ckpt=ckpt,
        gt_root=gt_root,
        out_dir=out_dir,
        run_name=run_name,
        index_path=index_path,
        num_context_views=num_context_views,
        export_format=export_format,
        experiment=experiment,
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
        gt_root=args.gt_root,
        out_dir=args.out_dir,
        run_name=args.run_name,
        index_path=args.index_path,
        num_context_views=args.num_context_views,
        export_format=args.export_format,
        experiment=args.experiment,
        extra_overrides=args.extra_override,
    )


if __name__ == "__main__":
    main()
