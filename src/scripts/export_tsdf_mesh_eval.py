import argparse

from .export_mesh_eval import run_mesh_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Re10K TSDF meshes for TriSplat checkpoints and evaluate against GT point clouds.",
    )
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
        help="Mesh file format written by TSDF export.",
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


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"re10k_{args.variant}_tsdf_mesh_eval"
    run_mesh_eval(
        export_mode="tsdf",
        ckpt=args.ckpt,
        gt_root=args.gt_root,
        out_dir=args.out_dir,
        run_name=run_name,
        index_path=args.index_path,
        num_context_views=args.num_context_views,
        export_format=args.export_format,
        experiment=args.experiment,
        extra_overrides=args.extra_override,
    )


if __name__ == "__main__":
    main()
