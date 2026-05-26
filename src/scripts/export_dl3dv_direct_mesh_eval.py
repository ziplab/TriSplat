import argparse

from .export_dl3dv_mesh_eval import run_mesh_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DL3DV direct triangle meshes for TriSplat checkpoints.",
    )
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
        help="Mesh file format written by direct export.",
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


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"dl3dv_{args.variant}_direct_mesh_eval"
    run_mesh_eval(
        export_mode="direct",
        ckpt=args.ckpt,
        dl3dv_root=args.dl3dv_root,
        gt_root=args.gt_root,
        out_dir=args.out_dir,
        run_name=run_name,
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
