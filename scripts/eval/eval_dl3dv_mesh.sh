#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval/eval_dl3dv_mesh.sh --ckpt PATH [options] [-- Hydra overrides...]

Options:
  --ckpt PATH          Triangle checkpoint (required)
  --data-root PATH     DL3DV packed dataset root (default: DL3DV_ROOT or ./data/dl3dv)
  --out-dir PATH       Output root (default: outputs/dl3dv_mesh_eval)
  --run-name NAME      Run name under out-dir (default: dl3dv_mesh_eval)
  --index-path PATH    Evaluation index (default: assets/dl3dv_start_0_distance_100_ctx_12v_tgt_8v_first20.json)
  --gpus IDS           CUDA_VISIBLE_DEVICES value (default: existing value or 0)
  --num-context-views N  Context views (default: 12)
  --image-shape H W    Model input/render shape (default: 224 448)
  --orig-shape H W     Original packed image shape (default: 270 480)
  --max-scenes N       Optional cap for smoke testing
  --skip-export        Only render existing meshes
  --skip-render        Only export meshes
  --help               Show this message

Defaults: direct mesh export, scale0.5, post-process pruning, mesh render metrics.
EOF
}

ckpt=""
data_root="${DL3DV_ROOT:-data/dl3dv}"
out_dir="outputs/dl3dv_mesh_eval"
run_name="dl3dv_mesh_eval"
index_path="assets/dl3dv_start_0_distance_100_ctx_12v_tgt_8v_first20.json"
gpus="${CUDA_VISIBLE_DEVICES:-0}"
num_context_views="12"
image_h="224"
image_w="448"
orig_h="270"
orig_w="480"
max_scenes=""
do_export="true"
do_render="true"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt) ckpt="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --run-name) run_name="$2"; shift 2 ;;
    --index-path) index_path="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --num-context-views) num_context_views="$2"; shift 2 ;;
    --image-shape) image_h="$2"; image_w="$3"; shift 3 ;;
    --orig-shape) orig_h="$2"; orig_w="$3"; shift 3 ;;
    --max-scenes) max_scenes="$2"; shift 2 ;;
    --skip-export) do_export="false"; shift ;;
    --skip-render) do_render="false"; shift ;;
    --help|-h) usage; exit 0 ;;
    --) shift; extra_args+=("$@"); break ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

[[ -n "${ckpt}" ]] || { echo "--ckpt is required" >&2; exit 1; }
[[ -f "${ckpt}" ]] || { echo "Checkpoint not found: ${ckpt}" >&2; exit 1; }
[[ -d "${data_root}" ]] || { echo "DL3DV root not found: ${data_root}" >&2; exit 1; }
[[ -f "${index_path}" ]] || { echo "Index not found: ${index_path}" >&2; exit 1; }

test_output="${out_dir}/${run_name}"
export_index="${index_path}"
tmp_dir=""
if [[ -n "${max_scenes}" ]]; then
  tmp_dir="$(mktemp -d)"
  python - "${index_path}" "${tmp_dir}/index.json" "${max_scenes}" <<'PY'
import json, sys
src, dst, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
data = json.load(open(src, "r", encoding="utf-8"))
items = list(data.items())[:limit]
json.dump(dict(items), open(dst, "w", encoding="utf-8"), indent=2)
PY
  export_index="${tmp_dir}/index.json"
fi

if [[ "${do_export}" == "true" ]]; then
  cmd=(
    python -m src.main
    "+experiment=trisplat_dl3dv_triangle_refiner_unet_10m_224x448_test"
    mode=test
    "checkpointing.load=${ckpt}"
    "dataset/view_sampler@dataset.dl3dv.view_sampler=evaluation"
    "dataset.dl3dv.roots=[${data_root}]"
    "dataset.dl3dv.test_roots=[${data_root}]"
    "dataset.dl3dv.view_sampler.index_path=${export_index}"
    "dataset.dl3dv.view_sampler.num_context_views=${num_context_views}"
    "dataset.dl3dv.input_image_shape=[${image_h},${image_w}]"
    "dataset.dl3dv.original_image_shape=[${orig_h},${orig_w}]"
    test.compute_scores=true
    test.align_pose=false
    test.save_image=true
    test.save_gt_image=true
    test.save_video=false
    test.save_compare=false
    test.save_context=false
    test.save_debug_info=false
    test.save_scene_ranking=true
    test.export_mesh=true
    "test.output_path=${out_dir}"
    "hydra.run.dir=${out_dir}/hydra/${run_name}"
    mesh.tsdf_gs2d.export_mode=direct
    mesh.tsdf_gs2d.export_format=both
    mesh.tsdf_gs2d.direct_post_process=true
    model.encoder.triangle_adapter.triangle_scale_min=0.25
    model.encoder.triangle_adapter.triangle_scale_max=9.0
    wandb.mode=disabled
    "wandb.name=${run_name}"
    "${extra_args[@]}"
  )
  printf 'Export: CUDA_VISIBLE_DEVICES=%q ' "${gpus}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  env CUDA_VISIBLE_DEVICES="${gpus}" "${cmd[@]}"
fi

if [[ "${do_render}" == "true" ]]; then
  python scripts/eval/render_mesh_open3d.py \
    --data_root "${data_root}" \
    --test_output "${test_output}" \
    --eval_index "${export_index}" \
    --mesh_file DIRECT_triangle_mesh_post.ply \
    --frames target \
    --image_shape "${image_h}" "${image_w}" \
    --orig_image_shape "${orig_h}" "${orig_w}" \
    --pose_norm_method max_pairwise_d \
    --relative_pose \
    --summary_name mesh_render_metrics_summary.json
fi

if [[ -n "${tmp_dir}" ]]; then
  rm -rf "${tmp_dir}"
fi
