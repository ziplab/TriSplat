#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/eval/render_mesh.sh --test-output PATH --data-root PATH [options]

Options:
  --test-output PATH  Directory containing per-scene mesh exports
  --data-root PATH    Packed dataset root
  --eval-index PATH   Evaluation index JSON (default: auto)
  --mesh-file NAME    Mesh file under scene/mesh (default: DIRECT_triangle_mesh_post.ply)
  --frames MODE       eval | target | saved_pred | all (default: target)
  --image-shape H W   Optional render/model image shape
  --orig-shape H W    Optional original packed image shape
  --help              Show this message
EOF
}

test_output=""
data_root=""
eval_index="auto"
mesh_file="DIRECT_triangle_mesh_post.ply"
frames="target"
image_shape=()
orig_shape=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-output) test_output="$2"; shift 2 ;;
    --data-root) data_root="$2"; shift 2 ;;
    --eval-index) eval_index="$2"; shift 2 ;;
    --mesh-file) mesh_file="$2"; shift 2 ;;
    --frames) frames="$2"; shift 2 ;;
    --image-shape) image_shape=("$2" "$3"); shift 3 ;;
    --orig-shape) orig_shape=("$2" "$3"); shift 3 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

[[ -n "${test_output}" ]] || { echo "--test-output is required" >&2; exit 1; }
[[ -n "${data_root}" ]] || { echo "--data-root is required" >&2; exit 1; }

cmd=(
  python scripts/eval/render_mesh_open3d.py
  --test_output "${test_output}"
  --data_root "${data_root}"
  --eval_index "${eval_index}"
  --mesh_file "${mesh_file}"
  --frames "${frames}"
)
if [[ ${#image_shape[@]} -eq 2 ]]; then
  cmd+=(--image_shape "${image_shape[0]}" "${image_shape[1]}")
fi
if [[ ${#orig_shape[@]} -eq 2 ]]; then
  cmd+=(--orig_image_shape "${orig_shape[0]}" "${orig_shape[1]}")
fi

printf '%q ' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"
