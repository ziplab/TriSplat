#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

require_source_file() {
    local path="$1"
    local name="$2"

    if [[ ! -f "${repo_root}/${path}" ]]; then
        echo "${name} source is missing or incomplete."
        exit 1
    fi
}

detect_torch_cuda_arch_list() {
    if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        return 0
    fi

    local detected_arch
    detected_arch="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | sed '/^[[:space:]]*$/d' \
            | head -n 1 \
            | tr -d '[:space:]'
    )"

    if [[ -n "${detected_arch}" ]]; then
        export TORCH_CUDA_ARCH_LIST="${detected_arch}"
    fi
}

if ! command -v python >/dev/null 2>&1; then
    echo "python was not found in PATH. Activate the project environment first."
    exit 1
fi

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
then
    echo "PyTorch is not installed in the active Python environment."
    exit 1
fi

if [[ -z "${CUDA_HOME:-}" && -n "${CONDA_PREFIX:-}" ]]; then
    export CUDA_HOME="${CONDA_PREFIX}"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export CUDA_PATH="${CUDA_HOME}"
fi
detect_torch_cuda_arch_list

torch_lib="$(python - <<'PY'
from pathlib import Path
import torch

print(Path(torch.__file__).resolve().parent / "lib")
PY
)"

if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${torch_lib}:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
else
    export LD_LIBRARY_PATH="${torch_lib}:${LD_LIBRARY_PATH:-}"
fi

cd "${repo_root}"

require_source_file "submodules/diff-triangle-rasterization/setup.py" "diff-triangle-rasterization"
require_source_file "submodules/diff-triangle-rasterization/third_party/glm/glm/glm.hpp" "diff-triangle-rasterization third_party/glm"
require_source_file "submodules/simple-knn/setup.py" "simple-knn"

tmp_dir="$(mktemp -d)"
cleanup() {
    rm -rf "${tmp_dir}"
}
trap cleanup EXIT

zip_path="${tmp_dir}/diff-gaussian-rasterization-w-pose-main.zip"
src_dir="${tmp_dir}/diff-gaussian-rasterization-w-pose-main"
curl -L --fail --retry 3 --connect-timeout 15 \
    https://codeload.github.com/rmurai0610/diff-gaussian-rasterization-w-pose/zip/refs/heads/main \
    -o "${zip_path}"
unzip -q "${zip_path}" -d "${tmp_dir}"
mkdir -p "${src_dir}/third_party/glm"
cp -r "${repo_root}/submodules/diff-triangle-rasterization/third_party/glm/." "${src_dir}/third_party/glm/"
pip install --no-build-isolation "${src_dir}"

cd "${repo_root}/submodules/diff-triangle-rasterization"
rm -rf build diff_triangle_rasterization.egg-info
pip install . --no-build-isolation

cd "${repo_root}/submodules/simple-knn"
pip install . --no-build-isolation

cd "${repo_root}/src/model/encoder/backbone/croco/curope"
pip install . --no-build-isolation
