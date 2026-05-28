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

patch_diff_gaussian_source() {
    local source_dir="$1"

    python - "${source_dir}" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
for path in list(src.rglob("*.h")) + list(src.rglob("*.hip")) + list(src.rglob("*.cu")):
    text = path.read_text()
    text = text.replace('#include "device_launch_parameters.h"\n', "")
    text = text.replace("#include <device_launch_parameters.h>\n", "")
    text = text.replace("#include <cooperative_groups/reduce.h>\n", "")
    text = text.replace("__trap();", "__builtin_trap();")
    text = text.replace("std::uintptr_t", "uintptr_t")
    text = text.replace("<< <", "<<<")
    text = text.replace(">> >", ">>>")
    if path.name in {"backward.hip", "backward.cu"}:
        reduce_marker = "template <typename T>\n__device__ void inline reduce_helper(int lane, int i, T *data) {"
        reduce_helpers = "\n".join([
            "__device__ void inline reduce_add(float &a, float b) { a += b; }",
            "__device__ void inline reduce_add(float2 &a, float2 b) { a.x += b.x; a.y += b.y; }",
            "__device__ void inline reduce_add(float3 &a, float3 b) { a.x += b.x; a.y += b.y; a.z += b.z; }",
            "__device__ void inline reduce_add(float4 &a, float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }",
            "",
        ])
        if reduce_marker in text and "reduce_add(float2" not in text:
            text = text.replace(reduce_marker, reduce_helpers + reduce_marker)
        text = text.replace("data[lane] += data[lane + i];", "reduce_add(data[lane], data[lane + i]);")
    if path.name == "helper_math.h":
        lines = text.splitlines()
        new_lines = []
        i = 0
        replacements = {
            "inline __device__ __host__ float2 smoothstep": [
                "inline __device__ __host__ float2 smoothstep(float2 a, float2 b, float2 x) {",
                "  return make_float2(smoothstep(a.x, b.x, x.x), smoothstep(a.y, b.y, x.y));",
                "}",
            ],
            "inline __device__ __host__ float3 smoothstep": [
                "inline __device__ __host__ float3 smoothstep(float3 a, float3 b, float3 x) {",
                "  return make_float3(smoothstep(a.x, b.x, x.x), smoothstep(a.y, b.y, x.y), smoothstep(a.z, b.z, x.z));",
                "}",
            ],
            "inline __device__ __host__ float4 smoothstep": [
                "inline __device__ __host__ float4 smoothstep(float4 a, float4 b, float4 x) {",
                "  return make_float4(smoothstep(a.x, b.x, x.x), smoothstep(a.y, b.y, x.y), smoothstep(a.z, b.z, x.z), smoothstep(a.w, b.w, x.w));",
                "}",
            ],
        }
        while i < len(lines):
            matched = False
            for prefix, repl in replacements.items():
                if lines[i].startswith(prefix):
                    new_lines.extend(repl)
                    while i < len(lines) and lines[i].strip() != "}":
                        i += 1
                    i += 1
                    matched = True
                    break
            if not matched:
                new_lines.append(lines[i])
                i += 1
        text = "\n".join(new_lines) + "\n"
    path.write_text(text)
PY
}

zip_path="${tmp_dir}/diff-gaussian-rasterization-w-pose-main.zip"
src_dir="${tmp_dir}/diff-gaussian-rasterization-w-pose-main"
curl -L --fail --retry 3 --connect-timeout 15 \
    https://codeload.github.com/rmurai0610/diff-gaussian-rasterization-w-pose/zip/refs/heads/main \
    -o "${zip_path}"
unzip -q "${zip_path}" -d "${tmp_dir}"
mkdir -p "${src_dir}/third_party/glm"
cp -r "${repo_root}/submodules/diff-triangle-rasterization/third_party/glm/." "${src_dir}/third_party/glm/"
patch_diff_gaussian_source "${src_dir}"
pip install --no-build-isolation "${src_dir}"

cd "${repo_root}/submodules/diff-triangle-rasterization"
rm -rf build diff_triangle_rasterization.egg-info
pip install . --no-build-isolation

cd "${repo_root}/submodules/simple-knn"
pip install . --no-build-isolation

cd "${repo_root}/src/model/encoder/backbone/croco/curope"
pip install . --no-build-isolation
