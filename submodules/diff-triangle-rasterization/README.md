# Differential Triangle Rasterization

CUDA extension used by Triangle Splatting style renderers. This submodule builds a
PyTorch extension named `diff_triangle_rasterization._C` and exposes the Python API
from `diff_triangle_rasterization/__init__.py`.

## What Changed

The local build configuration was hardened for newer GPUs:

- `setup.py` no longer hardcodes `sm_89`
- CUDA architecture selection is delegated to PyTorch's extension tooling
- `TORCH_CUDA_ARCH_LIST` is now the primary way to override the target architectures
- the Python wrapper raises a clearer import error when the extension is missing
- empty fallback tensors now stay on the same device as the main inputs

This matters on modern cards such as RTX 5090, where old binaries often fail with:

```text
CUDA error: no kernel image is available for execution on the device
```

## Requirements

- Python 3.10+
- PyTorch built for a CUDA version that supports your GPU
- CUDA toolkit and `nvcc` that support the same GPU architecture
- A compiler toolchain compatible with your PyTorch build

If `torch` itself prints that your GPU capability is unsupported, fix that first.
Rebuilding this extension alone will not solve a mismatched PyTorch install.

## Install

From the repository root:

```bash
conda activate yonosplat
python -m pip install -e submodules/diff-triangle-rasterization
```

Or from inside this directory:

```bash
conda activate yonosplat
python -m pip install -e .
```

## Build For A Specific GPU

PyTorch extensions honor `TORCH_CUDA_ARCH_LIST`. Use it when:

- you are building on one machine for another machine
- you want deterministic multi-arch builds
- auto-detection is not enough

Examples:

```bash
export TORCH_CUDA_ARCH_LIST="8.9"
python -m pip install -e .
```

```bash
export TORCH_CUDA_ARCH_LIST="12.0"
python -m pip install -e .
```

For custom compiler flags:

```bash
export DIFF_TRIANGLE_NVCC_FLAGS="--threads 4"
export DIFF_TRIANGLE_CXX_FLAGS="-fdiagnostics-color=always"
python -m pip install -e .
```

## Rebuild After PyTorch Or CUDA Changes

If you upgraded `torch`, changed CUDA, or switched GPUs, clear stale build artifacts
before rebuilding:

```bash
rm -rf build
rm -rf *.egg-info
rm -rf ~/.cache/torch_extensions
python -m pip install -e .
```

## Quick Verification

```bash
python - <<'PY'
import torch
import diff_triangle_rasterization as dtr

print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("module:", dtr.__file__)
PY
```

## Troubleshooting

### `no kernel image is available for execution on the device`

Typical causes:

- `torch` was built for an older CUDA stack and does not support your GPU
- the extension was compiled before a GPU or CUDA upgrade
- `TORCH_CUDA_ARCH_LIST` was set to the wrong architecture

Recommended checks:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
PY
nvcc --version
```

Then rebuild the extension after fixing the PyTorch/CUDA mismatch.

### `ImportError: diff_triangle_rasterization._C is unavailable`

The compiled extension is missing or incompatible with the current environment.
Reinstall this package with:

```bash
python -m pip install -e .
```

If it still fails, clear `~/.cache/torch_extensions` and rebuild.

### CPU Tensors Passed To The Rasterizer

The Python wrapper now checks this explicitly. All tensor inputs consumed by the CUDA
kernel must live on CUDA devices.

## CMake Build

If you use the CMake path directly, pass architectures explicitly for reproducible
builds:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
```

If `CMAKE_CUDA_ARCHITECTURES` is omitted, CMake will rely on compiler defaults.

## Citation

If you use this rasterizer in research, please cite the original work:

```bibtex
@misc{held2025triangle,
title={Triangle Splatting for Real-Time Radiance Field Rendering},
author={Jan Held and Renaud Vandeghen and Adrien Deliege and Abdullah Hamdi and Silvio Giancola and Anthony Cioppa and Andrea Vedaldi and Bernard Ghanem and Andrea Tagliasacchi and Marc Van Droogenbroeck},
year={2025},
eprint={2505.19175},
url={https://arxiv.org/abs/2505.19175},
}
```

```bibtex
@inproceedings{Held20253DConvex,
title = {{3D} Convex Splatting: Radiance Field Rendering with {3D} Smooth Convexes},
author = {Jan Held and Renaud Vandeghen and Abdullah Hamdi and Adrien Deli\`ege and Anthony Cioppa and Silvio Giancola and Andrea Vedaldi and Bernard Ghanem and Marc Van Droogenbroeck},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
address = {Nashville, TN, USA},
month = {June},
year = {2025}
}
```

```bibtex
@Article{kerbl3Dgaussians,
author = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
title = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
journal = {ACM Transactions on Graphics},
number = {4},
volume = {42},
month = {July},
year = {2023},
url = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
