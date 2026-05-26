<h1 align="center">TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2605.26115"><img src="https://img.shields.io/badge/Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://lhmd.top/trisplat"><img src="https://img.shields.io/badge/Project%20Page-000000?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/lhmd/TriSplat"><img src="https://img.shields.io/badge/Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Models"></a>
</p>

<p align="center">
  <a href="https://lhmd.top/">Weijie Wang</a><sup>1,*</sup>
  <a href="https://github.com/puLangMu">Zimu Li</a><sup>1,*</sup>
  <a href="https://chuan-10.github.io/">Jinchuan Shi</a><sup>1</sup>
  <a href="https://steve-zeyu-zhang.github.io/">Zeyu Zhang</a><sup>1</sup>
  <a href="https://botaoye.github.io/">Botao Ye</a><sup>2,3</sup>
  <a href="https://people.inf.ethz.ch/~pomarc/">Marc Pollefeys</a><sup>2,4</sup>
  <a href="https://donydchen.github.io/">Donny Y. Chen</a><sup>5</sup>
  <a href="https://bohanzhuang.github.io/">Bohan Zhuang</a><sup>1</sup>
</p>

<p align="center">
  <sup>1</sup>Zhejiang University
  <sup>2</sup>ETH Zurich
  <sup>3</sup>ETH AI Center
  <sup>4</sup>Microsoft
  <sup>5</sup>Monash University
</p>

<p align="center">
  <img src="https://lhmd.top/trisplat/assets/images/teaser.jpg" alt="TriSplat teaser" width="100%">
</p>

TriSplat is a feed-forward 3D reconstruction model that predicts simulation-ready triangle meshes from sparse, unposed images. Unlike Gaussian-splatting pipelines that require post-hoc mesh extraction, TriSplat directly predicts oriented triangle primitives, camera poses, point maps, and appearance attributes in one forward pass. We train on RealEstate10K and DL3DV, and evaluate zero-shot generalization on ScanNet with RE10K-trained models.

## Method

<p align="center">
  <img src="https://lhmd.top/trisplat/assets/figures/web/pipeline2.png" alt="TriSplat pipeline" width="100%">
</p>

Given sparse input views, TriSplat predicts dense local point maps, triangle attributes, camera poses, and optional intrinsics. Point-map geometry anchors triangle orientation through geometry normals, a learned normal refiner, and a monocular-normal bootstrap. A differentiable triangle rasterizer renders RGB, depth, and normals, while mesh export only needs opacity filtering, winding correction, and duplicate-vertex merging.

## Installation

Create the environment:

```bash
conda create -y -n trisplat python=3.10
conda activate trisplat
pip install --upgrade pip
```

Install PyTorch and Python dependencies:

```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt --no-build-isolation
```

Build CUDA extensions:

```bash
bash scripts/env/rebuild_extensions.sh
```

Download initialization weights used by the model:

```bash
mkdir -p pretrained_weights
wget -O pretrained_weights/pi3.safetensors \
  https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors
wget -O pretrained_weights/omnidata_dpt_normal_v2.ckpt \
  'https://zenodo.org/records/10447888/files/omnidata_dpt_normal_v2.ckpt?download=1'
```

## Models

Download released TriSplat checkpoints from [lhmd/TriSplat](https://huggingface.co/lhmd/TriSplat):

```bash
mkdir -p checkpoints
wget -O checkpoints/re10k_trisplat.ckpt \
  https://huggingface.co/lhmd/TriSplat/resolve/main/re10k_trisplat.ckpt
wget -O checkpoints/dl3dv_trisplat.ckpt \
  https://huggingface.co/lhmd/TriSplat/resolve/main/dl3dv_trisplat.ckpt
```

## Datasets

Packed `.torch` datasets default to:

```text
data/re10k
data/dl3dv
```

You can also set:

```bash
export RE10K_ROOT="$PWD/data/re10k"
export DL3DV_ROOT="$PWD/data/dl3dv"
```

See [data/README.md](data/README.md) for dataset layout and conversion notes.

## Training

Train on RealEstate10K:

```bash
bash scripts/train/train_re10k.sh --gpus 0,1,2,3,4,5,6,7 --wandb-mode offline
```

Train on DL3DV:

```bash
bash scripts/train/train_dl3dv.sh --gpus 0,1,2,3,4,5,6,7 --wandb-mode offline
```

Extra arguments after `--` are passed to Hydra. Use `--ckpt` to resume or initialize from a checkpoint.

## Evaluation

Evaluate and render RealEstate10K meshes:

```bash
bash scripts/eval/eval_re10k_mesh.sh \
  --ckpt checkpoints/re10k_trisplat.ckpt \
  --data-root "$RE10K_ROOT"
```

Evaluate and render DL3DV meshes:

```bash
bash scripts/eval/eval_dl3dv_mesh.sh \
  --ckpt checkpoints/dl3dv_trisplat.ckpt \
  --data-root "$DL3DV_ROOT"
```

<p align="center">
  <img src="https://lhmd.top/trisplat/assets/figures/web/efficiency.png" alt="TriSplat efficiency comparison" width="100%">
</p>

<p align="center">
  <img src="https://lhmd.top/trisplat/assets/figures/main_simulation_demo.png" alt="Simulation-ready mesh demo" width="100%">
</p>

## Simulation

TriSplat exports ordinary triangle meshes, so the output can be opened directly by common graphics and simulation tools. The evaluation scripts above write per-scene meshes under:

```text
outputs/<eval_root>/<run_name>/<scene>/mesh/DIRECT_triangle_mesh.ply
outputs/<eval_root>/<run_name>/<scene>/mesh/DIRECT_triangle_mesh.off
outputs/<eval_root>/<run_name>/<scene>/mesh/DIRECT_triangle_mesh_post.ply
outputs/<eval_root>/<run_name>/<scene>/mesh/DIRECT_triangle_mesh_post.off
```

The `_post` mesh is the default rendering and simulation output. It applies connected-component cleanup to the direct mesh, keeping the largest components and removing small disconnected floaters, unreferenced vertices, and degenerate triangles. For example, after running `scripts/eval/eval_re10k_mesh.sh`, use:

```bash
ls outputs/re10k_mesh_eval/re10k_mesh_eval/*/mesh/DIRECT_triangle_mesh_post.ply
```

The exported `_post.ply` mesh is vertex-colored and can be imported into [Blender](https://www.blender.org/), [Open3D](https://www.open3d.org/), [Isaac Sim](https://developer.nvidia.com/isaac/sim), [Unity](https://unity.com/), or [PyBullet](https://pybullet.org/) as a static triangle mesh. For simulation, use the `.ply` mesh for visual geometry and generate a collision mesh in your simulator if needed; for example, simplify or convex-decompose it before rigid-body simulation when the raw mesh is too dense.

## Citation

If you find this repository useful, please cite:

```bibtex
@article{wang2026trisplat,
  title={TriSplat: Simulation-Ready Feed-Forward 3D Scene Reconstruction},
  author={Wang, Weijie and Li, Zimu and Shi, Jinchuan and Zhang, Zeyu and Ye, Botao and Pollefeys, Marc and Chen, Donny Y. and Zhuang, Bohan},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgements

This codebase builds on open-source work including [YoNoSplat](https://github.com/justimyhxu/YoNoSplat), [MVSplat](https://github.com/donydchen/mvsplat), [pixelSplat](https://github.com/dcharatan/pixelsplat), [CroCo](https://github.com/naver/croco), [DINOv2](https://github.com/facebookresearch/dinov2), [Omnidata](https://github.com/EPFL-VILAB/omnidata), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), and [Triangle Splatting](https://github.com/trianglesplatting/triangle-splatting).
