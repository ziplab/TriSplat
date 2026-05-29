#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train/train_re10k.sh [options] [-- Hydra overrides...]

Options:
  --gpus IDS          CUDA_VISIBLE_DEVICES value (default: existing value or 0)
  --ckpt PATH         Optional checkpoint/weights path to load
  --run-name NAME     W&B/Hydra run name (default: trisplat_re10k_train)
  --wandb-mode MODE   disabled | offline | online (default: offline)
  --batch-size N      Training batch size (default: 1)
  --lr LR             Learning rate (default: 2e-4)
  --num-nodes N       Lightning num_nodes (default: 1)
  --context-views V   Context views, e.g. 6 or [2,32] (default: 6)
  --help              Show this message

Dataset root defaults to RE10K_ROOT or ./data/re10k.
Default experiment: trisplat_re10k_triangle_refiner_unet_10m_wide.
Anything after -- is forwarded to Hydra.
EOF
}

gpus="${CUDA_VISIBLE_DEVICES:-0}"
ckpt=""
run_name="trisplat_re10k_train"
wandb_mode="offline"
batch_size="1"
lr="2e-4"
num_nodes="1"
context_views="6"
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) gpus="$2"; shift 2 ;;
    --ckpt) ckpt="$2"; shift 2 ;;
    --run-name) run_name="$2"; shift 2 ;;
    --wandb-mode) wandb_mode="$2"; shift 2 ;;
    --batch-size) batch_size="$2"; shift 2 ;;
    --lr) lr="$2"; shift 2 ;;
    --num-nodes) num_nodes="$2"; shift 2 ;;
    --context-views) context_views="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    --) shift; extra_args+=("$@"); break ;;
    *) extra_args+=("$1"); shift ;;
  esac
done

cmd=(
  python -m src.main
  "+experiment=trisplat_re10k_triangle_refiner_unet_10m_wide"
  "trainer.num_nodes=${num_nodes}"
  "wandb.mode=${wandb_mode}"
  "wandb.name=${run_name}"
  "optimizer.lr=${lr}"
  "data_loader.train.batch_size=${batch_size}"
  "checkpointing.save_weights_only=false"
  "dataset.re10k.view_sampler.num_context_views=${context_views}"
)

if [[ -n "${ckpt}" ]]; then
  [[ -f "${ckpt}" ]] || { echo "Checkpoint not found: ${ckpt}" >&2; exit 1; }
  cmd+=("checkpointing.load=${ckpt}")
fi
cmd+=("${extra_args[@]}")

printf 'Command: CUDA_VISIBLE_DEVICES=%q ' "${gpus}"
printf '%q ' "${cmd[@]}"
printf '\n'
exec env CUDA_VISIBLE_DEVICES="${gpus}" "${cmd[@]}"
