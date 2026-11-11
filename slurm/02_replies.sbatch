#!/bin/bash
# Generate the tutor reply pool on the audit questions, then select ~1500 replies.
#   sbatch slurm/02_replies.sbatch
# Optional API model: export OPENAI_API_KEY and add `api` to MODELS below.
#SBATCH --job-name=tb-replies
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
activate_env tutorbias-gpu
nvidia-smi --query-gpu=name,memory.total --format=csv

MODELS="qwen1.5b qwen7b"
for m in $MODELS; do
  python -m tutorbias.replies.generate --config configs/replies.yaml --model "$m"
done
python -m tutorbias.replies.generate --config configs/replies.yaml --select
python -m tutorbias.labeling.label_set --config configs/calibration.yaml
