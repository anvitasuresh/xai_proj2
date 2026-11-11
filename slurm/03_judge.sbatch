#!/bin/bash
# Score items with the 7B scaffolding judge. Resumable, so it is safe on scavenger-gpu.
#   sbatch slurm/03_judge.sbatch                                         # label set (E1), with rationales
#   INPUT=... OUTPUT=... ID_FIELD=pair_id sbatch slurm/03_judge.sbatch   # any other pair file (stage 2)
#SBATCH --job-name=tb-judge
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

INPUT="${INPUT:-data/labels/label_set.jsonl}"
OUTPUT="${OUTPUT:-results/judge/label_set_scores.jsonl}"
ID_FIELD="${ID_FIELD:-label_id}"
JUDGE="${JUDGE:-primary}"
EXTRA="${EXTRA:---explain}"

python -m tutorbias.judge.score --config configs/judge.yaml --judge "$JUDGE" \
  --input "$INPUT" --output "$OUTPUT" --id-field "$ID_FIELD" $EXTRA
