#!/bin/bash
# Stage 2 reward-model audit (E2): build pairs, score them with the judge and each RewardBench RM,
# then analyze. ~30k pairs; every scorer is resumable, so resubmitting after preemption continues.
#   sbatch slurm/04_audit.sbatch
#   SCORERS="armorm" sbatch slurm/04_audit.sbatch     # rerun only some scorers
#SBATCH --job-name=tb-audit
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
activate_env tutorbias-gpu
nvidia-smi --query-gpu=name,memory.total --format=csv

SCORERS="${SCORERS:-judge skywork_v2 armorm}"
python -m tutorbias.audit.pairs --config configs/audit.yaml
mkdir -p results/audit/scores

for s in $SCORERS; do
  if [ "$s" = "judge" ]; then
    # Rationales are off here (30k pairs); E4 generates them on a subset in stage 3.
    python -m tutorbias.judge.score --config configs/judge.yaml --judge primary \
      --input data/processed/audit_pairs.jsonl --output results/audit/scores/judge.jsonl --id-field pair_id
  else
    python -m tutorbias.audit.reward_models --config configs/audit.yaml --rm "$s"
  fi
done

python -m tutorbias.audit.analyze --config configs/audit.yaml
