#!/bin/bash
# GPU stage "audit" on DCC. See scripts/run_stage.sh for what it runs and its env-var options.
#   sbatch slurm/04_audit.sbatch
#SBATCH --job-name=tb-audit
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
bash scripts/run_stage.sh audit
