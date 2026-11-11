#!/bin/bash
# Dialect + name variants as an 8-shard CPU array (Multi-VALUE is slow). Run items first:
#   bash scripts/run_stage.sh items && sbatch slurm/01_data.sbatch
#SBATCH --job-name=tb-data
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source slurm/env.sh
SHARD="$SLURM_ARRAY_TASK_ID" SHARDS=8 bash scripts/run_stage.sh variants
