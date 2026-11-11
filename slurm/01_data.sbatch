#!/bin/bash
# Stage 1 dialect variants, CPU only. Multi-VALUE is slow (stanza coref), so this runs as an array of
# shards. First build items and messages on the login node (takes seconds):
#   conda activate $CONDA_ENVS/tutorbias-data
#   python -m tutorbias.data.build items && python -m tutorbias.data.build messages
# then:
#   sbatch slurm/01_data.sbatch
#SBATCH --job-name=tb-data
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
source slurm/env.sh
activate_env tutorbias-data
N=8

test -f data/processed/messages_test.jsonl || { echo "Build items and messages first (see top of file)"; exit 1; }

python -m tutorbias.data.build variants --config configs/data.yaml --shard "$SLURM_ARRAY_TASK_ID" --num-shards $N

# The last shard to finish merges.
if [ "$(ls data/processed/variants.shard*-of-0$N.jsonl 2>/dev/null | wc -l)" -eq $N ]; then
  python -m tutorbias.data.build merge --config configs/data.yaml
  python -m tutorbias.data.build check-sample --config configs/data.yaml
fi
