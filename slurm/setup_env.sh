#!/bin/bash
# One-time setup on a DCC login node (needs internet; ~20 min). Same as scripts/setup_local.sh, plus a
# model pre-download so GPU jobs don't spend queue time downloading.
#   bash slurm/setup_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source slurm/env.sh
bash scripts/setup_local.sh all
.venv-gpu/bin/python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-7B-Instruct')]"
echo "Setup done. Optional: .venv-gpu/bin/wandb login"
