#!/bin/bash
# One-time setup on a DCC login node (needs internet; takes ~20 min).
#   bash slurm/setup_env.sh
# Then put your W&B key in ~/.netrc with `wandb login` (or export WANDB_API_KEY in ~/.bashrc).
set -euo pipefail
source "$(dirname "$0")/env.sh"

conda env create -p "$CONDA_ENVS/tutorbias-gpu" -f environment-gpu.yml || conda env update -p "$CONDA_ENVS/tutorbias-gpu" -f environment-gpu.yml
conda env create -p "$CONDA_ENVS/tutorbias-data" -f environment-data.yml || conda env update -p "$CONDA_ENVS/tutorbias-data" -f environment-data.yml

# Multi-VALUE resources: spaCy English model, NLTK corpora, stanza English (with coref).
activate_env tutorbias-data
python -m spacy download en_core_web_sm
python -c "import nltk; [nltk.download(x) for x in ('cmudict', 'wordnet', 'omw-1.4')]"
python -c "import stanza; stanza.download('en', processors='tokenize,mwt,pos,lemma,depparse,coref')"
# Smoke test: one sentence through each dialect we use.
python - <<'PY'
from multivalue import Dialects
for d in (Dialects.AfricanAmericanVernacular(), Dialects.IndianDialect()):
    print(d.dialect_name, "->", d.transform("I don't know what I am supposed to do next."))
PY

# Pre-download model weights so GPU jobs don't spend queue time downloading.
activate_env tutorbias-gpu
python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-7B-Instruct')]"
echo "Setup done."
