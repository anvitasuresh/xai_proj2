#!/bin/bash
# Set up the two Python environments without conda or SLURM: works on a laptop (CPU steps), Colab,
# or any rented Linux GPU machine (RunPod, Lambda, etc.).
#   bash scripts/setup_local.sh          # both envs
#   bash scripts/setup_local.sh data     # only the dialect env (enough for variants, labeling, calibration)
#
# Creates .venv-data (Python 3.10, Multi-VALUE) and .venv-gpu (Python 3.11, vLLM) with uv, which
# downloads the right Python version itself.
set -euo pipefail
cd "$(dirname "$0")/.."
WHICH="${1:-all}"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }

if [ "$WHICH" = "all" ] || [ "$WHICH" = "data" ]; then
  uv venv --python 3.10 .venv-data
  uv pip install --python .venv-data/bin/python -e ".[dialect,label,track,dev]"
  # spaCy model as a wheel (spacy download needs pip inside the venv)
  uv pip install --python .venv-data/bin/python \
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
  .venv-data/bin/python -c "import nltk; [nltk.download(x, quiet=True) for x in ('cmudict', 'wordnet', 'omw-1.4')]"
  .venv-data/bin/python -c "import stanza; stanza.download('en', processors='tokenize,mwt,pos,lemma,depparse,coref')"
  .venv-data/bin/python - <<'PY'
from multivalue import Dialects
for d in (Dialects.AfricanAmericanVernacular(), Dialects.IndianDialect()):
    print(d.dialect_name, "->", d.transform("I don't know what I am supposed to do next."))
PY
fi

if [ "$WHICH" = "all" ] || [ "$WHICH" = "gpu" ]; then
  uv venv --python 3.11 .venv-gpu
  uv pip install --python .venv-gpu/bin/python -e ".[gpu,api,track,dev]"
fi
echo "Done. Next: bash scripts/run_stage.sh --help"
