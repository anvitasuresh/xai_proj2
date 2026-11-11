#!/bin/bash
# Run one pipeline stage on whatever machine you're on. The SLURM jobs call this same script, so
# DCC, Colab, a rented GPU, and a laptop all run identical commands.
#
#   bash scripts/run_stage.sh items       CPU   GSM8K items + stuck messages           (seconds)
#   bash scripts/run_stage.sh variants    CPU   Multi-VALUE dialect + name variants    (~1-3 h; laptop is fine)
#   bash scripts/run_stage.sh replies     GPU   tutor reply pool + blinded label set   (~1 h on one 24 GB+ GPU)
#   bash scripts/run_stage.sh judge       GPU   judge scores + rationales on label set (~10 min)
#   bash scripts/run_stage.sh audit       GPU   E2 audit: pairs, judge + 2 RMs, analysis (~2-3 h)
#
# Env vars: SHARDS (variants; default = CPU count), SHARD (run only one shard, used by SLURM arrays),
# DATA_CONFIG (items/variants), MODELS (replies; default "qwen1.5b qwen7b"), SCORERS (audit; default "judge skywork_v2 armorm").
# Uses .venv-data / .venv-gpu if they exist (scripts/setup_local.sh); otherwise the active Python.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:---help}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
DATA_CFG="${DATA_CONFIG:-configs/data.yaml}"

py() {  # py data|gpu -> python executable for that env
  if [ -x ".venv-$1/bin/python" ]; then echo ".venv-$1/bin/python"; else echo "python"; fi
}

case "$STAGE" in
  items)
    P=$(py data)
    $P -m tutorbias.data.build items --config "$DATA_CFG"
    $P -m tutorbias.data.build messages --config "$DATA_CFG"
    ;;
  variants)
    P=$(py data)
    test -f data/processed/messages_test.jsonl || { echo "Run 'items' first."; exit 1; }
    N="${SHARDS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"
    if [ -n "${SHARD:-}" ]; then
      $P -m tutorbias.data.build variants --config "$DATA_CFG" --shard "$SHARD" --num-shards "$N"
    else
      rm -f data/processed/variants.shard*.jsonl
      for i in $(seq 0 $((N - 1))); do
        $P -m tutorbias.data.build variants --config "$DATA_CFG" --shard "$i" --num-shards "$N" \
          > "logs/variants-shard$i.log" 2>&1 &
      done
      wait
    fi
    # Merge once every shard is written (in a SLURM array, the last shard to finish does this).
    if [ "$(ls data/processed/variants.shard*-of-$(printf %02d "$N").jsonl 2>/dev/null | wc -l)" -eq "$N" ]; then
      $P -m tutorbias.data.build merge --config "$DATA_CFG"
      $P -m tutorbias.data.build check-sample --config "$DATA_CFG"
    fi
    ;;
  replies)
    P=$(py gpu)
    for m in ${MODELS:-qwen1.5b qwen7b}; do
      $P -m tutorbias.replies.generate --config configs/replies.yaml --model "$m"
    done
    $P -m tutorbias.replies.generate --config configs/replies.yaml --select
    $P -m tutorbias.labeling.label_set --config configs/calibration.yaml
    ;;
  judge)
    P=$(py gpu)
    $P -m tutorbias.judge.score --config configs/judge.yaml --judge "${JUDGE:-primary}" \
      --input "${INPUT:-data/labels/label_set.jsonl}" --output "${OUTPUT:-results/judge/label_set_scores.jsonl}" \
      --id-field "${ID_FIELD:-label_id}" ${EXTRA---explain}
    ;;
  audit)
    P=$(py gpu)
    $P -m tutorbias.audit.pairs --config configs/audit.yaml
    mkdir -p results/audit/scores
    for s in ${SCORERS:-judge skywork_v2 armorm}; do
      if [ "$s" = "judge" ]; then
        $P -m tutorbias.judge.score --config configs/judge.yaml --judge primary \
          --input data/processed/audit_pairs.jsonl --output results/audit/scores/judge.jsonl --id-field pair_id
      else
        $P -m tutorbias.audit.reward_models --config configs/audit.yaml --rm "$s"
      fi
    done
    $P -m tutorbias.audit.analyze --config configs/audit.yaml
    ;;
  *)
    sed -n '2,17p' "$0"
    ;;
esac
