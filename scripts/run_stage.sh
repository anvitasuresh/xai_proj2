#!/bin/bash
# Run one pipeline stage on whatever machine you're on. The SLURM jobs call this same script, so
# DCC, Colab, a rented GPU, and a laptop all run identical commands.
#
#   bash scripts/run_stage.sh items       CPU   GSM8K items + stuck messages           (seconds)
#   bash scripts/run_stage.sh variants    CPU   Multi-VALUE dialect + name variants    (~1-3 h; laptop is fine)
#   bash scripts/run_stage.sh validate    CPU   analyze native-speaker ratings of the variants
#   bash scripts/run_stage.sh replies     GPU   tutor reply pool + blinded label set   (~1 h on one 24 GB+ GPU)
#   bash scripts/run_stage.sh judge       GPU   judge scores + rationales on label set (~10 min)
#   bash scripts/run_stage.sh audit       GPU   E2 audit: pairs, judge + 2 RMs, analysis (~2-3 h)
#
# Env vars: SHARDS (variants; default = min(CPU count, 4)), SHARD (run only one shard, used by SLURM arrays),
# PROFILE (t4 = Kaggle 2x T4 settings), DATA_CONFIG (items/variants), MODELS (replies; default "qwen1.5b qwen7b"), SCORERS (audit; default "judge skywork_v2 armorm").
# Uses .venv-data / .venv-gpu if they exist (scripts/setup_local.sh); otherwise the active Python.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:---help}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
mkdir -p logs
DATA_CFG="${DATA_CONFIG:-configs/data.yaml}"
# Hardware profile: PROFILE=t4 uses configs/t4/<stage>.yaml where one exists (Kaggle 2x T4).
cfg() { if [ -n "${PROFILE:-}" ] && [ -f "configs/$PROFILE/$1.yaml" ]; then echo "configs/$PROFILE/$1.yaml"; else echo "configs/$1.yaml"; fi; }

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
    # Each shard loads its own spaCy + stanza pipelines (~1.5-2 GB RAM), so default to at most 4.
    NCPU="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
    N="${SHARDS:-$(( NCPU < 4 ? NCPU : 4 ))}"
    if [ -n "${SHARD:-}" ]; then
      $P -m tutorbias.data.build variants --config "$DATA_CFG" --shard "$SHARD" --num-shards "$N"
    else
      rm -f data/processed/variants.shard*.jsonl logs/variants-shard*.log
      echo "Running $N shards in parallel; progress in logs/variants-shard*.log"
      pids=()
      for i in $(seq 0 $((N - 1))); do
        $P -m tutorbias.data.build variants --config "$DATA_CFG" --shard "$i" --num-shards "$N" \
          > "logs/variants-shard$i.log" 2>&1 &
        pids+=($!)
      done
      failed=0
      for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || { echo "Shard $i FAILED, see logs/variants-shard$i.log:"; tail -5 "logs/variants-shard$i.log"; failed=1; }
      done
      [ "$failed" -eq 0 ] || exit 1
    fi
    # Merge once every shard is written (in a SLURM array, the last shard to finish does this).
    if [ "$(ls data/processed/variants.shard*-of-$(printf %02d "$N").jsonl 2>/dev/null | wc -l)" -eq "$N" ]; then
      $P -m tutorbias.data.build merge --config "$DATA_CFG"
      $P -m tutorbias.data.build check-sample --config "$DATA_CFG"
    fi
    ;;
  validate)
    # After native speakers fill in data/validation/ratings_<group>_<rater>.csv (docs/dialect_validation.md)
    $(py data) -m tutorbias.data.validate_variants --config "$DATA_CFG"
    ;;
  replies)
    P=$(py gpu)
    for m in ${MODELS:-qwen1.5b qwen7b}; do
      $P -m tutorbias.replies.generate --config "$(cfg replies)" --model "$m"
    done
    $P -m tutorbias.replies.generate --config "$(cfg replies)" --select
    $P -m tutorbias.labeling.label_set --config configs/calibration.yaml
    ;;
  judge)
    P=$(py gpu)
    $P -m tutorbias.judge.score --config "$(cfg judge)" --judge "${JUDGE:-primary}" \
      --input "${INPUT:-data/labels/label_set.jsonl}" --output "${OUTPUT:-results/judge/label_set_scores.jsonl}" \
      --id-field "${ID_FIELD:-label_id}" ${EXTRA---explain}
    ;;
  audit)
    P=$(py gpu)
    $P -m tutorbias.audit.pairs --config "$(cfg audit)"
    mkdir -p results/audit/scores
    for s in ${SCORERS:-judge skywork_v2 armorm}; do
      if [ "$s" = "judge" ]; then
        $P -m tutorbias.judge.score --config "$(cfg judge)" --judge primary \
          --input data/processed/audit_pairs.jsonl --output results/audit/scores/judge.jsonl --id-field pair_id
      else
        $P -m tutorbias.audit.reward_models --config "$(cfg audit)" --rm "$s"
      fi
    done
    $P -m tutorbias.audit.analyze --config "$(cfg audit)"
    ;;
  *)
    sed -n "2,/^set -euo/p" "$0" | grep "^#"
    ;;
esac
