"""Score (row, reply) pairs with the judge. Resumable: re-running skips ids already in the output.

  python -m tutorbias.judge.score --config configs/judge.yaml --input data/labels/label_set.jsonl \
      --output results/judge/label_set_scores.jsonl [--explain] [--judge primary|api]

Input rows need: an id field (config `id_field`, default "pair_id"), question, answer,
student_message, reply. Output rows are {id, ...JudgeScore fields} plus any passthrough fields.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from ..io import append_jsonl, iter_jsonl, load_config, sha256_file
from ..models.backends import load_backend
from ..tracking import init_run
from .judge import ScaffoldingJudge

PASSTHROUGH = ("variant_id", "reply_id", "item_id", "group", "axis", "label_id")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/judge.yaml")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--judge", default="primary", help="which judge block in the config to use")
    p.add_argument("--id-field", default=None)
    p.add_argument("--explain", action="store_true", help="also generate post-hoc rationales (E4)")
    p.add_argument("--limit", type=int, default=None, help="score only the first N rows (smoke test)")
    a = p.parse_args(argv)

    cfg = load_config(a.config)
    jcfg = cfg["judges"][a.judge]
    id_field = a.id_field or cfg.get("id_field", "pair_id")
    out = Path(a.output)

    done = {r["id"] for r in iter_jsonl(out)} if out.exists() else set()
    rows = [r for r in iter_jsonl(a.input) if r[id_field] not in done]
    if a.limit:
        rows = rows[: a.limit]
    print(f"[score] {len(done)} already scored, {len(rows)} to go")
    if not rows:
        return

    run = init_run("judge-score", {**cfg, "judge": a.judge, "input": a.input, "input_sha": sha256_file(a.input)})
    judge = ScaffoldingJudge(load_backend(jcfg["model"]), jcfg.get("rubric", "v1"), jcfg.get("mode", "logprob"))

    bs = cfg.get("batch_size", 256)
    t0, n_done, masses = time.time(), 0, []
    for i in range(0, len(rows), bs):
        batch = rows[i : i + bs]
        scores = judge.score(batch, explain=a.explain)
        recs = []
        for r, s in zip(batch, scores):
            rec = {"id": r[id_field], **{k: r[k] for k in PASSTHROUGH if k in r}, **s.to_dict()}
            recs.append(rec)
            masses.append(s.valid_mass_scaffolding)
        append_jsonl(out, recs)  # flushed per batch, so a preempted job loses at most one batch
        n_done += len(batch)
        rate = n_done / (time.time() - t0)
        run.log({"scored": n_done, "rows_per_sec": rate,
                 "valid_mass_scaffolding_mean": statistics.fmean(masses)})
        print(f"[score] {n_done}/{len(rows)}  {rate:.1f} rows/s  valid-mass {statistics.fmean(masses):.3f}")

    low = sum(m < cfg.get("min_valid_mass", 0.9) for m in masses)
    if low:
        print(f"[score] WARNING: {low} rows put <{cfg.get('min_valid_mass', 0.9)} probability on valid labels. "
              "Check the prompt format before trusting these scores.")
    run.summary({"n_scored": n_done, "low_valid_mass": low, **judge.meta})
    run.finish()


if __name__ == "__main__":
    main()
