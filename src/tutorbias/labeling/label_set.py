"""Build the blinded label set (~300 replies) for judge calibration (E1).

  python -m tutorbias.labeling.label_set --config configs/calibration.yaml

Writes:
  data/labels/label_set.jsonl   what the labeler sees: label_id, problem, answer, student message, reply.
                                No model name, no style, no judge score (blinding).
  data/labels/label_key.jsonl   label_id -> reply_id, style, model, dev/test split, double-label flag.
  data/labels/label_set.csv     same items as a spreadsheet, if you'd rather label in Sheets/Excel.

Design choices:
  * Stratified by intended style so all scaffolding levels appear (otherwise kappa is unstable).
  * dev/test split of the labels: iterate on the rubric using *dev only*. Report final agreement on
    *test*, which you don't look at until the rubric is frozen. Without this, rubric tuning overfits
    to the labels and E1 overstates agreement.
  * A double-label subset (~100, spread over dev and test) for the second labeler: human-human
    agreement is the ceiling E1 is compared against.
  * Labels are given on the MAE message. Assumption (stated in the README): the correct label for a
    reply doesn't depend on the student's dialect, so the same label is the ground truth for every
    counterfactual variant. That lets us measure judge-human agreement *per dialect group* with no
    extra labeling.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path
from typing import Any

from ..io import load_config, read_jsonl, write_jsonl


def build_label_set(pool: list[dict], items: dict[str, dict], msgs: dict[str, dict], cfg: dict[str, Any]):
    rng = random.Random(cfg["seed"])
    n, n_double, dev_frac = cfg["n_labels"], cfg["n_double"], cfg["dev_frac"]

    # At most one reply per question, so labels aren't clustered on a few problems.
    by_style: dict[str, list[dict]] = {}
    for r in pool:
        by_style.setdefault(r["style"], []).append(r)
    for v in by_style.values():
        rng.shuffle(v)
    styles = sorted(by_style)
    chosen, used_items = [], set()
    while len(chosen) < n and any(by_style.values()):
        for s in styles:
            while by_style[s]:
                r = by_style[s].pop()
                if r["item_id"] not in used_items:
                    chosen.append(r)
                    used_items.add(r["item_id"])
                    break
            if len(chosen) == n:
                break

    rng.shuffle(chosen)
    # Stratified dev/test and double-label assignment: alternate within each style.
    per_style_idx: Counter = Counter()
    double_left = n_double
    key, shown = [], []
    style_totals = Counter(r["style"] for r in chosen)
    for i, r in enumerate(chosen):
        k = per_style_idx[r["style"]]
        per_style_idx[r["style"]] += 1
        split = "dev" if k < round(style_totals[r["style"]] * dev_frac) else "test"
        label_id = f"L{i + 1:04d}"
        key.append({"label_id": label_id, "reply_id": r["reply_id"], "item_id": r["item_id"],
                    "style": r["style"], "source_model": r["source_model"], "split": split, "double": False})
        it, m = items[r["item_id"]], msgs[r["message_id"]]
        shown.append({"label_id": label_id, "question": it["question"], "answer": it["final_answer"],
                      "student_message": m["text"], "reply": r["text"]})
    # Double-label subset: every k-th item, so it is spread over styles and both splits.
    step = max(1, len(key) // max(1, n_double))
    for rec in key[::step]:
        if double_left == 0:
            break
        rec["double"] = True
        double_left -= 1
    return shown, key


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/calibration.yaml")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    paths, lcfg = cfg["paths"], cfg["label_set"]

    pool = read_jsonl(paths["reply_pool"])
    items = {r["item_id"]: r for r in read_jsonl(paths["items"])}
    msgs = {r["message_id"]: r for r in read_jsonl(paths["messages"])}
    shown, key = build_label_set(pool, items, msgs, lcfg)

    out = Path(paths["labels_dir"])
    write_jsonl(out / "label_set.jsonl", shown)
    write_jsonl(out / "label_key.jsonl", key)
    with open(out / "label_set.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[*shown[0].keys(), "scaffolding", "helpfulness", "notes"])
        w.writeheader()
        for r in shown:
            w.writerow({**r, "scaffolding": "", "helpfulness": "", "notes": ""})
    print(f"[label_set] {len(shown)} items; splits {dict(Counter(k['split'] for k in key))}; "
          f"double-labeled {sum(k['double'] for k in key)}; styles {dict(Counter(k['style'] for k in key))}")


if __name__ == "__main__":
    main()
