"""Build counterfactual audit pairs: every reply in the pool x every version of its student's message.

  python -m tutorbias.audit.pairs --config configs/audit.yaml

Within one reply, the problem and the reply text are identical across pairs; only the student's
message changes (dialect, paraphrase, or name). So any score difference between two pairs that share a
reply_id is caused by the student text alone.

Variants whose dialect transform changed a number (numbers_ok = False) are dropped here, because the
student would then be describing different math.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from ..io import load_config, read_jsonl, write_jsonl


def build_pairs(pool: list[dict], variants: list[dict], items: dict[str, dict]) -> tuple[list[dict], dict]:
    by_msg: dict[str, list[dict]] = {}
    for v in variants:
        by_msg.setdefault(v["message_id"], []).append(v)
    pairs, dropped = [], Counter()
    for r in pool:
        it = items[r["item_id"]]
        for v in by_msg.get(r["message_id"], []):
            if not v["numbers_ok"]:
                dropped[v["group"]] += 1
                continue
            pairs.append({
                "pair_id": f"{r['reply_id']}||{v['variant_id']}",
                "reply_id": r["reply_id"], "variant_id": v["variant_id"], "item_id": r["item_id"],
                "message_type": v["message_type"], "style": r["style"], "source_model": r["source_model"],
                "axis": v["axis"], "group": v["group"], "name": v.get("name"), "name_group": v.get("name_group"),
                "changed": v["changed"], "features": v["features"],
                "question": it["question"], "answer": it["final_answer"],
                "student_message": v["text"], "reply": r["text"],
            })
    return pairs, {"n_pairs": len(pairs), "dropped_numbers_changed": dict(dropped),
                   "by_group": dict(Counter(p["group"] for p in pairs))}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/audit.yaml")
    a = p.parse_args(argv)
    cfg: dict[str, Any] = load_config(a.config)
    paths = cfg["paths"]
    items = {r["item_id"]: r for r in read_jsonl(paths["items"])}
    variants = [v for v in read_jsonl(paths["variants"]) if v["split"] == "audit"]
    pairs, stats = build_pairs(read_jsonl(paths["reply_pool"]), variants, items)
    write_jsonl(paths["pairs"], pairs)
    print(stats)
    Path(paths["pairs"]).with_suffix(".stats.json").write_text(__import__("json").dumps(stats, indent=2))


if __name__ == "__main__":
    main()
