"""Load GSM8K, parse its worked solutions, filter, and make the three item splits.

Splits (disjoint by construction):
  rl_train : GRPO training prompts            (from GSM8K train)
  audit    : judge calibration + RM audit      (from GSM8K train, disjoint from rl_train)
  test     : held-out tutor evaluation (E5-E7) (from GSM8K test)

Keeping the audit questions out of RL training means the judge is never calibrated or audited
on the exact prompts the tutor is optimized on.
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import requests

from ..textutil import canonical_number, extract_numbers

GSM8K_RAW_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{split}.jsonl"

# GSM8K solutions mark each arithmetic step as <<48/2=24>>.
_STEP_RE = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def load_gsm8k(split: str, cache_dir: str | Path = "data/raw") -> list[dict[str, str]]:
    """Return raw GSM8K rows ({question, answer}) for 'train' or 'test'.

    Tries Hugging Face `datasets` first, then the original GitHub release (same content, MIT license).
    The file is cached under data/raw/ so later runs work offline.
    """
    import json

    cache = Path(cache_dir) / f"gsm8k_{split}.jsonl"
    if cache.exists():
        return [json.loads(line) for line in cache.read_text().splitlines() if line.strip()]
    rows: list[dict[str, str]]
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split=split)
        rows = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    except Exception as e:  # no `datasets`, no HF access, etc.
        print(f"[gsm8k] HF load failed ({type(e).__name__}); falling back to GitHub raw file.")
        resp = requests.get(GSM8K_RAW_URL.format(split=split), timeout=60)
        resp.raise_for_status()
        rows = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def parse_solution(answer_field: str) -> dict[str, Any]:
    """Split GSM8K's answer field into a clean solution, arithmetic steps, and the final answer."""
    body, _, final = answer_field.rpartition("####")
    steps = []
    for expr, result in _STEP_RE.findall(body):
        res = canonical_number(result)
        if res is None:
            continue
        steps.append({"expr": expr.strip(), "result": res})
    solution = _STEP_RE.sub("", body).strip()
    return {"solution": solution, "steps": steps, "final_answer": canonical_number(final) or final.strip()}


def make_item(raw: dict[str, str], source_split: str, idx: int) -> dict[str, Any]:
    parsed = parse_solution(raw["answer"])
    return {
        "item_id": f"gsm8k-{source_split}-{idx:05d}",
        "source": "gsm8k",
        "source_split": source_split,
        "question": raw["question"].strip(),
        **parsed,
    }


def keep_item(item: dict[str, Any], min_steps: int, max_steps: int) -> tuple[bool, str]:
    """Filter rules. Returns (keep, reason) so the datasheet can report how many were dropped and why."""
    ans = item["final_answer"]
    if canonical_number(ans) is None:
        return False, "non_numeric_answer"
    if "." in ans or ans.startswith("-"):
        # Integer, non-negative answers keep leak detection and the "wrong answer" templates simple.
        return False, "non_integer_answer"
    if not (min_steps <= len(item["steps"]) <= max_steps):
        return False, "step_count"
    if ans in extract_numbers(item["question"]):
        # If the answer is already printed in the question, "answer leak" can't be measured.
        return False, "answer_in_question"
    return True, "ok"


def build_splits(cfg: dict[str, Any], cache_dir: str | Path = "data/raw") -> tuple[dict[str, list], dict]:
    """Build rl_train / audit / test item lists. Returns (splits, filter_stats)."""
    rng = random.Random(cfg["seed"])
    stats: dict[str, dict[str, int]] = {}
    pools: dict[str, list[dict[str, Any]]] = {}
    for source_split in ("train", "test"):
        raw = load_gsm8k(source_split, cache_dir)
        kept, counts = [], {}
        for i, r in enumerate(raw):
            item = make_item(r, source_split, i)
            ok, reason = keep_item(item, cfg["min_steps"], cfg["max_steps"])
            counts[reason] = counts.get(reason, 0) + 1
            if ok:
                kept.append(item)
        rng.shuffle(kept)
        pools[source_split] = kept
        stats[source_split] = {"raw": len(raw), **counts}

    n_train, n_audit, n_test = cfg["n_rl_train"], cfg["n_audit"], cfg["n_test"]
    if n_train + n_audit > len(pools["train"]) or n_test > len(pools["test"]):
        raise ValueError(f"Not enough items after filtering: {stats}")
    splits = {
        "rl_train": pools["train"][:n_train],
        "audit": pools["train"][n_train : n_train + n_audit],
        "test": pools["test"][:n_test],
    }
    for name, items in splits.items():
        for it in items:
            it["split"] = name
    return splits, stats
