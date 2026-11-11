"""Generate the fixed pool of tutor replies used in the judge calibration and reward-model audit.

The audit needs replies that span the whole scaffolding range, from handing over the answer to a
pure hint, so we prompt models with explicit style instructions. The *intended* style is only a
sampling device: models (especially 1.5B) don't always follow it, which is why replies are labeled
by a human and scored by the judge rather than trusting the style tag.

Replies are written to the SAE base message. In the audit the *same* reply is then paired with every
counterfactual version of the student's message, so any score difference comes from the student text.

  python -m tutorbias.replies.generate --config configs/replies.yaml --model qwen1.5b
  python -m tutorbias.replies.generate --config configs/replies.yaml --select   # build the ~1500 pool
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from ..io import load_config, read_jsonl, stable_seed, write_jsonl
from ..models.backends import load_backend
from ..textutil import answer_leaked
from ..tracking import init_run

TUTOR_SYSTEM = "You are a friendly math tutor helping a middle-school student with a word problem."

# Style instructions. Kept short so they read like a plausible system prompt, not a jailbreak.
STYLES: dict[str, str] = {
    "direct_answer": "Tell the student the final answer directly, with a one-sentence explanation.",
    "worked_solution": "Show the student the full step-by-step solution, ending with the final answer.",
    "hint_with_leak": "Give the student a hint for the next step, and mention the final answer so they can check their work.",
    "partial_work": "Do the next one or two calculations for the student, then let them finish the last step.",
    "targeted_hint": "Give one short hint about the student's next step. Do not do any calculations for them and do not reveal the answer.",
    "socratic": "Ask the student one or two guiding questions that lead them to the next step. Do not give the answer or do calculations.",
    "vague": "Reply with brief general encouragement and a generic question. Do not mention anything specific about this problem.",
    "misleading": "Give a hint that sounds confident but points the student toward a wrong method. Do not reveal the answer.",
}


def tutor_conv(question: str, student_message: str, style: str | None) -> list[dict[str, str]]:
    system = TUTOR_SYSTEM + (" " + STYLES[style] if style else "")
    user = f"Here is the problem I'm working on:\n{question}\n\n{student_message}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def generate(cfg: dict[str, Any], model_key: str) -> None:
    mcfg = cfg["models"][model_key]
    pcfg = cfg["paths"]
    items = {r["item_id"]: r for r in read_jsonl(pcfg["items"])}
    msgs = read_jsonl(pcfg["messages"])
    run = init_run("replies-generate", {**cfg, "model_key": model_key})
    backend = load_backend(mcfg["model"])
    g = cfg["generation"]

    rows = []
    for style in cfg["styles"]:
        convs = [tutor_conv(items[m["item_id"]]["question"], m["text"], style) for m in msgs]
        outs = backend.generate(convs, max_tokens=g["max_tokens"], temperature=g["temperature"],
                                top_p=g["top_p"], seed=stable_seed(model_key, style, base=cfg["seed"]))
        for m, o in zip(msgs, outs):
            text = o[0].strip()
            it = items[m["item_id"]]
            rows.append({
                "reply_id": f"{m['item_id']}::{model_key}::{style}",
                "item_id": m["item_id"], "message_id": m["message_id"], "message_type": m["message_type"],
                "source_model": model_key, "style": style, "text": text,
                "n_words": len(text.split()), "answer_leaked": answer_leaked(text, it["final_answer"]),
            })
        print(f"[replies] {model_key} / {style}: {len(msgs)} replies")
    path = Path(pcfg["out_dir"]) / f"replies_{model_key}.jsonl"
    write_jsonl(path, rows)
    leak = Counter((r["style"], r["answer_leaked"]) for r in rows)
    run.summary({f"leak_rate/{s}": leak[(s, True)] / max(1, leak[(s, True)] + leak[(s, False)]) for s in cfg["styles"]})
    run.finish()


def select_pool(cfg: dict[str, Any]) -> None:
    """Pick the audit pool: `per_item` replies per question, styles and models balanced, empties dropped."""
    pcfg, scfg = cfg["paths"], cfg["selection"]
    rng = random.Random(cfg["seed"])
    rows = [r for f in sorted(Path(pcfg["out_dir"]).glob("replies_*.jsonl")) for r in read_jsonl(f)]
    rows = [r for r in rows if scfg["min_words"] <= r["n_words"] <= scfg["max_words"]]
    by_item: dict[str, list[dict]] = {}
    for r in rows:
        by_item.setdefault(r["item_id"], []).append(r)

    style_count: Counter = Counter()
    pool = []
    for item_id in sorted(by_item):
        cands = by_item[item_id]
        rng.shuffle(cands)
        chosen, used_styles = [], set()
        # Greedy: least-used style overall first, at most one reply per style per question.
        for r in sorted(cands, key=lambda r: style_count[r["style"]]):
            if r["style"] in used_styles:
                continue
            chosen.append(r)
            used_styles.add(r["style"])
            style_count[r["style"]] += 1
            if len(chosen) == scfg["per_item"]:
                break
        pool += chosen
    path = Path(pcfg["out_dir"]) / "reply_pool.jsonl"
    write_jsonl(path, pool)
    print(f"[replies] pool: {len(pool)} replies; styles {dict(style_count)}; "
          f"models {dict(Counter(r['source_model'] for r in pool))}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/replies.yaml")
    p.add_argument("--model", help="model key from the config to generate with")
    p.add_argument("--select", action="store_true", help="build reply_pool.jsonl from generated replies")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.select:
        select_pool(cfg)
    else:
        generate(cfg, a.model)


if __name__ == "__main__":
    main()
