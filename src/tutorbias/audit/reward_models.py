"""Score audit pairs with off-the-shelf reward models from RewardBench.

  python -m tutorbias.audit.reward_models --config configs/audit.yaml --rm skywork_v2

These are the kind of reward models people plug into RLHF pipelines. They are *audited only* (never
used for training here). Each RM sees the same conversation the tutor saw:
  system: tutor role   user: problem + student message   assistant: tutor reply
and returns one scalar reward. Scores are resumable, flushed per batch, like the judge.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from ..io import append_jsonl, iter_jsonl, load_config, sha256_file
from ..replies.generate import tutor_conv
from ..tracking import init_run


def rm_conversation(row: dict[str, Any]) -> list[dict[str, str]]:
    """Tutor conversation without a style instruction, plus the reply as the assistant turn."""
    return tutor_conv(row["question"], row["student_message"], style=None) + [
        {"role": "assistant", "content": row["reply"]}
    ]


class SequenceClassifierRM:
    """Reward models exposed as AutoModelForSequenceClassification with one output logit
    (Skywork-Reward, most RewardBench classifiers). ArmoRM uses remote code and returns `.score`."""

    def __init__(self, model: str, max_length: int = 4096, batch_size: int = 8, trust_remote_code: bool = False,
                 score_attr: str | None = None, **_: Any):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.name = model
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, torch_dtype=torch.bfloat16, device_map="cuda" if torch.cuda.is_available() else None,
            trust_remote_code=trust_remote_code, attn_implementation="sdpa",
        ).eval()
        self.max_length = max_length
        self.batch_size = batch_size
        self.score_attr = score_attr

    def score(self, rows: list[dict[str, Any]]) -> list[float]:
        texts = [self.tok.apply_chat_template(rm_conversation(r), tokenize=False) for r in rows]
        out: list[float] = []
        for i in range(0, len(texts), self.batch_size):
            enc = self.tok(texts[i : i + self.batch_size], return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_length, add_special_tokens=False).to(self.model.device)
            with self.torch.no_grad():
                res = self.model(**enc)
            vals = getattr(res, self.score_attr) if self.score_attr else res.logits[:, 0]
            out += vals.float().reshape(-1).cpu().tolist()
        return out


def check_batching(rm: SequenceClassifierRM, rows: list[dict[str, Any]], tol: float = 0.05) -> None:
    """Padding bugs in reward models silently change scores (the RM reads the wrong last token).
    Score a few rows batched and one at a time; stop if they disagree."""
    if rm.batch_size == 1 or len(rows) < 2:
        return
    batched = rm.score(rows)
    bs, rm.batch_size = rm.batch_size, 1
    single = rm.score(rows)
    rm.batch_size = bs
    worst = max(abs(a - b) for a, b in zip(batched, single))
    if worst > tol:
        raise RuntimeError(f"{rm.name}: batched vs single scores differ by {worst:.3f}. "
                           "Set batch_size: 1 for this RM in configs/audit.yaml.")
    print(f"[rm] batching check passed (max diff {worst:.4f})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/audit.yaml")
    p.add_argument("--rm", required=True, help="key under reward_models in the config")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    spec = cfg["reward_models"][a.rm]
    out = Path(cfg["paths"]["scores_dir"]) / f"{a.rm}.jsonl"

    done = {r["id"] for r in iter_jsonl(out)} if out.exists() else set()
    rows = [r for r in iter_jsonl(cfg["paths"]["pairs"]) if r["pair_id"] not in done]
    if a.limit:
        rows = rows[: a.limit]
    print(f"[rm:{a.rm}] {len(done)} done, {len(rows)} to go")
    if not rows:
        return
    run = init_run("audit-rm", {**cfg, "rm": a.rm, "pairs_sha": sha256_file(cfg["paths"]["pairs"])})
    rm = SequenceClassifierRM(**spec)
    check_batching(rm, rows[:8])
    # Sort by length so batches pad less; output order doesn't matter (keyed by id).
    rows.sort(key=lambda r: len(r["reply"]) + len(r["student_message"]))
    chunk, t0 = cfg.get("flush_every", 512), time.time()
    for i in range(0, len(rows), chunk):
        batch = rows[i : i + chunk]
        scores = rm.score(batch)
        append_jsonl(out, ({"id": r["pair_id"], "scorer": a.rm, "reward": s} for r, s in zip(batch, scores)))
        n = i + len(batch)
        run.log({"scored": n, "rows_per_sec": n / (time.time() - t0)})
        print(f"[rm:{a.rm}] {n}/{len(rows)}  {n / (time.time() - t0):.1f} rows/s")
    run.finish()


if __name__ == "__main__":
    main()
