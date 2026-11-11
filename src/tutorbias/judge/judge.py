"""The scaffolding judge (stage 1).

Scores a tutor reply on two axes using an LLM with a rubric prompt:
  scaffolding  -1 (gives the answer) .. +1 (useful hint or guiding question)
  helpfulness   1 .. 5

How a score is computed (the "logprob" mode, used for the reward and the audit):
  1. Pre-fill the assistant turn with "SCAFFOLDING: " and read the next-token probability of the
     labels "0", "1", "2" (rubric levels), mapped to -1, 0, +1.
  2. The expected value  E[s] = sum_k p_k * s_k  is a continuous score in [-1, 1]. This is smoother
     than a single sampled label (useful as an RL reward) and more sensitive to small shifts (useful
     for detecting bias: a dialect can move probability mass without flipping the top label).
     This follows G-Eval (Liu et al., 2023).
  3. Append the most likely scaffolding label and "\nHELPFULNESS: ", read p over "1".."5", E[h].
  4. Optional (E4): append the helpfulness label and "\nREASON:" and generate the rationale. Because the
     rationale comes *after* the scores on the same greedy path, it is a post-hoc explanation of exactly
     those scores, which is what the faithfulness analysis needs.

"generate" mode (for API models without log-probs or pre-fill): sample the full response once at
temperature 0 and parse it; probabilities become one-hot.

The rubric text is versioned in judge/prompts/rubric_<version>.txt. Every output row records the rubric
version and a hash of the rubric text, so scores from different rubric iterations never get mixed.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass, field
from importlib import resources
from typing import Any

from ..models.backends import Backend, Conv

SCAFF_LABELS = {"0": -1, "1": 0, "2": 1}  # rubric digit -> scaffolding score
HELP_LABELS = {str(k): k for k in range(1, 6)}
_PARSE_RE = re.compile(r"SCAFFOLDING:\s*([012])\s*\n\s*HELPFULNESS:\s*([1-5])(?:\s*\n\s*REASON:\s*(.*))?", re.S)


def load_rubric(version: str) -> str:
    return resources.files("tutorbias.judge").joinpath(f"prompts/rubric_{version}.txt").read_text().strip()


def build_conv(rubric: str, row: dict[str, Any]) -> Conv:
    """The judge sees: problem, correct answer, the student's message, the tutor's reply.

    The student message is the only thing that changes across counterfactual variants in the audit.
    """
    user = (
        f"PROBLEM:\n{row['question']}\n\n"
        f"CORRECT FINAL ANSWER (reference only): {row['answer']}\n\n"
        f"STUDENT MESSAGE:\n{row['student_message']}\n\n"
        f"TUTOR REPLY:\n{row['reply']}"
    )
    return [{"role": "system", "content": rubric}, {"role": "user", "content": user}]


def _normalize(logprobs: dict[str, float]) -> tuple[dict[str, float], float]:
    """Softmax over the valid labels. Also returns the raw probability mass those labels had."""
    mass = sum(math.exp(v) for v in logprobs.values() if v > float("-inf"))
    if mass == 0:
        # No valid label in the top-k: uniform, flagged by mass == 0.
        return {k: 1 / len(logprobs) for k in logprobs}, 0.0
    return {k: (math.exp(v) / mass if v > float("-inf") else 0.0) for k, v in logprobs.items()}, mass


@dataclass
class JudgeScore:
    scaffolding: float  # expected score in [-1, 1]
    scaffolding_label: int  # argmax in {-1, 0, 1}
    p_scaffolding: dict[str, float]  # keyed by score as string: "-1", "0", "1"
    helpfulness: float  # expected score in [1, 5]
    helpfulness_label: int
    p_helpfulness: dict[str, float]
    valid_mass_scaffolding: float  # probability on valid labels before renormalizing (format health check)
    valid_mass_helpfulness: float
    rationale: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScaffoldingJudge:
    def __init__(self, backend: Backend, rubric_version: str = "v1", mode: str = "logprob"):
        if mode == "logprob" and not backend.supports_prefill:
            raise ValueError(f"Backend {backend.name} can't pre-fill; use mode='generate'.")
        self.backend = backend
        self.rubric_version = rubric_version
        self.rubric = load_rubric(rubric_version)
        self.rubric_hash = hashlib.sha256(self.rubric.encode()).hexdigest()[:12]
        self.mode = mode

    @property
    def meta(self) -> dict[str, Any]:
        return {"judge_model": self.backend.name, "rubric_version": self.rubric_version,
                "rubric_hash": self.rubric_hash, "mode": self.mode}

    def score(self, rows: list[dict[str, Any]], explain: bool = False) -> list[JudgeScore]:
        convs = [build_conv(self.rubric, r) for r in rows]
        if self.mode == "generate":
            return self._score_generate(convs)
        return self._score_logprob(convs, explain)

    @staticmethod
    def _grouped(convs: list[Conv], prefixes: list[str], fn) -> list[Any]:
        """Run `fn(convs_subset, prefix)` once per distinct prefix (keeps vLLM batches large), then
        put results back in the original order."""
        out: list[Any] = [None] * len(convs)
        for pre in dict.fromkeys(prefixes):
            idx = [i for i, p in enumerate(prefixes) if p == pre]
            for i, res in zip(idx, fn([convs[i] for i in idx], pre)):
                out[i] = res
        return out

    def _score_logprob(self, convs: list[Conv], explain: bool) -> list[JudgeScore]:
        p1 = "SCAFFOLDING: "
        s_lp = self.backend.next_token_logprobs(convs, p1, list(SCAFF_LABELS))
        s_norm = [_normalize(lp) for lp in s_lp]
        s_top = [max(p, key=p.get) for p, _ in s_norm]

        p2 = [f"{p1}{d}\nHELPFULNESS: " for d in s_top]
        h_lp = self._grouped(convs, p2, lambda cs, pre: self.backend.next_token_logprobs(cs, pre, list(HELP_LABELS)))
        h_norm = [_normalize(lp) for lp in h_lp]
        h_top = [max(p, key=p.get) for p, _ in h_norm]

        rationales: list[str | None] = [None] * len(convs)
        if explain:
            p3 = [f"{a}{h}\nREASON:" for a, h in zip(p2, h_top)]
            gens = self._grouped(convs, p3, lambda cs, pre: self.backend.generate(cs, pre, max_tokens=120))
            rationales = [g[0].strip() for g in gens]

        out = []
        for (ps, ms), st, (ph, mh), ht, rat in zip(s_norm, s_top, h_norm, h_top, rationales):
            out.append(JudgeScore(
                scaffolding=sum(ps[d] * v for d, v in SCAFF_LABELS.items()),
                scaffolding_label=SCAFF_LABELS[st],
                p_scaffolding={str(SCAFF_LABELS[d]): round(ps[d], 6) for d in SCAFF_LABELS},
                helpfulness=sum(ph[d] * v for d, v in HELP_LABELS.items()),
                helpfulness_label=HELP_LABELS[ht],
                p_helpfulness={d: round(ph[d], 6) for d in HELP_LABELS},
                valid_mass_scaffolding=round(ms, 6),
                valid_mass_helpfulness=round(mh, 6),
                rationale=rat,
                meta=self.meta,
            ))
        return out

    def _score_generate(self, convs: list[Conv]) -> list[JudgeScore]:
        texts = [g[0] for g in self.backend.generate(convs, max_tokens=200, temperature=0.0)]
        out = []
        for t in texts:
            m = _PARSE_RE.search(t)
            if not m:
                # Unparseable: neutral score, zero valid mass so it is easy to filter and count.
                out.append(JudgeScore(0.0, 0, {"-1": 0, "0": 1, "1": 0}, 3.0, 3, {d: float(d == "3") for d in HELP_LABELS},
                                      0.0, 0.0, rationale=t.strip(), meta={**self.meta, "parse_error": True}))
                continue
            s, h = SCAFF_LABELS[m.group(1)], int(m.group(2))
            out.append(JudgeScore(
                float(s), s, {str(v): float(v == s) for v in SCAFF_LABELS.values()},
                float(h), h, {d: float(int(d) == h) for d in HELP_LABELS}, 1.0, 1.0,
                rationale=(m.group(3) or "").strip() or None, meta=self.meta,
            ))
        return out
