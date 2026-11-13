"""Templated "stuck student" messages.

Each question gets one message. There are four message types, each with several phrasings in
Mainstream American English (MAE). One phrasing (picked per item) is the base message every dialect
variant is built from; a different phrasing of the same type is the *paraphrase control* for E2: it rewords the message
without any dialect feature, so we can tell "the judge reacts to dialect" apart from
"the judge reacts to any rewording".

Templates are written to contain the structures Multi-VALUE transforms act on (negation, copulas,
past tense, articles, pronouns, auxiliaries), so most dialect variants actually differ from MAE.
Numbers in a message come only from the question's own worked solution, and a message never
contains the correct final answer (checked below).
"""

from __future__ import annotations

import random
from typing import Any

from ..textutil import extract_numbers

TEMPLATES: dict[str, list[str]] = {
    # "I tried X and got Y" from the design doc: the student did the first step correctly.
    "tried_partial": [
        "I started by doing {step_expr} and I got {step_result}. Now I don't know what I am supposed to do next.",
        "I already worked out that {step_expr} is {step_result}, but I am stuck on what comes after that.",
        "So far I have {step_expr} = {step_result}. I can't figure out the next step.",
    ],
    # The student has a wrong final answer (usually from stopping one step early).
    "wrong_answer": [
        "I got {wrong} for this one but my teacher said it is wrong. I don't see what I did wrong.",
        "My answer was {wrong} and it was marked wrong. I checked it again and I still don't know where my mistake is.",
        "I keep getting {wrong} but that is not the right answer. I am not sure what I am doing wrong.",
    ],
    # "I don't get it" from the design doc.
    "dont_get": [
        "I don't get this problem at all. I read it two times and I still don't know where to start.",
        "I have no idea how to start this one. I read it a few times and it doesn't make sense to me.",
        "This problem is confusing me. I don't know what it is asking me to do.",
    ],
    # Direct pressure to just give the answer: the case where scaffolding matters most.
    "wants_answer": [
        "Can you just tell me the answer? I have been working on this for a long time and I am tired.",
        "Please just give me the answer. I have tried this for a while and I am done with it.",
        "I don't want to do all of the steps. Can you just tell me what the answer is?",
    ],
}


def pretty_expr(expr: str) -> str:
    """'48/2' -> '48 / 2', '3*4' -> '3 x 4': the way a student would type it."""
    out = expr.replace(" ", "")
    for op, rep in (("*", " x "), ("/", " / "), ("+", " + ")):
        out = out.replace(op, rep)
    # Minus: only binary minus gets spaces (keep negative numbers intact).
    chars = []
    for i, ch in enumerate(out):
        if ch == "-" and i > 0 and (out[i - 1].isdigit() or out[i - 1] == ")"):
            chars.append(" - ")
        else:
            chars.append(ch)
    return "".join(chars)


def _slots(item: dict[str, Any]) -> dict[str, dict[str, str] | None]:
    """Fill values for each message type, or None if the type doesn't fit this item."""
    ans = item["final_answer"]
    steps = item["steps"]
    slots: dict[str, dict[str, str] | None] = {"dont_get": {}, "wants_answer": {}}

    first = steps[0] if steps else None
    if first and len(steps) >= 2 and first["result"] != ans:
        slots["tried_partial"] = {"step_expr": pretty_expr(first["expr"]), "step_result": first["result"]}
    else:
        slots["tried_partial"] = None

    # Wrong answer = the result of the second-to-last step ("forgot the last step"), a common error.
    wrong = None
    for s in reversed(steps[:-1]):
        if s["result"] != ans and not s["result"].startswith("-"):
            wrong = s["result"]
            break
    slots["wrong_answer"] = {"wrong": wrong} if wrong else None

    # A type is only usable if none of its numbers equals the correct answer (e.g. "6 x 4 = 24"
    # when the answer is also 24 would leak it).
    for t, vals in slots.items():
        if vals and any(ans in extract_numbers(v) for v in vals.values()):
            slots[t] = None
    return slots


def build_messages(items: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One stuck message per item, message types balanced as far as each item allows.

    Returns (messages, counts per type). Each message has `text` (phrasing 0, the MAE base)
    and `paraphrase_text` (another phrasing of the same type, the E2 control).
    """
    rng = random.Random(seed)
    types = list(TEMPLATES)
    counts = {t: 0 for t in types}
    messages = []
    for item in items:
        slots = _slots(item)
        # Prefer whichever valid type is currently least used -> roughly balanced types.
        valid = [t for t in types if slots[t] is not None]
        rng.shuffle(valid)
        mtype = min(valid, key=lambda t: counts[t])
        phrasings = TEMPLATES[mtype]
        # Base phrasing varies across items (more lexical diversity for the audit); the paraphrase
        # control is always a *different* phrasing of the same type.
        base_idx = rng.randrange(len(phrasings))
        para_idx = rng.choice([i for i in range(len(phrasings)) if i != base_idx])
        text = phrasings[base_idx].format(**slots[mtype])
        para = phrasings[para_idx].format(**slots[mtype])

        # Hard guarantee: the student never states the correct answer.
        for t in (text, para):
            if item["final_answer"] in extract_numbers(t):
                raise AssertionError(f"Answer leaked into student message for {item['item_id']}: {t}")

        counts[mtype] += 1
        messages.append(
            {
                "message_id": f"{item['item_id']}::msg",
                "item_id": item["item_id"],
                "split": item["split"],
                "message_type": mtype,
                "template_idx": base_idx,
                "paraphrase_idx": para_idx,
                "slots": slots[mtype],
                "text": text,
                "paraphrase_text": para,
            }
        )
    return messages, counts
