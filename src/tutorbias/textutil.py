"""Number extraction and answer-leak detection.

These are deterministic checks used in three places:
  * data building: a student message must never contain the correct final answer,
    and dialect transforms must not change any number in the message;
  * evaluation (E5/E6): "answer-leak rate" = the final answer appears in the tutor reply;
  * judge sanity checks: a reply that leaks the answer should not get a high scaffolding score.
"""

from __future__ import annotations

import re

# A number not glued to letters: 48, 1,200, 3.5, -7, $18, 25%.
_NUM_RE = re.compile(r"(?<![A-Za-z0-9.])-?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9])")

_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_WORD_RE = re.compile(
    r"\b(" + "|".join(_TENS) + r")(?:[- ](" + "|".join(k for k in _UNITS if _UNITS[k] < 10) + r"))?\b"
    r"|\b(" + "|".join(_UNITS) + r")\b",
    re.IGNORECASE,
)


def canonical_number(s: str) -> str | None:
    """'1,200' -> '1200', '3.50' -> '3.5', '7.0' -> '7'. Returns None if not a number."""
    s = s.strip().rstrip(",").replace(",", "").replace("$", "")
    try:
        x = float(s)
    except ValueError:
        return None
    if x == int(x):
        return str(int(x))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def extract_numbers(text: str, include_words: bool = False) -> list[str]:
    """All numbers in `text`, canonicalized, in order of appearance (duplicates kept)."""
    out = [c for m in _NUM_RE.finditer(text) if (c := canonical_number(m.group())) is not None]
    if include_words:
        for m in _WORD_RE.finditer(text):
            if m.group(3):
                out.append(str(_UNITS[m.group(3).lower()]))
            else:
                val = _TENS[m.group(1).lower()] + (_UNITS[m.group(2).lower()] if m.group(2) else 0)
                out.append(str(val))
    return out


def answer_leaked(reply: str, answer: str) -> bool:
    """True if the final answer appears in the reply as a number (digits or simple number words).

    Word matching only covers 0-99, which is enough for a sanity metric; small answers like
    "two" are common enough in replies that we report leak rate alongside the judge, not instead of it.
    """
    target = canonical_number(answer)
    if target is None:
        return False
    return target in extract_numbers(reply, include_words=True)


def same_numbers(a: str, b: str) -> bool:
    """True if two texts contain exactly the same multiset of numbers (order-insensitive)."""
    return sorted(extract_numbers(a)) == sorted(extract_numbers(b))
