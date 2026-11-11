"""Name-swap axis: the same SAE message, introduced by students with different names."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

NAME_PREFIX = "Hi, I'm {name}. "


def load_names(path: str | Path) -> dict[str, dict[str, Any]]:
    with open(path) as f:
        return yaml.safe_load(f)["groups"]


def pick_names(groups: dict[str, dict[str, Any]], per_gender: int, seed: int) -> list[dict[str, str]]:
    """Choose a fixed set of names per group, balanced by gender.

    The same names are used for every message, so name effects aren't confounded with which
    question a name happened to land on.
    """
    rng = random.Random(seed)
    chosen = []
    for group, spec in groups.items():
        for gender in ("female", "male"):
            pool = list(spec[gender])
            for name in rng.sample(pool, min(per_gender, len(pool))):
                chosen.append({"name": name, "name_group": group, "gender": gender, "name_status": spec["status"]})
    return chosen


def add_name(text: str, name: str) -> str:
    return NAME_PREFIX.format(name=name) + text
