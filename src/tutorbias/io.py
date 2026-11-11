"""Small file helpers shared by every stage. All intermediate data is JSON Lines."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. Every stage takes one config file so runs are reproducible."""
    with open(path) as f:
        return yaml.safe_load(f)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows atomically (temp file + rename) so a preempted job never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    """Append rows. Used for resumable scoring: each finished batch is flushed right away."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def sha256_file(path: str | Path) -> str:
    """Content hash, logged with every run so results can be traced to exact inputs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def stable_seed(*parts: Any, base: int = 0) -> int:
    """Deterministic per-item seed (Python's hash() is salted per process, so don't use it)."""
    key = "::".join(str(p) for p in parts)
    return (int(hashlib.md5(key.encode()).hexdigest()[:8], 16) + base) % (2**31 - 1)
