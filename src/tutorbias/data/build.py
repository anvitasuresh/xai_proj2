"""Stage 1 data CLI. Run the steps in order (see Makefile / slurm/data.sbatch):

  python -m tutorbias.data.build items     --config configs/data.yaml
  python -m tutorbias.data.build messages  --config configs/data.yaml
  python -m tutorbias.data.build variants  --config configs/data.yaml [--shard 0 --num-shards 4]
  python -m tutorbias.data.build merge     --config configs/data.yaml
  python -m tutorbias.data.build check-sample --config configs/data.yaml   # CSV for hand-checking naturalness

Outputs go to data/processed/. Every step writes a small *_stats.json next to its output; those
numbers feed DATASHEET.md.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
from pathlib import Path
from typing import Any

from ..io import load_config, read_jsonl, sha256_file, stable_seed, write_jsonl
from ..textutil import same_numbers
from ..tracking import init_run
from .dialects import make_transformer
from .gsm8k import build_splits
from .names import add_name, load_names, pick_names
from .stuck import build_messages

SPLITS = ("rl_train", "audit", "test")


def _out(cfg: dict[str, Any]) -> Path:
    return Path(cfg["paths"]["processed"])


def _write_stats(path: Path, stats: dict[str, Any]) -> None:
    path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def cmd_items(cfg: dict[str, Any]) -> None:
    splits, stats = build_splits(cfg["items"], cache_dir=cfg["paths"]["raw"])
    out = _out(cfg)
    for name, items in splits.items():
        write_jsonl(out / f"items_{name}.jsonl", items)
    stats["split_sizes"] = {k: len(v) for k, v in splits.items()}
    _write_stats(out / "items_stats.json", stats)


def cmd_messages(cfg: dict[str, Any]) -> None:
    out = _out(cfg)
    all_counts = {}
    for name in SPLITS:
        items = read_jsonl(out / f"items_{name}.jsonl")
        msgs, counts = build_messages(items, seed=stable_seed("messages", name, base=cfg["seed"]))
        write_jsonl(out / f"messages_{name}.jsonl", msgs)
        all_counts[name] = counts
    _write_stats(out / "messages_stats.json", {"message_types": all_counts})


def _variant(msg: dict[str, Any], group: str, axis: str, text: str, features: dict[str, int], **extra) -> dict:
    vid = f"{msg['item_id']}::{group}" + (f"::{extra['name']}" if extra.get("name") else "")
    return {
        "variant_id": vid,
        "message_id": msg["message_id"],
        "item_id": msg["item_id"],
        "split": msg["split"],
        "message_type": msg["message_type"],
        "axis": axis,  # dialect | control | name
        "group": group,
        "text": text,
        "base_text": msg["text"],
        "changed": text != msg["text"],
        "features": features,
        "n_feature_applications": sum(features.values()),
        "numbers_ok": same_numbers(text, msg["text"]),
        **extra,
    }


def cmd_variants(cfg: dict[str, Any], shard: int, num_shards: int) -> None:
    """Build all variants for every message. Shardable because Multi-VALUE is slow (stanza coref)."""
    vcfg = cfg["variants"]
    out = _out(cfg)
    run = init_run("data-variants", cfg, tags=[f"shard{shard}"])

    msgs = [m for name in SPLITS for m in read_jsonl(out / f"messages_{name}.jsonl")]
    msgs = msgs[shard::num_shards]

    transformers = {g: make_transformer(g, spec) for g, spec in vcfg["dialect_groups"].items()}
    names = pick_names(load_names(vcfg["names"]["file"]), vcfg["names"]["per_gender"], cfg["seed"])
    name_splits = set(vcfg["names"]["splits"])

    rows: list[dict[str, Any]] = []
    errors = 0
    for i, msg in enumerate(msgs):
        # 1) Dialect axis. Retry with new seeds if a probabilistic transform happened to change nothing.
        for group, tr in transformers.items():
            text, feats, attempts, err = msg["text"], {}, 0, None
            for attempts in range(1, vcfg["max_attempts"] + 1):
                try:
                    text, feats = tr.transform(msg["text"], seed=stable_seed(msg["message_id"], group, attempts))
                    err = None
                except Exception as e:  # Multi-VALUE can assert on odd tokenization; record and move on
                    text, feats, err = msg["text"], {}, f"{type(e).__name__}: {e}"[:200]
                if group == "mae" or (text != msg["text"] and same_numbers(text, msg["text"])):
                    break
            errors += err is not None
            rows.append(_variant(msg, group, "dialect", text, feats, attempts=attempts, transform_error=err))

        # 2) Paraphrase control: MAE, different wording, no dialect features.
        if vcfg.get("paraphrase_control", True):
            rows.append(_variant(msg, "paraphrase", "control", msg["paraphrase_text"], {}))

        # 3) Name axis (MAE text + a name), only where it's audited/evaluated.
        if msg["split"] in name_splits:
            for n in names:
                rows.append(
                    _variant(
                        msg, f"name_{n['name_group']}", "name", add_name(msg["text"], n["name"]), {},
                        name=n["name"], name_group=n["name_group"], gender=n["gender"],
                    )
                )
        if (i + 1) % 50 == 0:
            print(f"[variants] shard {shard}: {i + 1}/{len(msgs)} messages")
            run.log({"messages_done": i + 1})

    path = out / f"variants.shard{shard:02d}-of-{num_shards:02d}.jsonl"
    write_jsonl(path, rows)
    run.summary({"n_variants": len(rows), "transform_errors": errors})
    run.finish()
    print(f"[variants] wrote {len(rows)} rows to {path} ({errors} transform errors)")


def coverage_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per group: how often the variant differs from MAE, keeps all numbers, and which features fired."""
    stats: dict[str, Any] = {}
    for r in rows:
        g = stats.setdefault(r["group"], {"n": 0, "changed": 0, "numbers_ok": 0, "errors": 0, "features": {}})
        g["n"] += 1
        g["changed"] += r["changed"]
        g["numbers_ok"] += r["numbers_ok"]
        g["errors"] += bool(r.get("transform_error"))
        for f, c in r["features"].items():
            g["features"][f] = g["features"].get(f, 0) + c
    for g in stats.values():
        g["changed_rate"] = round(g["changed"] / g["n"], 3)
        g["numbers_ok_rate"] = round(g["numbers_ok"] / g["n"], 3)
        g["features"] = dict(sorted(g["features"].items(), key=lambda kv: -kv[1]))
    return stats


def cmd_merge(cfg: dict[str, Any]) -> None:
    out = _out(cfg)
    shards = sorted(glob.glob(str(out / "variants.shard*.jsonl")))
    if not shards:
        raise FileNotFoundError("No variant shards found; run `variants` first.")
    rows = [r for s in shards for r in read_jsonl(s)]
    rows.sort(key=lambda r: (r["split"], r["item_id"], r["axis"], r["group"], r.get("name", "")))
    path = out / "variants.jsonl"
    write_jsonl(path, rows)
    stats = {"n_rows": len(rows), "sha256": sha256_file(path), "by_group": coverage_stats(rows)}
    _write_stats(out / "variants_stats.json", stats)
    run = init_run("data-merge", cfg)
    run.log_artifact(str(path), "variants", "dataset")
    run.summary({"n_variants": len(rows)})
    run.finish()


def cmd_check_sample(cfg: dict[str, Any]) -> None:
    """Native-speaker rating sheets: one CSV per dialect group, for raters who speak that variety.

    Sampling covers features: variants are drawn so every allowed feature appears in several rated rows.
    Raters see only the MAE sentence and the variant (no feature names), and rate naturalness 1-5,
    whether it reads as stereotyped or exaggerated, and whether the meaning is unchanged.
    The key (which features each row contains) is kept in a separate file for the analysis.
    See docs/dialect_validation.md.
    """
    out = _out(cfg)
    vdir = Path(cfg["variants"].get("validation_dir", "data/validation"))
    vdir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in read_jsonl(out / "variants.jsonl")
            if r["axis"] == "dialect" and r["group"] != "mae" and r["changed"] and r["numbers_ok"]]
    rng = random.Random(cfg["seed"])
    n = cfg["variants"]["hand_check_per_group"]
    key = []
    for g in sorted({r["group"] for r in rows}):
        grp = [r for r in rows if r["group"] == g]
        rng.shuffle(grp)
        # Greedy feature coverage: prefer rows containing the least-sampled features so far.
        seen: dict[str, int] = {}
        sample, seen_text = [], set()
        while grp and len(sample) < n:
            grp.sort(key=lambda r: min((seen.get(f, 0) for f in r["features"]), default=99))
            r = grp.pop(0)
            if r["text"] in seen_text:
                continue
            seen_text.add(r["text"])
            sample.append(r)
            for f in r["features"]:
                seen[f] = seen.get(f, 0) + 1
        rng.shuffle(sample)
        with open(vdir / f"rating_sheet_{g}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "mae_sentence", "variant_sentence", "natural_1to5", "stereotyped_yn",
                        "meaning_same_yn", "notes"])
            for i, r in enumerate(sample, 1):
                w.writerow([f"{g}-{i:03d}", r["base_text"], r["text"], "", "", "", ""])
                key.append({"row": f"{g}-{i:03d}", "group": g, "variant_id": r["variant_id"],
                            "features": sorted(r["features"])})
    write_jsonl(vdir / "rating_key.jsonl", key)
    print(f"[check-sample] wrote rating sheets for {sorted({k['group'] for k in key})} to {vdir}/ ({len(key)} rows)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["items", "messages", "variants", "merge", "check-sample"])
    p.add_argument("--config", default="configs/data.yaml")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.step == "items":
        cmd_items(cfg)
    elif a.step == "messages":
        cmd_messages(cfg)
    elif a.step == "variants":
        cmd_variants(cfg, a.shard, a.num_shards)
    elif a.step == "merge":
        cmd_merge(cfg)
    else:
        cmd_check_sample(cfg)


if __name__ == "__main__":
    main()
