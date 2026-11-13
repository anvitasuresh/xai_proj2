"""Variant building with a fake dialect transformer (Multi-VALUE itself runs only in the dialect env)."""

import json

from tutorbias.data import build
from tutorbias.io import read_jsonl, write_jsonl


class FakeDialect:
    def __init__(self, name, spec):
        self.name = name
        self.mode = spec.get("mode", "ok")

    def transform(self, text, seed):
        if self.mode == "breaks_numbers":
            return text.replace("24", "25"), {"fake_feature": 1}
        if self.mode == "noop":
            return text, {}
        return text.replace("I don't", "I ain't").replace("I am", "I'm"), {"negative_concord": 1}


def _setup(tmp_path, monkeypatch, mode="ok"):
    msgs = [{
        "message_id": "q1::msg", "item_id": "q1", "split": "audit", "message_type": "tried_partial",
        "text": "I started by doing 48 / 2 and I got 24. Now I don't know what I am supposed to do next.",
        "paraphrase_text": "So far I have 48 / 2 = 24. I can't figure out the next step.",
    }]
    for s in build.SPLITS:
        write_jsonl(tmp_path / f"messages_{s}.jsonl", msgs if s == "audit" else [])
    names = tmp_path / "names.yaml"
    names.write_text("groups:\n  g1: {status: validated, female: [Ann], male: [Bob]}\n")
    cfg = {
        "seed": 1, "paths": {"processed": str(tmp_path)}, "wandb": {"enabled": False},
        "variants": {
            "dialect_groups": {"mae": {"kind": "identity"}, "fake": {"kind": "fake", "mode": mode}},
            "paraphrase_control": True, "max_attempts": 2, "hand_check_per_group": 5,
            "validation_dir": str(tmp_path / "validation"),
            "names": {"file": str(names), "per_gender": 1, "splits": ["audit"]},
        },
    }
    real = build.make_transformer
    monkeypatch.setattr(build, "make_transformer",
                        lambda g, spec: FakeDialect(g, spec) if spec["kind"] == "fake" else real(g, spec))
    return cfg


def test_variants_end_to_end(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch)
    build.cmd_variants(cfg, 0, 1)
    build.cmd_merge(cfg)
    rows = read_jsonl(tmp_path / "variants.jsonl")
    groups = {r["group"] for r in rows}
    assert groups == {"mae", "fake", "paraphrase", "name_g1"}
    fake = next(r for r in rows if r["group"] == "fake")
    assert fake["changed"] and fake["numbers_ok"] and fake["features"] == {"negative_concord": 1}
    names = [r for r in rows if r["axis"] == "name"]
    assert {r["name"] for r in names} == {"Ann", "Bob"}
    assert all(r["text"].startswith("Hi, I'm ") for r in names)
    stats = json.loads((tmp_path / "variants_stats.json").read_text())
    assert stats["by_group"]["fake"]["changed_rate"] == 1.0
    build.cmd_check_sample(cfg)
    assert (tmp_path / "validation" / "rating_sheet_fake.csv").exists()
    assert (tmp_path / "validation" / "rating_key.jsonl").exists()


def test_variants_flag_number_changes(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch, mode="breaks_numbers")
    build.cmd_variants(cfg, 0, 1)
    rows = read_jsonl(tmp_path / "variants.shard00-of-01.jsonl")
    fake = next(r for r in rows if r["group"] == "fake")
    assert not fake["numbers_ok"]
    assert fake["attempts"] == 2  # retried before giving up


def test_variants_unchanged_is_recorded(tmp_path, monkeypatch):
    cfg = _setup(tmp_path, monkeypatch, mode="noop")
    build.cmd_variants(cfg, 0, 1)
    fake = next(r for r in read_jsonl(tmp_path / "variants.shard00-of-01.jsonl") if r["group"] == "fake")
    assert not fake["changed"]
