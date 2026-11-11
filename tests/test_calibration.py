import csv
import random

import pytest

from tutorbias.labeling.calibrate import calibrate, load_labels, per_group_agreement
from tutorbias.labeling.label_set import build_label_set
from tutorbias.labeling.metrics import bootstrap_ci, kappa

STYLES = ["direct_answer", "socratic", "vague", "targeted_hint"]


def _pool(n_items=120):
    pool, items, msgs = [], {}, {}
    for i in range(n_items):
        iid = f"q{i}"
        items[iid] = {"item_id": iid, "question": f"Q{i}", "final_answer": "5"}
        msgs[f"{iid}::msg"] = {"message_id": f"{iid}::msg", "text": "I am stuck."}
        for s in STYLES:
            pool.append({"reply_id": f"{iid}::m::{s}", "item_id": iid, "message_id": f"{iid}::msg",
                         "style": s, "source_model": "m", "text": f"reply {s}"})
    return pool, items, msgs


def test_label_set_blinded_stratified_one_per_item():
    pool, items, msgs = _pool()
    cfg = {"seed": 0, "n_labels": 100, "n_double": 30, "dev_frac": 0.5}
    shown, key = build_label_set(pool, items, msgs, cfg)
    assert len(shown) == 100 and len({k["item_id"] for k in key}) == 100
    assert set(shown[0]) == {"label_id", "question", "answer", "student_message", "reply"}  # blinded
    assert sum(k["double"] for k in key) == 30
    assert abs(sum(k["split"] == "dev" for k in key) - 50) <= len(STYLES)  # per-style rounding
    counts = {s: sum(k["style"] == s for k in key) for s in STYLES}
    assert max(counts.values()) - min(counts.values()) <= 1


def test_kappa_perfect_and_bootstrap():
    a = [-1, 0, 1] * 20
    assert kappa(a, a) == pytest.approx(1.0)
    est, lo, hi = bootstrap_ci(kappa, a, a, n_boot=200)
    assert est == lo == hi == pytest.approx(1.0)


def _write_labels(path, labels):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label_id", "scaffolding", "helpfulness", "notes"])
        w.writeheader()
        for lid, (s, h) in labels.items():
            w.writerow({"label_id": lid, "scaffolding": s, "helpfulness": h, "notes": ""})


def test_calibrate_end_to_end(tmp_path):
    rng = random.Random(0)
    key, labels, scores = [], {}, {}
    for i in range(200):
        lid = f"L{i:04d}"
        s = rng.choice([-1, 0, 1])
        h = rng.randint(1, 5)
        key.append({"label_id": lid, "reply_id": f"r{i}", "style": "x", "split": "dev" if i % 2 else "test",
                    "double": i < 60})
        labels[lid] = (s, h)
        js = s if rng.random() < 0.85 else rng.choice([-1, 0, 1])  # judge agrees ~90%
        scores[lid] = {"scaffolding_label": js, "scaffolding": float(js), "helpfulness": h + rng.gauss(0, 0.5)}
    _write_labels(tmp_path / "a.csv", labels)
    _write_labels(tmp_path / "b.csv", {k: v for k, v in list(labels.items())[:60]})
    la, lb = load_labels(tmp_path / "a.csv"), load_labels(tmp_path / "b.csv")
    gate = {"min_scaffolding_kappa": 0.6, "min_helpfulness_spearman": 0.5}
    rep = calibrate(key, la, lb, scores, "test", gate, n_boot=200)
    assert rep["judge_vs_human"]["n"] == 100
    assert rep["judge_vs_human"]["scaffolding_kappa"] > 0.7
    assert rep["gate"]["passed"]
    assert rep["human_vs_human"]["scaffolding_kappa"] == pytest.approx(1.0)

    variants = [{"reply_id": f"r{i}", "group": g, "scaffolding_label": labels[f"L{i:04d}"][0]}
                for i in range(200) for g in ("sae", "aae")]
    pg = per_group_agreement(key, la, variants, n_boot=100, seed=0)
    assert set(pg) == {"sae", "aae"} and pg["aae"]["kappa"] == pytest.approx(1.0)


def test_load_labels_skips_blank_and_keeps_latest(tmp_path):
    p = tmp_path / "l.csv"
    p.write_text("label_id,scaffolding,helpfulness,notes\nL1,1,4,\nL2,,,skip\nL1,0,3,changed\n")
    out = load_labels(p)
    assert out == {"L1": {"scaffolding": 0, "helpfulness": 3, "notes": "changed"}}
