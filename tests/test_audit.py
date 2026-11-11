"""Audit analysis on synthetic scores with a known, injected bias."""

import numpy as np
import pandas as pd
import pytest

from tutorbias.audit.analyze import (
    by_style_table,
    cluster_bootstrap_mean,
    flip_table,
    gap_table,
    holm,
    label_change_table,
)
from tutorbias.audit.pairs import build_pairs
from tutorbias.audit.reward_models import rm_conversation

ACFG = {"baseline_group": "sae", "name_baseline_group": "name_white", "n_boot": 500, "alpha": 0.05}
GROUPS = ["sae", "aae", "paraphrase", "name_white", "name_black"]
EFFECT = {"sae": 0.0, "aae": -0.5, "paraphrase": 0.0, "name_white": 0.0, "name_black": 0.0}


def synthetic(n_items=80, n_replies=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_items):
        item_noise = rng.normal(0, 0.3)
        for k in range(n_replies):
            base = rng.normal(0, 1) + item_noise
            for g in GROUPS:
                score = base + EFFECT[g] + rng.normal(0, 0.05)
                rows.append({"pair_id": f"q{i}r{k}||{g}", "reply_id": f"q{i}r{k}", "item_id": f"q{i}",
                             "style": "hint" if k % 2 else "answer", "group": g,
                             "axis": "name" if g.startswith("name_") else ("control" if g == "paraphrase" else "dialect"),
                             "changed": g != "sae", "rm": score,
                             "judge_label": int(np.sign(base)) if g != "aae" else int(np.sign(base - 1))})
    return pd.DataFrame(rows)


def test_gap_detects_injected_bias_and_not_null_groups():
    gaps = gap_table(synthetic(), ["rm"], ACFG, seed=0)
    g = gaps[(gaps["subset"] == "all") & (gaps["vs"] == "sae")].set_index("group")
    sd = g.loc["aae", "sd_baseline"]
    assert g.loc["aae", "gap_raw"] == pytest.approx(-0.5, abs=0.02)
    assert g.loc["aae", "ci_hi"] < 0 and g.loc["aae", "p_holm"] < 0.01
    assert g.loc["aae", "net_of_paraphrase_std"] == pytest.approx(-0.5 / sd, abs=0.03)
    assert g.loc["paraphrase", "ci_lo"] < 0 < g.loc["paraphrase", "ci_hi"]
    assert g.loc["name_black", "p_holm"] > 0.05
    # name groups are also compared to the white-name baseline
    assert ((gaps["group"] == "name_black") & (gaps["vs"] == "name_white")).any()


def test_flip_rate_low_without_bias():
    flips = flip_table(synthetic(), ["rm"], ACFG, seed=0).set_index("group")
    # A constant shift never reverses a preference between two replies to the same question.
    assert flips.loc["aae", "flip_rate"] < 0.05
    assert flips.loc["paraphrase", "flip_rate"] < 0.05


def test_label_changes_and_style_table():
    df = synthetic()
    lab = label_change_table(df, ACFG, seed=0).set_index("group")
    assert lab.loc["aae", "label_change_rate"] > lab.loc["paraphrase", "label_change_rate"]
    st = by_style_table(df, ["rm"], ACFG, seed=0)
    assert set(st["style"]) == {"hint", "answer"}


def test_cluster_bootstrap_centered_on_mean():
    vals = np.arange(20, dtype=float)
    clusters = np.repeat(np.arange(5), 4)
    boot = cluster_bootstrap_mean(vals, clusters, n_boot=2000, seed=0)
    assert boot.mean() == pytest.approx(vals.mean(), abs=0.5)


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_build_pairs_drops_number_changes():
    pool = [{"reply_id": "r1", "item_id": "q1", "message_id": "m1", "style": "s", "source_model": "m", "text": "hint"}]
    variants = [
        {"variant_id": "v_sae", "message_id": "m1", "message_type": "t", "axis": "dialect", "group": "sae",
         "changed": False, "features": {}, "numbers_ok": True, "text": "I got 3"},
        {"variant_id": "v_bad", "message_id": "m1", "message_type": "t", "axis": "dialect", "group": "aae",
         "changed": True, "features": {"f": 1}, "numbers_ok": False, "text": "I got 4"},
    ]
    pairs, stats = build_pairs(pool, variants, {"q1": {"question": "Q", "final_answer": "5"}})
    assert [p["group"] for p in pairs] == ["sae"]
    assert stats["dropped_numbers_changed"] == {"aae": 1}
    assert pairs[0]["student_message"] == "I got 3" and pairs[0]["reply"] == "hint"


def test_rm_conversation_shape():
    conv = rm_conversation({"question": "Q", "student_message": "stuck", "reply": "hint"})
    assert [m["role"] for m in conv] == ["system", "user", "assistant"]
    assert conv[-1]["content"] == "hint"
