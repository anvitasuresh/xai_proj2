"""Dialect safeguards (attestation filter, blocklist, density, no slang pass) with a stand-in for Multi-VALUE,
plus the native-speaker validation rule."""

import sys
import types

import pytest

from tutorbias.data import dialects
from tutorbias.data.validate_variants import evaluate


class FakeMV:
    """Mimics the parts of a Multi-VALUE dialect our wrapper touches."""

    def __init__(self, transforms):
        self.morphosyntax_transforms = dict(transforms)
        self.modification_counter = {}

        def surface_contract(s):
            return s.replace("going to", "gonna-SLANG")

        def surface_fix_spacing(s):
            return s

        self.surface_subs = [surface_contract, surface_fix_spacing]

    def transform(self, text):
        self.modification_counter = {f: 1 for f in self.morphosyntax_transforms} | {"total": 9}
        for sub in self.surface_subs:
            text = sub(text)
        return text + " [" + ",".join(self.morphosyntax_transforms) + "]"


@pytest.fixture
def fake_multivalue(monkeypatch):
    mod = types.ModuleType("multivalue")
    mod.Dialects = types.SimpleNamespace(
        AfricanAmericanVernacular=lambda: FakeMV({"aint_be": 1.0, "negative_concord": 1.0, "yall": 1.0, "rare_one": 0.3}),
        DialectFromFeatureList=lambda feature_list, dialect_name: FakeMV({f: 1.0 for f in feature_list}),
    )
    monkeypatch.setitem(sys.modules, "multivalue", mod)
    monkeypatch.setattr(dialects, "_skip_spacy_download_if_installed", lambda: None)


def test_attestation_filter_and_blocklist(fake_multivalue):
    t = dialects.MultiValueTransformer("aae", {"kind": "multivalue", "dialect": "AfricanAmericanVernacular",
                                                "min_attestation": 1.0, "exclude": ["yall"], "feature_prob": 1.0})
    assert t.allowed == ["aint_be", "negative_concord"]  # rare (C-rated) and excluded features gone
    out, feats = t.transform("I am going to try.", seed=1)
    assert set(feats) == {"aint_be", "negative_concord"} and "total" not in feats
    assert "SLANG" not in out  # Multi-VALUE's slang contraction pass is removed


def test_feature_density_varies_and_is_reproducible(fake_multivalue):
    t = dialects.MultiValueTransformer("aae", {"kind": "multivalue", "dialect": "AfricanAmericanVernacular",
                                                "min_attestation": 1.0, "feature_prob": 0.5})
    sets = {frozenset(t.transform("x", seed=s)[1]) for s in range(40)}
    assert len(sets) > 2 and frozenset() not in sets  # different subsets, never empty
    assert t.transform("x", seed=7) == t.transform("x", seed=7)


def test_feature_list_group(fake_multivalue):
    t = dialects.MultiValueTransformer("l2", {"kind": "multivalue_features", "features": ["zero_plural", "uninflect"],
                                               "exclude": ["uninflect"], "feature_prob": 1.0})
    assert t.allowed == ["zero_plural"]


RULE = {"min_n": 3, "max_stereotyped_rate": 0.2, "min_naturalness": 3.0, "margin_stereotyped": 0.1,
        "margin_naturalness": 0.5}


def test_validation_rule_drops_flagged_features():
    # "bad" is in rows 0-3, "good" in rows 2-7, so they co-occur in rows 2-3.
    key = [{"row": f"a-{i}", "group": "aae", "features": (["good"] if i >= 2 else []) + (["bad"] if i < 4 else [])}
           for i in range(8)]
    ratings = []
    for i in range(8):
        for rater in ("r1", "r2"):
            bad = i < 4
            ratings.append({"row": f"a-{i}", "group": "aae", "rater": rater, "natural": 2.0 if bad else 4.5,
                            "stereotyped": 1.0 if bad else 0.0, "meaning_same": 1.0, "notes": ""})
    rep = evaluate(ratings, key, RULE)
    assert rep["flagged"]["aae"] == ["bad"]
    assert rep["groups"]["aae"]["features"]["good"]["drop"] is False
    assert rep["groups"]["aae"]["agreement"]["naturalness_kappa_quadratic"] == pytest.approx(1.0)


def test_validation_needs_min_n():
    key = [{"row": "a-0", "group": "aae", "features": ["rare"]}]
    ratings = [{"row": "a-0", "group": "aae", "rater": "r1", "natural": 1.0, "stereotyped": 1.0,
                "meaning_same": 1.0, "notes": ""}]
    assert evaluate(ratings, key, RULE)["flagged"]["aae"] == []
