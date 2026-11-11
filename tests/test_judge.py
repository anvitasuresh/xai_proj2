import math

import pytest

from tutorbias.judge.judge import ScaffoldingJudge, build_conv, load_rubric
from tutorbias.models.backends import FakeBackend

ROW = {"question": "Q?", "answer": "72", "student_message": "I am stuck.", "reply": "What is 48 / 2?"}


def lp(probs):
    return {k: math.log(v) if v > 0 else float("-inf") for k, v in probs.items()}


def fake_logprobs(conv, prefix, cands):
    if prefix.endswith("SCAFFOLDING: "):
        return lp({"0": 0.1, "1": 0.2, "2": 0.5})  # 0.8 valid mass
    if prefix.endswith("HELPFULNESS: "):
        return lp({"1": 0.0, "2": 0.0, "3": 0.5, "4": 0.5, "5": 0.0})
    raise AssertionError(prefix)


def test_rubric_loads_and_conv_contains_fields():
    rubric = load_rubric("v1")
    assert "SCAFFOLDING" in rubric and "HELPFULNESS" in rubric
    conv = build_conv(rubric, ROW)
    assert conv[0]["role"] == "system"
    for field in ("Q?", "72", "I am stuck.", "What is 48 / 2?"):
        assert field in conv[1]["content"]


def test_logprob_mode_expected_scores():
    judge = ScaffoldingJudge(FakeBackend(fake_logprobs), "v1")
    (s,) = judge.score([ROW])
    # renormalized: p(-1)=0.125, p(0)=0.25, p(+1)=0.625 -> E = 0.5
    assert s.scaffolding == pytest.approx(0.5)
    assert s.scaffolding_label == 1
    assert s.valid_mass_scaffolding == pytest.approx(0.8)
    assert s.helpfulness == pytest.approx(3.5)
    assert s.meta["rubric_version"] == "v1" and len(s.meta["rubric_hash"]) == 12


def test_helpfulness_prefix_conditions_on_top_label():
    be = FakeBackend(fake_logprobs)
    ScaffoldingJudge(be, "v1").score([ROW, ROW])
    prefixes = [p for kind, p in be.calls if kind == "lp"]
    assert "SCAFFOLDING: 2\nHELPFULNESS: " in prefixes


def test_explain_generates_rationale_after_scores():
    be = FakeBackend(fake_logprobs, gen_fn=lambda conv, prefix: " Asks a guiding question.")
    (s,) = ScaffoldingJudge(be, "v1").score([ROW], explain=True)
    assert s.rationale == "Asks a guiding question."
    gen_prefix = [p for kind, p in be.calls if kind == "gen"][0]
    assert gen_prefix.endswith("HELPFULNESS: 3\nREASON:")


def test_generate_mode_parses_and_flags_errors():
    outs = iter(["SCAFFOLDING: 0\nHELPFULNESS: 4\nREASON: States the answer.", "garbage"])
    be = FakeBackend(gen_fn=lambda c, p: next(outs))
    be.supports_prefill = False
    a, b = ScaffoldingJudge(be, "v1", mode="generate").score([ROW, ROW])
    assert a.scaffolding_label == -1 and a.helpfulness_label == 4 and a.rationale == "States the answer."
    assert b.meta.get("parse_error") and b.valid_mass_scaffolding == 0.0


def test_logprob_mode_requires_prefill():
    be = FakeBackend(fake_logprobs)
    be.supports_prefill = False
    with pytest.raises(ValueError):
        ScaffoldingJudge(be, "v1", mode="logprob")


def test_zero_valid_mass_is_uniform_and_flagged():
    be = FakeBackend(lambda c, p, cands: {k: float("-inf") for k in cands})
    (s,) = ScaffoldingJudge(be, "v1").score([ROW])
    assert s.valid_mass_scaffolding == 0.0
    assert s.scaffolding == pytest.approx(0.0)
