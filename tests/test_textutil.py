from tutorbias.textutil import answer_leaked, canonical_number, extract_numbers, same_numbers


def test_canonical_number():
    assert canonical_number("1,200") == "1200"
    assert canonical_number("3.50") == "3.5"
    assert canonical_number("7.0") == "7"
    assert canonical_number("abc") is None


def test_extract_numbers_basic():
    assert extract_numbers("I did 48 / 2 and got 24.") == ["48", "2", "24"]
    assert extract_numbers("It costs $1,200, not 3.5") == ["1200", "3.5"]
    assert extract_numbers("Q3 and 2nd") == []  # glued to letters: not a standalone number


def test_answer_leaked_digits_and_words():
    assert answer_leaked("So the answer is 72!", "72")
    assert answer_leaked("You should get seventy-two", "72")
    assert not answer_leaked("What is 48 / 2?", "72")
    assert not answer_leaked("There are 172 apples", "72")


def test_same_numbers_ignores_order():
    assert same_numbers("I got 3 then 5", "5 is what I get after 3")
    assert not same_numbers("I got 3", "I got 4")


def test_config_profile_inherits_base(tmp_path):
    from tutorbias.io import load_config

    (tmp_path / "base.yaml").write_text("a: 1\nm: {x: 1, y: 2}\n")
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "c.yaml").write_text("base: ../base.yaml\nm: {y: 3}\n")
    assert load_config(tmp_path / "p" / "c.yaml") == {"a": 1, "m": {"x": 1, "y": 3}}


def test_t4_profiles_keep_models_and_rubric():
    from tutorbias.io import load_config

    base, t4 = load_config("configs/judge.yaml"), load_config("configs/t4/judge.yaml")
    assert t4["judges"]["primary"]["model"]["model"] == base["judges"]["primary"]["model"]["model"]
    assert t4["judges"]["primary"]["rubric"] == base["judges"]["primary"]["rubric"]
    assert t4["judges"]["primary"]["model"]["dtype"] == "float16"
