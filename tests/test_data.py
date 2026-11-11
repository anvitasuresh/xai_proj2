import pytest

from tutorbias.data.gsm8k import keep_item, make_item, parse_solution
from tutorbias.data.names import add_name, pick_names
from tutorbias.data.stuck import TEMPLATES, build_messages, pretty_expr
from tutorbias.textutil import extract_numbers

RAW = {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
}


def test_parse_solution():
    p = parse_solution(RAW["answer"])
    assert p["final_answer"] == "72"
    assert p["steps"] == [{"expr": "48/2", "result": "24"}, {"expr": "48+24", "result": "72"}]
    assert "<<" not in p["solution"]


def test_keep_item_filters():
    item = make_item(RAW, "train", 0)
    assert keep_item(item, 2, 6) == (True, "ok")
    assert keep_item(item, 3, 6)[1] == "step_count"
    item2 = dict(item, final_answer="2.5")
    assert keep_item(item2, 2, 6)[1] == "non_integer_answer"
    item3 = dict(item, question=item["question"] + " (Hint: 72)")
    assert keep_item(item3, 2, 6)[1] == "answer_in_question"


def test_pretty_expr():
    assert pretty_expr("48/2") == "48 / 2"
    assert pretty_expr("3*4-1") == "3 x 4 - 1"
    assert pretty_expr("-3+4") == "-3 + 4"


def _items(n=40):
    items = []
    for i in range(n):
        it = make_item(RAW, "train", i)
        it["split"] = "audit"
        items.append(it)
    return items


def test_messages_never_contain_answer_and_are_balanced():
    msgs, counts = build_messages(_items(), seed=1)
    assert len(msgs) == 40
    for m in msgs:
        assert "72" not in extract_numbers(m["text"])
        assert "72" not in extract_numbers(m["paraphrase_text"])
        assert m["text"] != m["paraphrase_text"]
        assert m["text"] == TEMPLATES[m["message_type"]][m["template_idx"]].format(**m["slots"])
    assert max(counts.values()) - min(counts.values()) <= 1


def test_messages_deterministic():
    a, _ = build_messages(_items(), seed=5)
    b, _ = build_messages(_items(), seed=5)
    assert a == b


def test_wrong_answer_is_penultimate_step():
    msgs, _ = build_messages(_items(), seed=1)
    wrong = [m for m in msgs if m["message_type"] == "wrong_answer"]
    assert wrong and all(m["slots"]["wrong"] == "24" for m in wrong)


def test_pick_names_balanced():
    groups = {"a": {"status": "validated", "female": ["F1", "F2", "F3"], "male": ["M1", "M2", "M3"]},
              "b": {"status": "needs_check", "female": ["G1", "G2"], "male": ["N1", "N2"]}}
    names = pick_names(groups, per_gender=2, seed=0)
    assert len(names) == 8
    assert sum(n["gender"] == "female" for n in names) == 4
    assert add_name("I am stuck.", "F1") == "Hi, I'm F1. I am stuck."


def test_split_sizes_error():
    from tutorbias.data import gsm8k

    cfg = {"seed": 0, "min_steps": 2, "max_steps": 6, "n_rl_train": 5, "n_audit": 5, "n_test": 5}
    orig = gsm8k.load_gsm8k
    gsm8k.load_gsm8k = lambda split, cache_dir: [RAW] * 3
    try:
        with pytest.raises(ValueError):
            gsm8k.build_splits(cfg)
    finally:
        gsm8k.load_gsm8k = orig
