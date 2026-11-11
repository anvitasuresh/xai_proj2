from tutorbias.io import read_jsonl, write_jsonl
from tutorbias.replies import generate as gen
from tutorbias.replies.generate import STYLES, tutor_conv


def test_tutor_conv_includes_problem_and_style():
    conv = tutor_conv("Q?", "I am stuck.", "socratic")
    assert STYLES["socratic"] in conv[0]["content"]
    assert "Q?" in conv[1]["content"] and "I am stuck." in conv[1]["content"]
    assert STYLES["socratic"] not in tutor_conv("Q?", "x", None)[0]["content"]


def test_generate_and_select_pool(tmp_path, monkeypatch):
    items = [{"item_id": f"q{i}", "question": f"Q{i}", "final_answer": "7"} for i in range(10)]
    msgs = [{"message_id": f"q{i}::msg", "item_id": f"q{i}", "message_type": "dont_get", "text": "stuck"}
            for i in range(10)]
    write_jsonl(tmp_path / "items.jsonl", items)
    write_jsonl(tmp_path / "msgs.jsonl", msgs)
    cfg = {
        "seed": 0, "wandb": {"enabled": False}, "styles": list(STYLES),
        "paths": {"items": str(tmp_path / "items.jsonl"), "messages": str(tmp_path / "msgs.jsonl"),
                  "out_dir": str(tmp_path / "replies")},
        "generation": {"max_tokens": 50, "temperature": 0.7, "top_p": 0.95},
        "models": {"m1": {"model": {"backend": "fake"}}, "m2": {"model": {"backend": "fake"}}},
        "selection": {"per_item": 5, "min_words": 3, "max_words": 250},
    }
    from tutorbias.models.backends import FakeBackend

    def fake_reply(conv, prefix):
        return "The answer is 7, well done." if "final answer" in conv[0]["content"] else "What could you try first here?"

    monkeypatch.setattr(gen, "load_backend", lambda spec: FakeBackend(gen_fn=fake_reply))
    gen.generate(cfg, "m1")
    gen.generate(cfg, "m2")
    rows = read_jsonl(tmp_path / "replies" / "replies_m1.jsonl")
    assert len(rows) == 10 * len(STYLES)
    assert any(r["answer_leaked"] for r in rows) and not all(r["answer_leaked"] for r in rows)
    gen.select_pool(cfg)
    pool = read_jsonl(tmp_path / "replies" / "reply_pool.jsonl")
    assert len(pool) == 50
    for i in range(10):
        styles = [r["style"] for r in pool if r["item_id"] == f"q{i}"]
        assert len(styles) == len(set(styles)) == 5
