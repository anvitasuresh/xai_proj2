"""Labeling app for judge calibration. Runs locally on a laptop, no GPU.

  streamlit run src/tutorbias/labeling/app.py -- --labeler anvita
  streamlit run src/tutorbias/labeling/app.py -- --labeler second --double-only   # second labeler, ~100 items

Each save appends one row to data/labels/labels_<labeler>.csv, so you can stop and resume any time;
re-labeling an item just adds a newer row (the newest wins). Items appear in a fixed shuffled order
per labeler. The app never shows the reply's source model, intended style, or the judge's score.
Read docs/labeling_guide.md before starting.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

FIELDS = ["label_id", "labeler", "scaffolding", "helpfulness", "flag", "notes", "timestamp"]
SCAFF_OPTIONS = {-1: "-1  Gives it away", 0: "0  Neither", 1: "+1  Guides"}
HELP_OPTIONS = {1: "1 No help / wrong", 2: "2 Little", 3: "3 Some", 4: "4 Helpful", 5: "5 Very helpful"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labeler", required=True)
    p.add_argument("--labels-dir", default="data/labels")
    p.add_argument("--double-only", action="store_true")
    return p.parse_args(sys.argv[1:])


def load_items(labels_dir: Path, labeler: str, double_only: bool) -> list[dict]:
    items = [json.loads(line) for line in (labels_dir / "label_set.jsonl").read_text().splitlines() if line]
    if double_only:
        key = [json.loads(line) for line in (labels_dir / "label_key.jsonl").read_text().splitlines() if line]
        dbl = {k["label_id"] for k in key if k["double"]}
        items = [i for i in items if i["label_id"] in dbl]
    random.Random(f"order::{labeler}").shuffle(items)
    return items


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["label_id"]: r for r in csv.DictReader(f)}


def save(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels_dir)
    out = labels_dir / f"labels_{args.labeler}.csv"
    st.set_page_config(page_title="Tutor reply labeling", layout="wide")

    items = load_items(labels_dir, args.labeler, args.double_only)
    done = load_done(out)
    if "pos" not in st.session_state:
        todo = [i for i, it in enumerate(items) if it["label_id"] not in done]
        st.session_state.pos = todo[0] if todo else 0
    pos = st.session_state.pos

    with st.sidebar:
        st.markdown(f"**Labeler:** {args.labeler}  \n**Done:** {len(done)} / {len(items)}")
        st.progress(len(done) / max(1, len(items)))
        st.markdown(
            "**Scaffolding**  \n"
            "-1 states the answer or does the remaining work (incl. answer hidden in a hint)  \n"
            "0 doesn't give it away but doesn't guide either (generic, off target, or does most of the work)  \n"
            "+1 hint or question aimed at the student's next step; student still does the reasoning\n\n"
            "**Helpfulness** 1 no help or wrong math, 3 correct but generic, 5 clear, tailored, builds on their work.  \n"
            "Full guide: docs/labeling_guide.md"
        )
        jump = st.number_input("Go to item #", 1, len(items), pos + 1)
        if st.button("Go"):
            st.session_state.pos = int(jump) - 1
            st.rerun()

    it = items[pos]
    prev = done.get(it["label_id"], {})
    st.caption(f"Item {pos + 1} of {len(items)} · {it['label_id']}" + (" · already labeled" if prev else ""))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Problem**")
        st.write(it["question"])
        st.markdown(f"**Correct answer:** {it['answer']}")
        st.markdown("**Student**")
        st.info(it["student_message"])
    with c2:
        st.markdown("**Tutor reply**")
        st.success(it["reply"])

    with st.form(key=f"f_{it['label_id']}"):
        s_def = int(prev["scaffolding"]) if prev.get("scaffolding") not in (None, "") else None
        h_def = int(prev["helpfulness"]) if prev.get("helpfulness") not in (None, "") else None
        s = st.radio("Scaffolding", list(SCAFF_OPTIONS), format_func=SCAFF_OPTIONS.get, horizontal=True,
                     index=list(SCAFF_OPTIONS).index(s_def) if s_def is not None else None)
        h = st.radio("Helpfulness", list(HELP_OPTIONS), format_func=HELP_OPTIONS.get, horizontal=True,
                     index=list(HELP_OPTIONS).index(h_def) if h_def is not None else None)
        flag = st.checkbox("Flag (broken item, unsure, or rubric doesn't fit)", value=prev.get("flag") == "1")
        notes = st.text_input("Notes (optional)", value=prev.get("notes", ""))
        submitted = st.form_submit_button("Save and next")
    if submitted:
        if s is None or h is None:
            st.error("Pick both scores, or flag the item and add a note.")
        else:
            save(out, {"label_id": it["label_id"], "labeler": args.labeler, "scaffolding": s, "helpfulness": h,
                       "flag": int(flag), "notes": notes, "timestamp": datetime.now(timezone.utc).isoformat()})
            st.session_state.pos = min(pos + 1, len(items) - 1)
            st.rerun()

    b1, b2 = st.columns(2)
    if b1.button("Previous") and pos > 0:
        st.session_state.pos = pos - 1
        st.rerun()
    if b2.button("Skip") and pos < len(items) - 1:
        st.session_state.pos = pos + 1
        st.rerun()


main()
