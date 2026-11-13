"""E1: is the judge trustworthy? Compare judge scores with human labels.

  python -m tutorbias.labeling.calibrate --config configs/calibration.yaml \
      [--split dev]                      # while iterating on the rubric (default)
      [--split test]                     # once, after the rubric is frozen
      [--variant-scores results/audit/judge_variant_scores.jsonl]   # per-group agreement (after stage 2)

Reports (with bootstrap 95% CIs):
  * scaffolding: Cohen's kappa (nominal and quadratic-weighted), accuracy, confusion matrix,
    judge label = argmax of its label probabilities (no thresholds fit to the labels);
  * helpfulness: Spearman between the human 1-5 label and the judge's expected score;
  * human-human agreement on the double-labeled subset: the ceiling for the numbers above;
  * the pre-registered gate from the config: pass/fail.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ..io import load_config, read_jsonl
from ..tracking import init_run
from .metrics import agreement_report, bootstrap_ci, kappa, spearman


def load_labels(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read a labels CSV (from the labeling app or a filled-in label_set.csv). Latest row per item wins;
    rows without a scaffolding label (skipped/flagged) are dropped."""
    out: dict[str, dict[str, Any]] = {}
    if not Path(path).exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            s, h = (row.get("scaffolding") or "").strip(), (row.get("helpfulness") or "").strip()
            if s == "" or h == "":
                continue
            out[row["label_id"]] = {"scaffolding": int(float(s)), "helpfulness": int(float(h)),
                                    "notes": row.get("notes", "")}
    return out


def calibrate(key: list[dict], labels: dict, labels2: dict, scores: dict, split: str, gate: dict,
              n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    keep = [k for k in key if (split == "all" or k["split"] == split) and k["label_id"] in labels
            and k["label_id"] in scores]
    if not keep:
        raise ValueError(f"No items with both a label and a judge score in split '{split}'.")
    ids = [k["label_id"] for k in keep]
    rep: dict[str, Any] = {"split": split}
    rep["judge_vs_human"] = agreement_report(
        [labels[i]["scaffolding"] for i in ids], [scores[i]["scaffolding_label"] for i in ids],
        [labels[i]["helpfulness"] for i in ids], [scores[i]["helpfulness"] for i in ids], seed=seed, n_boot=n_boot)

    # Per intended style: where does the judge disagree? (debugging aid, not a headline number)
    by_style: dict[str, dict] = {}
    for k in keep:
        st = by_style.setdefault(k["style"], {"n": 0, "human_mean": 0.0, "judge_mean": 0.0})
        st["n"] += 1
        st["human_mean"] += labels[k["label_id"]]["scaffolding"]
        st["judge_mean"] += scores[k["label_id"]]["scaffolding"]
    for st in by_style.values():
        st["human_mean"] = round(st["human_mean"] / st["n"], 3)
        st["judge_mean"] = round(st["judge_mean"] / st["n"], 3)
    rep["by_style"] = by_style

    # Human-human ceiling on the double-labeled subset.
    dbl = [i for i in ids if i in labels2]
    if len(dbl) >= 10:
        a = [labels[i]["scaffolding"] for i in dbl]
        b = [labels2[i]["scaffolding"] for i in dbl]
        k_, lo, hi = bootstrap_ci(kappa, a, b, n_boot=n_boot, seed=seed)
        r_, rlo, rhi = bootstrap_ci(spearman, [labels[i]["helpfulness"] for i in dbl],
                                    [labels2[i]["helpfulness"] for i in dbl], n_boot=n_boot, seed=seed)
        rep["human_vs_human"] = {"n": len(dbl), "scaffolding_kappa": k_, "scaffolding_kappa_ci": [lo, hi],
                                 "helpfulness_spearman": r_, "helpfulness_spearman_ci": [rlo, rhi]}

    jh = rep["judge_vs_human"]
    rep["gate"] = {
        "min_scaffolding_kappa": gate["min_scaffolding_kappa"],
        "min_helpfulness_spearman": gate["min_helpfulness_spearman"],
        "passed": bool(jh["scaffolding_kappa"] >= gate["min_scaffolding_kappa"]
                       and jh["helpfulness_spearman"] >= gate["min_helpfulness_spearman"]),
    }
    return rep


def per_group_agreement(key: list[dict], labels: dict, variant_scores: list[dict], n_boot: int, seed: int) -> dict:
    """Judge-human kappa per counterfactual group, using the MAE human label as truth for every variant
    of the same reply. If the judge is fair, agreement should be the same in every group."""
    reply_to_label = {k["reply_id"]: labels[k["label_id"]] for k in key if k["label_id"] in labels}
    groups: dict[str, tuple[list, list]] = {}
    for s in variant_scores:
        lab = reply_to_label.get(s.get("reply_id"))
        if lab is None:
            continue
        h, j = groups.setdefault(s["group"], ([], []))
        h.append(lab["scaffolding"])
        j.append(s["scaffolding_label"])
    return {g: dict(zip(("kappa", "ci_lo", "ci_hi"), bootstrap_ci(kappa, h, j, n_boot=n_boot, seed=seed)), n=len(h))
            for g, (h, j) in sorted(groups.items())}


def plot_confusion(cm: list[list[int]], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.6, 3.2), dpi=150)
    ax.imshow(cm, cmap="Blues")
    ticks = ["-1 gives\nanswer", "0 neither", "+1 guides"]
    ax.set_xticks(range(3), ticks, fontsize=7)
    ax.set_yticks(range(3), ticks, fontsize=7)
    ax.set_xlabel("Judge")
    ax.set_ylabel("Human")
    ax.set_title(title, fontsize=9)
    vmax = max(max(r) for r in cm) or 1
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i][j], ha="center", va="center", color="white" if cm[i][j] > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_report(rep: dict, path: Path) -> None:
    jh = rep["judge_vs_human"]
    lines = [
        f"# Judge calibration (E1), split = {rep['split']}",
        "",
        f"Items: {jh['n']}",
        "",
        "| Metric | Judge vs human | 95% CI |",
        "|---|---|---|",
        f"| Scaffolding kappa | {jh['scaffolding_kappa']:.3f} | {jh['scaffolding_kappa_ci'][0]:.3f} to {jh['scaffolding_kappa_ci'][1]:.3f} |",
        f"| Scaffolding kappa (quadratic) | {jh['scaffolding_kappa_quadratic']:.3f} | {jh['scaffolding_kappa_quadratic_ci'][0]:.3f} to {jh['scaffolding_kappa_quadratic_ci'][1]:.3f} |",
        f"| Scaffolding accuracy | {jh['scaffolding_accuracy']:.3f} | |",
        f"| Helpfulness Spearman | {jh['helpfulness_spearman']:.3f} | {jh['helpfulness_spearman_ci'][0]:.3f} to {jh['helpfulness_spearman_ci'][1]:.3f} |",
    ]
    if "human_vs_human" in rep:
        hh = rep["human_vs_human"]
        lines += ["", f"Human-human ceiling (n={hh['n']}): scaffolding kappa {hh['scaffolding_kappa']:.3f}, "
                  f"helpfulness Spearman {hh['helpfulness_spearman']:.3f}"]
    g = rep["gate"]
    lines += ["", f"**Gate** (kappa >= {g['min_scaffolding_kappa']}, Spearman >= {g['min_helpfulness_spearman']}): "
              f"{'PASSED' if g['passed'] else 'NOT PASSED'}", "", "![confusion](confusion.png)"]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/calibration.yaml")
    p.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    p.add_argument("--scores", default=None, help="override judge scores path")
    p.add_argument("--variant-scores", default=None)
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    paths, ccfg = cfg["paths"], cfg["calibration"]

    key = read_jsonl(Path(paths["labels_dir"]) / "label_key.jsonl")
    labels = load_labels(Path(paths["labels_dir"]) / f"labels_{ccfg['primary_labeler']}.csv")
    labels2 = load_labels(Path(paths["labels_dir"]) / f"labels_{ccfg['second_labeler']}.csv")
    scores = {r["id"]: r for r in read_jsonl(a.scores or paths["judge_scores"])}

    rep = calibrate(key, labels, labels2, scores, a.split, ccfg["gate"], ccfg["n_boot"], cfg["seed"])
    if a.variant_scores:
        rep["per_group"] = per_group_agreement(key, labels, read_jsonl(a.variant_scores), ccfg["n_boot"], cfg["seed"])

    out = Path(paths["results_dir"]) / a.split
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(rep, indent=2))
    plot_confusion(rep["judge_vs_human"]["confusion_scaffolding"], out / "confusion.png",
                   f"Scaffolding, {a.split} (n={rep['judge_vs_human']['n']})")
    write_report(rep, out / "report.md")
    print((out / "report.md").read_text())

    run = init_run("calibrate", {**cfg, "split": a.split})
    jh = rep["judge_vs_human"]
    run.summary({f"{a.split}/{k}": v for k, v in jh.items() if isinstance(v, float)} | {"gate_passed": rep["gate"]["passed"]})
    run.finish()


if __name__ == "__main__":
    main()
