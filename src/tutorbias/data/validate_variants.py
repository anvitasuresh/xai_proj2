"""Analyze native-speaker ratings of the dialect variants and decide which features to drop.

  python -m tutorbias.data.validate_variants --config configs/data.yaml

Input: filled-in rating sheets saved as data/validation/ratings_<group>_<rater>.csv (copies of
rating_sheet_<group>.csv) plus rating_key.jsonl. Output: data/validation/report.md and
data/validation/flagged_features.yaml.

Decision rule (pre-registered in configs/data.yaml, `validation:`), applied per feature within a group,
over rated variants that contain the feature:
  drop the feature if it appears in >= min_n rated rows AND either
    stereotyped rate >= max_stereotyped_rate and at least margin_stereotyped above rows without it, OR
    mean naturalness < min_naturalness and at least margin_naturalness below rows without it.
  The "rows without it" comparison stops a fine feature from being dropped just because it often
  co-occurs with a bad one in the same sentence.
Dropped features go into that group's `exclude` list in configs/data.yaml, then variants are rebuilt.
A group whose overall mean naturalness stays below min_naturalness is reported as a limitation (and
should be dropped from headline claims), not quietly kept.

Agreement between raters of the same group: quadratic-weighted kappa on naturalness, Cohen's kappa
and raw agreement on the stereotyped flag.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..io import load_config, read_jsonl


def _yn(v: str) -> float | None:
    v = (v or "").strip().lower()
    if v in ("y", "yes", "1", "true"):
        return 1.0
    if v in ("n", "no", "0", "false"):
        return 0.0
    return None


def load_ratings(vdir: Path) -> list[dict[str, Any]]:
    """All ratings_<group>_<rater>.csv files -> one row per (rater, sheet row) with a naturalness score."""
    out = []
    for f in sorted(vdir.glob("ratings_*_*.csv")):
        group, rater = f.stem.split("_", 2)[1:]
        with open(f) as fh:
            for r in csv.DictReader(fh):
                nat = (r.get("natural_1to5") or "").strip()
                if not nat:
                    continue
                out.append({"row": r["row"], "group": group, "rater": rater, "natural": float(nat),
                            "stereotyped": _yn(r.get("stereotyped_yn", "")), "meaning_same": _yn(r.get("meaning_same_yn", "")),
                            "notes": r.get("notes", "")})
    return out


def _mean(xs: list[float | None]) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def agreement(ratings: list[dict], group: str) -> dict[str, Any]:
    from sklearn.metrics import cohen_kappa_score

    by_rater: dict[str, dict[str, dict]] = {}
    for r in ratings:
        if r["group"] == group:
            by_rater.setdefault(r["rater"], {})[r["row"]] = r
    raters = sorted(by_rater)
    if len(raters) < 2:
        return {"n_raters": len(raters)}
    a, b = by_rater[raters[0]], by_rater[raters[1]]
    common = sorted(set(a) & set(b))
    res: dict[str, Any] = {"n_raters": len(raters), "pair": raters[:2], "n_common": len(common)}
    if len(common) >= 5:
        na, nb = [int(a[k]["natural"]) for k in common], [int(b[k]["natural"]) for k in common]
        res["naturalness_kappa_quadratic"] = float(cohen_kappa_score(na, nb, weights="quadratic", labels=[1, 2, 3, 4, 5]))
        flags = [(a[k]["stereotyped"], b[k]["stereotyped"]) for k in common
                 if a[k]["stereotyped"] is not None and b[k]["stereotyped"] is not None]
        if flags:
            fa, fb = zip(*flags)
            res["stereotyped_agreement"] = float(np.mean(np.array(fa) == np.array(fb)))
            res["stereotyped_kappa"] = float(cohen_kappa_score(fa, fb, labels=[0.0, 1.0])) if len(set(fa) | set(fb)) > 1 else None
    return res


def evaluate(ratings: list[dict], key: list[dict], rule: dict[str, Any]) -> dict[str, Any]:
    feats_by_row = {k["row"]: k["features"] for k in key}
    groups = sorted({r["group"] for r in ratings})
    report: dict[str, Any] = {"groups": {}, "flagged": {}}
    for g in groups:
        rs = [r for r in ratings if r["group"] == g]
        gsum = {"n_ratings": len(rs), "n_rows": len({r["row"] for r in rs}),
                "naturalness_mean": _mean([r["natural"] for r in rs]),
                "stereotyped_rate": _mean([r["stereotyped"] for r in rs]),
                "meaning_same_rate": _mean([r["meaning_same"] for r in rs]),
                "agreement": agreement(ratings, g)}
        gsum["below_naturalness_bar"] = bool(gsum["naturalness_mean"] < rule["min_naturalness"])
        per_feat: dict[str, dict] = {}
        for r in rs:
            for f in feats_by_row.get(r["row"], []):
                d = per_feat.setdefault(f, {"nat": [], "ster": [], "rows": set()})
                d["nat"].append(r["natural"])
                d["ster"].append(r["stereotyped"])
                d["rows"].add(r["row"])
        feats = {}
        flagged = []
        for f, d in sorted(per_feat.items()):
            nat, ster = _mean(d["nat"]), _mean(d["ster"])
            n = len(d["rows"])
            # Rows mix several features, so a feature that only co-occurs with a bad one would look bad
            # too. Require it to be worse than the rows *without* it (when there are any).
            others = [r for r in rs if r["row"] not in d["rows"]]
            nat_wo, ster_wo = _mean([r["natural"] for r in others]), _mean([r["stereotyped"] for r in others])
            ster_bad = ster >= rule["max_stereotyped_rate"] and (ster_wo != ster_wo or ster - ster_wo >= rule["margin_stereotyped"])
            nat_bad = nat < rule["min_naturalness"] and (nat_wo != nat_wo or nat_wo - nat >= rule["margin_naturalness"])
            drop = n >= rule["min_n"] and (ster_bad or nat_bad)
            feats[f] = {"n_rows": n, "naturalness_mean": round(nat, 2),
                        "stereotyped_rate": round(ster, 3) if ster == ster else None,
                        "naturalness_without": round(nat_wo, 2) if nat_wo == nat_wo else None,
                        "stereotyped_without": round(ster_wo, 3) if ster_wo == ster_wo else None, "drop": bool(drop)}
            if drop:
                flagged.append(f)
        gsum["features"] = feats
        report["groups"][g] = gsum
        report["flagged"][g] = flagged
    return report


def write_report(rep: dict[str, Any], rule: dict[str, Any], path: Path) -> None:
    lines = ["# Native-speaker validation of dialect variants", "",
             f"Rule: drop a feature seen in >= {rule['min_n']} rated rows if its stereotyped rate is >= "
             f"{rule['max_stereotyped_rate']} (and {rule['margin_stereotyped']} above rows without it), or its mean "
             f"naturalness is < {rule['min_naturalness']} (and {rule['margin_naturalness']} below rows without it).", ""]
    for g, s in rep["groups"].items():
        ag = s["agreement"]
        lines += [f"## {g}", "",
                  f"{s['n_rows']} rows, {s['n_ratings']} ratings. Mean naturalness {s['naturalness_mean']:.2f}/5; "
                  f"stereotyped {s['stereotyped_rate']:.0%}; meaning unchanged {s['meaning_same_rate']:.0%}."
                  + (" **Below the naturalness bar: report as a limitation.**" if s["below_naturalness_bar"] else ""), ""]
        if ag.get("n_raters", 0) >= 2 and "naturalness_kappa_quadratic" in ag:
            lines += [f"Rater agreement ({ag['n_common']} shared rows): naturalness kappa (quadratic) "
                      f"{ag['naturalness_kappa_quadratic']:.2f}; stereotyped flag agreement {ag.get('stereotyped_agreement', float('nan')):.0%}.", ""]
        else:
            lines += [f"Raters: {ag.get('n_raters', 0)} (agreement needs 2 raters on the same rows).", ""]
        lines += ["| Feature | Rows | Naturalness | Stereotyped | Decision |", "|---|---|---|---|---|"]
        for f, d in s["features"].items():
            st = "" if d["stereotyped_rate"] is None else f"{d['stereotyped_rate']:.0%}"
            lines.append(f"| `{f}` | {d['n_rows']} | {d['naturalness_mean']:.2f} | {st} | {'**drop**' if d['drop'] else 'keep'} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/data.yaml")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    vdir = Path(cfg["variants"].get("validation_dir", "data/validation"))
    rule = cfg["validation"]
    ratings = load_ratings(vdir)
    if not ratings:
        raise SystemExit(f"No ratings found in {vdir}/ (expected ratings_<group>_<rater>.csv)")
    rep = evaluate(ratings, read_jsonl(vdir / "rating_key.jsonl"), rule)
    write_report(rep, rule, vdir / "report.md")
    (vdir / "flagged_features.yaml").write_text(yaml.safe_dump({"exclude_add": rep["flagged"]}, sort_keys=True))
    print((vdir / "report.md").read_text())
    print("Features to add to `exclude` in configs/data.yaml:", rep["flagged"])


if __name__ == "__main__":
    main()
