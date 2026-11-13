"""E2 analysis: do reward models score identical replies differently by student dialect or name?

  python -m tutorbias.audit.analyze --config configs/audit.yaml

For each scorer (the judge's scaffolding and helpfulness scores, and each off-the-shelf RM):

  gap(g) = mean over replies r of [ score(r, student in group g) - score(r, MAE student) ]

This is a *paired* difference: same problem, same reply, only the student's text changed.

  * Standardized gap: gap / SD of MAE scores for that scorer, so RMs with arbitrary scales are comparable.
  * Net of rewording: gap(g) - gap(paraphrase), per reply. The paraphrase control rewords the message
    in MAE with no dialect features, so this isolates the dialect part of the effect.
  * Clustered bootstrap CIs: the ~5 replies to one question share one student message, so they are not
    independent. We resample questions, not replies.
  * p-values from the bootstrap standard error, Holm-corrected across all (scorer, group) tests in the family.
  * Flip rate: for two replies to the same question, how often the RM's preference between them
    reverses when only the student's dialect changes. This is what matters for preference-based
    training (DPO, best-of-n): a flip means a different reply wins.
  * Heterogeneity by reply style: is the gap larger for answer-giving replies than for hints? If an RM
    penalizes hints *more* for some students, a tutor trained on it learns to give those students answers.

Every table is computed on all variants and on `changed_only` (variants that differ from MAE).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..io import load_config, read_jsonl
from ..tracking import init_run


# --------------------------------------------------------------------------------------------- loading
def load_scores(cfg: dict[str, Any], pairs: pd.DataFrame) -> pd.DataFrame:
    """Join every available scorer's scores onto the pairs table (one column per scorer)."""
    sdir = Path(cfg["paths"]["scores_dir"])
    df = pairs.copy()
    for name, spec in cfg["analysis"]["scorers"].items():
        f = sdir / spec["file"]
        if not f.exists():
            print(f"[analyze] skipping {name}: {f} not found")
            continue
        s = pd.DataFrame(read_jsonl(f))[["id", spec["column"]]].rename(columns={"id": "pair_id", spec["column"]: name})
        s = s.drop_duplicates("pair_id", keep="last")
        df = df.merge(s, on="pair_id", how="left")
    if "judge_scaffolding" in df and (sdir / "judge.jsonl").exists():
        lab = pd.DataFrame(read_jsonl(sdir / "judge.jsonl"))[["id", "scaffolding_label"]]
        df = df.merge(lab.rename(columns={"id": "pair_id", "scaffolding_label": "judge_label"}), on="pair_id", how="left")
    return df


def reply_by_group(df: pd.DataFrame, scorer: str) -> pd.DataFrame:
    """reply x group table of scores (names within a group are averaged per reply first)."""
    t = df.pivot_table(index=["reply_id", "item_id", "style"], columns="group", values=scorer, aggfunc="mean")
    return t.reset_index()


# ------------------------------------------------------------------------------------------- bootstrap
def cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap distribution of the mean, resampling clusters (questions) with replacement."""
    mask = ~np.isnan(values)
    values, clusters = values[mask], clusters[mask]
    if len(values) == 0:
        return np.array([np.nan])
    codes, uniq = pd.factorize(clusters)
    sums = np.bincount(codes, weights=values, minlength=len(uniq))
    counts = np.bincount(codes, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    return sums[idx].sum(1) / counts[idx].sum(1)


def summarize(values: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int, alpha: float) -> dict[str, float]:
    boot = cluster_bootstrap_mean(values, clusters, n_boot, seed)
    mean = float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")
    lo, hi = np.nanquantile(boot, [alpha / 2, 1 - alpha / 2])
    # Two-sided p-value for "mean = 0" from the bootstrap standard error (normal approximation). A raw
    # percentile p-value bottoms out at 1/n_boot, which is too coarse once Holm multiplies it by ~40 tests.
    se = float(np.nanstd(boot))
    p = float(2 * norm.sf(abs(mean) / se)) if se > 0 else (0.0 if mean != 0 else 1.0)
    return {"mean": mean, "ci_lo": float(lo), "ci_hi": float(hi), "p": p,
            "n_replies": int(np.sum(~np.isnan(values))), "n_items": int(pd.Series(clusters[~np.isnan(values)]).nunique())}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls family-wise error, no independence assumption)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


# -------------------------------------------------------------------------------------------- analyses
def gap_table(df: pd.DataFrame, scorers: list[str], acfg: dict[str, Any], seed: int) -> pd.DataFrame:
    base, nbase = acfg["baseline_group"], acfg.get("name_baseline_group")
    rows = []
    for subset in ("all", "changed_only"):
        sub = df if subset == "all" else df[df["changed"] | (df["group"] == base)]
        for s in scorers:
            t = reply_by_group(sub, s)
            if base not in t:
                continue
            sd = float(np.nanstd(t[base])) or 1.0
            groups = [g for g in t.columns if g not in ("reply_id", "item_id", "style", base)]
            for g in groups:
                comparisons = [(base, t[g] - t[base])]
                if g.startswith("name_") and nbase in t and g != nbase:
                    comparisons.append((nbase, t[g] - t[nbase]))
                for ref, d in comparisons:
                    d = d.to_numpy(float)
                    r = summarize(d / sd, t["item_id"].to_numpy(), acfg["n_boot"], seed, acfg["alpha"])
                    row = {"subset": subset, "scorer": s, "group": g, "vs": ref, "sd_baseline": sd,
                           "gap_raw": float(np.nanmean(d)), "gap_std": r["mean"], "ci_lo": r["ci_lo"],
                           "ci_hi": r["ci_hi"], "p": r["p"], "n_replies": r["n_replies"], "n_items": r["n_items"]}
                    if ref == base and "paraphrase" in t and g not in ("paraphrase",) and not g.startswith("name_"):
                        net = ((t[g] - t[base]) - (t["paraphrase"] - t[base])).to_numpy(float) / sd
                        rn = summarize(net, t["item_id"].to_numpy(), acfg["n_boot"], seed, acfg["alpha"])
                        row |= {"net_of_paraphrase_std": rn["mean"], "net_ci_lo": rn["ci_lo"], "net_ci_hi": rn["ci_hi"]}
                    rows.append(row)
    out = pd.DataFrame(rows)
    for subset in out["subset"].unique() if len(out) else []:
        m = out["subset"] == subset
        out.loc[m, "p_holm"] = holm(out.loc[m, "p"].tolist())
    return out


def flip_table(df: pd.DataFrame, scorers: list[str], acfg: dict[str, Any], seed: int, eps: float = 1e-6) -> pd.DataFrame:
    """Share of within-question reply pairs whose preference order reverses vs the MAE student."""
    base = acfg["baseline_group"]
    rows = []
    for s in scorers:
        t = reply_by_group(df, s)
        if base not in t:
            continue
        for g in [c for c in t.columns if c not in ("reply_id", "item_id", "style", base)]:
            flips, items = [], []
            for item_id, grp in t.groupby("item_id"):
                a, b = grp[base].to_numpy(float), grp[g].to_numpy(float)
                for i in range(len(a)):
                    for j in range(i + 1, len(a)):
                        da, db = a[i] - a[j], b[i] - b[j]
                        if np.isnan(da) or np.isnan(db) or abs(da) < eps:
                            continue
                        flips.append(float(np.sign(da) != np.sign(db)))
                        items.append(item_id)
            if flips:
                r = summarize(np.array(flips), np.array(items), acfg["n_boot"], seed, acfg["alpha"])
                rows.append({"scorer": s, "group": g, "flip_rate": r["mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                             "n_pairs": len(flips), "n_items": r["n_items"]})
    return pd.DataFrame(rows)


def label_change_table(df: pd.DataFrame, acfg: dict[str, Any], seed: int) -> pd.DataFrame:
    """Judge only: how often the discrete scaffolding label (-1/0/+1) changes vs the MAE student."""
    if "judge_label" not in df:
        return pd.DataFrame()
    base = acfg["baseline_group"]
    t = df[df["axis"] != "name"].pivot_table(index=["reply_id", "item_id"], columns="group", values="judge_label",
                                             aggfunc="first").reset_index()
    rows = []
    for g in [c for c in t.columns if c not in ("reply_id", "item_id", base)]:
        changed = (t[g] != t[base]).astype(float).to_numpy().copy()
        changed[t[g].isna() | t[base].isna()] = np.nan
        r = summarize(changed, t["item_id"].to_numpy(), acfg["n_boot"], seed, acfg["alpha"])
        rows.append({"group": g, "label_change_rate": r["mean"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                     "n_replies": r["n_replies"]})
    return pd.DataFrame(rows)


def by_style_table(df: pd.DataFrame, scorers: list[str], acfg: dict[str, Any], seed: int) -> pd.DataFrame:
    base = acfg["baseline_group"]
    rows = []
    for s in scorers:
        t = reply_by_group(df, s)
        if base not in t:
            continue
        sd = float(np.nanstd(t[base])) or 1.0
        for g in [c for c in t.columns if c not in ("reply_id", "item_id", "style", base) and not c.startswith("name_")]:
            for style, grp in t.groupby("style"):
                d = ((grp[g] - grp[base]) / sd).to_numpy(float)
                r = summarize(d, grp["item_id"].to_numpy(), acfg["n_boot"], seed, acfg["alpha"])
                rows.append({"scorer": s, "group": g, "style": style, "gap_std": r["mean"], "ci_lo": r["ci_lo"],
                             "ci_hi": r["ci_hi"], "n_replies": r["n_replies"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- figure
def forest_plot(gaps: pd.DataFrame, path: Path, subset: str = "all") -> None:
    """One panel per scorer: standardized gap vs MAE with 95% CI for each student group."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = gaps[(gaps["subset"] == subset) & (gaps["vs"] == "mae")]
    if g.empty:
        return
    scorers = list(dict.fromkeys(g["scorer"]))
    groups = list(dict.fromkeys(sorted(g["group"], key=lambda x: (x.startswith("name_"), x != "paraphrase", x))))
    fig, axes = plt.subplots(1, len(scorers), figsize=(2.6 * len(scorers) + 1, 0.35 * len(groups) + 1.2),
                             sharey=True, dpi=150, squeeze=False)
    for ax, s in zip(axes[0], scorers):
        d = g[g["scorer"] == s].set_index("group").reindex(groups)
        y = np.arange(len(groups))
        sig = d["p_holm"] < 0.05
        ax.axvline(0, color="#888", lw=0.8)
        ax.errorbar(d["gap_std"], y, xerr=[d["gap_std"] - d["ci_lo"], d["ci_hi"] - d["gap_std"]], fmt="none",
                    ecolor="#555", lw=1)
        ax.scatter(d["gap_std"], y, c=np.where(sig, "#c2410c", "#64748b"), s=18, zorder=3)
        ax.set_title(s.replace("_", " "), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("gap vs MAE (SD units)", fontsize=7)
    axes[0][0].set_yticks(np.arange(len(groups)), [x.replace("name_", "name: ") for x in groups])
    axes[0][0].invert_yaxis()
    fig.suptitle("Same reply, different student: score gap vs Mainstream American English (95% CI; orange = Holm p<.05)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_summary(gaps: pd.DataFrame, flips: pd.DataFrame, labels: pd.DataFrame, path: Path) -> None:
    lines = ["# Reward-model audit (E2)", "",
             "Gap = mean paired score difference vs the MAE student, in SD units of that scorer's MAE scores. "
             "CIs resample questions. p is Holm-corrected within each subset.", ""]
    g = gaps[(gaps["subset"] == "all") & (gaps["vs"] == "mae")]
    lines += ["| Scorer | Group | Gap (SD) | 95% CI | Holm p | Net of paraphrase |", "|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        net = f"{r['net_of_paraphrase_std']:+.3f}" if pd.notna(r.get("net_of_paraphrase_std")) else ""
        lines.append(f"| {r['scorer']} | {r['group']} | {r['gap_std']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] "
                     f"| {'<0.001' if r['p_holm'] < 0.001 else format(r['p_holm'], '.3f')} | {net} |")
    if not flips.empty:
        lines += ["", "## Preference flip rate", "", "| Scorer | Group | Flip rate | 95% CI |", "|---|---|---|---|"]
        for _, r in flips.iterrows():
            lines.append(f"| {r['scorer']} | {r['group']} | {r['flip_rate']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |")
    if not labels.empty:
        lines += ["", "## Judge scaffolding label changes", "", "| Group | Change rate | 95% CI |", "|---|---|---|"]
        for _, r in labels.iterrows():
            lines.append(f"| {r['group']} | {r['label_change_rate']:.3f} | [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] |")
    lines += ["", "![forest plot](forest_all.png)"]
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/audit.yaml")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    acfg, seed = cfg["analysis"], cfg["seed"]
    pairs = pd.DataFrame(read_jsonl(cfg["paths"]["pairs"]))
    df = load_scores(cfg, pairs)
    scorers = [s for s in acfg["scorers"] if s in df]
    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)

    gaps = gap_table(df, scorers, acfg, seed)
    flips = flip_table(df, scorers, acfg, seed)
    labels = label_change_table(df, acfg, seed)
    styles = by_style_table(df, scorers, acfg, seed)
    gaps.to_csv(out / "gaps.csv", index=False)
    flips.to_csv(out / "flips.csv", index=False)
    labels.to_csv(out / "judge_label_changes.csv", index=False)
    styles.to_csv(out / "gaps_by_style.csv", index=False)
    for subset in ("all", "changed_only"):
        forest_plot(gaps, out / f"forest_{subset}.png", subset)
    write_summary(gaps, flips, labels, out / "summary.md")
    print((out / "summary.md").read_text())

    run = init_run("audit-analyze", cfg)
    for _, r in gaps[(gaps["subset"] == "all") & (gaps["vs"] == "mae")].iterrows():
        run.summary({f"gap_std/{r['scorer']}/{r['group']}": r["gap_std"], f"p_holm/{r['scorer']}/{r['group']}": r["p_holm"]})
    run.log_table("gaps", json.loads(gaps.to_json(orient="records")))
    run.finish()


if __name__ == "__main__":
    main()
