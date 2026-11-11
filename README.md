# Who Does the Tutor Learn From?

**Auditing reward-model dialect bias, and whether it carries into an RL-trained AI tutor.**

An AI tutor trained with reinforcement learning learns whatever its reward model rewards. If the reward model scores the *same* tutoring help differently depending on whether the student writes in Standard American English, African American English, Indian English, or learner English, or on the student's name, the tutor can learn to give some students real guidance and hand others the answer. Bastani et al. (PNAS, 2025) found that answer-giving help improves practice scores but hurts learning, so that gap would fall on exactly the students AI-in-schools policies aim to protect.

This repo builds the whole pipeline:

1. **Scaffolding judge.** An LLM judge (Qwen2.5-7B-Instruct + rubric) scores tutor replies from -1 (gives the answer) to +1 (useful hint), plus 1-5 helpfulness, calibrated against human labels.
2. **Counterfactual reward-model audit.** Identical replies, with only the student's dialect or name changed. Any score gap is bias by construction.
3. **Explanation.** Which dialect features drive the gaps; whether the judge's stated reasons match them; linear probes on activations.
4. **Two GRPO tutors.** Qwen2.5-1.5B trained on the original judge (Tutor A) vs a debiased judge (Tutor B).
5. **Evaluation.** Scaffolding, helpfulness, reward hacking, and per-group gaps before and after RL.

> Status: **stages 1 (data, judge, labeling) and 2 (reward-model audit) are implemented.** Stages 3-5 are next. See [docs/timeline.md](docs/timeline.md).

Everything is synthetic or public. No real students and no minors are involved.

---

## Repo layout

```
configs/                 one YAML per stage (data, replies, judge, calibration)
src/tutorbias/
  data/                  GSM8K items, stuck-student templates, Multi-VALUE dialects, name swaps
  models/backends.py     vLLM / transformers / OpenAI-compatible / fake backends, one interface
  replies/               fixed tutor-reply pool for the audit
  judge/                 scaffolding judge + versioned rubric prompts + scoring CLI
  labeling/              blinded label set, Streamlit labeling app, calibration (E1)
  audit/                 counterfactual pairs, RewardBench RM scoring, gap/flip analysis (E2)
  textutil.py            number extraction, answer-leak detection
  tracking.py            optional Weights & Biases logging
scripts/                 setup_local.sh (envs) and run_stage.sh (one command per stage)
slurm/                   Duke DCC job scripts (thin wrappers around run_stage.sh)
data/                    names list, labels, DATASHEET.md (generated data is gitignored)
docs/                    labeling guide, design decisions, timeline
tests/                   pytest suite (CPU only, runs in CI)
```

## Setup

Two Python environments, because Multi-VALUE pins old spaCy/nltk/stanza versions that conflict with vLLM. One script creates both with [uv](https://docs.astral.sh/uv/) (no conda needed):

| Env | Python | Used for | Hardware |
|---|---|---|---|
| `.venv-data` | 3.10 | dialect variants, labeling app, calibration | any laptop |
| `.venv-gpu` | 3.11 | reply generation, judge scoring, RM audit, later GRPO | one GPU with 24 GB+ (L4, A10, A6000, A100) |

```bash
git clone https://github.com/anvitasuresh/xai_proj2 && cd xai_proj2
bash scripts/setup_local.sh data   # laptop: CPU steps only
bash scripts/setup_local.sh        # GPU machine: both envs
```

## Reproducing

Every stage is one command, and it runs the same way on a laptop, Colab, a rented GPU, or Duke DCC:

```bash
bash scripts/run_stage.sh items      # CPU  GSM8K items + stuck messages
bash scripts/run_stage.sh variants   # CPU  dialect + name variants (parallel shards; ~1-3 h)
bash scripts/run_stage.sh replies    # GPU  tutor reply pool + blinded label set
bash scripts/run_stage.sh judge      # GPU  judge scores + rationales on the label set
# label on a laptop, then calibrate (E1):
streamlit run src/tutorbias/labeling/app.py -- --labeler anvita
streamlit run src/tutorbias/labeling/app.py -- --labeler second --double-only
python -m tutorbias.labeling.calibrate --split dev    # iterate on the rubric
python -m tutorbias.labeling.calibrate --split test   # once, after the rubric is frozen
bash scripts/run_stage.sh audit      # GPU  E2 reward-model audit
```

All GPU steps are resumable: if a session dies, rerun the same command and it continues where it stopped.

**On Duke DCC:** run `bash slurm/setup_env.sh` once on a login node, then `sbatch slurm/01_data.sbatch` (after `items`), `02_replies`, `03_judge`, `04_audit`. Each job just calls `scripts/run_stage.sh`.

**Free GPUs (Kaggle, 2x T4):** `PROFILE=t4 bash scripts/run_stage.sh <stage>` runs the same models and rubric in float16, split across both GPUs. Step-by-step notebook setup: [docs/kaggle.md](docs/kaggle.md).

Stage 1-2 GPU work is roughly 3-5 A100-hours or 4-6 T4-hours. GRPO runs (stage 4) take longer and are planned for DCC.

## Stage 2: reward-model audit (after the judge passes the E1 gate)

```bash
bash scripts/run_stage.sh audit   # pairs -> judge + Skywork-Reward-V2 + ArmoRM scores -> analysis
# -> results/audit/summary.md, gaps.csv, flips.csv, gaps_by_style.csv, forest_all.png
```

Every reply in the pool is paired with every version of its student's message (~1500 replies x ~21 versions). Within a reply, only the student text changes, so a score gap is bias by construction. The analysis reports paired gaps in SD units with bootstrap CIs that resample *questions* (replies to one question share a message), Holm-corrected p-values, the gap net of the SAE paraphrase control, preference flip rates between replies, the judge's discrete label change rate, and gaps broken down by reply style. Everything is repeated on variants that actually differ from SAE (`changed_only`).

## How the judge scores a reply

The judge sees the problem, its correct answer, the student's message, and the tutor reply. The assistant turn is pre-filled with `SCAFFOLDING: `, and we read the probabilities of the labels `0/1/2` (mapped to -1/0/+1). The score is the expected value, a continuous number in [-1, 1] (as in G-Eval). Helpfulness is read the same way, and an optional rationale is generated **after** the scores so it explains exactly those scores (needed for the faithfulness test, E4). Each score row records the judge model, rubric version, and a hash of the rubric text, plus how much probability fell on valid labels (a check that the prompt format is working). Details and trade-offs: [docs/design_decisions.md](docs/design_decisions.md).

## Experiments

| # | Question | Metric | Control | Status |
|---|---|---|---|---|
| E1 | Is the judge trustworthy? | Cohen's kappa (scaffolding), Spearman (helpfulness), bootstrap CIs, per-group agreement | Human-human agreement on ~100 items | code ready |
| E2 | Are reward models biased by dialect or name? | Paired gap vs SAE (SD units) with clustered bootstrap CIs, Holm; preference flip rate; gap by reply style | SAE paraphrase (rewording, no dialect) | code ready |
| E3 | What drives the gaps? | Score change per ablated feature; probe accuracy by layer | Random-feature ablation; shuffled-label probe | planned |
| E4 | Are the judge's explanations faithful? | Share of real drivers the rationale mentions | Cases with no score gap | planned |
| E5 | Did RL teach scaffolding? | Scaffolding rate, helpfulness, answer-leak rate | Base model | planned |
| E6 | Did RL reward-hack? | Empty-question rate, hidden answers, length drift | Held-out judge | planned |
| E7 | Does bias propagate? | Per-group gap, Tutor A vs B, difference-in-differences | Base-model gap | planned |

## Limitations

- **Synthetic dialects.** Rule-based transforms are cleaner than real speech. Gaps show sensitivity to dialect *features*, not real-world effect sizes. ESL is a feature list that overlaps with AAE.
- **Small models.** 1.5B tutor, 7B judge: a mechanism demonstration, not a claim about deployed tutors.
- **My labels set the standard.** "Good scaffolding" reflects one rubric; second-labeler agreement shows how subjective it is.
- **Single-turn, math only.**
- **Fairness definition.** Counterfactual fairness (same help, same score) assumes every student should get the same kind of help, which may be wrong if some students need more direct help.
- **Names** partly come from unvalidated placeholder lists (marked in `data/names/names.yaml`).

## Data and licenses

GSM8K (MIT). Multi-VALUE (Apache 2.0). Qwen2.5 models (Apache 2.0 / Qwen license). Code: MIT. See [data/DATASHEET.md](data/DATASHEET.md).

## References

Bastani et al., *Generative AI without guardrails can harm learning*, PNAS 2025 · Hofmann et al., *AI generates covertly racist decisions about people based on their dialect*, Nature 2024 · Ziems et al., *Multi-VALUE*, ACL 2023 · Lambert et al., *RewardBench*, 2024 · Shao et al., *DeepSeekMath* (GRPO), 2024 · Liu et al., *G-Eval*, 2023 · Zheng et al., *Judging LLM-as-a-judge*, 2023 · Gao et al., *Scaling laws for reward model overoptimization*, 2022 · Turpin et al., *Language models don't always say what they think*, 2023 · Bertrand & Mullainathan, *Are Emily and Greg more employable than Lakisha and Jamal?*, AER 2004 · Cobbe et al., *GSM8K*, 2021.
