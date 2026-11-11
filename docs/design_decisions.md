# Design decisions

Short notes on choices that a reviewer might question. Each says what we did, why, and what it costs.

### Judge score = expected value over label probabilities, not a sampled label
The judge reads the next-token probability of each rubric label ("0"/"1"/"2" for scaffolding, "1"-"5" for helpfulness) after a pre-filled `SCAFFOLDING: ` and reports the expectation (G-Eval, Liu et al. 2023). This gives a continuous, deterministic score in [-1, 1]: smoother as an RL reward, and more sensitive to bias (a dialect can shift probability mass without flipping the top label). The discrete label (argmax) is used for agreement with humans, so E1 has no thresholds fit to the labels. Cost: needs open weights or an API with log-probs; the API sanity judge falls back to parsing generated text.

### Helpfulness is conditioned on the judge's top scaffolding label; rationale comes last
Prompt order is SCAFFOLDING, HELPFULNESS, REASON on one greedy path. The rationale is generated *after* the scores, so it is a post-hoc explanation of exactly those scores. That is what E4 needs (does the stated reason match what actually moves the score?). Asking for the reason first would make it chain-of-thought and change the scores.

### The rubric never mentions dialect or names
We audit the judge as a practitioner would naturally write it. Debiased variants come later (stage 4) and are separate rubric/method versions.

### Rubric versions are files; scores record a hash
`src/tutorbias/judge/prompts/rubric_v*.txt`. Every score row carries `rubric_version` and `rubric_hash`, so scores from different rubric iterations can't be mixed by accident.

### Labels split into dev and test
Iterating on a rubric while measuring agreement on the same labels overfits the rubric to those labels. We iterate on dev (50%) and report test once. The gate (kappa >= 0.6, Spearman >= 0.5) is in the config before any labels exist.

### Per-group agreement reuses the SAE labels
Labels are given on the SAE message. We assume the correct label for a reply doesn't depend on the student's dialect or name (that is the counterfactual-fairness definition the project uses). Under that assumption, the same label is ground truth for every variant, so judge-human agreement can be computed per group with no extra labeling. If someone argues a reply *should* be judged differently for a different student, this assumption is where to push.

### Three disjoint item splits
`rl_train` (2000, GSM8K train), `audit` (300, GSM8K train, disjoint), `test` (300, GSM8K test). The judge is calibrated and audited on questions the tutor never trains on, and tutor evaluation uses GSM8K's own test split.

### Item filtering
Integer non-negative answers, 2-6 solution steps, and the answer must not already appear in the question (otherwise answer-leak can't be measured). Counts of dropped items are in `data/processed/items_stats.json`.

### Stuck messages: 4 types x 3 phrasings, base phrasing varies by item
Types: tried a partial step, wrong answer (the second-to-last step's result, i.e. "forgot the last step"), "I don't get it", and "just tell me the answer". A message never contains the correct answer, and a message type is skipped for an item if any of its numbers equals the answer. The paraphrase control is a different SAE phrasing of the same type.

### Rule-based dialects via Multi-VALUE, in a separate environment
LLM rewrites "in AAVE" tend to produce caricature; Multi-VALUE applies documented eWAVE features and records which ones fired, which stage 3 needs for feature ablation. Multi-VALUE pins spaCy 3.7.4 / nltk 3.5 / stanza 1.8.2, so it lives in `tutorbias-data` (Python 3.10), separate from the GPU env. Variants that change any number are flagged (`numbers_ok`) and excluded from the audit; variants that change nothing are recorded (`changed`), and the audit reports results with and without them.

### ESL is a feature list, not a variety
Multi-VALUE has no learner-English variety. We use features documented in second-language English (article omission, plural and 3sg -s omission, auxiliary drop in questions, bare past). Several overlap with AAE features (e.g. 3sg -s absence), so the ESL and AAE groups are not independent. Stage 3 ablation by feature handles this better than group-level comparisons.

### Names
Black/white names are from Bertrand & Mullainathan (2004). Hispanic and South Asian lists are placeholders marked `needs_check` in `data/names/names.yaml` and should be verified (Rosenman et al. 2023 for Hispanic) before the final audit. The name appears as "Hi, I'm {name}." before the SAE message.

### Math only, no reading-comprehension items (for now)
The design doc mentions short reading items. They have no numeric answer, so answer-leak detection and the step-based stuck messages don't apply, and the scaffolding rubric gets fuzzier. We keep the core tier to GSM8K and add reading items only if time allows.

### Reply pool by style prompting
To span "gives the answer" to "pure hint", replies are generated with 8 style instructions (including vague and misleading ones) from Qwen2.5-1.5B and 7B, optionally an API model. The style tag is only a sampling device; human labels and the judge decide what a reply actually is. Pool: 5 replies per audit question, styles balanced, 1500 total.

### W&B is optional
Every stage writes its outputs and stats locally. W&B adds run tracking and dataset versioning when credentials exist and is a no-op otherwise (tests and CI run with `WANDB_MODE=disabled`).
