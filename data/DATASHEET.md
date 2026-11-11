# Datasheet

Following Gebru et al., "Datasheets for Datasets" (2021). Numbers marked TBD are filled from the `*_stats.json` files after the DCC runs.

## Motivation
Built to test whether reward models score identical tutoring help differently depending on how a student writes (dialect) or their name, and whether that bias carries into a tutor trained with RL. Not intended for training production tutors or for drawing conclusions about real students.

## Composition
- **Questions:** GSM8K (Cobbe et al., 2021; MIT license). 2000 RL-train and 300 audit questions from GSM8K train (disjoint), 300 test questions from GSM8K test. Filter: integer non-negative answer, 2-6 solution steps, answer not printed in the question. From 7473 train / 1319 test, 6435 / 1125 pass.
- **Stuck-student messages:** one per question, from 4 types x 3 phrasings of hand-written templates. Synthetic. Types are balanced (500/75/75 per type for rl_train/audit/test).
- **Counterfactual variants:** each message in SAE, African American English, Indian English (Multi-VALUE eWAVE varieties), an ESL feature list, and an SAE paraphrase control; audit and test messages also get 16 name variants (4 groups x 2 genders x 2 names). Every variant records the Multi-VALUE features applied. Coverage (share changed, share with numbers intact): TBD.
- **Tutor replies:** ~1500 model-generated replies to audit messages (Qwen2.5-1.5B/7B, optional API model), 8 prompted styles.
- **Human labels:** ~300 replies labeled for scaffolding (-1/0/+1) and helpfulness (1-5) by the project author; ~100 double-labeled by a second labeler.

## Collection and synthetic-data notes
No real students, no minors, no personal data. All student text is template-generated and transformed by rules. **Every variant is synthetic and is labeled as such in this release.** Dialect transforms are cleaner and less natural than real speech; a hand-checked sample (`variant_hand_check.csv`, 40 per group) reports naturalness and meaning preservation: TBD.

## Known limitations
- Rule-based dialect text is a proxy for dialect, not real speech; ESL overlaps with AAE features.
- Name lists for Hispanic and South Asian groups are unvalidated placeholders (see `names/names.yaml`).
- Names were drawn from adult-era sources; a name is a cue, not an identity.
- Math only; single-turn messages.

## Uses to avoid
Don't use the variants to "pass" a fairness check by tuning against them (fairness washing). Publish per-feature results with any summary number.
