# Timeline (finish Sat Nov 21, Expo Mon Nov 23, 2026)

The design doc's phases were in weeks; this fits them to 10 days. The bottlenecks are DCC queue time, GRPO wall-clock, and the 300 human labels, not coding time. Each gate is a go/no-go.

| Dates | Work | Gate |
|---|---|---|
| Wed Nov 11 | Repo, stage 1 code (done). DCC env setup; build items/messages; submit variant + reply jobs. | |
| Thu Nov 12 | Reply pool + label set ready. Judge scores the label set. Pilot-label 30 dev items, fix the guide. Line up the second labeler. | |
| Fri Nov 13 | Finish 300 labels (second labeler: 100). Calibrate on dev; revise rubric if needed; then test once. | **Gate 1:** judge trustworthy (kappa >= 0.6 on test). If not: rubric v2 by Sat, no audit until it passes. |
| Sat-Sun Nov 14-15 | Stage 2 audit: judge + 2 RewardBench RMs on every reply x variant. Stage 3: feature ablation, rationale faithfulness, probes. GRPO code + 50-step smoke run. | **Gate 2:** audit result exists. This is already a poster result even if RL fails. |
| Mon Nov 16 | Launch Tutor A (GRPO, original judge). | Training stable after ~100 steps? If not, fix reward/KL before scaling. |
| Tue Nov 17 | Pick the debiasing method from stage 2-3; launch Tutor B. Extra seeds if GPUs allow. | **Gate 3:** if Tutor A isn't stable by Tue night, ship core tier with Tutor A only. |
| Wed-Thu Nov 18-19 | Evaluation E5-E7, reward-hacking review, demo app. | |
| Fri Nov 20 | Poster draft, README results, figures. | |
| Sat Nov 21 | Done. Final reproducibility pass. | |
| Sun Nov 22 | Buffer. | |
