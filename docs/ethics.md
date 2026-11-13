# Ethics: dialects, identity, and how the variants are built

This project changes how a synthetic student writes and checks whether AI graders treat the same help differently. Done carelessly, that kind of work can reproduce the stereotypes it is meant to measure. These are the rules the project follows, and why.

## What we claim and what we don't

- **A dialect is not a race or a nationality.** African American English (AAE) is a rule-governed variety of English used by many, but not all, Black Americans, and by some people who are not Black. Indian English is a national variety used by speakers of many backgrounds. Our groups are defined by *grammatical features*, and every result is reported as "the grader responds to these features", never as "the grader is biased against Black or Indian students".
- **Synthetic text is not real speech.** Rule-based variants are cleaner and more uniform than how anyone actually writes. A score gap shows sensitivity to dialect features; it doesn't estimate real-world harm to real students.
- **Learner English is not a dialect.** The `l2` group uses features common in second-language English. It describes a stage of learning, not an ethnic or national group.
- **Names are cues, not identities.** A name signals perceived race, ethnicity, or gender to a model. We measure the model's reaction to that cue only.

## Wording

| We say | We avoid | Why |
|---|---|---|
| Mainstream American English (MAE) | "Standard" or "correct" English | Implies other varieties are substandard; MAE is the term used in current sociolinguistics and NLP fairness work |
| African American English (AAE) | "AAVE" in prose, "broken English", "slang" | AAE is the broader linguistic term; the code keeps Multi-VALUE's class name `AfricanAmericanVernacular` only because that's the library's API |
| dialect, variety | "nonstandard", "incorrect grammar" | Dialect features are systematic, not errors |
| learner (L2) English | "bad English" | Learner features are a normal part of acquiring a language |

## How variants are built, with safeguards against caricature

1. **Documented features only.** Variants are produced by Multi-VALUE (Ziems et al., 2023), which implements features from eWAVE, the electronic World Atlas of Varieties of English (Kortmann, Lunkenheimer & Ehret, 2020). We never ask an LLM to "write like" a group.
2. **Only pervasive features.** For AAE and Indian English we keep only features eWAVE rates **A: pervasive or obligatory** in that variety. Features rated B or C (less common or rare) are not used. The full list, with eWAVE numbers and examples, is in [dialect_features.md](dialect_features.md).
3. **A blocklist with reasons.** Features that are wrong in our context are removed, with the reason recorded in `configs/data.yaml`. Examples: *y'all* (our student addresses one tutor, so a plural pronoun would be ungrammatical) and *fixin' to* (a Southern US form that Multi-VALUE files under the same eWAVE feature as AAE *finna*).
4. **No stacking.** Each allowed feature is used in a given message with probability 0.5, so a message doesn't pile on every feature at once, which reads as exaggerated. The features used in each variant are recorded.
5. **No untracked slang.** Multi-VALUE also rewrites "going to" to "I'ma", "have to" to "gotta", and "trying to" to "tryna" for *every* dialect, outside any eWAVE feature. We disable those substitutions and keep only its spelling-repair rules.
6. **Meaning and numbers are preserved.** A variant that changes any number is excluded from the audit.
7. **Native-speaker validation** before the variants are used for any reported result: see [dialect_validation.md](dialect_validation.md). Features that raters flag as unnatural or stereotyped are dropped by a rule fixed in advance, and the variants are rebuilt.

## Remaining limitations (reported on the poster and in the README)

- Even pervasive features are not used by every speaker, and written messages to a tutor are a more formal register in which speakers may use fewer of them.
- AAE and learner English share some features (e.g. third-person *-s* absence), so group comparisons overlap. Stage 3 compares individual features.
- Validation uses a small number of raters (classmates or volunteers), not a representative sample of speakers.
- eWAVE ratings describe varieties at a regional level; they don't capture age, region, or individual variation.

## Misuse

The variants and audit could be used to tune a grader until it "passes" this test without being fair in general (fairness washing). We publish per-feature results and the full variant set (marked synthetic), not only a pass/fail number.

## References

- Ziems, C., Held, W., Yang, J., Dhamala, J., Gupta, R., & Yang, D. (2023). Multi-VALUE: A framework for cross-dialectal English NLP. *ACL*.
- Ziems, C., Chen, J., Harris, C., Anderson, J., & Yang, D. (2022). VALUE: Understanding dialect disparity in NLU. *ACL*.
- Kortmann, B., Lunkenheimer, K., & Ehret, K. (Eds.) (2020). *The Electronic World Atlas of Varieties of English* (eWAVE 3.0). https://ewave-atlas.org
- Hofmann, V., Kalluri, P. R., Jurafsky, D., & King, S. (2024). AI generates covertly racist decisions about people based on their dialect. *Nature*, 633, 147-154.
- Blodgett, S. L., Barocas, S., Daumé III, H., & Wallach, H. (2020). Language (technology) is power: A critical survey of "bias" in NLP. *ACL*.
- Sap, M., Card, D., Gabriel, S., Choi, Y., & Smith, N. A. (2019). The risk of racial bias in hate speech detection. *ACL*.
- Green, L. J. (2002). *African American English: A Linguistic Introduction*. Cambridge University Press.
- Rickford, J. R. (1999). *African American Vernacular English: Features, Evolution, Educational Implications*. Blackwell.
- Dulay, H., & Burt, M. (1974). Natural sequences in child second language acquisition. *Language Learning*, 24(1).
- Goldschneider, J. M., & DeKeyser, R. M. (2001). Explaining the "natural order of L2 morpheme acquisition" in English: A meta-analysis of multiple determinants. *Language Learning*, 51(1).
- Bertrand, M., & Mullainathan, S. (2004). Are Emily and Greg more employable than Lakisha and Jamal? *American Economic Review*, 94(4).

Check each citation against the original before it goes on the poster.
