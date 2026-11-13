# Native-speaker validation of the dialect variants

Before any dialect result is reported, speakers of each variety rate a sample of the variants. Features that read as unnatural or stereotyped are removed by a rule fixed in advance (`validation:` in `configs/data.yaml`), and the variants are rebuilt.

## Who rates
- **AAE sheet:** people who speak African American English themselves.
- **Indian English sheet:** people who grew up speaking Indian English.
- **L2 sheet:** English teachers or tutors, or people who learned English as an additional language.

Aim for **2 raters per sheet**. Agreement between them is part of the evidence. Raters only read short synthetic sentences and give opinions; no personal data is collected. Still, check with the course instructor whether this needs any review, since the design doc already asks about IRB for the stretch study. Ask raters for their name only if they want to be acknowledged.

## Steps
1. After `bash scripts/run_stage.sh variants`, the sheets are in `data/validation/rating_sheet_<group>.csv`: 40 rows each, chosen so every feature appears several times.
2. Give each rater a copy (a spreadsheet works). They fill three columns per row and save as `ratings_<group>_<rater>.csv` in `data/validation/` (e.g. `ratings_aae_r1.csv`).
3. Run `.venv-data/bin/python -m tutorbias.data.validate_variants`. It writes `data/validation/report.md` (naturalness, stereotyped rate, rater agreement, decision per feature) and `flagged_features.yaml`.
4. Add the flagged features to that group's `exclude` list in `configs/data.yaml`, rerun `variants`, and regenerate `docs/dialect_features.md` (`python -m tutorbias.data.feature_report`).
5. Put the report's headline numbers in the README and on the poster's limitations panel.

## Instructions to give raters

> You'll see pairs of short messages a student might send to a math tutor. The first is written in Mainstream American English. The second was changed by a computer program to use grammar features of [African American English / Indian English / learner English]. We are checking whether the program's output reads as natural, or as fake or stereotyped. For each pair:
>
> - **natural_1to5:** How natural is the second message as something a real [speaker] might write to a tutor? 1 = nobody would write this, 3 = possible but odd, 5 = completely natural.
> - **stereotyped_yn:** Does it read as exaggerated, mocking, or like a stereotype? (y/n)
> - **meaning_same_yn:** Does it still mean the same thing as the first message? (y/n)
> - **notes (optional):** Which word or phrase felt off, if any.
>
> There are no right answers. Your own judgment is what we need. It takes about 15 minutes.

## Decision rule (fixed before seeing ratings)
A feature is dropped if it appears in at least 5 rated rows and either
- at least 20% of ratings of rows containing it say *stereotyped*, and that is at least 10 points higher than for rows without it, or
- its mean naturalness is below 3/5, and at least 0.5 below rows without it.

A group whose overall mean naturalness stays below 3 is reported as a limitation, and its gap results are flagged as low-confidence.
