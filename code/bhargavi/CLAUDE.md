# Project: Amazon ML Challenge 2026 — Business Entity Resolution
Deadline: Sept 27, 2026, 11:59 PM IST. Speed matters more than perfection.

## Task
For each Source 1 record, find matching Source 2/Source 3 records (same real business).
Output: output/matching_results.tsv and output/candidate_pairs.tsv (tab-separated,
one row per S1 entity, comma-separated ID lists, empty if no match).
Metric: F0.5 macro-averaged per S1 entity. Precision-heavy. Singletons (no match)
score 1.0 if predicted empty, 0.0 otherwise.

## Rules (must follow)
- All files are TSV: always read with sep="\t", dtype=str, keep_default_na=False
- Test set has France (not in train). Never hard-code US/India only.
- No external APIs, geocoding, or web lookups of businesses (disqualification).
- Any pretrained model must be MIT/Apache 2.0 and <= 8B params.
- Validate outputs with: python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
- After finishing any step, update the Status section in CLAUDE.md.

## Setup
- Windows laptop, i7, 16 GB RAM, GTX 1650 4GB. Use CPU-friendly methods.
- Python 3.11 venv in ./venv. Code goes in code/business_entity_resolution/src/
- Data files are large: never print whole files, use .head() or samples.

## Plan
Normalize text -> TF-IDF char n-gram blocking (top-K per S1, same country)
-> pairwise features (rapidfuzz, TF-IDF cosine, zip/pin match)
-> LightGBM classifier -> tune threshold on validation F0.5
-> each S2/S3 record assigned to at most one S1.

## About me
I'm new to ML. Explain what you're doing briefly in plain language.

## Status (update as you go)
- [x] Data exploration: 5.6% singletons, most S1 have 2-5 matches,
      each S2/S3 matches at most one S1, ~25% of S2/S3 match nothing.
      Some S2/S3 names are gibberish (match by address only); some names/states
      are in Indian scripts (use anyascii to transliterate).
- [x] src/normalize.py, src/blocking.py written (from claude.ai chat)
- [x] Blocking recall measured on 10% train sample: recall@20 = 0.954,
      recall@30 = 0.960. Using K=20.
- [ ] All-empty test submission (checks format)
- [x] Features + LightGBM v1 trained, val F0.5 = 0.9617 at threshold 0.7
      (106 rounds). predict_test.py written, not yet run.
- [x] v1 leaderboard score = 0.778. Val 0.9617 was misleading: the 10% train
      sample is 10x less crowded (fewer candidates per S1) than test, so
      validation overstated precision. Team best so far: 0.906 (teammates).
- [x] v2: full-train blocking + retrain, val 0.79 (val tracks LB).
- [x] v3 (current best, team best): blocking v2 (TF-IDF words + word pairs, max_df=2000,
      top-20 per S1 + reverse top-3), recall 0.958 (US 0.972, India 0.937).
      lgbm_v3.txt, 126 rounds, val F0.5 0.9328 @ thr 0.7, LB 0.920 (rank 2118).
      Outputs backed up in output_v3/. Val runs ~1 pt above LB (steady offset).
      predict_test.py takes ~9 min on test.
- [x] Error analysis (explore.ipynb, cells at bottom): points lost FN 0.035, FP 0.0175,
      blocking 0.015 (ceiling 0.985). No ID leak. S2/S3 records with EMPTY address are
      3.3% of records but ~24% of FNs, 18% of FPs, 24% of blocking misses. Half of
      "stolen" FPs are exact same-name ambiguity between S1s.
- [x] v4 (CURRENT BEST, team best): 31 features (v3's 28 + same-name ambiguity features:
      name_exact, cand_n_same_name_s1, s1_n_same_name_cands), lgbm_v4.txt, 127 rounds, val logloss 0.0324
      (v3 0.0350). Val F0.5 0.9366 @ thr 0.7 (v3 0.9328). Test predict 12.5 min,
      6.6% S1 empty, validator PASS. LB 0.9238 (rank ~2410, drifts as others submit).
      Outputs backed up in output_v4/. Val runs ~1.3 pts above LB; val gains
      transfer ~1:1 to LB.
      v4 is what gets packaged. (To package v3 instead, restore BOTH
      features_v3.py and predict_test_v3.py.)
- [ ] Packaging by 7-8 PM: README, requirements.txt, Documentation_template.md, zip.