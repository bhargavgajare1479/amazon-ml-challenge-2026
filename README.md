Business Entity Resolution — Amazon ML Challenge 2026

End-to-end pipeline: normalize → block (candidate generation) → feature engineer → train LightGBM → calibrate → tune threshold for F0.5 → predict → write submission files.

CPU-only — no GPU needed (the matcher is LightGBM on similarity features, not a neural net).

Setup
bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
Data layout expected
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
Run

From this project's root folder (the one containing src/):

bash
python3 -m src.pipeline --train-dir dataset/train --test-dir dataset/test --output-dir output

Or, if your data already lives at ./dataset/train and ./dataset/test:

bash
python3 -m src.pipeline

This runs the full pipeline and writes output/matching_results.tsv and output/candidate_pairs.tsv. Along the way it prints:

blocking stats (avg/median/max candidates per entity, reduction ratio vs brute force) — candidate set size now counts toward the final ranking, alongside recall
blocking recall (the hard ceiling on your final score — fix this first if low)
LightGBM training log + feature importances
the tuned decision threshold and validation macro F0.5
local format-validation results (still run the official utils/validate_submission.py from the challenge repo before submitting — this is a quick sanity check, not a replacement)
Project structure
src/
├── config.py       # all tunable constants (blocking width, thresholds, model params)
├── normalize.py     # data loading + text cleaning (abbreviations, punctuation, tokens)
├── blocking.py       # candidate generation: TF-IDF cosine blocking + conditional token fallback
├── features.py         # pairwise similarity features + entity-level aggregate features
├── labels.py             # ground truth parsing
├── split.py                # entity-level train/val split
├── train.py                  # LightGBM training + isotonic calibration
├── evaluate.py                 # local F0.5 scorer + threshold tuning (mirrors the official metric)
├── predict.py                    # test inference, writes the two submission TSVs, local validator
└── pipeline.py                     # orchestrates everything, CLI entry point

Each module can also be imported and used individually — e.g. to only rerun blocking with new settings without retraining the model, import from src.blocking directly in a notebook or script.

Where to tune first

Edit src/config.py. In priority order:

Blocking recall vs. candidate-set size (MIN_SIM, ADAPTIVE_MARGIN, TOP_K_BLOCK, MAX_CANDIDATES_PER_ENTITY) — this is the hard ceiling on everything downstream, and now also graded directly on size. Loosen if recall is low; tighten if recall is already high but the average candidate count feels large.
Feature set (src/features.py) — add features based on error analysis on your worst validation entities.
Threshold grid (THRESHOLD_GRID) — usually doesn't need changing, but widen it if the tuned threshold lands at either edge of the current range.