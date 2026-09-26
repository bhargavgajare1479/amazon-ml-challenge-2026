# Bhargavi's solution — TF-IDF blocking + LightGBM

## Approach

1. **Normalize** business names and addresses (`src/normalize.py`) — lowercase,
   strip legal suffixes/punctuation, transliterate non-Latin scripts
   (Indian-language names/states) to ASCII with `anyascii`.
2. **Block** (`src/blocking.py`): TF-IDF word vectors, candidate pairs generated
   per country, top-20 nearest Source-1 neighbors per Source-2/3 record (K=20).
3. **Features** (`src/features.py`): 28 pairwise similarity features (RapidFuzz
   string metrics, TF-IDF cosine, address/zip/pin overlap, etc.) per candidate pair.
4. **Model**: LightGBM binary classifier on the pairwise features, scored to a
   match probability.
5. **Threshold**: 0.7 on the match probability, tuned on held-out validation for
   Macro F0.5. Each Source-2/3 record is assigned to at most one Source-1 entity
   (its highest-scoring match above threshold).

## Results

- Blocking recall@20 = 0.956 (candidate set contains the true match 95.6% of
  the time, measured on a 10% sample of train).
- Validation Macro F0.5 = 0.9617 at threshold 0.7.
- 11.8% of test Source-1 entities predicted as having no match (empty).

## Files

- `src/normalize.py` — text normalization / transliteration
- `src/blocking.py` — TF-IDF candidate generation
- `src/features.py` — pairwise similarity features
- `src/run_test_blocking.py` — runs blocking on the test set, writes candidate pairs
- `src/predict_test.py` — scores candidates with the trained model, writes both
  submission files
- `src/lgbm_v1.txt` — trained LightGBM model (text/booster format)
- `src/explore.ipynb` — data exploration notebook

## How to run

From this folder (`code/bhargavi/`), with `dataset/` and `output/` present two
levels up (repo root):

```bash
pip install -r requirements.txt

python src/run_test_blocking.py --data ../../dataset --k 20
python src/predict_test.py --model src/lgbm_v1.txt --threshold 0.7 --out ../../output
```

This writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`.
