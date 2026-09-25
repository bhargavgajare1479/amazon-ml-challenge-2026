# Amazon ML Challenge 2026: Official Project Status Report

**Document Path:** `docs/status.md`  
**Current Date:** 2026-09-25  
**Competition Phase:** Feature Engineering & Supervised Modeling (Phase 5 / 8)  
**Overall Status:** GREEN (On track, validation harness verified, baseline established)  

---

### 1. Executive Status Summary

| Metric Dimension | Current Value | Verification Source |
| :--- | :--- | :--- |
| **Best Local Validation Macro $F_{0.5}$** | **0.77132** | `reports/lgbm_matcher_val_results.json` |
| **Best Public Leaderboard Score** | **UNKNOWN** | No submission uploaded to portal yet |
| **Best Model Currently Available** | **LightGBM Matcher v1 ($\tau^*=0.85$)** | `models/lgbm_matcher.bin` |
| **Best Blocking Strategy Available** | **Multi-Channel Blocker v2** | `src/blocking.py` |
| **Current Candidate Recall** | **69.97%** | `reports/blocking_benchmark_india.json` |
| **Current Candidate Reduction Ratio** | **0.9999861 (99.9986%)** | `reports/blocking_benchmark_india.json` |
| **Mean Candidates per $S1$ Entity** | **57.65 candidates** | `reports/blocking_benchmark_india.json` |
| **Current Final Precision** | **85.47%** | LightGBM v1 on validation split |
| **Current Final Recall** | **63.32%** | LightGBM v1 on validation split |
| **Current Singleton Accuracy** | **86.59%** | LightGBM v1 on validation split |

---

### 2. Work Stream Progress

* **Completed:**
  * Exhaustive 34-point dataset audit (`reports/dataset_audit.md`).
  * Stratified 30,000 $S1$ validation split with zero entity leakage (`data/validation/`).
  * Official Macro $F_{0.5}$ evaluation harness with verified singleton rules (`src/evaluate.py`).
  * Multi-view normalization engine (`src/normalize.py`).
  * Multi-channel inverted index candidate blocker (`src/blocking.py`).
  * Baseline 0A and 0B benchmark evaluations (`scripts/evaluate_baseline0.py`).
  * Experiment tracking ledger (`experiments/experiments.csv`).
  * Comprehensive technical documentation suite (`docs/*`).
* **In Progress:**
  * Pairwise feature extraction module (`src/features.py`).
* **Blocked:**
  * None.

---

### 3. Top 5 Next Immediate Actions

1. Complete `code/business_entity_resolution/src/features.py` (RapidFuzz string distances, numeric overlap).
2. Generate balanced pairwise training dataset with hard negatives (`src/labels.py`).
3. Train first GBDT binary classifier (LightGBM) to beat the 0.383 baseline.
4. Execute conservative threshold line-search ($\tau^* \ge 0.75$) and apply target disjoint constraint.
5. Build test candidate generation and inference script (`src/submission.py`) and validate via `utils/validate_submission.py`.

---

### 4. Primary Known Technical Risks

1. **Candidate Recall Ceiling (70.0%):** Blocker must be expanded to character 3-grams to capture severe name typos and achieve $\ge 85\%$ recall.
2. **Singleton False Merges:** False matches on singletons yield score 0.0; model requires elevated probability threshold ($\tau \ge 0.75$).
3. **Open-Set France Market:** Test set contains 15% French records unseen in training; normalizers must remain language-agnostic.
