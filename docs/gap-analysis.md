# Amazon ML Challenge 2026: Gap Analysis

**Document Path:** `docs/gap-analysis.md`  
**Status:** ACTIVE AUDIT  

---

### Comparative Analysis: Requirement vs. Current Implementation

| ID | Competition Requirement | Current Implementation Status | Identified Gap | Severity | Recommended Action |
| :--- | :--- | :--- | :--- | :---: | :--- |
| **GAP-01** | High Candidate Recall ($\ge 85\%$) | Multi-Channel Blocker v2 reaches **69.97%** recall on India validation targets. | **~30% recall gap** due to severe name corruptions and absent numbers. | **HIGH** | Add character 3-gram inverted indexing to capture misspelled brand tokens. |
| **GAP-02** | Statistical Matching Model | Rule-based exact matchers (Baseline 0A and 0B). | **No machine learning model** currently scores candidate pairs; relies on binary rules. | **CRITICAL** | Implement `src/features.py` and train LightGBM pairwise binary classifier (`src/train.py`). |
| **GAP-03** | Calibrated Threshold for Singletons | Hardcoded binary exact matching. | **No continuous threshold tuning** ($\tau$) exists to balance precision and protect singletons. | **HIGH** | Build threshold grid search ($\tau \in [0.50, 0.95]$) optimizing Macro $F_{0.5}$ in `src/predict.py`. |
| **GAP-04** | Target Disjoint Assignment Constraint | Independent candidate selection in Baseline 0. | Multiple $S1$ entities can currently predict the same $S2/S3$ target, hurting precision. | **MEDIUM** | Implement argmax target assignment post-processing (`src/predict.py`). |
| **GAP-05** | Submission Artifact Generation | Manual scripting / validation utility. | **No automated pipeline** currently generates `candidate_pairs.tsv` and `matching_results.tsv` for test data. | **CRITICAL** | Implement `src/submission.py` executing test candidate generation and matching end-to-end. |
| **GAP-06** | Open-Set France Evaluation | Normalization handles accents; test set contains France; local validation split covers US and India only. | Local validation split does not contain French records because training set lacks France. | **MEDIUM** | Build a synthetic unit testing suite for French corporate formats and addresses. |
| **GAP-07** | Automated Submission Validation | Standalone `utils/validate_submission.py`. | Validator is executed manually rather than integrated as an automated pipeline gate. | **MEDIUM** | Embed `validate_submission.py` execution directly into `src/submission.py`. |
| **GAP-08** | Reproducibility Package & Dependencies | Unpinned local virtualenv (`.venv`). | No pinned `requirements.txt` or unified `run_pipeline.py` script. | **HIGH** | Freeze exact dependencies into `requirements.txt` and create end-to-end reproduction entrypoint. |
