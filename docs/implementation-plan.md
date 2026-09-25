# Amazon ML Challenge 2026: Master Implementation Plan

**Document Path:** `docs/implementation-plan.md`  
**Status:** ACTIVE OPERATIONAL PLAN  

---

### Phase Structure & Task Matrix

| Task ID | Phase | Description | Dep | Owner | Files Affected | Expected Output | Status | Priority |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- | :---: | :---: |
| **TSK-001** | Phase 0 | Comprehensive Workspace & State Reconstruction | None | TEAMMATE_A | `docs/*` | Full documentation set | **DONE** | P0 |
| **TSK-002** | Phase 1 | Comprehensive Data Audit & Autopsy | None | TEAMMATE_A | `reports/dataset_audit.md` | Full 34-point autopsy report | **DONE** | P0 |
| **TSK-003** | Phase 2 | Validation Split Design (30k S1 Stratified) | TSK-002 | TEAMMATE_B | `src/split.py`, `data/validation/` | Stratified holdout set | **DONE** | P0 |
| **TSK-004** | Phase 2 | Official Macro $F_{0.5}$ Evaluation Metric Harness | None | TEAMMATE_B | `src/evaluate.py` | Unit-tested scorer | **DONE** | P0 |
| **TSK-005** | Phase 3 | Multi-View Normalization Engine | TSK-002 | TEAMMATE_A | `src/normalize.py` | Accent, suffix, digit parsing | **DONE** | P1 |
| **TSK-006** | Phase 4 | Multi-Channel Inverted Index Candidate Blocker | TSK-005 | TEAMMATE_A | `src/blocking.py` | Inverted index blocker | **DONE** | P1 |
| **TSK-007** | Phase 6 | Baseline 0 Implementation & Validation Scoring | TSK-004 | TEAMMATE_B | `scripts/evaluate_baseline0.py` | Baseline 0A/0B scores | **DONE** | P1 |
| **TSK-008** | Phase 5 | Pairwise Feature Engineering Module (`src/features.py`)| TSK-005 | TEAMMATE_A | `src/features.py` | Vectorized RapidFuzz features | **IN PROGRESS** | P1 |
| **TSK-009** | Phase 4 | Supervised Pairwise Label & Candidate Dataset Generator | TSK-006 | TEAMMATE_B | `src/labels.py` | Balanced train pairs | **TODO** | P1 |
| **TSK-010** | Phase 7 | Supervised GBDT Matcher Training Pipeline | TSK-008,9 | TEAMMATE_A | `src/train.py` | Serialized GBDT model | **TODO** | P1 |
| **TSK-011** | Phase 8 | Conservative Threshold Line Search ($\tau^*$) | TSK-010 | TEAMMATE_B | `scripts/tune_threshold.py` | Calibrated $\tau^* \ge 0.75$ | **TODO** | P1 |
| **TSK-012** | Phase 8 | Target 1-to-1 Disjoint Assignment Post-Processor | TSK-010 | TEAMMATE_B | `src/predict.py` | Target uniqueness pruning | **TODO** | P1 |
| **TSK-013** | Phase 9 | Error Analysis: FP/FN Breakdown on Validation Set | TSK-011 | TEAMMATE_A | `reports/error_analysis.md` | Failure taxonomy report | **TODO** | P2 |
| **TSK-014** | Phase 10| 3-Gram Inverted Index for Corrupted Names | TSK-013 | TEAMMATE_A | `src/blocking.py` | $\ge 85\%$ Candidate Recall | **TODO** | P2 |
| **TSK-015** | Phase 11| End-to-End Test Inference Execution | TSK-011 | TEAMMATE_B | `src/predict.py` | Raw test predictions | **TODO** | P0 |
| **TSK-016** | Phase 11| Submission Artifact Generation (`output/*.tsv`) | TSK-015 | TEAMMATE_B | `src/submission.py` | Matching & Candidate TSVs | **TODO** | P0 |
| **TSK-017** | Phase 12| Official Submission Verification Run | TSK-016 | TEAMMATE_A | Terminal / Logs | `validate_submission.py` PASS | **TODO** | P0 |
| **TSK-018** | Phase 13| Complete Methodology Document | TSK-017 | TEAMMATE_A | `Documentation_template.md` | Technical write-up complete | **TODO** | P1 |
| **TSK-019** | Phase 13| Reproducibility Packaging & ZIP Generation | TSK-017,18| TEAMMATE_B | `<team_name>_submission.zip`| Verified standalone ZIP | **TODO** | P0 |

---

### Immediate Sprint (Next 3 Milestones)

```
[Milestone 1: Pairwise Features & Training Set]
  ├── Complete code/business_entity_resolution/src/features.py (TSK-008)
  └── Create balanced train candidate pairs via src/labels.py (TSK-009)

[Milestone 2: First ML Model & Threshold Tuning]
  ├── Train LightGBM model on extracted features (TSK-010)
  ├── Search optimal conservative threshold tau on validation split (TSK-011)
  └── Apply 1-to-1 target constraint post-processing (TSK-012)

[Milestone 3: Benchmark & Error Analysis]
  ├── Evaluate Macro F0.5 against Baseline 0 (Target: >= 0.65)
  └── Log detailed results in experiments/experiments.csv
```
