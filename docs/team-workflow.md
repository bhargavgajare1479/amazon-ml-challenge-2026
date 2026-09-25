# Amazon ML Challenge 2026: Team Collaboration & Workflow Guide

**Document Path:** `docs/team-workflow.md`  
**Status:** ACTIVE COLLABORATION PROTOCOL  

---

### 1. Work Streams & Ownership Matrix

To prevent code collisions and maximize experimental throughput, the project is structured into parallel engineering tracks:

| Track | Domain | Primary Focus | Modules Owned | Primary Owner | Secondary Owner |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Track A** | Data / Audit | Data integrity, cardinality, missingness | `reports/`, `scripts/audit_*.py` | TEAMMATE_A | TEAMMATE_B |
| **Track B** | Normalization | Multilingual text cleaning, tokenizers | `src/normalize.py` | TEAMMATE_A | TEAMMATE_C |
| **Track C** | Blocking | Inverted indexes, candidate recall | `src/blocking.py` | TEAMMATE_A | TEAMMATE_D |
| **Track D** | Feature Engineering | Pairwise similarity metrics, RapidFuzz | `src/features.py` | TEAMMATE_B | TEAMMATE_A |
| **Track E** | ML Modeling | Model training, tuning, calibration | `src/train.py`, `src/predict.py` | TEAMMATE_B | TEAMMATE_C |
| **Track F** | Evaluation | Validation harness, Macro $F_{0.5}$ | `src/evaluate.py`, `src/split.py` | TEAMMATE_C | TEAMMATE_B |
| **Track G** | Error Analysis | FP/FN categorization, hard negatives | `reports/error_analysis.md` | TEAMMATE_D | TEAMMATE_A |
| **Track H** | Submission / Repro | Packaging, validation, documentation | `src/submission.py`, `Documentation_template.md` | TEAMMATE_D | TEAMMATE_C |

---

### 2. Dependency Graph Across Tracks

```
Track A (Audit) ──► Track B (Normalization) ──► Track C (Blocking)
                                                        │
                         ┌──────────────────────────────┴───────────────┐
                         ▼                                              ▼
               Track D (Features)                              Track F (Validation)
                         │                                              │
                         ▼                                              ▼
               Track E (ML Modeling) ──────────────────────────► Track G (Error Analysis)
                         │
                         ▼
               Track H (Submission & Reproducibility)
```

---

### 3. File Coordination & Anti-Collision Rules

1. **Single-Writer Rule:** Never edit another teammate’s active module without coordination.
   * `src/normalize.py` and `src/blocking.py` $\rightarrow$ Lead: **TEAMMATE_A**.
   * `src/features.py` and `src/train.py` $\rightarrow$ Lead: **TEAMMATE_B**.
   * `src/evaluate.py` and `src/split.py` $\rightarrow$ Lead: **TEAMMATE_C**.
   * `src/submission.py` and packaging $\rightarrow$ Lead: **TEAMMATE_D**.
2. **Read-Only Data Layer:** Raw data in `dataset/` is strictly immutable. No script may modify or write into `dataset/`.
3. **Derived Data Location:** All intermediate tables, candidate lists, and validation splits must be written to `data/` or `output/`.

---

### 4. Experiment Naming & Tracking Conventions

Every experimental run must be logged in `experiments/experiments.csv` using the standard format:

* **ID Format:** `EXP-XXX` (e.g. `EXP-004`).
* **Required Metadata:**
  * `date`: YYYY-MM-DD
  * `model`: Descriptive name (e.g. `LightGBM_v1_RapidFuzz`)
  * `features`: Summary of feature set (e.g. `levenshtein+jaccard+numeric`)
  * `blocking`: Active blocking configuration (e.g. `multi_channel_v2`)
  * `validation_scheme`: E.g. `Stratified_30k_S1_Val`
  * `cv_score`: Official Macro $F_{0.5}$
  * `precision`: Macro Precision
  * `recall`: Macro Recall
  * `candidate_recall`: Blocker candidate recall
  * `candidate_reduction`: Blocker reduction ratio
  * `status`: `DONE` / `FAILED` / `IN_PROGRESS`
  * `notes`: Key findings and failure modes

---

### 5. Shared Decision-Making Protocol

1. **Leaderboard vs. Local Validation:**
   * Local validation on the stratified 30k split is the single source of truth.
   * A technique is never deployed to test inference unless it demonstrates statistically significant gain on local validation.
2. **Blocking vs. Model Trade-Off:**
   * Candidate blocking recall must be $\ge 80\%$ before tuning model hyperparameters. If the candidate net misses a true match, no model can recover it.
3. **Threshold Selection:**
   * Thresholds must be chosen to protect singletons (optimizing Macro $F_{0.5}$, never raw accuracy).
