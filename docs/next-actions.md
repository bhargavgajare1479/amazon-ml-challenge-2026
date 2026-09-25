# Amazon ML Challenge 2026: Prioritized Next Actions

**Document Path:** `docs/next-actions.md`  
**Status:** ACTIVE SPRINT BACKLOG  

---

### Priority Ranking Rationale

Actions are ordered strictly by **expected contribution to competition performance (Macro $F_{0.5}$) and submission validity**.

```
[P0: Submission Correctness & Fatal Pipeline Gates]
       ▲
[P1: Core ML Model, False-Positive Control & Blocking Recall]
       ▲
[P2: Feature Engineering & Computational Optimization]
       ▲
[P3: Documentation Polish & Non-Functional Refactoring]
```

---

### Top 5 Immediate Technical Actions

#### Action 1: Implement Pairwise Feature Extractor (`code/business_entity_resolution/src/features.py`)
* **Priority:** **P1**
* **Rationale:** The candidate blocker successfully isolates ~57 plausible suspects per entity, but currently we have no way to evaluate fuzzy similarity, token overlap, or numeric agreement.
* **Deliverable:** Vectorized RapidFuzz similarity features (Levenshtein, Token-Sort, Token-Set, Jaro-Winkler) and address numeric overlap functions.
* **Dependencies:** None (`src/normalize.py` is complete).

#### Action 2: Build Supervised Training Dataset & Train First GBDT Model (`src/train.py`)
* **Priority:** **P1**
* **Rationale:** A machine learning model is necessary to advance beyond the 0.383 baseline. GBDT will learn non-linear combinations of name similarity and address evidence.
* **Deliverable:** Balanced candidate training dataset with hard negatives (10:1 ratio) and serialized LightGBM binary classifier (`models/lgbm_matcher.bin`).
* **Dependencies:** Action 1.

#### Action 3: Calibrated Threshold Search & Target Disjoint Post-Processing (`src/predict.py`)
* **Priority:** **P1**
* **Rationale:** Essential for false-positive control. Under Macro $F_{0.5}$, singletons receive score 0.0 for any false match. A conservative threshold ($\tau \ge 0.75$) combined with target uniqueness enforcement will protect singletons and maximize precision.
* **Deliverable:** Vectorized line-search script evaluating Macro $F_{0.5}(\tau)$ on the 30k validation split.
* **Dependencies:** Action 2.

#### Action 4: Character 3-Gram Inverted Indexing for Corrupted Names
* **Priority:** **P1**
* **Rationale:** Recovers the remaining ~30% of candidate blocking misses driven by severe typographical errors and dropped brand tokens, pushing candidate recall towards $\ge 85\%$.
* **Deliverable:** Optional 3-gram index channel integrated into `src/blocking.py`.
* **Dependencies:** Action 3 (benchmark baseline must be established first).

#### Action 5: End-to-End Submission Pipeline & Validator Verification (`src/submission.py`)
* **Priority:** **P0**
* **Rationale:** Verifies the complete test generation loop (`dataset/test/*` $\rightarrow$ `output/candidate_pairs.tsv` + `output/matching_results.tsv`) and validates outputs against `utils/validate_submission.py`.
* **Deliverable:** Working `src/submission.py` achieving official **`PASS`** on test outputs.
* **Dependencies:** Action 3.
