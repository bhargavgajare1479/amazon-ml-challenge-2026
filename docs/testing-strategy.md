# Amazon ML Challenge 2026: Testing & Quality Assurance Strategy

**Document Path:** `docs/testing-strategy.md`  
**Status:** ACTIVE TESTING SUITE  

---

### 1. Multi-Tier Testing Pyramid

The testing strategy spans four distinct levels of verification:

```
[Level 4: Official Scorer Validation (utils/validate_submission.py)]
                   ▲
[Level 3: End-to-End Pipeline Integration Tests]
                   ▲
[Level 2: Golden Regression & Difficult-Pair Benchmarks]
                   ▲
[Level 1: Fast Component Unit Tests (Normalization, Metric, Features)]
```

---

### 2. Level 1: Unit Testing

Unit tests run in seconds and guard core algorithmic building blocks:

#### A. Metric Verification (`src/evaluate.py`)
* **Test Case 1 (README Verification):**
  * Ground Truth: `['S2-00047', 'S3-00812']`
  * Prediction: `['S2-00047', 'S2-00193', 'S3-00812']`
  * Assertion: $F_{0.5} == 0.71429 \pm 0.0001$. (**PASSED**)
* **Test Case 2 (Singleton True Empty):**
  * Ground Truth: `[]`, Prediction: `[]`
  * Assertion: $F_{0.5} == 1.0$. (**PASSED**)
* **Test Case 3 (Singleton False Positive):**
  * Ground Truth: `[]`, Prediction: `['S2-00001']`
  * Assertion: $F_{0.5} == 0.0$. (**PASSED**)

#### B. Normalization Verification (`src/normalize.py`)
* **Accents:** `"L'Étoile SARL"` $\rightarrow$ `"letoile"` (accent stripped, legal suffix stripped).
* **Digit Separation:** `"No127 Arcot Road"` $\rightarrow$ `"no 127 arcot road"` (unpacked number `127`).
* **Domain Normalization:** `"telefutureindia.com"` $\rightarrow$ compact `"telefutureindia"`.
* **Acronym Generation:** `"Primary Care Group"` $\rightarrow$ `"pc"`.

#### C. Feature Extraction Verification (`src/features.py`)
* Zero division handling on empty strings.
* Symmetrical string distance properties.
* Boolean numeric flags evaluate to valid floats in $[0, 1]$.

---

### 3. Level 2: Golden Regression Tests

Maintains a curated set of known difficult entity pairs extracted from the audit:

| Entity Pair Case | S1 Input | Target Input | Challenge | Expected Pipeline Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **Domain Name** | `Tele Future (India) Ltd` | `telefutureindia.com` | Name transmuted into URL | Blocker compact channel captures pair; feature extractor computes domain match. |
| **Acronym Collision** | `Primary Care Group` | `Primary Care (CA)` | Same name, different state | Model uses address token distance to reject match. |
| **Address Typo** | `85 Wayne Ave, NY` | `85 Wanye Ave, New York` | Street & state typo | House number `85` and Levenshtein score link records. |
| **Missing Address** | `Maure Williams Inc` | `Maure Wilblims (No Addr)` | Address is null in target | Missing address flag set; model evaluates strong name evidence. |

---

### 4. Level 3: Integration Tests

Simulates pipeline execution across a small end-to-end slice (1,000 $S1$ entities):
1. Candidate Blocker generates candidates.
2. Feature Extractor builds feature matrix $\mathbf{X}$.
3. Model generates probability predictions.
4. Post-processing cleans multi-assignments.
5. Evaluator scores output and asserts Macro $F_{0.5} > 0.0$.

---

### 5. Level 4: Official Submission Validation

Before any file is submitted or packaged, it must be validated with `utils/validate_submission.py`:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

#### Automated Gate Checks Enforced:
1. **File Format:** Strict tab-separation (`\t`); rejection if comma-separated.
2. **Row Count:** Exactly 1,732,544 rows in both output files.
3. **No Duplicate Rows:** Exactly one prediction line per test $S1$ entity.
4. **Valid ID Syntax:** Target IDs must begin with `S2-` or `S3-`. Self-matches to `S1-` fail immediately.
5. **ID Existence:** All predicted IDs must exist in `test_source2.tsv` or `test_source3.tsv`.
6. **Subset Verification:** Final matches must be a subset of candidate pairs.
