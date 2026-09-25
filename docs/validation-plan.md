# Amazon ML Challenge 2026: Validation Plan & Evaluation Harness

**Document Path:** `docs/validation-plan.md`  
**Status:** ACTIVE  

---

### 1. Validation Design Objectives

Entity resolution validation must satisfy three fundamental principles to avoid catastrophic offline-to-online divergence:
1. **Zero Entity Leakage:** Validation entities must be split strictly at the Source 1 reference entity level. Never randomly split candidate pairs.
2. **Realistic Distractor Pressure:** Searching for matches among only validation targets creates an artificially easy problem. Validation $S1$ entities must search against the **full multi-million target pool** ($S2 + S3$) to experience real-world distractor density and false positive competition.
3. **Exact Metric Replication:** Score models exclusively using the official entity-level Macro $F_{0.5}$ metric with full singleton credit/penalties.

---

### 2. Validation Split Strategy

Implemented in `code/business_entity_resolution/src/split.py`:
* **Sample Size:** Exactly **30,000 Source 1 entities** (~1.36% of training data).
* **Random Seed:** Fixed seed `42` (`np.random.default_rng(42)`).
* **Joint Stratification:** Stratified across 10 strata combining `country` $\times$ `match_count_bin`:
  * `US__0_singleton` (3.35%)
  * `US__1_match` (3.25%)
  * `US__2_matches` (10.21%)
  * `US__3_4_matches` (27.61%)
  * `US__5plus_matches` (15.56%)
  * `India__0_singleton` (2.24%)
  * `India__1_match` (2.15%)
  * `India__2_matches` (6.79%)
  * `India__3_4_matches` (18.38%)
  * `India__5plus_matches` (10.46%)

#### Validation vs. Population Fidelity Comparison

| Metric | Full Training Set | Validation Split (30k) | Delta |
| :--- | :---: | :---: | :---: |
| **Total S1 Entities** | 2,206,821 | 30,000 | - |
| **Singleton Percentage** | **5.585%** | **5.587%** | $+0.002\%$ |
| **US Entity Share** | **59.98%** | **59.98%** | $0.000\%$ |
| **India Entity Share** | **40.02%** | **40.02%** | $0.000\%$ |
| **Mean Matches / S1** | **3.461** | **3.462** | $+0.001$ |
| **Total True Matches** | 7,638,365 | 103,858 | - |

---

### 3. Step-by-Step Validation Simulation Protocol

```
Step 1: Load 30,000 Validation S1 records (val_source1.tsv)
Step 2: Stream & Index ALL 4.13M India & 6.19M US Target records (S2 + S3)
Step 3: Query Blocker -> Candidate Sets per S1
Step 4: Compute Candidate Recall & Candidate Reduction Ratio
Step 5: Extract Pairwise Features for all Candidate Pairs
Step 6: Run ML Model Inference -> Probability p for each pair
Step 7: Search Optimal Threshold tau in [0.50, 0.95]
Step 8: Enforce Target 1-to-1 Disjoint Assignment Constraint
Step 9: Compute Macro F0.5, Macro Precision, Macro Recall, Singleton Acc
Step 10: Record Experiment in experiments/experiments.csv
```

---

### 4. Official Evaluation Implementation Details

Implemented in `src/evaluate.py`:
* **Mathematical Formula:**
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
* **Singleton Edge Cases:**
  * True matches $= \emptyset$, Predicted matches $= \emptyset \implies \text{Score} = 1.0$.
  * True matches $= \emptyset$, Predicted matches $\neq \emptyset \implies \text{Score} = 0.0$.
* **Non-Singleton Edge Cases:**
  * True matches $\neq \emptyset$, Predicted matches $= \emptyset \implies \text{Score} = 0.0$.
  * True matches $\neq \emptyset$, True Positives $= 0 \implies \text{Score} = 0.0$.

---

### 5. Identified Validation Risks & Weaknesses

1. **Absence of French Ground Truth in Training:**  
   * *Issue:* The training dataset contains only US and India records; France appears exclusively in the test set. Local validation cannot evaluate French entity resolution directly.
   * *Mitigation:* Ensure normalization and blocking channels do not contain US/India-specific conditional branches; test French parsing using synthetic unit tests.
2. **Current Benchmark Partitioning:**  
   * *Issue:* `benchmark_blocking.py` was initially evaluated on the India partition (12,006 entities).
   * *Action:* Extend the benchmark script to run end-to-end across both India and US partitions simultaneously.
