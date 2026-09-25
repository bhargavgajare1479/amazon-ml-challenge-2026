# Amazon ML Challenge 2026: Experiment Plan & Tracking Ledger

**Document Path:** `docs/experiment-plan.md`  
**Tracking File:** `experiments/experiments.csv`  
**Status:** ACTIVE  

---

### 1. Completed Experiments

| Exp ID | Date | Description | Cand Recall | Macro Precision | Macro Recall | Macro $F_{0.5}$ | Verdict |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **EXP-001** | 2026-09-25 | Baseline 0A: Exact Stripped Name Only | 35.60% | 42.47% | 35.60% | **0.38328** | Exact name suffers from generic medical/retail collisions across cities. |
| **EXP-002** | 2026-09-25 | Baseline 0B: Exact Name + Exact Clean Address | 8.40% | 16.14% | 8.40% | **0.12290** | Confirmed: requiring exact address matching destroys recall (8.4%). |
| **EXP-003** | 2026-09-25 | Multi-Channel Blocker v2 (Single-pass index) | **69.97%** | N/A | N/A | N/A | Captures 70% of true matches with 57 cands/S1 across 4.13M targets. |

---

### 2. Current In-Progress Experiments

* **EXP-004:** **Pairwise Feature Engineering Module (`src/features.py`)**
  * *Hypothesis:* Combining RapidFuzz string similarities with exact street number equality and postal code matching will provide sufficient signal for a GBDT model to separate true matches from generic same-name imposters.
  * *Metrics:* Feature extraction throughput (pairs/sec), feature correlation with ground truth.

---

### 3. Prioritized Planned Experiments

#### EXP-005: First GBDT Pairwise Matcher (LightGBM Baseline)
* **Priority:** P1 (Highest Information Gain)
* **Hypothesis:** A Gradient Boosted Decision Tree trained on pairwise features will dramatically outperform rule-based baselines by learning joint name-address interactions (e.g. strong name + weak address vs weak name + strong address).
* **Baseline to Beat:** Baseline 0A (Macro $F_{0.5} = 0.38328$).
* **Metric:** Macro $F_{0.5}$, Singleton Accuracy.
* **Expected Outcome:** Macro $F_{0.5} \ge 0.58 - 0.65$.
* **Computational Cost:** Moderate (~5 minutes training on 100k pairs, ~2 minutes inference).
* **Success Criterion:** Macro $F_{0.5} > 0.50$ on validation set.
* **Rollback Criterion:** Macro $F_{0.5} \le 0.38$.

#### EXP-006: Conservative Decision Threshold Optimization ($\tau^*$)
* **Priority:** P1
* **Hypothesis:** Because singletons constitute 5.6% of entities and score $0.0$ on any false merge, raising the classification threshold $\tau$ from default $0.50$ to $\approx 0.75 - 0.85$ will reduce false merges and maximize Macro $F_{0.5}$.
* **Baseline:** Default threshold $\tau = 0.50$.
* **Metric:** Macro $F_{0.5}$ across $\tau \in [0.50, 0.95]$ in increments of $0.05$.
* **Expected Outcome:** $+0.04$ to $+0.08$ boost in Macro $F_{0.5}$ due to singleton protection.
* **Computational Cost:** Negligible (vectorized threshold search over cached probabilities).
* **Success Criterion:** Monotonic improvement in Macro $F_{0.5}$ at elevated $\tau$.

#### EXP-007: Target 1-to-1 Disjoint Assignment Constraint
* **Priority:** P1
* **Hypothesis:** Enforcing that every target record $S2/S3$ is assigned to at most one $S1$ entity (matching empirical ground truth where `max_s1_per_target = 1`) will eliminate duplicate false positive assignments and boost precision.
* **Baseline:** Independent unconstrained candidate selection.
* **Metric:** Macro Precision and Macro $F_{0.5}$.
* **Expected Outcome:** $+0.02$ to $+0.04$ boost in Macro Precision.
* **Computational Cost:** Low (hash-map argmax grouping).
* **Success Criterion:** Precision improvement with zero decrease in valid true positives.

#### EXP-008: Character 3-Gram Inverted Index for Corrupted Names
* **Priority:** P2
* **Hypothesis:** The remaining 30% of candidate blocking misses are driven by severe name corruptions (e.g. `Maure Wilblims` vs `Maure Williams`) that miss exact token keys. Adding a character 3-gram inverted index will push candidate recall from 70% to $>85\%$.
* **Baseline:** Multi-Channel Blocker v2 (69.97% recall).
* **Metric:** Candidate Recall, Candidates per $S1$.
* **Expected Outcome:** Candidate recall $\ge 85\%$ with candidates/S1 $\le 90$.
* **Computational Cost:** Moderate (+2 GB RAM for 3-gram index).
* **Success Criterion:** Candidate recall $\ge 80\%$ with candidates/S1 $\le 100$.
* **Rollback Criterion:** Candidate explosion (candidates/S1 $> 150$).

#### EXP-009: CatBoost vs. LightGBM Model Comparison
* **Priority:** P2
* **Hypothesis:** CatBoost handles heterogeneous categorical and numerical ER interactions more smoothly than standard LightGBM.
* **Baseline:** EXP-005 (LightGBM).
* **Metric:** Macro $F_{0.5}$, training speed, memory footprint.
* **Success Criterion:** $+0.015$ gain in Macro $F_{0.5}$.
