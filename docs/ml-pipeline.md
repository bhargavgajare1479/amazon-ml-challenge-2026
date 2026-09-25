# Amazon ML Challenge 2026: Machine Learning Pipeline Specification

**Document Path:** `docs/ml-pipeline.md`  
**Status:** ACTIVE  

---

### 1. Mathematical Formulation

The task is formulated as a two-stage entity linkage system:
1. **Candidate Retrieval (Blocking):** Map each reference entity $S1_i$ to a sparse set of plausible candidate target records:
   $$\mathcal{C}(S1_i) \subset \{S2 \cup S3\}$$
   such that $|\mathcal{C}(S1_i)| \ll |S2 \cup S3|$ while maximizing the probability that all true matches $\mathcal{T}(S1_i) \subseteq \mathcal{C}(S1_i)$.
2. **Pairwise Probability Estimation (Scoring):** Learn a parameterized mapping:
   $$f_\theta(\mathbf{x}_{ij}) = P(\text{match} \mid \mathbf{x}(S1_i, T_j)) \in [0, 1]$$
   where $\mathbf{x}_{ij}$ is a vector of heterogeneous similarity features between $S1_i$ and candidate target $T_j$.
3. **Decision & Multi-Match Aggregation:**
   $$\hat{\mathcal{M}}(S1_i) = \{ T_j \in \mathcal{C}(S1_i) \mid f_\theta(\mathbf{x}_{ij}) \ge \tau \}$$
   subject to the target uniqueness constraint:
   $$\forall T_j, \quad |\{ S1_i \mid T_j \in \hat{\mathcal{M}}(S1_i) \}| \le 1$$

---

### 2. Metric Hierarchy & Definitions

It is critical to distinguish the three tiers of recall and accuracy in this system:

```
Candidate Recall (Stage 1 Ceiling)
       │  (Proportion of true matches captured by blocker)
       ▼
Matching Recall (Stage 2 Model Coverage)
       │  (Proportion of true matches scored >= threshold)
       ▼
Entity-Level Macro F0.5 (Stage 3 Competition Objective)
          (Harmonic mean favoring precision 2x, averaged per S1)
```

#### A. Candidate Recall (Blocking Ceiling)
$$\text{Recall}_{\text{cand}} = \frac{\sum_{i=1}^N |\mathcal{T}(S1_i) \cap \mathcal{C}(S1_i)|}{\sum_{i=1}^N |\mathcal{T}(S1_i)|}$$
*Current Benchmark:* **69.97%** on India validation partition.

#### B. Pairwise Matching Precision & Recall
$$\text{Precision}_{\text{pair}} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall}_{\text{pair}} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

#### C. Official Entity-Level Macro $F_{0.5}$
For each individual entity $i$:
$$F_{0.5}^{(i)} = \begin{cases} 
1.0 & \text{if } |\mathcal{T}_i| = 0 \text{ and } |\hat{\mathcal{M}}_i| = 0 \\ 
0.0 & \text{if } |\mathcal{T}_i| = 0 \text{ and } |\hat{\mathcal{M}}_i| > 0 \\ 
\frac{1.25 \times \text{Prec}_i \times \text{Rec}_i}{0.25 \times \text{Prec}_i + \text{Rec}_i} & \text{if } |\mathcal{T}_i| > 0 \text{ and } \text{TP}_i > 0 \\ 
0.0 & \text{otherwise} 
\end{cases}$$

Macro Score across all $N$ entities:
$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5}^{(i)}$$

---

### 3. Normalization Pipeline

Implemented in `src/normalize.py`:
* **Unicode / Accents:** NFKD accent removal converts `é`, `è`, `ô`, `ç` into standard ASCII equivalents `e`, `e`, `o`, `c`.
* **Digit Uncoupling:** Splits fused alphanumeric tokens (`No127` $\rightarrow$ `No 127`, `Plot4B` $\rightarrow$ `Plot 4 B`).
* **Legal Suffixes:** Strips corporate markers (`Inc`, `LLP`, `SARL`, `Pvt Ltd`) into a parallel `stripped_legal` view.
* **Domain Normalization:** Strips `.com`, `.in`, `.fr`, `www.` to match URLs with brand names.

---

### 4. Candidate Generation (Blocking)

Operates per country partition with 7 complementary inverted index channels:
1. `Exact Name`: Matches stripped legal names ($\ge 3$ chars).
2. `Compact Name`: Matches space-stripped alphanumeric strings (e.g. `telefutureindia`).
3. `Name Bigram`: Matches first two significant name words (`tele_future`).
4. `Exact Address`: Clean normalized address.
5. `Address Number + Word`: Matches house number + first address token (`127_sriananthammalcompx`).
6. `Significant Name Token`: Matches specific tokens ($\ge 5$ chars, frequency $\le 15,000$).
7. `Name Prefix + Location`: 6-char name prefix + location token.

**Capacity Cap:** $\le 120$ candidates per $S1$ entity.

---

### 5. Pairwise Feature Engineering Plan (`src/features.py`)

For each candidate pair $(S1_i, T_j)$, the model extracts:

| Feature Name | Category | Computation / Library | Hypothesis |
| :--- | :--- | :--- | :--- |
| `name_levenshtein_ratio` | Name String | `RapidFuzz.distance.Levenshtein.normalized_similarity` | Captures minor typos |
| `name_token_sort_ratio` | Name Token | `RapidFuzz.fuzz.token_sort_ratio` / 100.0 | Word-order invariance |
| `name_token_set_ratio` | Name Token | `RapidFuzz.fuzz.token_set_ratio` / 100.0 | Handles substring/extra legal words |
| `name_jaro_winkler` | Name String | `RapidFuzz.distance.JaroWinkler.similarity` | Favors common prefixes |
| `name_exact_clean` | Name Exact | Boolean flag $(N_1 == N_2)$ | High precision match indicator |
| `name_compact_match` | Name Domain | Boolean flag $(\text{compact}_1 == \text{compact}_2)$ | Domain URL matches |
| `name_acronym_match` | Name Acronym| Boolean flag $(\text{acronym}_1 == N_2 \lor \dots)$ | Captures `PC` vs `Primary Care` |
| `addr_token_jaccard` | Address Token | $|Tokens_1 \cap Tokens_2| / |Tokens_1 \cup Tokens_2|$ | Robust address similarity |
| `addr_levenshtein_ratio`| Address String | Normalized Levenshtein similarity | Spelling errors in street names |
| `addr_exact_clean` | Address Exact | Boolean flag $(A_1 == A_2)$ | Exact address match |
| `addr_missing` | Address Missing| Boolean flag $(A_2 == \text{NaN} \lor \text{len} == 0)$ | Instructs model to rely on name |
| `exact_number_match` | Numeric | $|Numbers_1 \cap Numbers_2| > 0$ | Shared street/house number |
| `exact_postal_match` | Numeric | $|Postal_1 \cap Postal_2| > 0$ | Shared postal/PIN code |
| `name_x_addr_sim` | Interaction | `name_token_sort_ratio` $\times$ `addr_token_jaccard` | Cross-field joint confidence |

---

### 6. Training Data & Negative Sampling

1. **Positive Pairs ($y=1$):**
   * Pairs generated by blocking that exist in `train_ground_truth.tsv`.
2. **Hard Negative Pairs ($y=0$):**
   * Pairs generated by blocking that do *not* exist in `train_ground_truth.tsv`.
   * These are exceptionally high-quality hard negatives because they already share identical street numbers, acronyms, or name tokens with the reference entity.
3. **Subsampling Strategy:**
   * To prevent memory bloat and severe class imbalance, sample hard negatives at a **10:1 ratio** to positive pairs for training.

---

### 7. Model Selection & Configuration

* **Primary Candidate:** **LightGBM Binary Classifier** (`LGBMClassifier`)
  * License: MIT (Complies with competition rules).
  * Parameter Count: $< 1 \text{ Million}$ parameters (Well below 8 Billion ceiling).
  * Objective: `binary:logloss`
  * Metric: `binary_logloss`, `auc`
  * Key Hyperparameters: `num_leaves=31`, `max_depth=6`, `learning_rate=0.05`, `n_estimators=600`, `subsample=0.8`, `colsample_bytree=0.8`.

---

### 8. Threshold Optimization & Decision Logic

* Default classification threshold of $0.50$ is suboptimal because false merges on singletons zero out scores.
* Optimal threshold $\tau^*$ is found via line search over validation split:
  $$\tau^* = \arg\max_{\tau \in [0.50, 0.95]} \text{Macro } F_{0.5}(\tau)$$
* Initial empirical expectation: $\tau^* \in [0.75, 0.88]$.

---

### 9. Post-Processing: Target Uniqueness Constraint

* Discovered Invariant: Every target record belongs to at most one $S1$ record.
* Conflict Resolution Algorithm:
  If a target ID $T_k$ is predicted for multiple $S1$ entities $\{S1_a, S1_b, \dots\}$:
  $$S1^*(T_k) = \arg\max_{S1} P(\text{match}(S1, T_k))$$
  Assign $T_k$ exclusively to $S1^*$ and prune it from all other candidate lists.
