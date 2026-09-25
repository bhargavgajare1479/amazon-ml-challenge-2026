# Amazon ML Challenge 2026: Product Requirements Document (PRD)

**Document Path:** `docs/PRD.md`  
**System Target:** Reproducible Business Entity Resolution Pipeline  
**Evaluation Standard:** Macro $F_{0.5}$ (Leaderboard Evaluated)  
**Status:** ACTIVE  

---

### 1. Executive Summary
The Amazon ML Challenge 2026 Business Entity Resolution challenge requires building an automated, high-precision machine learning system that maps records from three noisy, disparate data sources into a canonical reference catalog. Given reference entities from Source 1 ($S1$), the system must identify all matching entity records from Source 2 ($S2$) and Source 3 ($S3$), producing two official tab-separated artifacts: `matching_results.tsv` (final matches) and `candidate_pairs.tsv` (pre-scoring candidate sets).

---

### 2. Problem Definition
Commercial platforms ingest business information from multiple channels: corporate registries, merchant self-registration, billing records, and delivery locations. These records share no universal foreign keys and exhibit severe noise:
* Typos and character transpositions
* Dropped or altered legal suffixes (`Inc`, `LLP`, `SARL`)
* Transmuted names (e.g. domain names `company.com` entered as business names)
* Missing address attributes (~3.3% of records lack addresses entirely)
* Ambiguous generic names (*"Primary Care"*, *"Physical Therapy"*) occurring at hundreds of independent locations

The system must solve a $1 \rightarrow \{0, 1, \dots, K\}$ entity linkage task across millions of records with high precision.

---

### 3. Business Context
In real-world commercial master data management, **false merges are catastrophic**: merging two distinct merchants into one causes severe operational errors (routing payouts to the wrong entity, commingling tax liabilities, or leaking inventory). Conversely, missing a link simply leaves two separate records unlinked. This business reality directly motivates the competition's choice of a precision-heavy evaluation metric ($F_{0.5}$).

---

### 4. User & Stakeholder Definition
* **Primary Evaluator:** Amazon ML Challenge Automated Leaderboard Scorer.
* **Secondary Reviewers:** Competition Technical Audit Committee (inspecting code reproducibility, licenses, and candidate blocking files).
* **Engineering Users:** Competition participants maintaining, training, and running the pipeline.

---

### 5. Competition Objective
Maximize the macro-averaged $F_{0.5}$ score across all test Source 1 entities while producing fully verifiable, reproducible code adhering to all licensing, parameter, and external data constraints.

---

### 6. Input Data Specifications
Input files are provided in TSV format:
* `dataset/train/train_source1.tsv` (2,206,821 rows, deduplicated reference)
* `dataset/train/train_source2.tsv` (5,034,616 rows, noisy feed)
* `dataset/train/train_source3.tsv` (5,285,603 rows, noisy feed)
* `dataset/train/train_ground_truth.tsv` (2,206,821 rows, linkage labels)
* `dataset/test/test_source1.tsv` (1,732,544 rows, target reference entities)
* `dataset/test/test_source2.tsv` (4,887,273 rows, noisy feed)
* `dataset/test/test_source3.tsv` (5,082,316 rows, noisy feed)

Columns per source file: `entity_id`, `business_name`, `business_address`, `country`.

---

### 7. Output Data Specifications
The pipeline must produce two mandatory files in `output/`:
1. **`matching_results.tsv`**
   * Columns: `source1_entity_id`, `matched_entity_ids`
   * Row Count: Exactly 1,732,544 rows (every test $S1$ entity must be present).
   * Values: Comma-separated list of valid $S2$/$S3$ entity IDs, or blank for singletons.
2. **`candidate_pairs.tsv`**
   * Columns: `source1_entity_id`, `candidate_entity_ids`
   * Row Count: Exactly 1,732,544 rows.
   * Values: Comma-separated list of candidate $S2$/$S3$ IDs fed directly into the model.
   * Requirement: Every matched ID in `matching_results.tsv` must be a subset of `candidate_pairs.tsv`.

---

### 8. Functional Requirements
* **FR-01 (TSV Ingestion):** Ingest tab-delimited files strictly with `\t` delimiter, handling embedded commas and apostrophes correctly.
* **FR-02 (Country Partitioning):** Enforce country-based partitioning on all candidate and matching logic.
* **FR-03 (Candidate Generation):** Filter 10 million target candidates down to $\le 120$ candidates per $S1$ entity using inverted index blocking.
* **FR-04 (Feature Extraction):** Extract numeric, token, and string similarity metrics for all candidate pairs.
* **FR-05 (Prediction & Thresholding):** Score candidate pairs using an ML model and apply a calibrated decision threshold.
* **FR-06 (Post-Processing Enforcement):** Ensure no target record is assigned to multiple $S1$ entities.
* **FR-07 (Submission Validation):** Validate outputs against `utils/validate_submission.py` prior to delivery.

---

### 9. ML Requirements
* Formulate as pairwise binary classification: $P(\text{match} \mid \text{Pairwise Evidence})$.
* Model must handle heterogeneous tabular features: string distances, length discrepancies, shared token counts, and exact numeric matches.
* Gradient Boosted Decision Trees (LightGBM, CatBoost, or XGBoost) chosen for efficiency and tabular performance.

---

### 10. Matching Cardinality Requirements
* The system must natively support:
  * $S1 \rightarrow \emptyset$ (Singletons, ~5.6% of entities)
  * $S1 \rightarrow \{S2\}$ or $\{S3\}$ (Single matches, ~5.4%)
  * $S1 \rightarrow \{S2_1, \dots, S2_a, S3_1, \dots, S3_b\}$ (Multi-matches up to 11, ~89.0%)
* $S2$ and $S3$ target records must enforce $\le 1$ parent $S1$ entity.

---

### 11. Candidate Generation Requirements
* Candidate Recall Target: $\ge 85\%$ of true matches captured.
* Candidate Reduction Ratio: $\ge 99.998\%$ reduction from Cartesian product.
* Target Candidate Budget: $30 \le \text{candidates/S1} \le 120$.

---

### 12. Evaluation & Metric Requirements
* **Metric:** Macro-averaged $F_{0.5}$.
* **Entity Calculation:**
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
* **Singleton Credit:**
  * True matches $= \emptyset$ and Predicted matches $= \emptyset \implies F_{0.5} = 1.0$.
  * True matches $= \emptyset$ and Predicted matches $\neq \emptyset \implies F_{0.5} = 0.0$.
* Macro average computed over all $N$ entities in the test set.

---

### 13. Precision / Recall Priority
* $\beta = 0.5$ weights precision twice as heavily as recall:
  $$\frac{\partial F_{0.5}}{\partial \text{Precision}} > \frac{\partial F_{0.5}}{\partial \text{Recall}}$$
* The system must prioritize high confidence over speculative matches.

---

### 14. Singleton Requirements
* Singletons constitute ~5.6% of $S1$ entities.
* Predicting false matches on singletons destroys score with $0.0$.
* The pipeline must feature a conservative threshold ensuring singletons predict empty lists.

---

### 15. Data Integrity Requirements
* Raw data files in `dataset/` must remain immutable.
* Missing address fields (~3.3% in $S2/S3$) must be imputed or processed safely without triggering exceptions.

---

### 16. Open-Set Country Requirement
* Country `France` appears in Test (15% of records) but is absent from Train.
* The pipeline must not hardcode allowed country sets (`{US, India}`).
* Text normalizers must support French characters, street indicators (`Rue`, `Boulevard`), and corporate suffixes (`SARL`, `SAS`).

---

### 17. Fair-Play & Integrity Constraints
* **STRICTLY PROHIBITED:** External APIs, Google Maps, web scraping, geocoders, commercial registries, external entity resolution systems.
* All matching signals must be derived strictly from the competition datasets.

---

### 18. Model Licensing & Parameter Constraints
* Maximum model size: **8 Billion parameters**.
* License: Permissive open source (**MIT** or **Apache 2.0**).
* Selected tools: `LightGBM` (MIT), `CatBoost` (Apache 2.0), `RapidFuzz` (MIT), `Scikit-learn` (BSD-3).

---

### 19. Reproducibility Requirements
* Fixed random seeds across all randomized components.
* Strict dependency pinning in `requirements.txt`.
* Execution must be end-to-end runnable from a command line via `run_pipeline.py`.

---

### 20. Submission Package Requirements
The submission archive `<team_name>_submission.zip` must contain:
```
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

---

### 21. Non-Functional Requirements
* **Memory Ceiling:** Must execute reliably within 32 GB RAM without swapping.
* **Inference Runtime:** Full test set inference (1.73M entities against 10M targets) must finish within $\le 4$ hours on a 16-core CPU.
* **Storage:** Temporary candidate storage must remain within disk bounds ($\le 25 \text{ GB}$).

---

### 22. Success Criteria
* Validation Macro $F_{0.5} \ge 0.70$.
* Test submission passes `validate_submission.py` with 0 errors.
* Candidate blocking recall $\ge 85\%$.
* Complete technical documentation filled in `Documentation_template.md`.

---

### 23. Risks & Mitigations
* *Risk:* Address missingness causes candidate drop $\rightarrow$ *Mitigation:* Multi-channel name blocking handles records with missing addresses.
* *Risk:* Memory exhaustion indexing 10M records $\rightarrow$ *Mitigation:* Single-pass streaming ingestion and country partitioning.

---

### 24. Out of Scope
* Real-time API serving endpoints.
* Web interfaces or interactive dashboards.
* External geocoding or entity enrichment.
