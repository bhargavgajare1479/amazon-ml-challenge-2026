# Amazon ML Challenge 2026: Project Reconstruction & Current State Audit

**Document Path:** `docs/project-reconstruction.md`  
**Date:** 2026-09-25  
**Author:** Senior ML Competition Engineer & Research Partner  
**Status:** FACTUAL RECONSTRUCTION (Based on Repository Evidence & Experiment Outputs)  

---

## A. Project Objective
The objective is to develop a high-precision, reproducible machine learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution Challenge**. Given three distinct datasets of business entities:
* **Source 1 ($S1$):** A deduplicated reference catalog of commercial entities.
* **Source 2 ($S2$):** A noisy feed of business records with typographical errors, missing attributes, and abbreviations.
* **Source 3 ($S3$):** A secondary noisy feed of business records with varied representations, domain URLs, and structural discrepancies.

The task is to map every Source 1 entity to its corresponding representations in Source 2 and Source 3:
$$S1_i \longrightarrow \{S2_j, S3_k, \dots\}$$
A Source 1 entity may match zero records (a singleton), one record, or multiple records across $S2$ and $S3$. The primary evaluation metric is **Macro $F_{0.5}$** across all test $S1$ entities, weighting precision $2\times$ over recall and heavily penalizing false merges.

---

## B. Competition Constraints & Guardrails
1. **Strictly No External Data / Lookups:** Hard prohibition against external APIs, Google Search, Google Maps, geocoding endpoints, corporate registries (e.g. MCA, SEC), commercial entity resolution software, web scraping, and external address libraries. All knowledge must originate solely from provided competition TSVs.
2. **Model Parameter & Licensing Ceiling:** Maximum 8 Billion parameters; open-source licensing strictly limited to **MIT** or **Apache 2.0**.
3. **Open-Set Country Distribution:** Training data contains records from `US` and `India`. Test data contains an unseen third country, `France` (~15% of test records). Pipelines must remain country-adaptive without hardcoded country splits.
4. **Mandatory Output Contracts:**
   * `output/matching_results.tsv`: Header `['source1_entity_id', 'matched_entity_ids']`. Exactly one row per test $S1$ entity; comma-separated $S2$/$S3$ IDs (empty for singletons).
   * `output/candidate_pairs.tsv`: Header `['source1_entity_id', 'candidate_entity_ids']`. Represents the final filtered candidate set entering the ML scoring stage. Every matched ID must be a strict subset of candidates.
   * Both files must pass `utils/validate_submission.py`.

---

## C. Existing Architecture
The established system follows a multi-stage funnel:

```
[Raw Sources: S1, S2, S3]
         │
         ▼
[Country-Partitioning & Gentle Normalization] (code/business_entity_resolution/src/normalize.py)
         │
         ▼
[Multi-Channel Inverted Index Blocking] (code/business_entity_resolution/src/blocking.py)
         │
         ▼
[Refined Candidate Pairs] (candidate_pairs.tsv format)
         │
         ▼
[Pairwise Feature Engineering] (RAPIDFUZZ + Token + Numeric Overlap)  <-- [IN PROGRESS]
         │
         ▼
[Supervised Match Classifier (GBDT)]                                <-- [TODO]
         │
         ▼
[Conservative Thresholding & 1-to-1 Target Assignment]              <-- [TODO]
         │
         ▼
[Final Matches] (matching_results.tsv)
```

---

## D. Existing Data Pipeline
* **Source TSVs Ingestion:** Tab-separated ingestion using explicit `sep="\t"` and string dtypes to prevent float conversions of numeric IDs and postal codes.
* **Country Partitions:** Empirical ground truth audit confirmed **100.000% country conservation** across 7,638,365 true pairs (0 cross-country matches). Data ingestion partitions execution by country (`US`, `India`, `France`).
* **Storage Structure:** Raw data remains immutable in `dataset/train/` and `dataset/test/`. Intermediate validation artifacts reside in `data/validation/`.

---

## E. Existing Normalization
Implemented in `code/business_entity_resolution/src/normalize.py`:
* **Accents & Encoding:** Unicode NFKD decomposition (`strip_accents`), converting accented characters (e.g. French `é`, `è`, `ç` $\rightarrow$ `e`, `c`).
* **Digit-Letter Uncoupling:** Regex separation of concatenated numbers and letters (e.g. Indian address markers `No127` $\rightarrow$ `No 127`, `Plot4B` $\rightarrow$ `Plot 4 B`).
* **Legal Suffix Stripping:** Identification and stripping of corporate suffixes (`inc`, `corp`, `llc`, `llp`, `ltd`, `pvt`, `sarl`, `sas`, `sci`).
* **Domain Normalization:** Stripping URL artifacts (`.com`, `.org`, `.net`, `.in`, `.fr`, `www.`).
* **Multi-View Representations:** Every record generates:
  * `raw`: Preserved original string.
  * `clean`: Lowercase, accent-stripped, normalized whitespace.
  * `stripped_legal`: Clean string with legal suffixes removed.
  * `compact`: Alphanumeric-only representation without spaces or dots.
  * `bigram`: First two significant name words (`tele_future`).
  * `acronym`: First letters of significant tokens (`PC` for `Primary Care`).
  * `numbers`: Extracted numeric tokens (house/flat numbers, postal codes).

---

## F. Existing Blocking / Candidate Generation
Implemented in `code/business_entity_resolution/src/blocking.py`:
* **Target Indexing:** Single-pass streaming ingestion indexing target records ($S2 + S3$) into inverted dictionaries.
* **Active Blocking Channels:**
  1. `name_exact_index`: Stripped clean name.
  2. `name_compact_index`: Compact alphanumeric string (captures web domains).
  3. `name_bigram_index`: Word pairs (captures multi-word business names).
  4. `addr_exact_index`: Clean address string.
  5. `addr_num_token_index`: House number + first significant address word (e.g., `127_sriananthammalcompx`).
  6. `rare_token_index`: Significant name tokens with length $\ge 5$ and frequency $\le 15,000$.
  7. `name_prefix_index`: First 6 letters of name + city/region token.
* **Capacity Constraints:** Candidates capped at $\le 120$ per $S1$ entity to prevent long-tail computational explosion.

---

## G. Existing Features
* **Current Status:** Basic string equality and token overlap exist inside baseline scripts (`scripts/evaluate_baseline0.py`).
* **Pending Work:** Comprehensive pairwise feature engineering module (`code/business_entity_resolution/src/features.py`) incorporating character Levenshtein, token-sort ratio, Jaro-Winkler, numeric address comparison, and cross-field interaction terms is planned for Phase 5.

---

## H. Existing Models
* **Baseline 0A (Rule-based):** Exact stripped name matcher.
* **Baseline 0B (Rule-based):** Exact stripped name + exact clean address matcher.
* **Machine Learning Model (Phase 8):** Not yet trained. GBDT (CatBoost / LightGBM) is selected as the first candidate model.

---

## I. Existing Validation Methodology
Implemented in `code/business_entity_resolution/src/split.py` and `code/business_entity_resolution/src/evaluate.py`:
* **Stratified Holdout Split:** 30,000 $S1$ entities partitioned from `dataset/train/train_source1.tsv` (Seed 42).
  * Stratified jointly by Country (`US` vs `India`) and Match Cardinality bins (`0_singleton`, `1_match`, `2_matches`, `3_4_matches`, `5plus_matches`).
  * Validation split properties exactly match the population: 5.587% singletons, 59.98% US, 40.02% India, 3.462 mean matches/entity.
  * Stored in `data/validation/val_source1.tsv` and `data/validation/val_ground_truth.json`.
* **Realistic Simulation:** Validation $S1$ records are searched against the **entire pool** of $S2$ and $S3$ records (4.13M targets for India, 6.19M for US), faithfully replicating test inference conditions.
* **Official Metric Implementation:** `src/evaluate.py` implements the exact competition formula:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Includes strict singleton scoring: predicting empty on a true singleton yields $1.0$; any false positive yields $0.0$. Passes verification against the official README test case (`0.71429`).

---

## J. Existing Threshold / Decision Logic
* Currently rule-based in Baseline 0:
  * Policy 0A: Predict candidate if exact name matches.
  * Policy 0B: Predict candidate if exact name matches AND address matches (or target address is missing).
* Systematic probability threshold tuning $\tau \in [0.50, 0.95]$ is planned for post-model evaluation.

---

## K. Existing Submission Generation
* Verification harness available at `utils/validate_submission.py`.
* End-to-end pipeline script to generate `output/matching_results.tsv` and `output/candidate_pairs.tsv` is not yet created.

---

## L. Existing Experiment History
Logged in `experiments/experiments.csv`:

| Exp ID | Date | Model / Configuration | Candidate Recall | Precision | Recall | Macro $F_{0.5}$ | Status | Key Takeaways |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **EXP-001** | 2026-09-25 | Baseline 0A (Exact Name Only) | 35.60% | 42.47% | 35.60% | **0.38328** | DONE | Exact name alone suffers from false merges on generic names (e.g. *"Primary Care"*). |
| **EXP-002** | 2026-09-25 | Baseline 0B (Exact Name + Addr) | 8.40% | 16.14% | 8.40% | **0.12290** | DONE | Severe recall collapse confirms exact address equality is impossible for noisy ER. |
| **EXP-003** | 2026-09-25 | Multi-Channel Blocker v2 | **69.97%** | N/A | N/A | N/A | DONE | Generated 57.65 cands/entity across 4.13M targets with 99.9986% reduction ratio. |

---

## M. Best Known Validation Results
* **Best Candidate Recall:** **69.97%** (Multi-Channel Blocker v2 on India validation partition).
* **Best End-to-End Macro $F_{0.5}$:** **0.38328** (Baseline 0A, Exact Name Matcher).

---

## N. Best Known Public Leaderboard Results
* **Current Status:** `UNKNOWN` (No submission uploaded to portal yet).

---

## O. Known Failures & Root Causes
1. **Exact Address Match Failure:** 71.24% of true matches share neither exact normalized name nor address. Requiring exact address matching collapses recall to 8.4%.
2. **Generic Business Name Collisions:** Medical practices, trade services, and retail shops share identical names across different cities (e.g., *"Physical Therapy"*, *"Urgent Care"*). Matching on name alone triggers massive precision penalties.
3. **Concatenated Numeric Prefix Parsing:** Address strings formatted as `No127` failed standard word boundary regex `\b\d+\b`. Fixed via explicit digit-letter separation regex in `src/normalize.py`.
4. **Memory Spike in Initial Blocker Draft:** A 2-pass indexing loop storing 4.13M intermediate tuples caused an 11.5 GB memory spike. Refactored into a single-pass streaming index, dropping memory to 2.5–3.6 GB.

---

## P. Known Weaknesses
1. **Candidate Recall Gap:** Candidate blocking currently captures ~70.0% of true matches. The remaining ~30% are missed due to severe name corruptions (e.g., DBA transmutations like `Drxkor` vs `Maure Williams Colombier Inc`) or absent address street numbers.
2. **Missing Pairwise ML Scorer:** Currently, predictions jump directly from blocking to heuristic filtering without a statistical model weighing multi-field evidence.
3. **Target Multi-Claim Unchecked in Baseline:** In Baseline 0A, an $S2$ entity could be predicted for multiple $S1$ entities, violating the empirical ground-truth property where each target belongs to $\le 1$ $S1$ entity.

---

## Q. Technical Debt
1. **Country Partition Scripting:** Benchmark script currently evaluates India targets explicitly; US target indexing needs to be generalized into an automated country-agnostic loop.
2. **Package Namespace Collision:** `code` is a Python built-in standard library module. Importing `code.business_entity_resolution` directly fails unless `sys.path` is explicitly configured. We must maintain consistent import paths via `PYTHONPATH` or `sys.path.insert`.
3. **No Centralized Configuration:** File paths and hyperparameter constants are partially hardcoded across scripts rather than consolidated into a single `config.py`.

---

## R. Incomplete Components
* `src/config.py`: Centralized configuration.
* `src/features.py`: Pairwise similarity feature extraction.
* `src/train.py`: Supervised training pipeline.
* `src/predict.py`: Batch candidate scoring and threshold application.
* `src/submission.py`: Pipeline to generate validated submission TSVs.
* `pipeline.py`: Unified end-to-end execution script.

---

## S. Current Project Status
* **Phase 1 (Data Audit):** **DONE** (Complete report at `reports/dataset_audit.md`).
* **Phase 2 (Validation Design):** **DONE** (Stratified 30k split in `data/validation/`).
* **Phase 3 (Normalization):** **DONE** (Multi-view engine in `src/normalize.py`).
* **Phase 4 (Candidate Blocking):** **DONE** (Benchmark at 70.0% recall, 57 cands/entity).
* **Phase 5 (Official Metric):** **DONE** (Verified Macro $F_{0.5}$ in `src/evaluate.py`).
* **Phase 6 & 7 (Baseline Models):** **DONE** (Baseline 0A: 0.38328 Macro $F_{0.5}$).
* **Phase 8 (Pairwise Feature Engineering & ML Matcher):** **READY TO START**.

---

## T. Immediate Next Actions
1. Formulate the comprehensive documentation suite (`PRD`, `TRD`, `Architecture`, `Implementation Plan`, `Risk Register`).
2. Implement `src/features.py` using vector-accelerated `RapidFuzz` string metrics and token overlaps.
3. Train our first tree-based classifier (LightGBM / CatBoost) to advance beyond the 0.383 baseline.
