# Amazon ML Challenge 2026: Project Decision Log

**Document Path:** `docs/decision-log.md`  
**Date:** 2026-09-25  
**Status:** ACTIVE RECORD  

This document logs every technical, architectural, and algorithmic decision made throughout the project lifecycle.

---

### Record DEC-001: Country as an Absolute Hard Partition Key
* **Date:** 2026-09-25
* **Decision:** Partition all blocking, indexing, and matching strictly by `country`. Never compare records across different country labels.
* **Reason:** Reduces the theoretical search space from 17.27 Trillion pairs to 6.72 Trillion pairs without any loss in true match recall.
* **Evidence:** Exhaustive audit of all 7,638,365 ground-truth pairs in `train_ground_truth.tsv` revealed **0 cross-country matches (100.000% consistency)**.
* **Alternatives Considered:** Soft cross-country fuzzy matching (rejected: computationally wasteful and contradicted by 100% empirical ground truth conservation).
* **Current Status:** IMPLEMENTED in `src/blocking.py` and benchmark scripts.

---

### Record DEC-002: Multi-View Gentle Normalization over Destructive Cleaning
* **Date:** 2026-09-25
* **Decision:** Maintain multiple representations (`raw`, `clean`, `stripped_legal`, `compact`, `bigram`, `acronym`, `numbers`) instead of overwriting raw text with a single heavily normalized string.
* **Reason:** Aggressive cleaning destroys discriminative signal (e.g. collapsing *"Apple Store"* and *"Apple Inc"* into *"Apple"*). Different blocking and matching stages require different representations.
* **Evidence:** In ground truth, web domains (`telefutureindia.com`), acronyms (`PC`), and legal suffix variants coexist. Multi-view enables compact alphanumeric keys for domains while preserving raw tokens for feature scoring.
* **Alternatives Considered:** Single destructive preprocessing pass replacing the raw columns (rejected: irrevocably destroys critical distinguishing features).
* **Current Status:** IMPLEMENTED in `code/business_entity_resolution/src/normalize.py`.

---

### Record DEC-003: Entity-Level Stratified Validation Split (30,000 $S1$)
* **Date:** 2026-09-25
* **Decision:** Partition 30,000 Source 1 entities from `train_source1.tsv` stratified jointly across `country` and `match_count` bins. Evaluate them against the **entire multi-million** target pool ($S2 + S3$).
* **Reason:** Random pair-level splitting causes massive data leakage (the model memorizes entity names). Searching against the full target pool accurately reproduces test inference conditions, including distractor noise and precision pressure.
* **Evidence:** Validation split mirrors full population metrics precisely: 5.587% singletons (vs 5.585% full), 59.98% US, 40.02% India, 3.462 matches/entity.
* **Alternatives Considered:** 5-fold cross-validation on full 2.2M records (rejected: full 24M record scoring takes hours per iteration, severely throttling experimental velocity).
* **Current Status:** IMPLEMENTED in `code/business_entity_resolution/src/split.py`.

---

### Record DEC-004: Strict Singleton-Aware Macro $F_{0.5}$ Metric Harness
* **Date:** 2026-09-25
* **Decision:** Build an exact evaluation function enforcing the competition's macro-averaged $F_{0.5}$ formula with official singleton credit ($1.0$ for empty, $0.0$ for false merge).
* **Reason:** Optimizing pairwise accuracy or pairwise $F_1$ leads to over-matching. Because $F_{0.5}$ weights precision $2\times$ over recall and penalizes false merges on singletons severely, evaluation must operate at the entity level.
* **Evidence:** Verified against official competition README example (reproduces exact $0.71429$ score).
* **Alternatives Considered:** Scikit-learn's standard `fbeta_score(average='binary')` on pair rows (rejected: mathematically incorrect for entity resolution with singletons).
* **Current Status:** IMPLEMENTED in `code/business_entity_resolution/src/evaluate.py`.

---

### Record DEC-005: Multi-Channel Inverted Index Candidate Blocking
* **Date:** 2026-09-25
* **Decision:** Use a multi-channel inverted index combining exact names, compact names, word bigrams, exact addresses, and address number + first token. Cap candidate output at $\le 120$ per $S1$.
* **Reason:** Naive exact matching captures only 28.76% of true pairs (missing 71.24%). Multi-channel blocking widens the net to capture typos, domains, and abbreviations while keeping candidate pool size computationally tractable ($\approx 57$ candidates per entity).
* **Evidence:** Benchmark achieved **69.97% candidate recall** with a **99.9986% candidate reduction ratio** across 4.13M targets.
* **Alternatives Considered:** Exhaustive Cartesian product (rejected: 17.27 Trillion pairs in test is impossible); embedding-based vector search / dense ANN (rejected: slow on CPU, high memory footprint, violates 8B parameter / offline deployment constraints).
* **Current Status:** IMPLEMENTED in `code/business_entity_resolution/src/blocking.py`.

---

### Record DEC-006: Single-Pass Streaming Target Indexer
* **Date:** 2026-09-25
* **Decision:** Stream target records directly into inverted index dictionaries in a single pass without storing intermediate normalized tuples in a Python list.
* **Reason:** Storing 4.13M tuples containing nested dictionaries and sets consumed 11.5 GB of RAM and triggered Python garbage collector thrashing.
* **Evidence:** Memory usage dropped from 11.5 GB to 2.5–3.6 GB (an ~80% reduction), and indexing execution time improved by 27 seconds.
* **Alternatives Considered:** Loading entire target sets into SQLite or DuckDB (held as viable alternative if memory limits are exceeded on the full 10M record set).
* **Current Status:** IMPLEMENTED in `src/blocking.py`.

---

### Record DEC-007: Target-Side 1-to-1 Partitioning Post-Processing
* **Date:** 2026-09-25
* **Decision:** Enforce that a single Source 2 or Source 3 record cannot be assigned to more than one Source 1 record in final predictions.
* **Reason:** In real-world entity resolution and in the competition ground truth, noisy records are observations of a single underlying business.
* **Evidence:** Ground truth analysis confirmed `max_s1_per_target = 1` for both $S2$ and $S3$ across all 7.6M true pairs (0 violations).
* **Alternatives Considered:** Independent candidate scoring with unrestricted multi-assignment (rejected: produces duplicate target predictions and hurts precision).
* **Current Status:** FORMULATED; scheduled for post-processing implementation in Phase 8/9.

---

### Record DEC-008: Open-Set Multilingual Handling for France
* **Date:** 2026-09-25
* **Decision:** Design tokenizers, accent strippers, and legal suffix registries to natively support French entity syntax (`SARL`, `SAS`, `Rue`, `Boulevard`, `Arrondissement`) without conditional hardcoding of country lists.
* **Reason:** Test set contains ~15% French records, whereas training set contains only US and India. Any hardcoded assumptions like `if country in ['US', 'India']` will fail on test data.
* **Evidence:** Dataset audit revealed 259,452 French $S1$ entities and >1.4M French $S2/S3$ records in test.
* **Alternatives Considered:** Ignoring country or treating France with a fallback pipeline (rejected: France must be scored identically to US and India).
* **Current Status:** IMPLEMENTED in `src/normalize.py`.
