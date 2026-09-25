# Amazon ML Challenge 2026: Comprehensive Risk Register

**Document Path:** `docs/risk-register.md`  
**Status:** ACTIVE RISK AUDIT  

---

### Risk Assessment Matrix

| Risk ID | Risk Description | Prob | Impact | Detection Method | Mitigation Strategy | Current Status |
| :--- | :--- | :---: | :---: | :--- | :--- | :---: |
| **RSK-001** | **Low Blocking Recall Ceiling:** True matches missed in blocking can never be recovered by the ML model. | HIGH | HIGH | `evaluate_blocking_candidates()` on validation split. | Implement multi-channel blocking (compact names, bigrams, address numbers, character 3-grams). | **MITIGATED (70% recall reached; 3-gram planned)** |
| **RSK-002** | **Candidate Explosion:** Inverted index yields thousands of candidates per entity, exhausting RAM. | MEDIUM | HIGH | Track `candidates_per_s1_max` and total candidates. | Enforce hard frequency thresholding on tokens (`max_token_freq`) and hard candidate cap ($\le 120$/S1). | **CONTROLLED (Mean 57.6 cands/S1)** |
| **RSK-003** | **False Merges on Generic Names:** Imposters with identical names (e.g. *"Primary Care"*) merged across cities. | HIGH | HIGH | Track precision and false positive breakdown in error analysis. | Use cross-field interaction features (name similarity $\times$ address similarity) and numeric equality. | **OPEN (Requires GBDT model)** |
| **RSK-004** | **Singleton False Positives:** Over-predicting matches on true singletons zeroes out their score ($F_{0.5}=0.0$). | HIGH | HIGH | `singleton_accuracy` tracking in `evaluate.py`. | Elevate classification threshold $\tau \ge 0.75 - 0.85$; predict empty list when confidence is marginal. | **OPEN (Threshold search in Phase 8)** |
| **RSK-005** | **Validation Leakage:** Leaking entity names between train and validation creates over-optimistic offline metrics. | LOW | HIGH | Inspect entity overlap between train and validation splits. | Strict Source 1 entity-level stratified partitioning (`src/split.py`). | **MITIGATED (Zero entity overlap)** |
| **RSK-006** | **Public Leaderboard Overfitting:** Repeatedly tuning models to public leaderboard noise. | MEDIUM | HIGH | Track local CV vs. public leaderboard delta. | Treat local 30k stratified split as single source of truth; limit leaderboard submissions to verified gains. | **MANAGED** |
| **RSK-007** | **Unseen Market (France) Covariate Shift:** French records in Test fail due to country-specific parsing assumptions. | HIGH | HIGH | Unit tests with French addresses (`Rue`, `Boulevard`, `SARL`). | Universal normalization: accent stripping, language-agnostic tokenizers, country-partitioned execution. | **MITIGATED in `src/normalize.py`** |
| **RSK-008** | **Transliteration & Typo Collisions:** Severe spelling corruption in Indian and US names misses tokens. | MEDIUM | MEDIUM | Diagnose missed true pairs in validation. | Character-level n-grams and fuzzy Levenshtein distance metrics via RapidFuzz. | **IN PROGRESS** |
| **RSK-009** | **Address Ambiguity / Missingness:** ~3.3% of target records lack address entirely. | HIGH | MEDIUM | Missing address count tracking. | Feature extractor flags `addr_missing` as explicit Boolean feature; model learns strong-name matching logic. | **PLANNED in `src/features.py`** |
| **RSK-010** | **RAM Exhaustion on 10M Records:** Loading all pairs simultaneously causes OOM crash. | MEDIUM | HIGH | Memory profiling with `psutil`. | Stream target indexing in a single pass; partition execution strictly by country; process pairs in batches. | **CONTROLLED (Memory reduced from 11.5GB to 3.6GB)** |
| **RSK-011** | **Dependency / Version Incompatibilities:** Script fails on different Python version or missing package. | LOW | HIGH | Verification in clean virtualenv. | Strict version pinning in `requirements.txt`. | **PLANNED** |
| **RSK-012** | **Model Licensing Violation:** Using models that violate MIT / Apache 2.0 competition rules. | LOW | CRITICAL | License review of all imported dependencies. | Restricted to verified open-source packages: LightGBM (MIT), CatBoost (Apache 2.0), RapidFuzz (MIT). | **COMPLIANT** |
| **RSK-013** | **Output Formatting / Validator Rejection:** File formatting error (e.g. CSV comma instead of TSV tab) causes rejection. | MEDIUM | CRITICAL | Run `utils/validate_submission.py` locally before submission. | Automated validation step built directly into submission pipeline script. | **PLANNED in `src/submission.py`** |
| **RSK-014** | **Unreproducible Submission Package:** Reviewers unable to run pipeline from scratch. | MEDIUM | CRITICAL | Clean environment reproduction dry run. | Deliver single `run_pipeline.py` script and comprehensive `README.md`. | **PLANNED for Phase 13** |
