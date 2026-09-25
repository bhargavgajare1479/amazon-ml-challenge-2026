# Amazon ML Challenge 2026: Definition of Done (DoD)

**Document Path:** `docs/definition-of-done.md`  
**Status:** ACTIVE QUALITY GATE  

---

### 1. Task-Level Definition of Done

A development task is marked **DONE** if and only if all of the following criteria are satisfied:
1. **Working Implementation:** Source code exists in the designated repository path (under `code/business_entity_resolution/src/`).
2. **Deterministic Execution:** The script or module executes cleanly without runtime errors or memory overflow.
3. **Automated Verification:** Unit tests or validation benchmarks have been executed and passed.
4. **Recorded Evidence:** Results, throughput, and metrics are documented in `experiments/experiments.csv`.
5. **Zero Regression:** The change does not degrade previously established validation scores without documented rationale.
6. **Documentation Updated:** Relevant entries in `docs/` and docstrings are maintained.

---

### 2. Final Competition Solution Definition of Done

The end-to-end competition submission is marked **DONE** and ready for final submission only when all 11 conditions are met:

- [ ] **1. Test Inference Completes:** Full inference runs over all 1,732,544 test $S1$ entities and ~10M test targets without manual intervention.
- [ ] **2. `matching_results.tsv` Exists:** Output file created in `output/` with header `['source1_entity_id', 'matched_entity_ids']`.
- [ ] **3. `candidate_pairs.tsv` Exists:** Output file created in `output/` with header `['source1_entity_id', 'candidate_entity_ids']`.
- [ ] **4. Exact S1 Entity Completeness:** Exactly 1,732,544 rows exist in both files—every single test $S1$ entity appears exactly once.
- [ ] **5. ID Validity:** Every predicted ID begins with `S2-` or `S3-` and exists in `test_source2.tsv` or `test_source3.tsv`. No self-matches to `S1-`.
- [ ] **6. Candidate Subset Invariant:** For every $S1$ entity, `matched_entity_ids` is a strict subset of `candidate_entity_ids`.
- [ ] **7. Official Validator Passes:** `utils/validate_submission.py` outputs **`PASS` with exit code 0**.
- [ ] **8. Pinned Dependencies:** `requirements.txt` contains pinned package versions capable of recreating the exact environment.
- [ ] **9. Standalone Reproducibility:** A reviewer can run `python run_pipeline.py` from scratch to regenerate both outputs.
- [ ] **10. Methodology Document Complete:** `Documentation_template.md` is fully filled with technical depth, architecture diagrams, and empirical results.
- [ ] **11. Submission ZIP Archive Verified:** The final archive `<team_name>_submission.zip` matches the exact required directory structure.
