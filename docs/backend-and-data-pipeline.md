# Amazon ML Challenge 2026: Backend & Data Pipeline Design

**Document Path:** `docs/backend-and-data-pipeline.md`  
**Architecture Type:** Batch Data Processing & Offline ML Pipeline  
**Status:** ACTIVE  

---

### 1. Architectural Statement

**The system does NOT use a client-server, microservice, or REST API backend architecture.**  
The solution is strictly an offline batch data processing and machine learning pipeline designed to transform large-scale static TSV files into competition submission artifacts.

---

### 2. Input Datasets

All inputs are tab-separated values (TSV) stored under `dataset/`:

| File Path | Description | Records | Size | Access Mode |
| :--- | :--- | :--- | :--- | :--- |
| `dataset/train/train_source1.tsv` | Canonical reference catalog | 2,206,821 | 200.34 MB | Read-Only |
| `dataset/train/train_source2.tsv` | Noisy merchant feed | 5,034,616 | 466.63 MB | Read-Only |
| `dataset/train/train_source3.tsv` | Noisy merchant feed | 5,285,603 | 480.37 MB | Read-Only |
| `dataset/train/train_ground_truth.tsv` | True linkage mapping | 2,206,821 | 121.13 MB | Read-Only |
| `dataset/test/test_source1.tsv` | Test reference entities | 1,732,544 | 166.91 MB | Read-Only |
| `dataset/test/test_source2.tsv` | Test noisy feed | 4,887,273 | 485.86 MB | Read-Only |
| `dataset/test/test_source3.tsv` | Test noisy feed | 5,082,316 | 482.56 MB | Read-Only |

---

### 3. Pipeline Stages & Preprocessing
1. **TSV Streaming Ingestion:**  
   Records are ingested via Pandas streaming chunks (`chunksize=500000`) with explicit string dtype enforcement to guarantee no ID mangling.
2. **Country Splitting:**  
   Data streams are routed into country-specific processors (`US`, `India`, `France`).
3. **Multi-View Normalization:**  
   Functions in `src/normalize.py` produce clean text, stripped legal entities, compact strings, and numeric sets on the fly without writing huge intermediate preprocessed tables to disk.

---

### 4. Intermediate Artifacts

Intermediate artifacts are persisted in designated directories to enable reproducibility and fast restarts:

| Artifact Path | Format | Description | Generation Stage |
| :--- | :--- | :--- | :--- |
| `data/validation/val_s1_ids.json` | JSON | 30,000 holdout $S1$ entity IDs | `src/split.py` |
| `data/validation/val_ground_truth.json` | JSON | Ground truth mappings for validation entities | `src/split.py` |
| `data/validation/val_source1.tsv` | TSV | Reference metadata for validation $S1$ | `src/split.py` |
| `data/validation/val_summary.json` | JSON | Summary metrics of validation partition | `src/split.py` |
| `reports/dataset_audit.md` | Markdown | Exhaustive data autopsy report | Phase 1 Audit |
| `experiments/experiments.csv` | CSV | Structured experiment tracking ledger | Continuous |

---

### 5. Model Artifacts

Trained model weights and configurations are serialized under `models/`:
* `models/lgbm_matcher.bin`: LightGBM binary booster artifact.
* `models/model_config.json`: Feature names, hyperparameters, and optimal probability threshold $\tau^*$.

---

### 6. Output Files (Deliverables)

Placed in `output/` and submitted directly to the portal:

| Deliverable Path | Format | Required Schema | Row Count |
| :--- | :--- | :--- | :--- |
| `output/candidate_pairs.tsv` | TSV | `source1_entity_id\tcandidate_entity_ids` | Exactly 1,732,544 |
| `output/matching_results.tsv` | TSV | `source1_entity_id\tmatched_entity_ids` | Exactly 1,732,544 |

---

### 7. Central Configuration Management

All pipeline parameters will be centralized in `code/business_entity_resolution/src/config.py`:
```python
# System Paths
DATASET_DIR = "dataset"
TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
OUTPUT_DIR = "output"

# Blocking Hyperparameters
MAX_TOKEN_FREQ = 15000
MAX_CANDIDATES_PER_S1 = 120

# Model Hyperparameters
SEED = 42
OPTIMAL_THRESHOLD = 0.80
```
