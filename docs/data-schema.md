# Amazon ML Challenge 2026: Data Schema Specification

**Document Path:** `docs/data-schema.md`  
**Status:** ACTIVE  

---

### 1. Source 1 Schema (`*_source1.tsv`)
Canonical reference catalog. Every row represents a distinct, deduplicated real-world business entity.

| Field Name | Type | Nullable? | Meaning & Domain | Example | Used in Blocking? | Used in Model? |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| `entity_id` | `string` | No (0%) | Unique reference ID prefixed with `S1-` | `S1-965667` | Yes (Key) | Yes (Key) |
| `business_name` | `string` | No (0%) | Official registered business name | `Maure Williams Colombier Inc` | Yes | Yes |
| `business_address`| `string` | No (0%) | Street address, city, state, postal code | `85 Wayne Avenue, Ticonderoga, NY` | Yes | Yes |
| `country` | `string` | No (0%) | ISO country string (`US`, `India`, `France`) | `US` | Yes (Hard Partition) | Yes |

---

### 2. Source 2 Schema (`*_source2.tsv`)
Noisy business records ingested from secondary merchant channels.

| Field Name | Type | Nullable? | Meaning & Domain | Example | Used in Blocking? | Used in Model? |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| `entity_id` | `string` | No (0%) | Unique record ID prefixed with `S2-` | `S2-681193310` | Yes (Target Key) | Yes (Target Key) |
| `business_name` | `string` | Yes ($<0.001\%$) | Noisy name, typos, dropped suffixes | `Maure Wilblims Colombier` | Yes | Yes |
| `business_address`| `string` | Yes (~3.35%) | Missing or abbreviated address | `85 Wanye Ave, Ticonderoga` | Yes | Yes |
| `country` | `string` | No (0%) | Country label (`US`, `India`, `France`) | `US` | Yes (Hard Partition) | Yes |

---

### 3. Source 3 Schema (`*_source3.tsv`)
Noisy business records ingested from web and commercial feeds.

| Field Name | Type | Nullable? | Meaning & Domain | Example | Used in Blocking? | Used in Model? |
| :--- | :--- | :---: | :--- | :--- | :---: | :---: |
| `entity_id` | `string` | No (0%) | Unique record ID prefixed with `S3-` | `S3-11291185` | Yes (Target Key) | Yes (Target Key) |
| `business_name` | `string` | Yes ($<0.001\%$) | Domain names, acronyms, or DBA names | `maurewilliamscolombier.com` | Yes | Yes |
| `business_address`| `string` | Yes (~3.33%) | Street variants, landmark references | `Wayne Ave, Ticonderoga Townshiip, NY`| Yes | Yes |
| `country` | `string` | No (0%) | Country label (`US`, `India`, `France`) | `US` | Yes (Hard Partition) | Yes |

---

### 4. Ground Truth Schema (`train_ground_truth.tsv`)
Provides supervised linkage labels for the training set.

| Field Name | Type | Nullable? | Meaning & Domain | Formatting Contract |
| :--- | :--- | :---: | :--- | :--- |
| `source1_entity_id`| `string` | No | ID of a Source 1 record | Exactly matches an `S1-` ID in `train_source1.tsv` |
| `matched_entity_ids`| `string` | Yes (5.58% empty) | Comma-separated list of matching $S2$/$S3$ IDs | E.g. `S2-681193310,S2-743505751,S3-11291185`. Empty string for singletons. |

---

### 5. Candidate Pairs Schema (`candidate_pairs.tsv`)
Pre-scoring candidate set produced by the final blocking stage before model scoring.

| Field Name | Type | Nullable? | Description & Constraints |
| :--- | :--- | :---: | :--- |
| `source1_entity_id` | `string` | No | Exactly one row for every test $S1$ entity (1,732,544 rows). |
| `candidate_entity_ids`| `string` | Yes (Allowed empty) | Comma-separated list of candidate $S2$ and $S3$ entity IDs considered plausible matches by the blocker. Must contain only test $S2$/$S3$ IDs. |

**Example Row:**
```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

---

### 6. Final Matching Results Schema (`matching_results.tsv`)
The final prediction file uploaded to the competition leaderboard.

| Field Name | Type | Nullable? | Description & Constraints |
| :--- | :--- | :---: | :--- |
| `source1_entity_id` | `string` | No | Exactly one row for every test $S1$ entity (1,732,544 rows). |
| `matched_entity_ids` | `string` | Yes (Allowed empty) | Comma-separated list of predicted matching $S2$ and $S3$ entity IDs. Every ID must be a strict subset of `candidate_entity_ids`. |

**Example Row:**
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	S3-00004
S1-00003	
```

---

### 7. Formatted ID List Invariants & Rules
1. **Delimiter:** Fields within rows are separated by a **single tab (`\t`)**. Comma delimiters are prohibited at the column level.
2. **List Separator:** IDs within the second column are separated by a **comma (`,`) without spaces**.
3. **No Quoting:** Do not enclose ID strings in double quotes (`"S2-123"` is invalid; `S2-123` is valid).
4. **No Internal Duplicates:** An ID must never appear twice in the same list (e.g. `S2-01,S2-01` is strictly invalid).
5. **Subset Constraint:** For every $S1$ record:
   $$\text{matched\_entity\_ids} \subseteq \text{candidate\_entity\_ids}$$
