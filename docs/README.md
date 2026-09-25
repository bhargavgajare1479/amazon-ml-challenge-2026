# Amazon ML Challenge 2026: Technical Documentation Index

**Root Index Path:** `docs/README.md`  
**Challenge:** Business Entity Resolution  
**Evaluation Standard:** Entity-Level Macro $F_{0.5}$  
**Repository Version:** Active Development  

---

### Welcome to the Amazon ML Challenge 2026 Engineering Workspace

This directory contains the complete technical specifications, architectural blueprints, data contracts, and execution plans for our competitive entity resolution solution. If you are joining the project today, start by reading this index.

---

### Core Documentation Sitemap

| Document | Primary Audience | Key Questions Answered |
| :--- | :--- | :--- |
| **[project-reconstruction.md](file:///z:/amazon-ml-2026/docs/project-reconstruction.md)** | All Teammates | Where does the project currently stand? What works, what failed, and what are the verified numbers? |
| **[PRD.md](file:///z:/amazon-ml-2026/docs/PRD.md)** | Engineering & Product | What are the exact competition requirements, constraints, metrics, and deliverable contracts? |
| **[TRD.md](file:///z:/amazon-ml-2026/docs/TRD.md)** | Core ML Engineers | What is the technical specification of the data flow, normalization, blocking, features, and model? |
| **[architecture.md](file:///z:/amazon-ml-2026/docs/architecture.md)** | Systems Engineers | How do data streams flow through the funnel? How do training, validation, and inference differ? |
| **[data-schema.md](file:///z:/amazon-ml-2026/docs/data-schema.md)** | Data Engineers | What are the schemas, nullability rules, and formatting invariants for all raw and output TSV files? |
| **[ml-pipeline.md](file:///z:/amazon-ml-2026/docs/ml-pipeline.md)** | ML / Research Engineers | What is the mathematical formulation of matching, metric hierarchy, and model configuration? |
| **[validation-plan.md](file:///z:/amazon-ml-2026/docs/validation-plan.md)** | ML Engineers | How do we evaluate models offline without data leakage? How is Macro $F_{0.5}$ calculated? |
| **[experiment-plan.md](file:///z:/amazon-ml-2026/docs/experiment-plan.md)** | ML / Research Engineers | What experiments have completed, what is currently running, and what are the prioritized next hypotheses? |
| **[implementation-plan.md](file:///z:/amazon-ml-2026/docs/implementation-plan.md)** | Project Leads | What is the complete task breakdown, dependencies, status, and milestone schedule? |
| **[team-workflow.md](file:///z:/amazon-ml-2026/docs/team-workflow.md)** | All Teammates | Who owns which modules? How do we coordinate branches, experiments, and avoid code collisions? |
| **[definition-of-done.md](file:///z:/amazon-ml-2026/docs/definition-of-done.md)** | All Teammates | When is a task truly done? What are the 11 strict conditions for a final submission? |
| **[testing-strategy.md](file:///z:/amazon-ml-2026/docs/testing-strategy.md)** | QA / Engineers | What unit tests, regression tests, and official submission validations guard the codebase? |
| **[risk-register.md](file:///z:/amazon-ml-2026/docs/risk-register.md)** | Project Leads | What are the highest-severity failure modes (recall drops, singleton penalties, OOMs) and mitigations? |
| **[gap-analysis.md](file:///z:/amazon-ml-2026/docs/gap-analysis.md)** | Project Leads | What is the exact delta between competition requirements and our current code implementation? |
| **[next-actions.md](file:///z:/amazon-ml-2026/docs/next-actions.md)** | Core Developers | What are the top 5 technical priorities that must be implemented next, ranked strictly by impact? |
| **[decision-log.md](file:///z:/amazon-ml-2026/docs/decision-log.md)** | All Teammates | Why did we make specific algorithmic, architectural, and optimization choices? |
| **[status.md](file:///z:/amazon-ml-2026/docs/status.md)** | Stakeholders | What is the official current project status, best validation score, and benchmark metrics? |
| **[UI-UX.md](file:///z:/amazon-ml-2026/docs/UI-UX.md)** | Reviewers | Formal statement explaining why UI/UX is not applicable to this offline batch ML pipeline. |
| **[app-flow.md](file:///z:/amazon-ml-2026/docs/app-flow.md)** | Systems Engineers | Execution flowchart showing the offline data and model flow from raw TSVs to final submission. |
| **[backend-and-data-pipeline.md](file:///z:/amazon-ml-2026/docs/backend-and-data-pipeline.md)** | Systems Engineers | Technical document describing the batch data processing architecture (non-server backend). |

---

### Executive Project Summary

* **Competition Goal:** Business Entity Resolution across three data sources (reference $S1$ to noisy $S2$ and $S3$).
* **Evaluation Metric:** Entity-Level Macro $F_{0.5}$ (Precision weighted $2\times$ over recall; singletons earn $1.0$ for empty predictions, $0.0$ for false merges).
* **Current Baseline Macro $F_{0.5}$:** **0.38328** (Rule-based exact name matcher).
* **Current Blocker Recall:** **69.97%** with **57.65 candidates per entity** across 4.13M targets.
* **Immediate Sprint Focus:** Implement `src/features.py` (RapidFuzz), train first LightGBM model, and optimize threshold $\tau^*$ to surpass the 0.383 baseline.
