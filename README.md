# FINRISK NEXUS

## Enterprise Customer & Financial Risk Intelligence Platform

**FINRISK NEXUS** is an end-to-end AI/ML platform designed to identify, quantify, explain, and investigate financial risks using classical machine learning, deep learning, temporal analytics, graph intelligence, and AI-assisted investigation.

**Tech Stack:** Python, Scikit-learn, XGBoost, LightGBM, PyTorch, PyTorch Geometric, PostgreSQL, DuckDB, Polars, Parquet, FastAPI, Streamlit, Docker, MLflow, and Git.

**Project Status:** Modular implementation in progress. Home Credit and PaySim modelling evaluations are frozen, and the initial inference API is implemented and tested. Temporal, graph, AML, monitoring, and advanced investigation capabilities remain under development.

---

## 1. Executive Overview

FINRISK NEXUS integrates credit/default risk, transaction fraud, AML/network risk, temporal intelligence, explainable AI, and risk scoring into a unified decision-support platform. It aims to transform raw financial data into actionable risk insights through reproducible pipelines, validated models, interpretable predictions, and investigation workflows.

## 2. Business Problem

Financial institutions face challenges identifying defaults, fraudulent transactions, suspicious networks, and evolving behavioral patterns. Traditional rule-based systems can struggle with complex relationships, changing behavior, and large data volumes. FINRISK NEXUS addresses these challenges through predictive analytics, network analysis, and evidence-based investigation.

## 3. Solution

The platform ingests and validates data, engineers features, trains specialized models, generates risk predictions, explains model outputs, and presents findings through dashboards and APIs. Future capabilities include real-time scoring, scenario simulation, and AI-assisted investigation with human oversight.

## 4. Target Users

- Credit and risk analysts.
- Fraud detection teams.
- AML investigators.
- Risk managers.
- Machine learning engineers.
- Model validation and governance teams.

## 5. Business Questions

The platform aims to answer:

- Which customers have elevated default risk?
- Which transactions warrant investigation?
- Which entities participate in suspicious networks?
- How does risk change over time?
- What factors explain a prediction?
- Which cases should analysts prioritize?

## 6. System Architecture

The architecture comprises data ingestion, validation, canonical data modeling, feature engineering, ML services, risk scoring, explainability, API serving, dashboards, investigation workflows, and monitoring. Each component is designed for modularity, independent testing, and future scalability.

## 7. Data Architecture

The core datasets are:

- **Home Credit Default Risk:** Credit/default prediction.
- **PaySim:** Synthetic transaction fraud detection.
- **IBM AMLSim:** Synthetic AML and network analysis.

Parquet, Polars, and DuckDB support analytical processing, while PostgreSQL supports structured application data. Source-specific mappings preserve dataset boundaries; unrelated datasets must not be treated as sharing real-world customer identities.

## 8. ML Architecture

The ML lifecycle includes data validation, feature engineering, train/validation/test splitting, baseline modeling, hyperparameter tuning, probability calibration, threshold selection, final evaluation, artifact versioning, and inference. Temporal tasks require time-aware validation and strict prevention of future-data leakage.

## 9. Classical ML

Candidate algorithms include Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, and Isolation Forest.

Models are evaluated using ROC-AUC, PR-AUC, precision, recall, F1, calibration, cost-sensitive metrics, and inference latency. Selection depends on the task and operational requirements.

## 10. Deep Learning

PyTorch-based neural networks, sequence models, and graph neural networks will be evaluated against classical ML baselines. Deep learning will be retained where experiments demonstrate meaningful improvements in predictive performance, generalization, or operational value.

## 11. Temporal Intelligence

Temporal features capture transaction frequency, rolling amounts, behavioral deviations, counterparty diversity, and changes across multiple time windows. Candidate models include temporal neural networks and sequence models. Time-based evaluation ensures that features contain only information available at prediction time.

## 12. Graph/Network Intelligence

Entities become graph nodes, while documented transactions and relationships become edges. Graph statistics, community detection, GraphSAGE, and Graph Attention Networks may identify unusual relationships and suspicious networks.

Network connections are investigative signals, not proof of criminal activity.

## 13. Explainable AI

Explainability includes global feature importance, individual prediction explanations, SHAP analysis, and graph-based evidence. Explanations help analysts understand model behavior but do not establish causation. Explanation quality, stability, and consistency must be evaluated.

## 14. Risk Scoring

The proposed **FINRISK SCORE** consolidates validated risk signals into an interpretable score and Low, Medium, or High risk bands.

Credit, fraud, and AML outputs require appropriate normalization and validation before combination. Scoring policies and thresholds must be documented and versioned.

## 15. Investigation Intelligence

The proposed AI Risk Investigator combines structured evidence retrieval, model explanations, temporal patterns, and graph relationships. RAG, LangGraph, SQL tools, graph queries, and LLMs may support evidence-grounded summaries.

Human approval remains essential for consequential decisions.

## 16. Real-Time Inference

The initial REST inference layer is implemented using FastAPI.

It supports:
- Home Credit application default-risk prediction.
- PaySim transaction fraud prediction.
- Model metadata retrieval.
- API health checks.
- Input schema validation and prediction response validation.

Available endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | API health status |
| GET | `/models` | Loaded model metadata |
| POST | `/predict/paysim` | PaySim fraud prediction |
| POST | `/predict/home-credit` | Home Credit default-risk prediction |

Interactive API documentation is available at `/docs`.

The current implementation loads frozen model artifacts during application startup. Event streaming, online feature retrieval, and production-scale serving remain future enhancements.

## 17. MLOps

MLOps includes reproducible experiments, versioned artifacts, configuration management, automated testing, controlled model promotion, and rollback. MLflow, Docker, GitHub Actions, and data-validation tools are candidate technologies.

Each model release must satisfy defined quality gates.

## 18. Monitoring

Monitoring covers data quality, feature drift, predictive performance, calibration, API availability, latency, resource usage, and investigation quality.

Performance metrics should be calculated when reliable labels become available. Drift alone does not prove declining model accuracy.

## 19. Security

Security measures include access control, input validation, secret management, protected datasets, dependency validation, audit logging, and restricted agent tools.

Sensitive actions require human approval. Implementation and independent testing are necessary before claiming production security or regulatory compliance.

## 20. Testing

Testing includes unit, integration, API, data-quality, model, calibration, temporal-leakage, security, performance, and container tests.

Automated checks validate schemas, prediction outputs, threshold boundaries, artifact loading, and reproducibility.

## 21. Model Performance

Each risk task will have a separate benchmark covering ROC-AUC, PR-AUC, precision, recall, F1, confusion matrices, Brier score, calibration, cost-sensitive metrics, and inference latency.

Results will include dataset versions, evaluation methodology, and thresholds. Only measured results will be published.

## 22. Business KPIs

Business success will be assessed through detection yield, false-alert burden, investigation preparation time, evidence traceability, service reliability, and cost per processed record.

Financial savings and productivity improvements will be claimed only when supported by measured evidence.

## 23. Limitations

Public and synthetic datasets may not represent live financial behavior. Labels can be incomplete, risk patterns evolve, and model explanations do not prove causality.

Combining independent datasets and heterogeneous risk scores requires explicit validation. Local deployment does not establish production-scale readiness.

## 24. Future Enhancements

The roadmap covers ten phases:

1. Data foundation.
2. Classical ML.
3. Temporal intelligence.
4. Graph intelligence.
5. Real-time inference.
6. Risk Scenario Lab.
7. AI Risk Investigator.
8. Production engineering.
9. Observability.
10. Research.

Each phase will have measurable acceptance criteria, reproducible experiments, and documented limitations.

## 25. How to Run

### Prerequisites

- Python environment compatible with the pinned project dependencies.
- Git.
- Model artifacts provisioned locally as described below.

### Clone the Repository

```powershell
git clone https://github.com/cruzer-shravan/FINRISK-NEXUS.git
cd FINRISK-NEXUS
```

### Create and Activate a Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

### Model Artifact Provisioning

Serialized model artifacts (`.joblib`) are excluded from Git.

The inference API requires the following frozen artifacts:

| Dataset | Required artifact |
|---|---|
| PaySim | `models/paysim/tree_fraud_v1/random_forest.joblib` |
| Home Credit | `reports/home_credit/model_selection_locks/20261107_xgboost_sigmoid/model/calibrated_xgboost_tuned_sigmoid.joblib` |

Expected SHA256 checksums:

**PaySim**
```text
f78ff716a62665aa2c3524b39ee7ac3a909895b18ae13c55abc8f729dee002fb
```

**Home Credit**
```text
21b24d48dd54a03080f30d3421d807d74ae04fd7f734c34da82d3e59e33d644b
```

Before starting the API:

1. Obtain the corresponding frozen model artifacts from the project release archive or original development environment.
2. Place each artifact at the exact path specified above.
3. Verify its SHA256 checksum.
4. Do not substitute a newly trained or differently calibrated model without creating a new model release and updating its contract.

The model contracts under `configs/model_contracts/` define the expected artifact paths, feature schemas, model checksums, and decision thresholds.

Example checksum verification:

```powershell
Get-FileHash models/paysim/tree_fraud_v1/random_forest.joblib -Algorithm SHA256

Get-FileHash reports/home_credit/model_selection_locks/20261107_xgboost_sigmoid/model/calibrated_xgboost_tuned_sigmoid.joblib -Algorithm SHA256
```

### Start the Inference API

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The API loads and verifies the configured model artifacts during startup.

Available endpoints:

- Health: `http://127.0.0.1:8000/health`
- Model metadata: `http://127.0.0.1:8000/models`
- Interactive API documentation: `http://127.0.0.1:8000/docs`

Prediction endpoints:

- `POST /predict/paysim`
- `POST /predict/home-credit`

Prediction requests contain a `records` array with between 1 and 1,000 records. Each record must satisfy the corresponding model feature contract.

### Run Tests

```powershell
python -m pytest -q
```

Run code-quality checks:

```powershell
python -m ruff check api src/inference tests
python -m ruff format --check api src/inference tests
```

### Release Evidence and Reproducibility

The repository contains curated model evaluation reports, model-selection records, release records, feature contracts, checksum manifests, and selected validation-freeze documentation.

Complete validation-freeze bundles and large serialized artifacts may be retained in the original development environment and are not necessarily included in a Git clone.

A checksum manifest documents the integrity of its referenced files; it does not imply that every referenced file is committed to the repository.

For reproducibility:

- Preserve frozen model artifacts and their SHA256 checksums.
- Preserve the associated feature contracts and decision thresholds.
- Do not tune thresholds or select replacement models using frozen test results.
- Treat future model changes as new, independently versioned releases.
- Do not interpret experimental evaluation results as production approval.


---

## Project Philosophy

FINRISK NEXUS aims to move beyond risk prediction toward explainable, reproducible, and evidence-driven financial risk intelligence, combining robust ML engineering with responsible investigation and human oversight.
