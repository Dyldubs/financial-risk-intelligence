# Model Card — Financial Risk Intelligence Platform

**Version:** 1.0.0  
**Last updated:** July 2026  
**Status:** Development / Portfolio  
**Owner:** Dylan Wubs  
**Contact:** dylan.w96@gmail.com

---

## 1. Model Overview

| Field | Details |
|---|---|
| **Model name** | Financial Risk Intelligence Platform |
| **Model type** | Ensemble: Isolation Forest (unsupervised) + XGBoost (supervised) |
| **Task** | Transaction anomaly detection and fraud probability scoring |
| **Domain** | Retail banking — payment fraud and financial crime detection |
| **Output** | Continuous risk score ∈ [0, 1], risk tier (LOW / MEDIUM / HIGH / CRITICAL), SHAP-based explanation, RAG-grounded risk brief |
| **Regulatory context** | Australian AML/CTF Act 2006, APRA CPG 220, AUSTRAC reporting obligations, OFAC/DFAT sanctions screening |

---

## 2. Intended Use

### Primary use cases

- **Real-time transaction risk scoring** — flag individual transactions for analyst review before settlement
- **Batch screening** — score overnight transaction files against AML/fraud typology rules
- **Alert prioritisation** — rank open alerts by risk score to maximise analyst efficiency (highest-risk cases reviewed first)
- **Policy grounding** — retrieve the most relevant AML, fraud, and sanctions policy excerpts for each flagged transaction via RAG

### Out-of-scope uses

- **Credit decisioning** — the model scores transaction anomalies, not creditworthiness. It must not be used to accept or decline credit applications.
- **Customer identity verification** — the model has no KYC or biometric capability.
- **Automated transaction blocking** — the model is designed as a decision-support tool. Blocking or freezing transactions must involve human review. No automated action should be taken solely on model output.
- **Regulatory reporting substitution** — model output does not constitute a Suspicious Matter Report (SMR) or Threshold Transaction Report (TTR) under the AML/CTF Act. Human review is required before any regulatory lodgement.
- **Cross-border sanctions screening** — while the RAG pipeline references OFAC/DFAT/UNSC materials, this model is not a substitute for a validated sanctions screening system integrated with live sanctions lists.

---

## 3. Training Data

### Dataset

| Property | Synthetic dataset | Real dataset (optional) |
|---|---|---|
| **Source** | Generated via `src/data/loader.py` | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Size** | 10,000 transactions | 284,807 transactions |
| **Fraud rate** | 2.0% (200 fraud / 9,800 legitimate) | 0.17% (492 fraud / 284,315 legitimate) |
| **Time period** | N/A (synthetic) | September 2013, European cardholders |
| **Geography** | Australian context (AUD amounts, APRA/AUSTRAC policy references) | European (anonymised) |

### Features

| Feature | Type | Description |
|---|---|---|
| `amount` | Continuous | Transaction amount in AUD |
| `hour` | Ordinal | Hour of day the transaction occurred (0–23) |
| `merchant_risk_tier` | Ordinal | Merchant risk category (0 = low, 1 = medium, 2 = high) |
| `velocity_1h` | Count | Number of transactions by the same account in the preceding hour |
| `velocity_24h` | Count | Number of transactions by the same account in the preceding 24 hours |
| `high_risk_country` | Binary | 1 if transaction involves a FATF high-risk or sanctioned jurisdiction |
| `amount_vs_avg_ratio` | Continuous | Transaction amount divided by account's 30-day rolling average |
| `days_since_account_open` | Count | Age of the account at transaction time |
| `is_weekend` | Binary | 1 if transaction occurred on Saturday or Sunday |

### Engineered features

| Feature | Derivation |
|---|---|
| `log_amount` | `log₁₊₁(amount)` — reduces right skew from large outlier transactions |
| `is_night` | 1 if `hour` ∈ {23, 0, 1, 2, 3, 4, 5} — captures high-risk overnight window |

### Preprocessing

- `StandardScaler` applied to all numeric features after log-transform and binary flag engineering
- Pipeline fitted on training data only; test set transformed using training statistics (no data leakage)
- Train/test split: 80/20 stratified on `is_fraud`, `random_state=42`

### Known data limitations

- The synthetic dataset is generated with fixed distributional assumptions. Real fraud patterns are adversarial and evolve continuously; a model trained solely on synthetic data will underperform on novel typologies.
- The Kaggle dataset is from 2013 European card transactions. Payment fraud patterns have changed significantly since then. PCA-anonymised features (`V1`–`V28`) limit interpretability and domain feature engineering.
- Neither dataset includes customer demographic information, relationship history, device fingerprinting, or network/graph features — all of which are standard signals in production bank fraud systems.

---

## 4. Model Architecture

### Model 1: Isolation Forest (unsupervised)

| Parameter | Value |
|---|---|
| Algorithm | `sklearn.ensemble.IsolationForest` |
| Hyperparameter search | Optuna (30 trials, `direction=maximize`, `metric=AUC-PR`) |
| Search space | `n_estimators` ∈ [100, 400], `max_samples` ∈ [0.5, 1.0], `contamination` ∈ [0.005, 0.05], `max_features` ∈ [0.5, 1.0] |
| Best params (synthetic data) | `n_estimators=184`, `max_samples=0.884`, `contamination=0.040`, `max_features=0.543` |
| Experiment tracking | MLflow (SQLite backend) |
| Saved artifact | `models/detector.joblib` |

**How it works:** Isolation Forest isolates anomalies by recursively partitioning the feature space using random splits. Anomalous points require fewer splits to isolate and receive a lower `decision_function` score. Scores are inverted and normalised to [0, 1] where 1 = most anomalous. No fraud labels are required.

### Model 2: XGBoost Classifier (supervised)

| Parameter | Value |
|---|---|
| Algorithm | `xgboost.XGBClassifier` |
| Imbalance handling | `scale_pos_weight` searched in range [imbalance_ratio × 0.5, imbalance_ratio × 2.0] |
| Internal eval metric | `aucpr` (precision-recall AUC) |
| Hyperparameter search | Optuna (30 trials) |
| Search space | `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `scale_pos_weight` |
| Experiment tracking | MLflow (same experiment as Isolation Forest) |
| Saved artifact | `models/xgb_classifier.joblib` |

**How it works:** XGBoost builds an additive ensemble of decision trees using gradient boosting. It outputs a calibrated fraud probability ∈ [0, 1] directly. Requires `is_fraud` labels at training time.

### Explainability

- **SHAP TreeExplainer** is applied to both models post-hoc
- Per-transaction waterfall plots show each feature's contribution to the risk score
- Global importance plots (bar chart + beeswarm) are generated at training time
- The top-3 SHAP drivers are injected into the LLM prompt to ground the risk brief

### RAG pipeline

- 4 policy documents embedded using `sentence-transformers/all-MiniLM-L6-v2` (runs locally, no external API)
- Stored in ChromaDB vector store
- At scoring time, top-3 most relevant policy chunks are retrieved via cosine similarity
- Retrieved context is passed to Claude (`claude-haiku-4-5-20251001`) to generate a structured risk brief

---

## 5. Performance

All metrics are computed on the 20% held-out test set (stratified, `random_state=42`). The dataset is not shuffled between runs — results are reproducible.

### Primary metric: AUC-PR

AUC-PR (area under the precision-recall curve) is the primary evaluation metric for this model. With a ~2% fraud rate, a naive classifier that scores all transactions as the majority class achieves **AUC-PR ≈ 0.020** (the baseline). Accuracy and AUC-ROC are reported for completeness but are not used for model selection.

| Metric | Isolation Forest | XGBoost | Naive baseline |
|---|---|---|---|
| **AUC-PR** | 0.974 | 0.999 | 0.020 |
| **AUC-ROC** | — | — | 0.500 |
| **Fraud caught @ 10% review rate** | see notebook | see notebook | ~10% |

*Full precision-recall curves, ROC curves, calibration plots, and operational lift charts are in `notebooks/02_model_comparison.ipynb`.*

### Threshold-dependent metrics

The model does not have a fixed classification threshold. Threshold selection is a **business decision** that depends on the analyst team's review capacity and the relative cost of missed fraud vs. false positives. The comparison notebook provides precision, recall, and F1 at all thresholds to support that decision.

### Risk tier mapping

| Tier | Score range | Default action |
|---|---|---|
| LOW | < 0.30 | Log for audit trail only |
| MEDIUM | 0.30 – 0.60 | Add to daily review queue |
| HIGH | 0.60 – 0.85 | Same-day analyst review |
| CRITICAL | ≥ 0.85 | Immediate escalation; freeze transaction pending review |

Tier thresholds are configurable via environment variables (`LOW_RISK_THRESHOLD`, `MEDIUM_RISK_THRESHOLD`, `HIGH_RISK_THRESHOLD`) and should be recalibrated against the institution's fraud loss data and analyst capacity.

---

## 6. Limitations

**Synthetic training data:** The default training data is generated from fixed statistical distributions. Real fraud patterns are adversarial — fraudsters adapt to detection systems. A production deployment must be trained on real, institution-specific transaction data.

**Feature set:** The 9-feature input is intentionally minimal for portfolio purposes. Production systems typically incorporate 50–200+ features including device fingerprints, IP geolocation, merchant category codes, customer relationship tenure, network graph features, and behavioural biometrics.

**No real-time features:** `velocity_1h` and `velocity_24h` are assumed to be pre-computed and supplied as inputs. In production, these require a real-time feature store (e.g. Redis, Tecton, Feast) capable of computing rolling aggregations at sub-second latency.

**Temporal validity:** Both models are static snapshots. Fraud patterns drift over time (concept drift). The model's discriminative ability will degrade without periodic retraining. No automated retraining pipeline is included in this version.

**Isolation Forest score calibration:** The Isolation Forest's normalised anomaly score is not a calibrated probability. It should be used for ranking/prioritisation, not interpreted as "probability this transaction is fraud." The XGBoost model's output is a calibrated probability and is more appropriate where a probabilistic interpretation is required.

**LLM hallucination risk:** The Claude-generated risk brief is grounded in retrieved policy context but is not guaranteed to be factually accurate. All LLM-generated content should be treated as a **draft for analyst review**, not a definitive compliance determination.

**Sanctions list currency:** The sanctions content embedded in ChromaDB is static. Real sanctions screening requires integration with live OFAC/DFAT/UNSC list feeds updated daily or in real-time.

---

## 7. Bias and Fairness

### Protected attributes

The model does not use, and was not trained on, any legally protected attributes including:

- Customer name, gender, age, or date of birth
- Race, ethnicity, or nationality
- Religion or political affiliation
- Disability status

### Proxy risk

`high_risk_country` is a jurisdiction-level risk flag based on FATF and government-designated lists, not customer nationality. However, if a customer population is disproportionately composed of individuals transacting with particular jurisdictions, this feature could act as an indirect proxy for national origin. This should be monitored in deployment.

`days_since_account_open` may correlate with customer demographic segments (e.g. recent migrants who are new to the Australian banking system). This feature's impact should be audited against customer demographic data before production deployment.

### Bias evaluation

A formal disparate impact analysis across protected demographic groups has not been performed, as the training data does not contain protected attributes. **This analysis is a mandatory requirement before production deployment** under the Australian Consumer Law and APRA's Prudential Practice Guide CPG 229 (Climate and other emerging risks; extrapolated to model risk).

---

## 8. Model Risk Management

*This section is structured in alignment with APRA Prudential Practice Guide CPG 220 — Model Risk.*

### Model risk tier

This model is classified as **Tier 2 (Material)** under a standard bank model risk taxonomy:
- Output influences financial crime compliance decisions
- Incorrect output could result in regulatory breach or financial loss
- Not fully automated — human review is required before action

### Model validation requirements (pre-production)

Before deployment in a production environment, the following independent validation activities are required:

- **Conceptual soundness review** — assessment of whether the modelling approach is appropriate for the use case
- **Data quality assessment** — review of training data completeness, accuracy, and representativeness
- **Performance benchmarking** — independent replication of reported metrics on held-out data
- **Sensitivity analysis** — testing model output stability under feature perturbation and distributional shift
- **Champion/challenger testing** — parallel running against the incumbent detection system
- **Stress testing** — evaluation under scenarios representing novel fraud typologies not present in training data
- **Disparate impact analysis** — as described in Section 7

### Model governance controls (production)

| Control | Description | Frequency |
|---|---|---|
| Performance monitoring | Track AUC-PR, precision, recall on labelled production data | Monthly |
| Population Stability Index (PSI) | Detect distributional shift in input features | Monthly |
| KS statistic monitoring | Monitor score distribution stability | Monthly |
| Threshold review | Reassess tier thresholds against analyst capacity and fraud loss data | Quarterly |
| Full model revalidation | End-to-end independent validation | Annually or after material change |
| Retraining trigger | PSI > 0.25 on any input feature or AUC-PR drop > 5% | As triggered |

### Model inventory

This model should be registered in the institution's Model Inventory with the following attributes: model name, version, owner, validation status, risk tier, use case, deployment environment, review date, and next revalidation date.

---

## 9. Monitoring Plan

### Metrics to monitor in production

| Metric | Description | Alert threshold |
|---|---|---|
| Input PSI | Population Stability Index per feature; detects covariate shift | PSI > 0.25 (high shift) |
| Score PSI | PSI on the output risk score distribution | PSI > 0.20 |
| KS statistic | Separation between fraud and legitimate score distributions | KS drop > 10% from baseline |
| Alert volume | Number of HIGH/CRITICAL flags per day | ±30% from 30-day rolling average |
| False positive rate | Proportion of HIGH/CRITICAL flags confirmed as legitimate after review | > 80% sustained over 2 weeks |
| SHAP stability | Mean absolute SHAP value per feature; detects feature importance drift | Feature rank change of 3+ positions |

### Retraining triggers

- Any monitored metric breaches its alert threshold for two consecutive reporting periods
- A new fraud typology is identified that the current model demonstrably misses
- Material change to the institution's product suite or customer base
- Scheduled annual model review

---

## 10. Ethical Considerations

**Human oversight:** This model is a decision-support tool. No transaction should be blocked, frozen, or reported to AUSTRAC based solely on model output without human review. The recommended action outputs are starting points for analyst assessment, not directives.

**Transparency to customers:** Customers subject to adverse action (e.g. transaction freeze) based in part on this model's output have rights under the Australian Consumer Law. The institution must be able to provide a human-intelligible explanation for any adverse action. SHAP explanations support this requirement.

**Right to contestation:** Model-assisted decisions must be contestable. The institution should maintain a process for customers to dispute a fraud flag and have it reviewed by a human who has access to the full evidence.

**Feedback loop risk:** If the model's flags consistently determine which transactions are investigated, uninvestigated fraud will never generate training labels. This creates survivorship bias in future retraining data. Production systems should include a random sampling mechanism to ensure a proportion of low-scoring transactions are also reviewed.

---

## 11. Version History

| Version | Date | Changes |
|---|---|---|
| 1.0.0 | July 2026 | Initial version. Isolation Forest + XGBoost. SHAP explainability. RAG pipeline over 4 policy documents. FastAPI + Streamlit. |

---

## 12. References

- APRA Prudential Practice Guide CPG 220 — Model Risk (November 2023)
- APRA Prudential Standard CPS 220 — Risk Management
- Australian Transaction Reports and Analysis Centre (AUSTRAC) — AML/CTF Act 2006
- Financial Action Task Force (FATF) — High-risk and other monitored jurisdictions
- OFAC Sanctions List / DFAT Consolidated List / UNSC Sanctions List
- Ribeiro et al. (2016) — "Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME)
- Lundberg & Lee (2017) — A Unified Approach to Interpreting Model Predictions (SHAP)
- Mitchell et al. (2019) — Model Cards for Model Reporting (Google)
- Lewis (1994) — An Introduction to Credit Scoring
