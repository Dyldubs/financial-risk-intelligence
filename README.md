# Financial Risk Intelligence Platform

An end-to-end transaction risk scoring system combining **Isolation Forest anomaly detection**, **SHAP explainability**, a **RAG pipeline over policy documents**, and **Claude-powered risk summaries**.

Built as a portfolio project targeting data science and risk analytics roles in Australian financial services.

---

## What it does

1. **Detects** anomalous transactions using an Isolation Forest ensemble
2. **Explains** each anomaly using SHAP feature attributions
3. **Retrieves** relevant AML/fraud policy context using RAG (ChromaDB + local embeddings)
4. **Summarises** findings in plain English using Claude via the Anthropic API
5. **Serves** everything via a FastAPI REST endpoint and a Streamlit dashboard

---

## Tech Stack

| Layer | Tools |
|---|---|
| Anomaly detection | scikit-learn (Isolation Forest), XGBoost |
| Explainability | SHAP |
| Experiment tracking | MLflow + Optuna |
| RAG | LangChain, ChromaDB, sentence-transformers |
| LLM | Claude (Anthropic API) |
| API | FastAPI + Pydantic |
| Dashboard | Streamlit + Plotly |

---

## Quickstart

### 1. Set up environment
```bash
cd financial-risk-intelligence
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Train the model
```bash
python scripts/train.py
# Uses synthetic data by default. Add --data path/to/creditcard.csv for real data.
# Runs 30 Optuna trials and saves the best model to models/detector.joblib
```

### 4. Ingest policy documents
```bash
python scripts/ingest_policies.py
# Embeds the 4 policy documents in data/policies/ into ChromaDB
```

### 5. Launch the dashboard
```bash
streamlit run app/streamlit_app.py
```

### 6. (Optional) Launch the API
```bash
uvicorn src.api.main:app --reload --port 8000
# Interactive docs at: http://localhost:8000/docs
```

---

## Using real data (Kaggle)

Download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in `data/raw/`:

```bash
python scripts/train.py --data data/raw/creditcard.csv
```

---

## Project structure

```
financial-risk-intelligence/
├── data/
│   ├── raw/              # Place creditcard.csv here (gitignored)
│   ├── policies/         # AML, fraud, sanctions policy documents
│   └── processed/        # Auto-generated processed features
├── src/
│   ├── config.py         # Central config from .env
│   ├── data/             # Loader + feature engineering pipeline
│   ├── models/           # Isolation Forest + SHAP explainer
│   ├── rag/              # ChromaDB ingestion + retrieval
│   ├── llm/              # Claude risk summary generation
│   └── api/              # FastAPI endpoints
├── app/
│   └── streamlit_app.py  # Streamlit dashboard (3 tabs)
├── scripts/
│   ├── train.py          # Run once: trains + saves model
│   └── ingest_policies.py# Run once: embeds policies into ChromaDB
├── models/               # Auto-created: saved model files
├── mlruns/               # Auto-created: MLflow experiment logs
└── requirements.txt
```

---

## API usage

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [{
      "transaction_id": "TXN-001",
      "amount": 9800.00,
      "hour": 2,
      "velocity_1h": 4,
      "velocity_24h": 12,
      "high_risk_country": 1,
      "amount_vs_avg_ratio": 8.5,
      "merchant_risk_tier": 2,
      "days_since_account_open": 15,
      "is_weekend": 1
    }],
    "include_summaries": true
  }'
```

---

## Skills demonstrated

- End-to-end ML pipeline with scikit-learn Pipelines (reproducible preprocessing)
- Unsupervised anomaly detection for imbalanced data
- SHAP explainability for model transparency in regulated environments
- Hyperparameter optimisation with Optuna and MLflow tracking
- RAG architecture: document chunking, local embeddings, vector retrieval
- LLM integration via Anthropic API with structured prompting
- Production-grade API design with FastAPI and Pydantic validation
- Stakeholder-facing dashboard with Streamlit and Plotly
