"""
FastAPI service exposing the risk intelligence pipeline as a REST API.

Endpoints:
  POST /analyse        — score a batch of transactions and return risk summaries
  GET  /health         — health check
  GET  /model/info     — model metadata

Start with: uvicorn src.api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np

from src.models.detector import load_model, score, _normalise_scores
from src.models.explainer import build_explainer, get_shap_values, top_drivers_text
from src.rag.ingestion import load_vectorstore
from src.rag.retriever import retrieve_policy_context, build_retrieval_query
from src.llm.summariser import generate_risk_summary
from src.data.features import get_feature_names

app = FastAPI(
    title="Financial Risk Intelligence API",
    description="Anomaly detection + RAG + LLM risk summaries for transaction monitoring.",
    version="1.0.0",
)

# Module-level singletons — loaded once on startup
_model = None
_pipeline = None
_explainer = None
_vectorstore = None


@app.on_event("startup")
def load_resources():
    global _model, _pipeline, _explainer, _vectorstore
    try:
        _model, _pipeline = load_model()
        print("Model and pipeline loaded.")
    except FileNotFoundError as e:
        print(f"WARNING: {e}. Train the model first with: python scripts/train.py")

    try:
        _vectorstore = load_vectorstore()
        print("ChromaDB vectorstore loaded.")
    except Exception as e:
        print(f"WARNING: Could not load vectorstore: {e}. Run: python scripts/ingest_policies.py")


# --- Request / Response schemas ---

class Transaction(BaseModel):
    transaction_id: Optional[str] = Field(default=None, description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in AUD")
    hour: int = Field(default=12, ge=0, le=23, description="Hour of day (0–23)")
    merchant_risk_tier: int = Field(default=0, ge=0, le=2, description="0=low, 1=medium, 2=high")
    velocity_1h: int = Field(default=1, ge=0, description="Number of transactions in last 1 hour")
    velocity_24h: int = Field(default=5, ge=0, description="Number of transactions in last 24 hours")
    high_risk_country: int = Field(default=0, ge=0, le=1, description="1 if high-risk jurisdiction")
    amount_vs_avg_ratio: float = Field(default=1.0, ge=0, description="Amount / 30-day account avg")
    days_since_account_open: int = Field(default=365, ge=0)
    is_weekend: int = Field(default=0, ge=0, le=1)


class RiskResult(BaseModel):
    transaction_id: Optional[str]
    amount: float
    risk_score: float
    risk_tier: str
    top_drivers: str
    policy_context: str
    risk_summary: str
    recommended_action: str


class AnalyseRequest(BaseModel):
    transactions: list[Transaction]
    include_summaries: bool = Field(default=True, description="Set False to skip LLM calls (faster)")


class AnalyseResponse(BaseModel):
    total: int
    flagged: int
    results: list[RiskResult]


# --- Endpoints ---

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "vectorstore_loaded": _vectorstore is not None,
    }


@app.get("/model/info")
def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_type": type(_model).__name__,
        "n_estimators": getattr(_model, "n_estimators", None),
        "contamination": getattr(_model, "contamination", None),
    }


@app.post("/analyse", response_model=AnalyseResponse)
def analyse(request: AnalyseRequest):
    if _model is None or _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")

    df = pd.DataFrame([t.model_dump() for t in request.transactions])
    scored_df = score(df, _model, _pipeline)

    feature_names = get_feature_names(df)
    feature_names = [f for f in feature_names if f in df.columns]
    X_transformed = _pipeline.transform(df[feature_names])

    results = []
    for idx, row in scored_df.iterrows():
        txn = row.to_dict()

        # SHAP explanation
        shap_vals = get_shap_values(
            build_explainer(_model, X_transformed),
            X_transformed[idx:idx+1]
        )
        drivers = top_drivers_text(shap_vals[0], feature_names)

        # RAG retrieval
        policy_ctx = ""
        if _vectorstore is not None:
            query = build_retrieval_query(txn, drivers)
            policy_ctx = retrieve_policy_context(query, _vectorstore)

        # LLM summary
        summary = ""
        if request.include_summaries:
            summary = generate_risk_summary(txn, drivers, policy_ctx)

        results.append(RiskResult(
            transaction_id=txn.get("transaction_id"),
            amount=txn["amount"],
            risk_score=round(txn["risk_score"], 4),
            risk_tier=txn["risk_tier"],
            top_drivers=drivers,
            policy_context=policy_ctx,
            risk_summary=summary,
            recommended_action=txn["recommended_action"],
        ))

    flagged = sum(1 for r in results if r.risk_tier in ("HIGH", "CRITICAL"))
    return AnalyseResponse(total=len(results), flagged=flagged, results=results)
