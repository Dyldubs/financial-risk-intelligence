"""
Streamlit dashboard for the Financial Risk Intelligence Platform.

Tabs:
  1. Upload & Scan — upload a CSV, score all transactions, view flagged ones
  2. Transaction Detail — click a transaction to see SHAP + policy + AI risk brief
  3. Model Insights — global feature importance and risk distribution

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import risk_tier, TIER_ACTIONS, ROOT_DIR
from src.data.loader import load_transactions
from src.data.features import get_feature_names, prepare_training_data
from src.models.detector import load_model, score
from src.models.explainer import (
    build_explainer, get_shap_values, top_drivers_text,
    save_global_importance_plot, save_waterfall_plot,
)
from src.rag.ingestion import load_vectorstore
from src.rag.retriever import retrieve_policy_context, build_retrieval_query
from src.llm.summariser import generate_risk_summary

# --- Page config ---
st.set_page_config(
    page_title="Financial Risk Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

TIER_COLOURS = {
    "LOW": "#22c55e",
    "MEDIUM": "#f59e0b",
    "HIGH": "#ef4444",
    "CRITICAL": "#7c3aed",
}


# --- Cached resource loading ---

@st.cache_resource(show_spinner="Loading model...")
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None, None


@st.cache_resource(show_spinner="Loading policy database...")
def get_vectorstore():
    try:
        return load_vectorstore()
    except Exception:
        return None


@st.cache_data(show_spinner="Scoring transactions...")
def score_transactions(df_hash, _df):
    model, pipeline = get_model()
    if model is None:
        return None
    return score(_df, model, pipeline)


# --- Sidebar ---

with st.sidebar:
    st.title("🔍 Risk Intelligence")
    st.markdown("---")

    model, pipeline = get_model()
    vectorstore = get_vectorstore()

    st.markdown("**System Status**")
    st.markdown(f"{'✅' if model else '❌'} Anomaly Detection Model")
    st.markdown(f"{'✅' if vectorstore else '⚠️'} Policy Knowledge Base")
    st.markdown(f"{'✅' if True else '❌'} LLM Summariser")

    st.markdown("---")
    st.markdown("**Risk Tiers**")
    for tier, colour in TIER_COLOURS.items():
        st.markdown(f"<span style='color:{colour}'>■</span> **{tier}**", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("Load Demo Data"):
        st.session_state["demo_mode"] = True
        st.session_state["df"] = None  # force reload


# --- Main content ---

tab1, tab2, tab3 = st.tabs(["📤 Upload & Scan", "🔎 Transaction Detail", "📊 Model Insights"])


# ===== TAB 1: UPLOAD & SCAN =====
with tab1:
    st.header("Upload Transactions & Scan for Risk")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload a CSV of transactions",
            type=["csv"],
            help="Must include columns: amount, hour, velocity_1h, velocity_24h, "
                 "high_risk_country, amount_vs_avg_ratio, merchant_risk_tier, "
                 "days_since_account_open, is_weekend",
        )

    with col2:
        st.markdown("**Or use demo data:**")
        if st.button("Generate 500 sample transactions"):
            st.session_state["demo_mode"] = True

    # Load data
    if uploaded:
        df = pd.read_csv(uploaded)
        st.session_state["df"] = df
    elif st.session_state.get("demo_mode"):
        df = load_transactions()
        df = df.sample(min(500, len(df)), random_state=42).reset_index(drop=True)
        st.session_state["df"] = df

    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]

        if model is None:
            st.error("❌ Model not loaded. Run `python scripts/train.py` first.")
        else:
            with st.spinner("Scoring transactions..."):
                scored = score(df, model, pipeline)
                st.session_state["scored"] = scored

            # Summary metrics
            flagged_high = scored[scored["risk_tier"].isin(["HIGH", "CRITICAL"])]
            flagged_medium = scored[scored["risk_tier"] == "MEDIUM"]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Transactions", f"{len(scored):,}")
            m2.metric("Critical / High", len(flagged_high), delta=None)
            m3.metric("Medium Risk", len(flagged_medium))
            m4.metric("Avg Risk Score", f"{scored['risk_score'].mean():.3f}")

            # Risk distribution chart
            tier_counts = scored["risk_tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Count"]
            tier_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            tier_counts["Tier"] = pd.Categorical(tier_counts["Tier"], categories=tier_order, ordered=True)
            tier_counts = tier_counts.sort_values("Tier")

            fig = px.bar(
                tier_counts, x="Tier", y="Count",
                color="Tier",
                color_discrete_map=TIER_COLOURS,
                title="Risk Tier Distribution",
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

            # Flagged transactions table
            st.subheader("Flagged Transactions (High + Critical)")
            display_cols = ["transaction_id", "amount", "risk_score", "risk_tier",
                            "velocity_1h", "high_risk_country", "recommended_action"]
            display_cols = [c for c in display_cols if c in flagged_high.columns]

            if not flagged_high.empty:
                st.dataframe(
                    flagged_high[display_cols].style.background_gradient(
                        subset=["risk_score"], cmap="Reds"
                    ),
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key="flagged_table",
                )

                # Store selected row for detail tab
                if st.session_state.get("flagged_table") and \
                   st.session_state["flagged_table"].get("selection", {}).get("rows"):
                    selected_idx = st.session_state["flagged_table"]["selection"]["rows"][0]
                    st.session_state["selected_txn"] = flagged_high.iloc[selected_idx].to_dict()
                    st.info("Row selected — switch to the **Transaction Detail** tab to see the full analysis.")
            else:
                st.success("No high or critical risk transactions detected in this batch.")


# ===== TAB 2: TRANSACTION DETAIL =====
with tab2:
    st.header("Transaction Detail & Risk Brief")

    if "selected_txn" not in st.session_state:
        st.info("Select a transaction from the **Upload & Scan** tab to see its full analysis here.")
    else:
        txn = st.session_state["selected_txn"]
        scored = st.session_state.get("scored")

        tier = txn.get("risk_tier", "LOW")
        colour = TIER_COLOURS.get(tier, "#6b7280")

        st.markdown(
            f"<h3 style='color:{colour}'>⚠️ {tier} RISK — Transaction {txn.get('transaction_id', '')}</h3>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Score", f"{txn.get('risk_score', 0):.4f}")
        col2.metric("Amount", f"${txn.get('amount', 0):,.2f}")
        col3.metric("Velocity (1h)", txn.get("velocity_1h", "N/A"))

        st.divider()

        # SHAP explanation
        col_shap, col_summary = st.columns([1, 1])

        with col_shap:
            st.subheader("🔬 Anomaly Drivers (SHAP)")
            if model and scored is not None:
                df = st.session_state.get("df")
                feature_names = get_feature_names(df)
                feature_names = [f for f in feature_names if f in df.columns]
                txn_df = pd.DataFrame([txn])[feature_names]
                X_t = pipeline.transform(txn_df)
                explainer = build_explainer(model, X_t)
                shap_vals = get_shap_values(explainer, X_t)
                drivers = top_drivers_text(shap_vals[0], feature_names)

                # Simple bar chart of SHAP values
                shap_df = pd.DataFrame({
                    "Feature": feature_names,
                    "SHAP Value": shap_vals[0],
                }).sort_values("SHAP Value", key=abs, ascending=True).tail(8)

                fig = px.bar(
                    shap_df, x="SHAP Value", y="Feature",
                    orientation="h",
                    color="SHAP Value",
                    color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
                    title="Top Feature Contributions",
                )
                fig.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"**Key drivers:**\n\n{drivers}")
            else:
                st.warning("Load model to see SHAP explanation.")
                drivers = "SHAP explanation unavailable."

        with col_summary:
            st.subheader("📋 AI Risk Brief")
            with st.spinner("Generating risk brief..."):
                if vectorstore:
                    query = build_retrieval_query(txn, drivers)
                    policy_ctx = retrieve_policy_context(query, vectorstore)
                else:
                    policy_ctx = "Policy knowledge base not available."

                summary = generate_risk_summary(txn, drivers, policy_ctx)

            st.markdown(summary)

            with st.expander("📄 Retrieved Policy Context"):
                st.markdown(policy_ctx)

        st.divider()
        st.subheader("🎯 Recommended Action")
        st.info(txn.get("recommended_action", TIER_ACTIONS.get(tier, "")))


# ===== TAB 3: MODEL INSIGHTS =====
with tab3:
    st.header("Model Performance & Feature Insights")

    if "scored" not in st.session_state:
        st.info("Score a batch of transactions first in the **Upload & Scan** tab.")
    else:
        scored = st.session_state["scored"]

        col1, col2 = st.columns(2)

        with col1:
            # Risk score distribution
            fig = px.histogram(
                scored, x="risk_score", nbins=50,
                color_discrete_sequence=["#3b82f6"],
                title="Risk Score Distribution",
                labels={"risk_score": "Risk Score"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Tier pie chart
            tier_counts = scored["risk_tier"].value_counts()
            fig = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                color=tier_counts.index,
                color_discrete_map=TIER_COLOURS,
                title="Risk Tier Breakdown",
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Risk score vs amount scatter
        st.subheader("Risk Score vs Transaction Amount")
        sample = scored.sample(min(500, len(scored)), random_state=42)
        fig = px.scatter(
            sample, x="amount", y="risk_score",
            color="risk_tier",
            color_discrete_map=TIER_COLOURS,
            hover_data=["transaction_id"] if "transaction_id" in sample.columns else None,
            log_x=True,
            labels={"amount": "Transaction Amount (log scale)", "risk_score": "Risk Score"},
            title="Risk Score vs Transaction Amount",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Global importance plot (if saved)
        importance_path = ROOT_DIR / "app" / "static" / "shap_global_importance.png"
        if importance_path.exists():
            st.subheader("Global Feature Importance (SHAP)")
            st.image(str(importance_path))
        else:
            st.info("Train the model and generate SHAP plots with `python scripts/train.py` to see global importance here.")
