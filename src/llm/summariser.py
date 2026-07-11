"""
LLM-powered risk summary generation using Claude (Anthropic).

Takes a flagged transaction, its SHAP explanation, and retrieved policy context,
and generates a concise, professional risk brief suitable for a risk analyst.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from src.config import ANTHROPIC_API_KEY, LLM_MODEL


SYSTEM_PROMPT = """You are a senior financial crime risk analyst at an Australian bank.
Your job is to write clear, concise risk briefs for flagged transactions.
You are precise, professional, and always cite the specific policy that applies.
You write for an audience of risk analysts who need to make fast, accurate decisions.
Never speculate beyond what the data shows. Be direct."""


def build_risk_prompt(transaction: dict, shap_drivers: str, policy_context: str) -> str:
    return f"""A transaction has been flagged by our anomaly detection system.

TRANSACTION DETAILS:
- Transaction ID: {transaction.get('transaction_id', 'N/A')}
- Amount: ${transaction.get('amount', 0):,.2f}
- Risk Score: {transaction.get('risk_score', 0):.2f} / 1.00
- Risk Tier: {transaction.get('risk_tier', 'UNKNOWN')}
- Hour of day: {transaction.get('hour', 'N/A')}:00
- High-risk country flag: {'Yes' if transaction.get('high_risk_country') else 'No'}
- Transactions in last 1h: {transaction.get('velocity_1h', 'N/A')}
- Transactions in last 24h: {transaction.get('velocity_24h', 'N/A')}
- Amount vs. account average: {transaction.get('amount_vs_avg_ratio', 1):.1f}x

ANOMALY DRIVERS (SHAP explanation):
{shap_drivers}

RELEVANT POLICY CONTEXT:
{policy_context}

Write a risk brief with exactly three parts:
1. SUMMARY (1–2 sentences): What makes this transaction suspicious.
2. POLICY RELEVANCE (1–2 sentences): Which specific policy applies and why, citing the policy by name.
3. RECOMMENDED ACTION (1 sentence): What the analyst should do next.

Be specific. Use numbers from the transaction details. Do not pad with generic statements."""


def generate_risk_summary(
    transaction: dict,
    shap_drivers: str,
    policy_context: str,
) -> str:
    """
    Generate a risk brief for a single flagged transaction.

    Args:
        transaction: Dict of transaction fields (including risk_score, risk_tier, etc.)
        shap_drivers: Human-readable string of top SHAP drivers from explainer.py
        policy_context: Retrieved policy text from the RAG retriever

    Returns:
        A formatted risk brief string.
    """
    if not ANTHROPIC_API_KEY:
        return _fallback_summary(transaction, shap_drivers)

    llm = ChatAnthropic(
        model=LLM_MODEL,
        anthropic_api_key=ANTHROPIC_API_KEY,
        max_tokens=400,
        temperature=0.2,
    )

    prompt = build_risk_prompt(transaction, shap_drivers, policy_context)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    response = llm.invoke(messages)
    return response.content.strip()


def _fallback_summary(transaction: dict, shap_drivers: str) -> str:
    """
    Fallback summary when no API key is configured.
    Uses a template-based approach so the app still works without LLM access.
    """
    tier = transaction.get("risk_tier", "UNKNOWN")
    score = transaction.get("risk_score", 0)
    txn_id = transaction.get("transaction_id", "N/A")

    return f"""**SUMMARY:** Transaction {txn_id} has been flagged with a risk score of {score:.2f} \
({tier} tier) based on anomalous patterns in the transaction data.

**POLICY RELEVANCE:** Review against AML transaction monitoring thresholds and velocity \
screening rules. Manual review required for {tier} tier transactions.

**RECOMMENDED ACTION:** {transaction.get('recommended_action', 'Review required.')}

*(LLM summary unavailable — set ANTHROPIC_API_KEY in .env for AI-generated briefs.)*"""
