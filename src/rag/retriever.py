"""
RAG retrieval: given a transaction context, find the most relevant policy chunks.
"""

from langchain_chroma import Chroma


def retrieve_policy_context(
    query: str,
    vectorstore: Chroma,
    k: int = 3,
) -> str:
    """
    Retrieve the top-k most relevant policy chunks for a given query.

    Args:
        query: A natural language description of the anomaly (built from transaction features).
        vectorstore: The loaded Chroma vectorstore.
        k: Number of chunks to retrieve.

    Returns:
        A formatted string of retrieved policy excerpts with source citations.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)

    if not results:
        return "No relevant policy context found."

    formatted = []
    for doc, score in results:
        source = doc.metadata.get("source", "Unknown Policy")
        source_name = source.split("/")[-1].replace(".txt", "").replace("_", " ").title()
        formatted.append(
            f"[{source_name}] (relevance: {1 - score:.2f})\n{doc.page_content.strip()}"
        )

    return "\n\n---\n\n".join(formatted)


def build_retrieval_query(transaction: dict, shap_drivers: str) -> str:
    """
    Build a natural language query from transaction details and SHAP drivers.
    This is what gets sent to the vector store for retrieval.
    """
    parts = [f"Transaction anomaly with risk score {transaction.get('risk_score', 0):.2f}."]

    if transaction.get("high_risk_country"):
        parts.append("Transaction involves a high-risk or sanctioned jurisdiction.")
    if transaction.get("velocity_1h", 0) > 3:
        parts.append("Unusually high transaction velocity — multiple transactions in a short period.")
    if transaction.get("amount_vs_avg_ratio", 1) > 3:
        parts.append("Transaction amount is significantly above the customer's historical average.")
    if transaction.get("is_night"):
        parts.append("Transaction occurred during overnight high-risk hours.")

    parts.append(f"Key anomaly drivers:\n{shap_drivers}")

    return " ".join(parts)
