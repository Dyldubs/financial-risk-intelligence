"""
Ingest policy documents into ChromaDB.

Run once after training:
    python scripts/ingest_policies.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.ingestion import ingest_policies

if __name__ == "__main__":
    print("Ingesting policy documents into ChromaDB...")
    vectorstore = ingest_policies(force_rebuild=True)
    print(f"Done. Collection contains {vectorstore._collection.count()} chunks.")
