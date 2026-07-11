"""
RAG ingestion pipeline.

Loads policy documents from data/policies/, chunks them,
embeds them with a local sentence-transformers model (no API key needed),
and stores them in ChromaDB.

Run once: python scripts/ingest_policies.py
"""

from pathlib import Path
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import POLICIES_DIR, CHROMA_DIR


EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # ~80MB, runs locally, no API key required


def get_embeddings():
    """Return the sentence-transformers embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def ingest_policies(force_rebuild: bool = False) -> Chroma:
    """
    Ingest all .txt files in data/policies/ into ChromaDB.

    Args:
        force_rebuild: If True, delete and rebuild the vector store.

    Returns:
        Chroma vectorstore ready for retrieval.
    """
    if CHROMA_DIR.exists() and not force_rebuild:
        print(f"ChromaDB already exists at {CHROMA_DIR}. Loading existing store.")
        return load_vectorstore()

    print(f"Ingesting policy documents from {POLICIES_DIR}...")
    loader = DirectoryLoader(str(POLICIES_DIR), glob="**/*.txt", loader_cls=TextLoader)
    raw_docs = loader.load()

    if not raw_docs:
        raise ValueError(f"No .txt files found in {POLICIES_DIR}. Add policy documents first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "],
    )
    docs = splitter.split_documents(raw_docs)
    print(f"  Split into {len(docs)} chunks from {len(raw_docs)} documents.")

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    # ChromaDB >= 0.4 auto-persists; .persist() was removed
    print(f"  Stored in ChromaDB at {CHROMA_DIR}")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB vectorstore."""
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )
