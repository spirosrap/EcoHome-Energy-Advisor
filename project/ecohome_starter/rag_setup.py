"""
Utility to build the EcoHome vector knowledge base from local documents."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

try:  # Allow imports whether run as package or script
    from .tools import search_energy_tips  # noqa: F401  # Ensures dependencies installed
    from .models import energy  # noqa: F401
except ImportError:  # pragma: no cover
    pass

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

PACKAGE_ROOT = Path(__file__).resolve().parent
DOCUMENT_DIR = PACKAGE_ROOT / "data" / "documents"
VECTOR_DIR = PACKAGE_ROOT / "data" / "vectorstore"


def _resolve_embedding_kwargs() -> Dict[str, str]:
    """Build kwargs for OpenAIEmbeddings from environment settings."""
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("VOCAREUM_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OpenAI credentials. Set OPENAI_API_KEY or VOCAREUM_API_KEY "
            "before initializing the vector store."
        )

    base_url = (
        os.getenv("OPENAI_API_BASE")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OPENAI_API_HOST")
    )
    if not base_url and os.getenv("VOCAREUM_API_KEY"):
        base_url = "https://openai.vocareum.com/v1"

    kwargs: Dict[str, str] = {"openai_api_key": api_key}
    if base_url:
        kwargs["openai_api_base"] = base_url
    return kwargs


def load_documents(directory: Path = DOCUMENT_DIR) -> List:
    """Load all .txt documents from the documents directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Document directory {directory} does not exist.")

    documents = []
    for path in sorted(directory.glob("*.txt")):
        loader = TextLoader(str(path))
        docs = loader.load()
        documents.extend(docs)
    if not documents:
        raise RuntimeError(f"No documents found in {directory}")
    return documents


def build_vector_store(force: bool = False) -> Dict[str, int]:
    """
    Create or refresh the Chroma vector store with energy documents.

    Args:
        force: When True, rebuilds the store even if it already exists.

    Returns:
        Mapping with counts of source documents and stored chunks.
    """
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    persist_path = VECTOR_DIR / "chroma.sqlite3"

    embeddings = OpenAIEmbeddings(**_resolve_embedding_kwargs())

    if not force and persist_path.exists():
        vectorstore = Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embeddings,
        )
        chunk_count = vectorstore._collection.count()  # type: ignore[attr-defined]
        return {"documents": 0, "chunks": chunk_count}

    documents = load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(VECTOR_DIR),
    )
    chunk_count = vectorstore._collection.count()  # type: ignore[attr-defined]

    return {"documents": len(documents), "chunks": chunk_count}


def vector_store_status() -> Tuple[int, Path]:
    """Return the number of stored chunks and the persistence path."""
    persist_path = VECTOR_DIR / "chroma.sqlite3"
    if not persist_path.exists():
        return 0, persist_path

    embeddings = OpenAIEmbeddings(**_resolve_embedding_kwargs())
    vectorstore = Chroma(
        persist_directory=str(VECTOR_DIR),
        embedding_function=embeddings,
    )
    count = vectorstore._collection.count()  # type: ignore[attr-defined]
    return count, persist_path


if __name__ == "__main__":
    summary = build_vector_store(force=True)
    total_chunks, db_path = vector_store_status()
    print(
        "Vector store initialization complete.\n"
        f"Documents processed: {summary['documents']}\n"
        f"Chunks stored: {total_chunks}\n"
        f"Persisted at: {db_path}"
    )
