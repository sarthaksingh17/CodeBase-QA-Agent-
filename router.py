"""
Router: Orchestration layer connecting retrieval, generation, and ingestion.
Streamlit app calls this module — no FastAPI needed.
"""

from qdrant_client import QdrantClient
from retriever import retrieve
from generator import generate
from ingest import ingest_repo as _ingest_repo, get_repo_name
from config import QDRANT_URL, QDRANT_API_KEY

# ── Qdrant connection (for collection management) ──────────────
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def ask(
    question: str,
    collection_name: str,
    top_k_search: int = 20,
    top_k_rerank: int = 5,
) -> dict:
    """
    Full pipeline: question → hybrid search → rerank → LLM → answer.

    Returns dict with: answer, sources, model.
    """
    # Retrieve and rerank
    ranked_chunks = retrieve(
        query=question,
        collection_name=collection_name,
        top_k_search=top_k_search,
        top_k_rerank=top_k_rerank,
    )

    if not ranked_chunks:
        return {
            "answer": "❌ No relevant code found for your question. "
                      "Try rephrasing or check if the repository has been ingested.",
            "sources": [],
            "model": "N/A",
        }

    # Generate answer
    return generate(question, ranked_chunks)


def list_collections() -> list[str]:
    """List all available repo collections in Qdrant."""
    collections = client.get_collections().collections
    return [c.name for c in collections]


def get_collection_info(collection_name: str) -> dict:
    """Get stats for a specific collection."""
    try:
        info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "status": info.status.value if info.status else "unknown",
        }
    except Exception as e:
        return {
            "name": collection_name,
            "error": str(e),
        }


def ingest(github_url: str, recreate: bool = False) -> str:
    """
    Ingest a GitHub repo into the pipeline.
    Returns the collection name.
    """
    _ingest_repo(github_url, recreate=recreate)
    repo_name = get_repo_name(github_url)
    return f"repo_{repo_name}".lower().replace("-", "_")


def delete_collection(collection_name: str) -> bool:
    """Delete a collection from Qdrant."""
    try:
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False
