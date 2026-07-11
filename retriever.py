"""
Retriever: Hybrid search (dense + sparse + RRF fusion) + CrossEncoder reranking.
Stages 7, 8, 9 of the pipeline.
"""

from qdrant_client import QdrantClient, models
from qdrant_client.models import SparseVector
from embeddings import embed_dense, embed_sparse, get_reranker
from config import QDRANT_URL, QDRANT_API_KEY

# ── Qdrant connection ───────────────────────────────────────────
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


def hybrid_search(
    query: str,
    collection_name: str,
    top_k: int = 20,
) -> list:
    """
    Stage 7 + 8: Embed query with same models as ingestion,
    then run hybrid dense+sparse search with RRF fusion on Qdrant.

    Returns list of ScoredPoint objects with payloads.
    """
    # Embed query with both models
    dense_vec = embed_dense([query])[0].tolist()
    sparse_vecs = embed_sparse([query])
    sparse_vec = sparse_vecs[0]

    # Hybrid search: dense + sparse branches fused via RRF
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k,
            ),
            models.Prefetch(
                query=SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist(),
                ),
                using="sparse",
                limit=top_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    return results.points


def rerank(
    query: str,
    candidates: list,
    top_k: int = 5,
) -> list[tuple]:
    """
    Stage 9: Rerank candidates using CrossEncoder.

    Takes (query, candidate_chunks) and returns list of (chunk, score) tuples
    sorted by relevance, keeping only top_k.
    """
    if not candidates:
        return []

    reranker = get_reranker()

    # Build (query, code) pairs for cross-encoder scoring
    pairs = [
        (query, candidate.payload.get("code", ""))
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    # Pair chunks with their reranker scores and sort descending
    scored = list(zip(candidates, scores.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def retrieve(
    query: str,
    collection_name: str,
    top_k_search: int = 20,
    top_k_rerank: int = 5,
) -> list[tuple]:
    """
    Full retrieval pipeline: hybrid search → rerank → top results.

    Returns list of (ScoredPoint, rerank_score) tuples.
    """
    # Stage 7 + 8: Hybrid search
    candidates = hybrid_search(query, collection_name, top_k=top_k_search)

    if not candidates:
        print("⚠️  No results found in vector search")
        return []

    print(f"🔍 Retrieved {len(candidates)} candidates, reranking...")

    # Stage 9: Rerank
    ranked = rerank(query, candidates, top_k=top_k_rerank)

    print(f"✅ Top {len(ranked)} chunks after reranking:")
    for chunk, score in ranked:
        name = chunk.payload.get("name", "unknown")
        file = chunk.payload.get("file", "unknown")
        print(f"   {score:.3f} | {name} → {file}")

    return ranked
