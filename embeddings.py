"""
Shared embedding models — singleton pattern.
Both ingest.py and retriever.py import from here to avoid double-loading.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer, CrossEncoder
from fastembed import SparseTextEmbedding

# ── Model names ─────────────────────────────────────────────────
DENSE_MODEL_NAME = "all-MiniLM-L6-v2"
DENSE_DIM = 384
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Singleton instances ─────────────────────────────────────────
_dense_model = None
_sparse_model = None
_reranker_model = None


def get_dense_model() -> SentenceTransformer:
    """Load or return cached dense embedding model."""
    global _dense_model
    if _dense_model is None:
        print(f"🔄 Loading dense model: {DENSE_MODEL_NAME}...")
        _dense_model = SentenceTransformer(DENSE_MODEL_NAME)
        print(f"✅ Dense model loaded ({DENSE_DIM}-dim)")
    return _dense_model


def get_sparse_model() -> SparseTextEmbedding:
    """Load or return cached sparse embedding model (BM25)."""
    global _sparse_model
    if _sparse_model is None:
        print(f"🔄 Loading sparse model: {SPARSE_MODEL_NAME}...")
        _sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
        print(f"✅ Sparse model loaded")
    return _sparse_model


def get_reranker() -> CrossEncoder:
    """Load or return cached reranker model (lazy — only when retriever needs it)."""
    global _reranker_model
    if _reranker_model is None:
        print(f"🔄 Loading reranker: {RERANKER_MODEL_NAME}...")
        _reranker_model = CrossEncoder(RERANKER_MODEL_NAME)
        print(f"✅ Reranker loaded")
    return _reranker_model


def embed_dense(texts: list[str], batch_size: int = 64) -> list:
    """Embed texts using dense model. Returns list of numpy arrays (384-dim each)."""
    model = get_dense_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def embed_sparse(texts: list[str], batch_size: int = 64) -> list:
    """Embed texts using sparse BM25 model. Returns list of sparse vectors."""
    model = get_sparse_model()
    return list(model.embed(texts, batch_size=batch_size))
