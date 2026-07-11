"""
Ingestion pipeline: Clone repos → Tree-sitter chunking → Embed → Store in Qdrant.
Uses all-MiniLM-L6-v2 (384-dim dense) + BM25 (sparse) for hybrid search.
"""

import os
import uuid
import git
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    PointStruct, SparseVector
)
from chunker import chunk_repo
from embeddings import embed_dense, embed_sparse, DENSE_DIM
from config import QDRANT_URL, QDRANT_API_KEY

# ── Qdrant connection ───────────────────────────────────────────
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# ── Your GitHub repos ───────────────────────────────────────────
MY_REPOS = [
    "https://github.com/pallets/click",
]

CLONE_DIR = "./cloned_repos"


def get_repo_name(github_url: str) -> str:
    """Extract repo name from GitHub URL."""
    return github_url.rstrip("/").split("/")[-1]


def clone_repo(github_url: str, repo_name: str) -> str:
    """Clone repo locally, pull if already exists."""
    repo_path = os.path.join(CLONE_DIR, repo_name)

    if os.path.exists(repo_path):
        print(f"📂 {repo_name} already cloned, pulling latest...")
        repo = git.Repo(repo_path)
        repo.remotes.origin.pull()
    else:
        print(f"⬇️  Cloning {repo_name}...")
        git.Repo.clone_from(github_url, repo_path)

    print(f"✅ {repo_name} ready at {repo_path}")
    return repo_path


def create_collection(collection_name: str, recreate: bool = False):
    """Create Qdrant collection with dense (384-dim) + sparse vectors."""
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        if recreate:
            print(f"🗑️  Deleting existing collection '{collection_name}'...")
            client.delete_collection(collection_name)
        else:
            print(f"📦 Collection '{collection_name}' already exists, skipping...")
            return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=DENSE_DIM,        # 384 for all-MiniLM-L6-v2
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )
    print(f"✅ Collection '{collection_name}' created (dense={DENSE_DIM}-dim + sparse)")


def embed_and_index(chunks: list[dict], collection_name: str):
    """Embed chunks and store in Qdrant."""
    if not chunks:
        print("⚠️  No chunks to index")
        return

    print(f"🔄 Embedding {len(chunks)} chunks...")

    texts = [chunk["enriched"] for chunk in chunks]

    # Dense embeddings via sentence-transformers (all-MiniLM-L6-v2)
    dense_embeddings = embed_dense(texts, batch_size=64)

    # Sparse embeddings via fastembed (BM25)
    sparse_embeddings = embed_sparse(texts, batch_size=64)

    print(f"✅ Embeddings done, indexing into Qdrant...")

    points = []
    for i, chunk in enumerate(chunks):
        sparse_vec = sparse_embeddings[i]

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense": dense_embeddings[i].tolist(),
                "sparse": SparseVector(
                    indices=sparse_vec.indices.tolist(),
                    values=sparse_vec.values.tolist()
                )
            },
            payload={
                "code":       chunk["code"],
                "name":       chunk["name"],
                "type":       chunk["type"],
                "file":       chunk["file"],
                "repo":       chunk["repo"],
                "language":   chunk["language"],
                "start_line": chunk["start_line"],
                "end_line":   chunk["end_line"],
            }
        )
        points.append(point)

    # Upload in batches of 100
    upload_batch_size = 100
    for i in range(0, len(points), upload_batch_size):
        batch = points[i:i + upload_batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"📤 Indexed batch {i // upload_batch_size + 1} / {-(-len(points) // upload_batch_size)}")

    print(f"✅ {len(points)} chunks indexed into '{collection_name}'")


def ingest_repo(github_url: str, recreate: bool = False):
    """Full ingestion pipeline for one repo."""
    repo_name       = get_repo_name(github_url)
    collection_name = f"repo_{repo_name}".lower().replace("-", "_")

    print(f"\n{'='*50}")
    print(f"🚀 Ingesting: {repo_name}")
    print(f"{'='*50}")

    repo_path = clone_repo(github_url, repo_name)
    chunks    = chunk_repo(repo_path, repo_name)

    if not chunks:
        print(f"⚠️  No chunks found for {repo_name}")
        return

    create_collection(collection_name, recreate=recreate)
    embed_and_index(chunks, collection_name)

    print(f"\n✅ {repo_name} fully ingested!")
    print(f"   Chunks indexed : {len(chunks)}")
    print(f"   Collection     : {collection_name}")


def ingest_all(recreate: bool = False):
    """Ingest all repos in MY_REPOS list."""
    os.makedirs(CLONE_DIR, exist_ok=True)
    print(f"🚀 Starting ingestion of {len(MY_REPOS)} repos...")

    for url in MY_REPOS:
        ingest_repo(url, recreate=recreate)

    print(f"\n🎉 All repos ingested successfully!")


if __name__ == "__main__":
    # recreate=True to re-ingest with new 384-dim vectors
    ingest_all(recreate=True)