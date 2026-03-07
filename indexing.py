"""
Vector Indexing Module

Builds and manages a FAISS HNSW index for fast ANN search.

RATIONALE FOR FAISS HNSW:
- FAISS: Robust open-source nearest neighbor library with pre-built Windows
    wheels and excellent CPU performance.
- HNSW graph: Retains the same scalable performance characteristics as previous
    HNSW implementations while simplifying deployment.

INDEX PARAMETERS:
- M (links per node): Controls graph connectivity vs memory (default 16).
- ef_construction: Candidate pool size during build (default 200).
- ef_search: Candidate pool size during queries (default 50, tunable per query).

VALIDATION:
- Build index on 1000 vectors → should take <5 seconds
- Query time should be <50ms for top-10 search
- Recall@10 should be >0.85 vs exact search
- Index size should be ~(M × n_vectors × 4 bytes × 2) for storage
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import json

try:
    import faiss
except ImportError:
    faiss = None

from config import (
    HNSW_M,
    HNSW_EF_CONSTRUCTION,
    HNSW_EF_SEARCH,
    HNSW_MAX_ELEMENTS,
    EMBEDDING_DIM,
    INDEX_FILE_PATH,
    METADATA_FILE_PATH,
)

logger = logging.getLogger(__name__)


class VectorIndex:
    """
    HNSW-based vector index for fast approximate nearest neighbor search
    
    DESIGN CHOICES:
    - Separate embedding storage from metadata (efficiency)
    - Support for incremental addition (add_items can be called multiple times)
    - Persistence to disk (avoid rebuilding on restart)
    - Space: 'cosine' for normalized embeddings, 'l2' for raw embeddings
    
    PERFORMANCE CHARACTERISTICS:
    - Index build: O(n × M × log(n))  → ~1s for 1000 chunks
    - Query time: O(ef_search × log(n)) → ~50ms for top-10
    - Memory: ~M × n_vectors × 8 bytes  → ~130KB for 1000 chunks with M=16
    - Disk size: Similar to memory + metadata
    """
    
    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        max_elements: int = HNSW_MAX_ELEMENTS,
        m: int = HNSW_M,
        ef_construction: int = HNSW_EF_CONSTRUCTION,
        ef_search: int = HNSW_EF_SEARCH,
        space: str = "cosine",  # or "l2" for euclidean distance
    ):
        """
        Args:
            dim: Embedding dimension
            max_elements: Maximum number of vectors (can resize)
            m: HNSW M parameter (links per node)
            ef_construction: Build quality parameter
            ef_search: Query quality parameter
            space: Distance metric ("cosine" or "l2")
        """
        self.dim = dim
        self.max_elements = max_elements
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.space = space
        
        if faiss is None:
            raise ImportError(
                "FAISS is not installed. Run: pip install faiss-cpu"
            )

        # Initialize FAISS HNSW index
        self._init_faiss()
        
        # Metadata storage (id → metadata dict)
        self.metadata = {}
        
        # Track current size
        self.current_size = 0
        
        logger.info(
            f"Initialized FAISS HNSW index: dim={dim}, M={m}, "
            f"ef_construction={ef_construction}, ef_search={ef_search}"
        )
    
    def _init_faiss(self):
        """Initialize FAISS backend"""
        # FAISS HNSW index
        if self.space == "cosine":
            # For cosine similarity with normalized vectors, use inner product
            self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_INNER_PRODUCT)
        else:
            # L2 distance
            self.index = faiss.IndexHNSWFlat(self.dim, self.m, faiss.METRIC_L2)
        
        # Set construction parameter (FAISS uses efConstruction)
        self.index.hnsw.efConstruction = self.ef_construction
        # Set search parameter
        self.index.hnsw.efSearch = self.ef_search
    
    def add_items(
        self,
        embeddings: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Add vectors to index
        
        Args:
            embeddings: numpy array of shape (n, dim)
            ids: List of n string IDs (must be unique)
            metadata: Optional list of n metadata dicts
            
        IMPORTANT: IDs must be unique across all add_items calls
        
        Example:
            >>> index = VectorIndex()
            >>> embeddings = np.random.rand(100, 384)
            >>> ids = [f"chunk_{i}" for i in range(100)]
            >>> index.add_items(embeddings, ids)
        """
        if len(embeddings) == 0:
            logger.warning("No embeddings to add")
            return
        
        if len(embeddings) != len(ids):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings but {len(ids)} ids"
            )
        
        if metadata and len(metadata) != len(ids):
            raise ValueError(
                f"Mismatch: {len(metadata)} metadata but {len(ids)} ids"
            )
        
        # Normalize for cosine similarity if using FAISS with inner product
        if self.space == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-9)
        
        # Convert string IDs to integer labels
        start_label = self.current_size
        labels = np.arange(start_label, start_label + len(ids))

        # Add to FAISS index (it grows automatically)
        self.index.add(embeddings.astype(np.float32))
        
        # Store metadata mapping
        for i, (chunk_id, label) in enumerate(zip(ids, labels)):
            meta = metadata[i] if metadata else {}
            self.metadata[int(label)] = {
                "chunk_id": chunk_id,
                **meta,
            }
        
        self.current_size += len(embeddings)
        
        logger.info(
            f"Added {len(embeddings)} vectors to index "
            f"(total: {self.current_size})"
        )
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
    ) -> Tuple[List[str], List[float], List[Dict]]:
        """
        Search for k nearest neighbors
        
        Args:
            query_embedding: 1D numpy array of shape (dim,)
            k: Number of results to return
            ef_search: Override default ef_search for this query
            
        Returns:
            Tuple of (ids, distances, metadata)
            - ids: List of chunk IDs (strings)
            - distances: List of distances (lower = more similar)
            - metadata: List of metadata dicts
            
        NOTE ON DISTANCES:
        - For cosine space: distance = 1 - cosine_similarity
          * distance=0 means identical (similarity=1)
          * distance=2 means opposite (similarity=-1)
          * Convert to similarity: similarity = 1 - distance
        - For l2 space: distance = euclidean distance
        
        Example:
            >>> query_emb = embedder.embed_query("What is AI?")
            >>> ids, distances, meta = index.search(query_emb, k=5)
            >>> for id, dist in zip(ids, distances):
            ...     similarity = 1 - dist  # For cosine space
            ...     print(f"{id}: {similarity:.3f}")
        """
        if self.current_size == 0:
            logger.warning("Index is empty, no results")
            return [], [], []
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity if using FAISS
        if self.space == "cosine":
            norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_embedding = query_embedding / (norm + 1e-9)
        
        k = min(k, self.current_size)

        old_ef = None
        if ef_search is not None:
            old_ef = self.index.hnsw.efSearch
            self.index.hnsw.efSearch = ef_search

        distances, labels = self.index.search(query_embedding.astype(np.float32), k)
        labels = labels[0]
        distances = distances[0]

        if ef_search is not None and old_ef is not None:
            self.index.hnsw.efSearch = old_ef

        # Convert FAISS inner product to distance for consistency
        if self.space == "cosine":
            distances = 1 - distances
        
        chunk_ids = []
        metadata_list = []
        
        for label in labels:
            meta = self.metadata.get(int(label), {})
            chunk_id = meta.get("chunk_id", f"unknown_{label}")
            chunk_ids.append(chunk_id)
            metadata_list.append(meta)
        
        return chunk_ids, distances.tolist(), metadata_list
    
    def save(
        self,
        index_path: Path = INDEX_FILE_PATH,
        metadata_path: Path = METADATA_FILE_PATH,
    ):
        """
        Save index and metadata to disk
        
        Args:
            index_path: Path to save index (.bin or .faiss)
            metadata_path: Path to save metadata (.json)
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        
        # Create directories
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump({
                "metadata": self.metadata,
                "config": {
                    "dim": self.dim,
                    "max_elements": self.max_elements,
                    "m": self.m,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                    "space": self.space,
                    "current_size": self.current_size,
                },
            }, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
    
    @classmethod
    def load(
        cls,
        index_path: Path = INDEX_FILE_PATH,
        metadata_path: Path = METADATA_FILE_PATH,
    ) -> "VectorIndex":
        """
        Load index and metadata from disk
        
        Args:
            index_path: Path to index (.bin or .faiss)
            metadata_path: Path to metadata (.json)
            
        Returns:
            Loaded VectorIndex instance
        """
        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        # Load metadata
        with open(metadata_path, "r") as f:
            data = json.load(f)
        
        config = data["config"]
        metadata = data["metadata"]

        obj = cls(
            dim=config["dim"],
            max_elements=config["max_elements"],
            m=config["m"],
            ef_construction=config["ef_construction"],
            ef_search=config["ef_search"],
            space=config["space"],
        )

        obj.index = faiss.read_index(str(index_path))
        
        # Restore metadata
        obj.metadata = {int(k): v for k, v in metadata.items()}
        obj.current_size = config["current_size"]
        
        logger.info(
            f"Loaded FAISS index from {index_path} "
            f"({obj.current_size} vectors)"
        )
        
        return obj


def build_index(
    chunk_ids: List[str],
    embeddings: np.ndarray,
    metadata: Optional[List[Dict]] = None,
    save_path: Optional[Path] = None,
) -> VectorIndex:
    """
    Convenience function to build and optionally save an index
    
    Args:
        chunk_ids: List of chunk IDs
        embeddings: numpy array of embeddings
        metadata: Optional metadata for each chunk
        save_path: If provided, save index to this path
        
    Returns:
        Built VectorIndex
        
    Example:
        >>> from embedding import embed_chunks
        >>> from chunking import chunk_submission
        >>> 
        >>> chunks = chunk_submission("doc1", "Long text...")
        >>> chunk_ids, embeddings = embed_chunks(chunks)
        >>> index = build_index(chunk_ids, embeddings)
        >>> 
        >>> # Search
        >>> query_emb = embedder.embed_query("What is AI?")
        >>> ids, dists, meta = index.search(query_emb, k=5)
    """
    index = VectorIndex()
    index.add_items(embeddings, chunk_ids, metadata)
    
    if save_path:
        index.save(save_path)
    
    return index


# ==============================================================================
# VALIDATION & TESTING
# ==============================================================================

def validate_indexing():
    """
    Self-test function to verify vector index module
    
    TEST CASES:
    1. Index creation and adding vectors
    2. Search accuracy (recall vs exact search)
    3. Search speed (<100ms for 1000 vectors)
    4. Save and load persistence
    5. Incremental addition (add in batches)
    6. Edge cases (empty index, k > num_vectors)
    """
    print("=" * 70)
    print("VECTOR INDEX VALIDATION")
    print("=" * 70)
    
    # Test 1: Index creation
    print("\n[TEST 1] Index creation and adding vectors")
    n_vectors = 1000
    embeddings = np.random.rand(n_vectors, EMBEDDING_DIM).astype(np.float32)
    
    # Normalize embeddings (required for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    ids = [f"chunk_{i:04d}" for i in range(n_vectors)]
    metadata = [{"index": i, "group": i % 10} for i in range(n_vectors)]
    
    index = VectorIndex()
    index.add_items(embeddings, ids, metadata)
    
    print(f"✓ Created index with {index.current_size} vectors")
    assert index.current_size == n_vectors, "Size mismatch"
    
    # Test 2: Search accuracy (recall test)
    print("\n[TEST 2] Search accuracy (recall vs exact)")
    query_idx = 42
    query_emb = embeddings[query_idx]
    
    # HNSW search
    hnsw_ids, hnsw_dists, _ = index.search(query_emb, k=10)
    
    # Exact search (brute force)
    if index.space == "cosine":
        # Cosine distance = 1 - dot product (for normalized vectors)
        scores = embeddings @ query_emb
        exact_indices = np.argsort(-scores)[:10]  # Top 10
    else:
        # L2 distance
        dists = np.linalg.norm(embeddings - query_emb, axis=1)
        exact_indices = np.argsort(dists)[:10]
    
    exact_ids = [ids[i] for i in exact_indices]
    
    # Calculate recall@10
    recall = len(set(hnsw_ids) & set(exact_ids)) / 10
    print(f"✓ Recall@10: {recall:.2%}")
    print(f"  HNSW results: {hnsw_ids[:5]}")
    print(f"  Exact results: {exact_ids[:5]}")
    
    if recall < 0.80:
        print("⚠ Low recall! Consider increasing ef_search or M")
    else:
        print("✓ Good recall (>80%)")
    
    # Test 3: Search speed
    print("\n[TEST 3] Search speed")
    import time
    
    n_queries = 100
    start = time.time()
    for i in range(n_queries):
        query = embeddings[i % n_vectors]
        index.search(query, k=10)
    elapsed = (time.time() - start) / n_queries * 1000  # ms per query
    
    print(f"✓ Average query time: {elapsed:.2f} ms")
    if elapsed > 100:
        print("⚠ Slow queries! Consider decreasing ef_search")
    else:
        print("✓ Fast queries (<100ms)")
    
    # Test 4: Save and load
    print("\n[TEST 4] Persistence (save and load)")
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "test_index.bin"
        metadata_path = Path(tmpdir) / "test_metadata.json"
        
        # Save
        index.save(index_path, metadata_path)
        print(f"✓ Saved index to {index_path}")
        
        # Load
        loaded_index = VectorIndex.load(index_path, metadata_path)
        print(f"✓ Loaded index with {loaded_index.current_size} vectors")
        
        # Verify same results
        loaded_ids, loaded_dists, _ = loaded_index.search(query_emb, k=10)
        assert loaded_ids == hnsw_ids, "Results differ after load"
        print("✓ Search results match after load")
    
    # Test 5: Incremental addition
    print("\n[TEST 5] Incremental addition")
    index2 = VectorIndex()
    
    # Add in 3 batches
    batch_size = n_vectors // 3
    for i in range(3):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size if i < 2 else n_vectors
        
        batch_embs = embeddings[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]
        batch_meta = metadata[start_idx:end_idx]
        
        index2.add_items(batch_embs, batch_ids, batch_meta)
    
    print(f"✓ Added vectors in 3 batches → total {index2.current_size}")
    assert index2.current_size == n_vectors, "Size mismatch"
    
    # Test 6: Edge cases
    print("\n[TEST 6] Edge cases")
    
    # Empty index search
    empty_index = VectorIndex()
    empty_results = empty_index.search(query_emb, k=10)
    assert len(empty_results[0]) == 0, "Empty index should return no results"
    print("✓ Empty index handled correctly")
    
    # k > num_vectors
    large_k_ids, _, _ = index.search(query_emb, k=10000)
    assert len(large_k_ids) == index.current_size, \
        "Should return all vectors when k > size"
    print(f"✓ Large k handled correctly (returned {len(large_k_ids)} results)")
    
    # Test 7: Distance to similarity conversion
    print("\n[TEST 7] Distance to similarity conversion")
    if index.space == "cosine":
        # Query should be most similar to itself
        self_ids, self_dists, _ = index.search(query_emb, k=5)
        self_similarity = 1 - self_dists[0]  # Convert distance to similarity
        
        print(f"✓ Self-similarity: {self_similarity:.6f}")
        assert self_similarity > 0.999, "Self-similarity should be ~1.0"
        print("✓ Cosine distance conversion correct")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nINDEX STATS:")
    print(f"- Vectors: {index.current_size}")
    print(f"- Dimension: {EMBEDDING_DIM}")
    print(f"- M (links per node): {HNSW_M}")
    print(f"- ef_construction: {HNSW_EF_CONSTRUCTION}")
    print(f"- ef_search: {HNSW_EF_SEARCH}")
    print(f"- Recall@10: {recall:.2%}")
    print(f"- Query speed: {elapsed:.2f} ms")
    print("\nTUNING RECOMMENDATIONS:")
    if recall < 0.85:
        print("- Increase ef_search (50 → 100) for better recall")
    if elapsed > 50:
        print("- Decrease ef_search (50 → 30) for faster queries")
    if recall >= 0.90 and elapsed < 50:
        print("- ✓ Optimal configuration!")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    validate_indexing()
