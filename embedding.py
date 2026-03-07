"""
Embedding Module

Generates dense vector embeddings for text chunks using sentence-transformers.

RATIONALE:
- sentence-transformers: Best open-source library for semantic embeddings
- all-MiniLM-L6-v2: 384-dim, 80MB, fast CPU inference, good quality
  * Tradeoff: Smaller dim (384 vs 768) = faster + less memory, slightly lower quality
  * Alternative: all-mpnet-base-v2 (768-dim, better quality, 2x slower)
- Normalization: CRITICAL for cosine similarity (used in HNSW)
- Batch processing: 32 chunks/batch = good CPU utilization
- Same model for docs & queries: Prevents embedding mismatch (major cause of poor retrieval)

VALIDATION:
- Test semantic similarity: "dog" and "puppy" should have high similarity (>0.7)
- Test dimension: All embeddings should be 384-dim
- Test normalization: All embeddings should have L2 norm ≈ 1.0
- Test batch consistency: Same text in batch vs solo should produce same embedding
"""

import logging
from typing import List, Union, Optional, TYPE_CHECKING
import numpy as np
from pathlib import Path
import json

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = None

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    if not TYPE_CHECKING:
        SentenceTransformer = _SentenceTransformer
except ImportError:
    if not TYPE_CHECKING:
        SentenceTransformer = None

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    NORMALIZE_EMBEDDINGS,
    CACHE_DIR,
    CACHE_EMBEDDINGS,
)

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence-transformers embedding model
    
    DESIGN CHOICES:
    - Lazy loading: Model loaded on first use (saves memory if not needed)
    - Batch processing: Efficient for multiple chunks
    - Normalization: Required for cosine similarity in vector search
    - Caching: Save embeddings to disk to avoid recomputation
    
    PARAMETER TUNING:
    - EMBEDDING_BATCH_SIZE=32: Good default for CPU
      * Increase to 64-128 if using GPU
      * Decrease to 8-16 if running out of memory
    - EMBEDDING_DEVICE="cpu": Use "cuda" if GPU available (5-10x faster)
    """
    
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        device: str = EMBEDDING_DEVICE,
        normalize: bool = NORMALIZE_EMBEDDINGS,
    ):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model name
            device: "cpu" or "cuda"
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self._model: Optional["SentenceTransformer"] = None
    
    @property
    def model(self) -> "SentenceTransformer":
        """Lazy load model on first access"""
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded on device: {self.device}")
            
            # Verify dimensions
            test_embedding = self._model.encode(["test"], convert_to_numpy=True)
            actual_dim = test_embedding.shape[1]
            if actual_dim != EMBEDDING_DIM:
                logger.warning(
                    f"Model dimension mismatch: expected {EMBEDDING_DIM}, "
                    f"got {actual_dim}. Update EMBEDDING_DIM in config.py"
                )
        
        return self._model
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single string or list of strings
            batch_size: Number of texts to process at once
            show_progress: Show progress bar for large batches
            
        Returns:
            numpy array of shape (n_texts, embedding_dim)
            
        Example:
            >>> embedder = EmbeddingModel()
            >>> embedding = embedder.embed("This is a test")
            >>> print(embedding.shape)  # (1, 384)
            >>> embeddings = embedder.embed(["text 1", "text 2"])
            >>> print(embeddings.shape)  # (2, 384)
        """
        # Convert single string to list
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([]).reshape(0, EMBEDDING_DIM)
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        
        logger.debug(
            f"Generated {len(embeddings)} embeddings "
            f"(dim={embeddings.shape[1]})"
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string
        
        NOTE: For some models (e.g., e5-base-v2), queries need a "query: " prefix
        For all-MiniLM-L6-v2, no prefix needed (symmetric encoding)
        
        Args:
            query: Query string
            
        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        # Check if model requires query prefix
        if "e5-" in self.model_name.lower():
            query = f"query: {query}"
        
        embedding = self.embed([query])[0]  # Get first (and only) embedding
        return embedding
    
    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = EMBEDDING_BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed multiple documents (for indexing)
        
        NOTE: For some models (e.g., e5-base-v2), documents need a "passage: " prefix
        For all-MiniLM-L6-v2, no prefix needed
        
        Args:
            documents: List of document strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            2D numpy array of shape (n_documents, embedding_dim)
        """
        # Check if model requires passage prefix
        if "e5-" in self.model_name.lower():
            documents = [f"passage: {doc}" for doc in documents]
        
        return self.embed(documents, batch_size, show_progress)
    
    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        NOTE: Only valid if embeddings are normalized!
        If normalized, cosine similarity = dot product
        
        Args:
            embedding1: First embedding (1D array)
            embedding2: Second embedding (1D array)
            
        Returns:
            Similarity score in [-1, 1] (higher = more similar)
        """
        if self.normalize:
            # Fast: dot product (already normalized)
            return float(np.dot(embedding1, embedding2))
        else:
            # Slow: compute cosine similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


class EmbeddingCache:
    """
    Cache embeddings to disk to avoid recomputation
    
    RATIONALE:
    - Embeddings are expensive to compute (100ms/chunk on CPU)
    - Documents don't change often, so cache is valid
    - Storage: 384 floats × 4 bytes = ~1.5KB per chunk (acceptable)
    
    FORMAT: JSONL with one line per chunk:
        {"chunk_id": "doc1_chunk_0000", "embedding": [0.1, 0.2, ...]}
    """
    
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings_cache.jsonl"
        
        # In-memory cache (chunk_id → embedding)
        self._cache = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        if not self.cache_file.exists():
            return
        
        logger.info(f"Loading embedding cache from {self.cache_file}")
        try:
            with open(self.cache_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    chunk_id = data["chunk_id"]
                    embedding = np.array(data["embedding"], dtype=np.float32)
                    self._cache[chunk_id] = embedding
            
            logger.info(f"Loaded {len(self._cache)} cached embeddings")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
    
    def get(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get cached embedding for chunk_id"""
        return self._cache.get(chunk_id)
    
    def put(self, chunk_id: str, embedding: np.ndarray):
        """Store embedding in cache"""
        self._cache[chunk_id] = embedding
    
    def save(self):
        """Persist cache to disk"""
        logger.info(f"Saving {len(self._cache)} embeddings to cache")
        with open(self.cache_file, "w") as f:
            for chunk_id, embedding in self._cache.items():
                data = {
                    "chunk_id": chunk_id,
                    "embedding": embedding.tolist(),
                }
                f.write(json.dumps(data) + "\n")
        logger.info(f"Cache saved to {self.cache_file}")
    
    def has(self, chunk_id: str) -> bool:
        """Check if chunk_id is cached"""
        return chunk_id in self._cache


def embed_chunks(
    chunks: List,  # List of Chunk objects from chunking.py
    use_cache: bool = CACHE_EMBEDDINGS,
) -> tuple[List[str], np.ndarray]:
    """
    Convenience function to embed a list of chunks
    
    Args:
        chunks: List of Chunk objects (from chunking.py)
        use_cache: Whether to use embedding cache
        
    Returns:
        Tuple of (chunk_ids, embeddings)
        - chunk_ids: List of chunk IDs
        - embeddings: numpy array of shape (n_chunks, embedding_dim)
        
    Example:
        >>> from chunking import chunk_submission
        >>> chunks = chunk_submission("doc1", "Long text...")
        >>> chunk_ids, embeddings = embed_chunks(chunks)
        >>> print(f"Embedded {len(chunk_ids)} chunks")
        >>> print(f"Embedding shape: {embeddings.shape}")
    """
    if not chunks:
        return [], np.array([]).reshape(0, EMBEDDING_DIM)
    
    embedder = EmbeddingModel()
    cache = EmbeddingCache() if use_cache else None
    
    # Separate cached and uncached chunks
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]
    
    embeddings = []
    uncached_indices = []
    uncached_texts = []
    
    for i, (chunk_id, text) in enumerate(zip(chunk_ids, texts)):
        if cache and cache.has(chunk_id):
            # Use cached embedding
            embeddings.append(cache.get(chunk_id))
        else:
            # Mark for computation
            embeddings.append(None)  # Placeholder
            uncached_indices.append(i)
            uncached_texts.append(text)
    
    # Compute uncached embeddings
    if uncached_texts:
        logger.info(
            f"Computing {len(uncached_texts)} new embeddings "
            f"({len(chunks) - len(uncached_texts)} cached)"
        )
        new_embeddings = embedder.embed_documents(
            uncached_texts,
            show_progress=len(uncached_texts) > 10,
        )
        
        # Fill in computed embeddings
        for idx, new_emb in zip(uncached_indices, new_embeddings):
            embeddings[idx] = new_emb
            if cache:
                cache.put(chunk_ids[idx], new_emb)
    
    # Save cache
    if cache and uncached_texts:
        cache.save()
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    logger.info(f"Embedded {len(chunks)} chunks → shape {embeddings_array.shape}")
    
    return chunk_ids, embeddings_array


# ==============================================================================
# VALIDATION & TESTING
# ==============================================================================

def validate_embedding():
    """
    Self-test function to verify embedding module
    
    TEST CASES:
    1. Model loading and dimension check
    2. Single text embedding
    3. Batch embedding
    4. Semantic similarity (dog vs puppy should be similar)
    5. Normalization check (L2 norm should be ~1.0)
    6. Embedding consistency (same text should give same embedding)
    """
    print("=" * 70)
    print("EMBEDDING MODULE VALIDATION")
    print("=" * 70)
    
    # Test 1: Model loading
    print(f"\n[TEST 1] Loading model: {EMBEDDING_MODEL_NAME}")
    embedder = EmbeddingModel()
    test_embedding = embedder.embed("test")
    print("✓ Model loaded successfully")
    print(f"✓ Embedding dimension: {test_embedding.shape[1]}")
    assert test_embedding.shape[1] == EMBEDDING_DIM, \
        f"Dimension mismatch: {test_embedding.shape[1]} != {EMBEDDING_DIM}"
    
    # Test 2: Single text embedding
    print("\n[TEST 2] Single text embedding")
    text = "This is a test sentence about machine learning."
    embedding = embedder.embed(text)
    print(f"✓ Embedded single text → shape {embedding.shape}")
    assert embedding.shape == (1, EMBEDDING_DIM), "Wrong shape for single text"
    
    # Test 3: Batch embedding
    print("\n[TEST 3] Batch embedding")
    texts = [
        "Machine learning is a subset of AI.",
        "Natural language processing uses neural networks.",
        "Student submissions need to be reviewed.",
    ]
    embeddings = embedder.embed(texts)
    print(f"✓ Embedded {len(texts)} texts → shape {embeddings.shape}")
    assert embeddings.shape == (len(texts), EMBEDDING_DIM), "Wrong batch shape"
    
    # Test 4: Semantic similarity
    print("\n[TEST 4] Semantic similarity test")
    emb_dog = embedder.embed("dog")[0]
    emb_puppy = embedder.embed("puppy")[0]
    emb_car = embedder.embed("car")[0]
    
    sim_dog_puppy = embedder.get_similarity(emb_dog, emb_puppy)
    sim_dog_car = embedder.get_similarity(emb_dog, emb_car)
    
    print(f"✓ Similarity(dog, puppy): {sim_dog_puppy:.3f}")
    print(f"✓ Similarity(dog, car): {sim_dog_car:.3f}")
    
    assert sim_dog_puppy > sim_dog_car, \
        "Dog-puppy should be more similar than dog-car"
    print("✓ Semantic similarity test passed")
    
    # Test 5: Normalization check
    print("\n[TEST 5] Normalization check")
    if NORMALIZE_EMBEDDINGS:
        norms = [np.linalg.norm(emb) for emb in embeddings]
        print(f"✓ L2 norms: {[f'{n:.3f}' for n in norms]}")
        for norm in norms:
            assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: {norm}"
        print("✓ All embeddings are normalized (L2 norm ≈ 1.0)")
    else:
        print("⚠ Normalization disabled (NORMALIZE_EMBEDDINGS=False)")
    
    # Test 6: Consistency check
    print("\n[TEST 6] Embedding consistency")
    text = "Consistency test sentence"
    emb1 = embedder.embed([text])[0]
    emb2 = embedder.embed([text])[0]
    
    similarity = embedder.get_similarity(emb1, emb2)
    print(f"✓ Similarity of same text with itself: {similarity:.6f}")
    assert similarity > 0.999, \
        f"Same text should have similarity ~1.0, got {similarity}"
    print("✓ Embedding is deterministic")
    
    # Test 7: Query vs document embedding
    print("\n[TEST 7] Query and document embedding")
    query = "What is machine learning?"
    doc = "Machine learning is a field of artificial intelligence."
    
    query_emb = embedder.embed_query(query)
    doc_emb = embedder.embed_documents([doc])[0]
    
    print(f"✓ Query embedding shape: {query_emb.shape}")
    print(f"✓ Document embedding shape: {doc_emb.shape}")
    
    similarity = embedder.get_similarity(query_emb, doc_emb)
    print(f"✓ Query-doc similarity: {similarity:.3f}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nMODEL INFO:")
    print(f"- Model: {EMBEDDING_MODEL_NAME}")
    print(f"- Dimension: {EMBEDDING_DIM}")
    print(f"- Device: {EMBEDDING_DEVICE}")
    print(f"- Normalized: {NORMALIZE_EMBEDDINGS}")
    print("\nRECOMMENDATIONS:")
    print("- For GPU: Set EMBEDDING_DEVICE='cuda' in config.py")
    print("- For better quality: Switch to all-mpnet-base-v2 (768-dim)")
    print("- For faster: Keep all-MiniLM-L6-v2 (384-dim, current)")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    validate_embedding()
