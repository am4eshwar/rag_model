"""
Retrieval and Reranking Module

Combines ANN search with optional cross-encoder reranking for high-precision retrieval.

TWO-STAGE RETRIEVAL:
1. STAGE 1 (Fast): HNSW retrieves top-K candidates (e.g., K=20)
   - Purpose: Quickly narrow down from 100K documents to 20 candidates
   - Speed: ~50ms on CPU
   - Method: Bi-encoder embeddings + cosine similarity
   
2. STAGE 2 (Accurate): Cross-encoder reranks top-K → top-N (e.g., N=5)
   - Purpose: Refine top candidates with more precise scoring
   - Speed: ~2-3s on CPU for 20 pairs
   - Method: Transformer that directly scores (query, doc) pairs

RATIONALE FOR RERANKING:
- Bi-encoders (HNSW): Fast but independent encoding (query and doc separate)
  * Can't capture query-document interaction
  * Good for recall, moderate precision
  
- Cross-encoders: Slow but joint encoding (query + doc together)
  * Captures attention between query and document
  * Excellent precision but can't pre-compute (must run at query time)
  
- Two-stage = Best of both: Fast recall + accurate precision

WHEN TO USE RERANKING:
- Enable if: Precision@5 < 0.70 or user needs highest quality
- Disable if: Latency must be <1s (e.g., real-time chat)
- Cost: Adds ~2-3s latency for 20 candidates on CPU

PARAMETERS:
- RETRIEVAL_TOP_K=20: Fetch from HNSW (enough diversity for reranking)
- FINAL_TOP_K=5: Send to LLM (fits in context, high precision)
- MIN_SIMILARITY_THRESHOLD=0.3: Filter very low relevance
"""

import logging
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
except ImportError:
    _CrossEncoder = None  # type: ignore

from config import (
    USE_RERANKER,
    RERANKER_MODEL_NAME,
    RERANKER_BATCH_SIZE,
    RERANKER_DEVICE,
    RETRIEVAL_TOP_K,
    FINAL_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Represents a retrieved chunk with relevance score
    
    Attributes:
        chunk_id: Unique chunk identifier
        text: Chunk text content
        score: Relevance score (higher = more relevant)
        rank: Position in results (0-indexed)
        metadata: Additional metadata (doc_id, page, etc.)
        retrieval_stage: "ann" or "reranked"
    """
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: Dict
    retrieval_stage: str = "ann"
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": self.score,
            "rank": self.rank,
            "metadata": self.metadata,
            "retrieval_stage": self.retrieval_stage,
        }


class Reranker:
    """
    Cross-encoder based reranker for refining retrieval results
    
    CROSS-ENCODER vs BI-ENCODER:
    
    Bi-encoder (used in HNSW):
        encode(query) → query_vec
        encode(doc) → doc_vec
        similarity = dot(query_vec, doc_vec)
        + Can pre-compute doc_vec
        + Fast at query time
        - Independent encoding (no interaction)
    
    Cross-encoder (this class):
        score = model([query, doc])  # Joint encoding
        - Can't pre-compute (query-dependent)
        - Slow at query time (must run model)
        + Captures query-doc interaction
        + Much higher accuracy
    
    MODEL CHOICE:
    - ms-marco-MiniLM-L-6-v2: 80MB, 6 layers, trained on MS MARCO
      * Fast: ~10ms per (query, doc) pair on CPU
      * Good quality: ~5-10% better than bi-encoder alone
    - Alternative: ms-marco-MiniLM-L-12-v2 (12 layers, slower, better)
    
    WHEN TO ADJUST:
    - Use L-12-v2 if precision@5 < 0.70 and latency is acceptable
    - Disable reranking if latency > 5s is unacceptable
    """
    
    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        device: str = RERANKER_DEVICE,
        batch_size: int = RERANKER_BATCH_SIZE,
    ):
        """
        Args:
            model_name: HuggingFace cross-encoder model
            device: "cpu" or "cuda"
            batch_size: Number of pairs to score at once
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None
    
    @property
    def model(self) -> "CrossEncoder":
        """Lazy load model on first access"""
        if self._model is None:
            if _CrossEncoder is None:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
            
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = _CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Reranker loaded on device: {self.device}")
        
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Rerank documents by relevance to query
        
        Args:
            query: Query string
            documents: List of document strings
            top_k: Return only top-k (None = all)
            
        Returns:
            Tuple of (indices, scores)
            - indices: Sorted indices into documents (best first)
            - scores: Corresponding relevance scores
            
        Example:
            >>> reranker = Reranker()
            >>> docs = ["AI is...", "ML is...", "Python is..."]
            >>> indices, scores = reranker.rerank("What is AI?", docs)
            >>> best_doc = docs[indices[0]]
            >>> print(f"Best: {best_doc} (score: {scores[0]:.3f})")
        """
        if not documents:
            return [], []
        
        # Ensure query is a string
        query = str(query) if query is not None else ""
        
        # Ensure all documents are strings...
        documents = [str(doc) if doc is not None else "" for doc in documents]
        
        # Create (query, doc) pairs
        pairs = [[query, doc] for doc in documents]
        
        # Score pairs
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=len(pairs) > 10,
        )
        
        # Sort by score (descending)
        sorted_indices = np.argsort(-scores)  # Negative for descending
        sorted_scores = scores[sorted_indices]
        
        # Limit to top-k if specified
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
            sorted_scores = sorted_scores[:top_k]
        
        logger.debug(
            f"Reranked {len(documents)} documents → "
            f"top score: {sorted_scores[0]:.3f}"
        )
        
        return sorted_indices.tolist(), sorted_scores.tolist()


class Retriever:
    """
    High-level retriever that combines HNSW search + optional reranking
    
    RETRIEVAL PIPELINE:
    1. Embed query using embedding model
    2. HNSW ANN search → top-K candidates (e.g., K=20)
    3. (Optional) Cross-encoder rerank → top-N (e.g., N=5)
    4. Filter by minimum similarity threshold
    5. Return RetrievalResult objects
    
    DESIGN CHOICES:
    - Separate chunk text from embeddings (text stored in metadata)
    - Score normalization: Convert HNSW distance to similarity (1 - distance)
    - Threshold filtering: Remove very low relevance chunks
    - Metadata propagation: Preserve doc_id, page_num for citations
    """
    
    def __init__(
        self,
        vector_index,  # VectorIndex instance
        embedding_model,  # EmbeddingModel instance
        use_reranker: bool = USE_RERANKER,
        reranker: Optional[Reranker] = None,
    ):
        """
        Args:
            vector_index: Initialized VectorIndex
            embedding_model: Initialized EmbeddingModel
            use_reranker: Whether to use cross-encoder reranking
            reranker: Reranker instance (created if None and use_reranker=True)
        """
        self.vector_index = vector_index
        self.embedding_model = embedding_model
        self.use_reranker = use_reranker
        
        if use_reranker:
            self.reranker = reranker or Reranker()
        else:
            self.reranker = None
    
    def retrieve(
        self,
        query: str,
        top_k: int = FINAL_TOP_K,
        retrieval_k: Optional[int] = None,
        min_similarity: float = MIN_SIMILARITY_THRESHOLD,
        ef_search: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve most relevant chunks for query
        
        Args:
            query: Query string
            top_k: Number of final results to return
            retrieval_k: Number of candidates to fetch from HNSW (default: RETRIEVAL_TOP_K)
            min_similarity: Minimum similarity threshold (0-1)
            ef_search: Override HNSW ef_search parameter
            
        Returns:
            List of RetrievalResult objects, sorted by relevance (best first)
            
        PARAMETER TUNING:
        - retrieval_k: Increase to 50 if reranker needs more candidates
        - top_k: Increase to 10 if LLM needs more context
        - min_similarity: Decrease to 0.2 for broad queries, increase to 0.5 for precise
        
        Example:
            >>> retriever = Retriever(index, embedder)
            >>> results = retriever.retrieve("What is machine learning?", top_k=5)
            >>> for r in results:
            ...     print(f"[{r.rank}] {r.chunk_id}: {r.score:.3f}")
            ...     print(f"  {r.text[:100]}...")
        """
        if retrieval_k is None:
            retrieval_k = RETRIEVAL_TOP_K if self.use_reranker else top_k
        
        # Step 1: Embed query
        query_embedding = self.embedding_model.embed_query(query)
        
        # Step 2: HNSW search
        chunk_ids, distances, metadata_list = self.vector_index.search(
            query_embedding,
            k=retrieval_k,
            ef_search=ef_search,
        )
        
        if not chunk_ids:
            logger.warning("No results from vector search")
            return []
        
        # Convert distances to similarities (for cosine space)
        if self.vector_index.space == "cosine":
            similarities = [1 - d for d in distances]
        else:
            # For L2, convert to similarity (inverse of distance)
            max_dist = max(distances) if distances else 1
            similarities = [1 - (d / max_dist) for d in distances]
        
        logger.info(
            f"Retrieved {len(chunk_ids)} candidates from HNSW "
            f"(top score: {similarities[0]:.3f})"
        )
        
        # Step 3: Optional reranking
        if self.use_reranker and self.reranker is not None:
            # Get chunk texts from metadata
            chunk_texts = [meta.get("text", "") for meta in metadata_list]
            
            # Rerank
            reranked_indices, reranked_scores = self.reranker.rerank(
                query,
                chunk_texts,
                top_k=top_k,
            )
            
            # Reorder results
            chunk_ids = [chunk_ids[i] for i in reranked_indices]
            similarities = reranked_scores  # Use cross-encoder scores
            metadata_list = [metadata_list[i] for i in reranked_indices]
            
            retrieval_stage = "reranked"
            
            logger.info(
                f"Reranked to top-{len(chunk_ids)} "
                f"(top score: {similarities[0]:.3f})"
            )
        else:
            # No reranking, just take top-k
            chunk_ids = chunk_ids[:top_k]
            similarities = similarities[:top_k]
            metadata_list = metadata_list[:top_k]
            retrieval_stage = "ann"
        
        # Step 4: Filter by threshold
        # ADJUSTMENT: If using reranker, scores are logits (can be negative), 
        # so we shouldn't use the cosine similarity threshold (0.0-1.0).
        effective_threshold = min_similarity
        if self.use_reranker and retrieval_stage == "reranked":
            # For CrossEncoder, 0 is typically a neutral score. 
            # We lower this significantly to ensure we pass *something* to the LLM.
            effective_threshold = -999.0 

        results = []
        for rank, (chunk_id, score, meta) in enumerate(
            zip(chunk_ids, similarities, metadata_list)
        ):
            if score < effective_threshold:
                logger.debug(
                    f"Filtered out {chunk_id} (score {score:.3f} < threshold {effective_threshold})"
                )
                continue
            
            # Get chunk text (should be in metadata)
            text = meta.get("text", "")
            
            result = RetrievalResult(
                chunk_id=chunk_id,
                text=text,
                score=score,
                rank=rank,
                metadata=meta,
                retrieval_stage=retrieval_stage,
            )
            results.append(result)
        
        logger.info(
            f"Returning {len(results)} results (after threshold filter)"
        )
        
        return results


def retrieve_for_query(
    query: str,
    vector_index,
    embedding_model,
    top_k: int = FINAL_TOP_K,
    use_reranker: bool = USE_RERANKER,
) -> List[RetrievalResult]:
    """
    Convenience function for retrieval
    
    Args:
        query: Query string
        vector_index: VectorIndex instance
        embedding_model: EmbeddingModel instance
        top_k: Number of results
        use_reranker: Whether to use reranking
        
    Returns:
        List of RetrievalResult objects
        
    Example:
        >>> from indexing import VectorIndex
        >>> from embedding import EmbeddingModel
        >>> 
        >>> index = VectorIndex.load()
        >>> embedder = EmbeddingModel()
        >>> 
        >>> results = retrieve_for_query(
        ...     "What is the main argument?",
        ...     index,
        ...     embedder,
        ...     top_k=5,
        ... )
    """
    retriever = Retriever(
        vector_index=vector_index,
        embedding_model=embedding_model,
        use_reranker=use_reranker,
    )
    
    return retriever.retrieve(query, top_k=top_k)


# ==============================================================================
# VALIDATION & TESTING
# ==============================================================================

def validate_retrieval():
    """
    Self-test function to verify retrieval and reranking
    
    TEST CASES:
    1. Basic retrieval without reranking
    2. Retrieval with reranking
    3. Threshold filtering
    4. Empty query handling
    5. Reranking quality (should improve order)
    """
    print("=" * 70)
    print("RETRIEVAL & RERANKING MODULE VALIDATION")
    print("=" * 70)
    
    # Setup: Create test index
    print("\n[SETUP] Creating test index...")
    from embedding import EmbeddingModel
    from indexing import VectorIndex
    
    embedder = EmbeddingModel()
    
    # Test documents with varying relevance
    test_docs = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are used in deep learning.",
        "Python is a popular programming language.",
        "Data science involves statistics and machine learning.",
        "The weather today is sunny and warm.",
        "Supervised learning uses labeled training data.",
        "Basketball is a popular sport in America.",
        "Transformers revolutionized natural language processing.",
    ]
    
    # Embed and index
    embeddings = embedder.embed_documents(test_docs)
    chunk_ids = [f"doc_{i}" for i in range(len(test_docs))]
    
    # Store text in metadata for retrieval
    metadata = [{"text": doc, "index": i} for i, doc in enumerate(test_docs)]
    
    index = VectorIndex()
    index.add_items(embeddings, chunk_ids, metadata)
    
    print(f"✓ Created index with {len(test_docs)} documents")
    
    # Test 1: Basic retrieval (no reranking)
    print("\n[TEST 1] Basic retrieval (ANN only)")
    retriever_no_rerank = Retriever(
        vector_index=index,
        embedding_model=embedder,
        use_reranker=False,
    )
    
    query = "What is machine learning?"
    results_ann = retriever_no_rerank.retrieve(query, top_k=3)
    
    print(f"✓ Retrieved {len(results_ann)} results")
    for r in results_ann:
        print(f"  [{r.rank}] {r.chunk_id}: {r.score:.3f} - {r.text[:60]}...")
    
    assert len(results_ann) > 0, "Should retrieve results"
    assert results_ann[0].retrieval_stage == "ann", "Should be ANN stage"
    
    # Test 2: Retrieval with reranking
    print("\n[TEST 2] Retrieval with cross-encoder reranking")
    if USE_RERANKER:
        retriever_rerank = Retriever(
            vector_index=index,
            embedding_model=embedder,
            use_reranker=True,
        )
        
        results_reranked = retriever_rerank.retrieve(query, top_k=3, retrieval_k=5)
        
        print(f"✓ Retrieved {len(results_reranked)} reranked results")
        for r in results_reranked:
            print(f"  [{r.rank}] {r.chunk_id}: {r.score:.3f} - {r.text[:60]}...")
        
        assert results_reranked[0].retrieval_stage == "reranked", "Should be reranked"
        print("✓ Reranking applied successfully")
    else:
        print("⚠ Reranking disabled in config (USE_RERANKER=False)")
    
    # Test 3: Threshold filtering
    print("\n[TEST 3] Threshold filtering")
    results_strict = retriever_no_rerank.retrieve(
        query,
        top_k=10,
        min_similarity=0.5,  # High threshold
    )
    
    results_loose = retriever_no_rerank.retrieve(
        query,
        top_k=10,
        min_similarity=0.1,  # Low threshold
    )
    
    print(f"✓ Strict threshold (0.5): {len(results_strict)} results")
    print(f"✓ Loose threshold (0.1): {len(results_loose)} results")
    assert len(results_strict) <= len(results_loose), \
        "Strict threshold should return fewer results"
    
    # Test 4: Edge case - empty query
    print("\n[TEST 4] Edge case handling")
    try:
        empty_results = retriever_no_rerank.retrieve("", top_k=3)
        print(f"✓ Empty query handled: {len(empty_results)} results")
    except Exception as e:
        print(f"⚠ Empty query raised error: {e}")
    
    # Test 5: Verify relevance
    print("\n[TEST 5] Relevance verification")
    # Top result should be about machine learning
    if results_ann:
        top_text = results_ann[0].text.lower()
        is_relevant = "machine learning" in top_text or "artificial intelligence" in top_text
        
        if is_relevant:
            print("✓ Top result is relevant to query")
        else:
            print(f"⚠ Top result may not be most relevant: {top_text[:80]}")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nRETRIEVAL SETTINGS:")
    print(f"- Use reranker: {USE_RERANKER}")
    print(f"- Retrieval top-K: {RETRIEVAL_TOP_K}")
    print(f"- Final top-K: {FINAL_TOP_K}")
    print(f"- Min similarity: {MIN_SIMILARITY_THRESHOLD}")
    print(f"- Reranker model: {RERANKER_MODEL_NAME}")
    print("\nRECOMMENDATIONS:")
    if not USE_RERANKER:
        print("- Enable USE_RERANKER for better precision (adds ~2s latency)")
    print("- Adjust MIN_SIMILARITY_THRESHOLD based on your domain")
    print("- Increase RETRIEVAL_TOP_K if reranker needs more candidates")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    validate_retrieval()
