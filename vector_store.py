"""
Vector Storage Module with ChromaDB

Manages vector storage using ChromaDB alongside FAISS indexing for fast retrieval.

RATIONALE:
- ChromaDB: Modern vector database with built-in persistence and metadata management
- Automatic persistence: No need for manual save/load operations
- Metadata integration: Store chunk metadata alongside vectors
- Easy querying: Built-in similarity search with filtering
- FAISS integration: Use FAISS for fast ANN search, ChromaDB for storage

DESIGN:
- ChromaDB stores: vectors + metadata + chunk text
- FAISS provides: fast approximate nearest neighbor search
- Combined approach: Best of both worlds (storage + speed)

VALIDATION:
- Test adding vectors and metadata
- Test querying with filters
- Test persistence across restarts
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_DISTANCE_METRIC,
    EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Vector storage using ChromaDB
    
    FEATURES:
    - Persistent storage with automatic save
    - Metadata storage alongside vectors
    - Built-in similarity search
    - Document text storage
    - Filtering by metadata
    
    DESIGN CHOICES:
    - Collection per document set: Allows easy management
    - Cosine distance: For normalized embeddings
    - Persist directory: Automatic disk persistence
    """
    
    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION_NAME,
        persist_directory: Path = CHROMA_PERSIST_DIR,
        distance_metric: str = CHROMA_DISTANCE_METRIC,
    ):
        """
        Initialize ChromaDB vector store
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
            distance_metric: Distance metric ("cosine", "l2", or "ip")
        """
        if chromadb is None:
            raise ImportError(
                "ChromaDB not installed. "
                "Run: pip install chromadb"
            )
        
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.distance_metric = distance_metric
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": distance_metric},
        )
        
        logger.info(
            f"Initialized ChromaDB collection '{collection_name}' "
            f"at {persist_directory} with {self.collection.count()} vectors"
        )
    
    def add_documents(
        self,
        chunk_ids: List[str],
        embeddings: np.ndarray,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
    ):
        """
        Add documents to the vector store
        
        Args:
            chunk_ids: List of unique chunk IDs
            embeddings: numpy array of shape (n, dim)
            texts: List of chunk texts
            metadata: Optional list of metadata dicts
            
        Example:
            >>> store = ChromaVectorStore()
            >>> store.add_documents(
            ...     chunk_ids=["chunk_1", "chunk_2"],
            ...     embeddings=np.array([[0.1, 0.2, ...], [0.3, 0.4, ...]]),
            ...     texts=["First chunk text", "Second chunk text"],
            ...     metadata=[{"page": 1}, {"page": 2}]
            ... )
        """
        if len(chunk_ids) != len(embeddings) or len(chunk_ids) != len(texts):
            raise ValueError(
                f"Mismatch in lengths: {len(chunk_ids)} ids, "
                f"{len(embeddings)} embeddings, {len(texts)} texts"
            )
        
        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in chunk_ids]
        
        # Add to collection
        self.collection.add(
            ids=chunk_ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
        )
        
        logger.info(
            f"Added {len(chunk_ids)} documents to ChromaDB "
            f"(total: {self.collection.count()})"
        )
    
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10,
        where: Optional[Dict] = None,
    ) -> Tuple[List[str], List[str], List[float], List[Dict]]:
        """
        Query the vector store for similar documents
        
        Args:
            query_embedding: Query vector (1D array)
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"page": 1})
            
        Returns:
            Tuple of (ids, documents, distances, metadatas)
            
        Example:
            >>> ids, docs, dists, metas = store.query(
            ...     query_embedding=np.array([0.1, 0.2, ...]),
            ...     n_results=5,
            ...     where={"doc_id": "essay_001"}
            ... )
        """
        # Ensure query is 1D
        if query_embedding.ndim > 1:
            query_embedding = query_embedding.flatten()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
        
        # Extract results (ChromaDB returns lists of lists)
        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []
        
        return ids, documents, distances, metadatas
    
    def get_by_ids(self, ids: List[str]) -> Tuple[List[str], List[Dict]]:
        """
        Retrieve documents by their IDs
        
        Args:
            ids: List of document IDs
            
        Returns:
            Tuple of (documents, metadatas)
        """
        results = self.collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
        
        return results["documents"], results["metadatas"]
    
    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None):
        """
        Delete documents by IDs or metadata filter
        
        Args:
            ids: List of IDs to delete
            where: Metadata filter for deletion
        """
        if ids:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents from ChromaDB")
        elif where:
            self.collection.delete(where=where)
            logger.info(f"Deleted documents matching filter: {where}")
    
    def reset(self):
        """Reset the collection (delete all documents)"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )
        logger.info(f"Reset collection '{self.collection_name}'")
    
    def count(self) -> int:
        """Get the number of documents in the collection"""
        return self.collection.count()
    
    def get_all_ids(self) -> List[str]:
        """Get all document IDs in the collection"""
        results = self.collection.get(include=[])
        return results["ids"]


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_vector_store(
    collection_name: str = CHROMA_COLLECTION_NAME,
) -> ChromaVectorStore:
    """
    Create a new vector store instance
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        ChromaVectorStore instance
    """
    return ChromaVectorStore(collection_name=collection_name)


def add_chunks_to_store(
    store: ChromaVectorStore,
    chunks: List,  # List of Chunk objects
    embeddings: np.ndarray,
):
    """
    Add chunks and their embeddings to the vector store
    
    Args:
        store: ChromaVectorStore instance
        chunks: List of Chunk objects (from chunking.py)
        embeddings: numpy array of embeddings
        
    Example:
        >>> from chunking import chunk_submission
        >>> from embedding import embed_chunks
        >>> 
        >>> chunks = chunk_submission("doc1", "Long text...")
        >>> chunk_ids, embeddings = embed_chunks(chunks)
        >>> 
        >>> store = create_vector_store()
        >>> add_chunks_to_store(store, chunks, embeddings)
    """
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    texts = [chunk.text for chunk in chunks]
    metadata = [
        {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "page_num": chunk.page_num,
            **(chunk.metadata or {}),
        }
        for chunk in chunks
    ]
    
    store.add_documents(chunk_ids, embeddings, texts, metadata)


# ==============================================================================
# VALIDATION & TESTING
# ==============================================================================

def validate_vector_store():
    """
    Self-test function to verify ChromaDB vector store
    
    TEST CASES:
    1. Create store and add documents
    2. Query for similar documents
    3. Retrieve by IDs
    4. Filter by metadata
    5. Persistence across restarts
    6. Delete operations
    """
    print("=" * 70)
    print("CHROMADB VECTOR STORE VALIDATION")
    print("=" * 70)
    
    # Test 1: Create store and add documents
    print("\n[TEST 1] Create store and add documents")
    
    # Use a test collection
    test_store = ChromaVectorStore(collection_name="test_collection")
    test_store.reset()  # Clear any existing data
    
    # Create sample data
    n_docs = 100
    embeddings = np.random.rand(n_docs, EMBEDDING_DIM).astype(np.float32)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)
    
    chunk_ids = [f"test_chunk_{i:04d}" for i in range(n_docs)]
    texts = [f"This is test document number {i}" for i in range(n_docs)]
    metadata = [{"index": i, "page": (i % 5) + 1} for i in range(n_docs)]
    
    test_store.add_documents(chunk_ids, embeddings, texts, metadata)
    
    print(f"✓ Added {n_docs} documents")
    print(f"✓ Collection count: {test_store.count()}")
    assert test_store.count() == n_docs, "Count mismatch"
    
    # Test 2: Query for similar documents
    print("\n[TEST 2] Query for similar documents")
    query_emb = embeddings[0]  # Use first embedding as query
    
    ids, docs, dists, metas = test_store.query(query_emb, n_results=5)
    
    print(f"✓ Query returned {len(ids)} results")
    print(f"  Top result ID: {ids[0]}")
    print(f"  Top result distance: {dists[0]:.4f}")
    print(f"  Top result text: {docs[0][:50]}...")
    
    # First result should be the query itself (distance ~0)
    assert ids[0] == chunk_ids[0], "Top result should be query itself"
    assert dists[0] < 0.01, f"Distance to self should be ~0, got {dists[0]}"
    print("✓ Query retrieves correct results")
    
    # Test 3: Retrieve by IDs
    print("\n[TEST 3] Retrieve by IDs")
    retrieve_ids = chunk_ids[:3]
    retrieved_docs, retrieved_metas = test_store.get_by_ids(retrieve_ids)
    
    print(f"✓ Retrieved {len(retrieved_docs)} documents by ID")
    assert len(retrieved_docs) == len(retrieve_ids), "Should retrieve all IDs"
    
    # Test 4: Filter by metadata
    print("\n[TEST 4] Filter by metadata")
    page_filter = {"page": 1}
    ids_filtered, docs_filtered, dists_filtered, metas_filtered = test_store.query(
        query_emb, n_results=100, where=page_filter
    )
    
    print(f"✓ Filtered query returned {len(ids_filtered)} results")
    print(f"  Expected ~{n_docs // 5} results for page 1")
    
    # Check all results have page=1
    for meta in metas_filtered:
        assert meta["page"] == 1, f"Filter failed: got page {meta['page']}"
    print("✓ Metadata filter works correctly")
    
    # Test 5: Persistence
    print("\n[TEST 5] Persistence across restarts")
    count_before = test_store.count()
    
    # Create new instance (simulates restart)
    test_store2 = ChromaVectorStore(collection_name="test_collection")
    count_after = test_store2.count()
    
    print(f"✓ Count before: {count_before}")
    print(f"✓ Count after reload: {count_after}")
    assert count_before == count_after, "Persistence failed"
    print("✓ Data persisted correctly")
    
    # Test 6: Delete operations
    print("\n[TEST 6] Delete operations")
    delete_ids = chunk_ids[:5]
    test_store.delete(ids=delete_ids)
    
    new_count = test_store.count()
    print(f"✓ Deleted {len(delete_ids)} documents")
    print(f"✓ New count: {new_count}")
    assert new_count == n_docs - len(delete_ids), "Delete failed"
    
    # Clean up
    test_store.reset()
    print("✓ Test collection cleaned up")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nCHROMADB INFO:")
    print(f"- Persist directory: {CHROMA_PERSIST_DIR}")
    print(f"- Collection name: {CHROMA_COLLECTION_NAME}")
    print(f"- Distance metric: {CHROMA_DISTANCE_METRIC}")
    print(f"- Embedding dimension: {EMBEDDING_DIM}")
    print("\nFEATURES:")
    print("- ✓ Automatic persistence")
    print("- ✓ Metadata storage and filtering")
    print("- ✓ Fast similarity search")
    print("- ✓ Document text storage")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    validate_vector_store()
