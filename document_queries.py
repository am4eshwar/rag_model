"""
Document-Specific Query System

Enables querying individual documents or filtering queries to specific students/submissions.
Essential for Google Classroom integration where teachers need to:
- Query a specific student's submission
- Compare specific documents
- Generate per-document reports

FEATURES:
- Document-filtered retrieval
- Multi-document queries
- Student-specific context isolation
- Metadata-based filtering

USAGE:
    # Query single document
    results = query_document(doc_id="student_123_essay", query="What is the main argument?")
    
    # Query multiple documents
    results = query_documents(doc_ids=["doc1", "doc2"], query="Compare thesis statements")
    
    # Query all documents by student
    results = query_by_student(student_id="student_123", query="Analyze writing style")
"""

import logging
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import numpy as np

from embedding import EmbeddingModel
from vector_store import ChromaVectorStore
from retrieval import RetrievalResult, Reranker
from config import USE_RERANKER, RETRIEVAL_TOP_K, FINAL_TOP_K

logger = logging.getLogger(__name__)


@dataclass
class DocumentQueryResult:
    """
    Results from a document-specific query
    
    Attributes:
        doc_id: Document ID that was queried
        query: Original query text
        results: List of retrieval results
        metadata: Document metadata
        total_chunks: Total chunks in document
        retrieved_chunks: Number of chunks retrieved
    """
    doc_id: str
    query: str
    results: List[RetrievalResult]
    metadata: Dict
    total_chunks: int
    retrieved_chunks: int
    
    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
            "total_chunks": self.total_chunks,
            "retrieved_chunks": self.retrieved_chunks,
        }


class DocumentQuerySystem:
    """
    Query system with document-level filtering
    
    Allows teachers to:
    1. Query specific student submissions
    2. Filter results to specific documents
    3. Compare across multiple documents
    4. Isolate context per student
    """
    
    def __init__(self):
        """Initialize query system components"""
        self.embedder = EmbeddingModel()
        self.vector_store = ChromaVectorStore()
        self.reranker = Reranker() if USE_RERANKER else None
        logger.info("Initialized DocumentQuerySystem")
    
    def query_single_document(
        self,
        doc_id: str,
        query: str,
        top_k: int = FINAL_TOP_K,
    ) -> DocumentQueryResult:
        """
        Query a specific document
        
        Args:
            doc_id: Document ID to query
            query: Query text
            top_k: Number of results to return
            
        Returns:
            DocumentQueryResult with filtered results
        """
        logger.info(f"Querying document: {doc_id}")
        logger.info(f"Query: {query}")
        
        # Check if document exists
        doc_chunks = self._get_document_chunks(doc_id)
        if not doc_chunks:
            logger.warning(f"Document {doc_id} not found in vector store")
            return DocumentQueryResult(
                doc_id=doc_id,
                query=query,
                results=[],
                metadata={},
                total_chunks=0,
                retrieved_chunks=0,
            )
        
        logger.info(f"Document has {len(doc_chunks)} chunks")
        
        # Embed query
        query_embedding = self.embedder.embed_query(query)
        
        # Retrieve with document filter
        chunk_ids, docs, distances, metadatas = self.vector_store.query(
            query_embedding,
            n_results=min(RETRIEVAL_TOP_K, len(doc_chunks)),
            where={"doc_id": doc_id}  # Filter by document
        )
        
        logger.info(f"Retrieved {len(chunk_ids)} chunks from document")
        
        # Create retrieval results
        results = []
        for i, (cid, doc, dist, meta) in enumerate(zip(chunk_ids, docs, distances, metadatas)):
            score = 1 - dist if dist <= 1 else 0.0
            results.append(RetrievalResult(
                chunk_id=cid,
                text=doc,
                score=score,
                rank=i,
                metadata=meta,
                retrieval_stage="ann"
            ))
        
        # Rerank if enabled
        if self.reranker and len(results) > 0:
            logger.info("Reranking results...")
            results = self.reranker.rerank(query, results, top_k=top_k)
        else:
            results = results[:top_k]
        
        # Get document metadata from first chunk
        doc_metadata = metadatas[0] if metadatas else {}
        
        return DocumentQueryResult(
            doc_id=doc_id,
            query=query,
            results=results,
            metadata=doc_metadata,
            total_chunks=len(doc_chunks),
            retrieved_chunks=len(results),
        )
    
    def query_multiple_documents(
        self,
        doc_ids: List[str],
        query: str,
        top_k_per_doc: int = 5,
    ) -> List[DocumentQueryResult]:
        """
        Query multiple documents and return results per document
        
        Args:
            doc_ids: List of document IDs
            query: Query text
            top_k_per_doc: Results per document
            
        Returns:
            List of DocumentQueryResult, one per document
        """
        logger.info(f"Querying {len(doc_ids)} documents")
        
        results = []
        for doc_id in doc_ids:
            doc_result = self.query_single_document(doc_id, query, top_k_per_doc)
            results.append(doc_result)
        
        logger.info(f"Completed queries for {len(results)} documents")
        return results
    
    def query_by_student(
        self,
        student_id: str,
        query: str,
        top_k: int = FINAL_TOP_K,
    ) -> List[DocumentQueryResult]:
        """
        Query all documents submitted by a specific student
        
        Args:
            student_id: Student identifier
            query: Query text
            top_k: Results per document
            
        Returns:
            List of results for all student's documents
        """
        logger.info(f"Querying all documents for student: {student_id}")
        
        # Get all documents for this student
        doc_ids = self._get_student_documents(student_id)
        
        if not doc_ids:
            logger.warning(f"No documents found for student: {student_id}")
            return []
        
        logger.info(f"Found {len(doc_ids)} documents for student")
        return self.query_multiple_documents(doc_ids, query, top_k)
    
    def compare_documents(
        self,
        doc_ids: List[str],
        query: str,
        top_k_per_doc: int = 3,
    ) -> Dict:
        """
        Compare multiple documents based on a query
        
        Useful for:
        - Comparing different students' answers to same question
        - Checking consistency across submissions
        - Finding similarities/differences
        
        Args:
            doc_ids: Documents to compare
            query: Comparison query
            top_k_per_doc: Results per document
            
        Returns:
            Comparison results with side-by-side analysis
        """
        logger.info(f"Comparing {len(doc_ids)} documents")
        
        # Query each document
        doc_results = self.query_multiple_documents(doc_ids, query, top_k_per_doc)
        
        # Create comparison structure
        comparison = {
            "query": query,
            "documents": [r.to_dict() for r in doc_results],
            "summary": {
                "total_documents": len(doc_ids),
                "documents_with_results": sum(1 for r in doc_results if r.results),
                "total_chunks_retrieved": sum(r.retrieved_chunks for r in doc_results),
            }
        }
        
        return comparison
    
    def _get_document_chunks(self, doc_id: str) -> List[str]:
        """Get all chunk IDs for a document"""
        try:
            # Query with document filter to get all chunks
            # Use a dummy embedding just to get document chunks
            dummy_embedding = np.zeros(self.embedder.model.get_sentence_embedding_dimension())
            chunk_ids, _, _, _ = self.vector_store.query(
                dummy_embedding,
                n_results=1000,  # Large number to get all chunks
                where={"doc_id": doc_id}
            )
            return chunk_ids
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            return []
    
    def _get_student_documents(self, student_id: str) -> List[str]:
        """Get all document IDs for a student"""
        try:
            # Query with student filter
            dummy_embedding = np.zeros(self.embedder.model.get_sentence_embedding_dimension())
            _, _, _, metadatas = self.vector_store.query(
                dummy_embedding,
                n_results=1000,
                where={"student_id": student_id}
            )
            
            # Extract unique document IDs
            doc_ids = set()
            for meta in metadatas:
                if "doc_id" in meta:
                    doc_ids.add(meta["doc_id"])
            
            return list(doc_ids)
        except Exception as e:
            logger.error(f"Error getting student documents: {e}")
            return []
    
    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """Get metadata for a document"""
        chunks = self._get_document_chunks(doc_id)
        if not chunks:
            return None
        
        # Get metadata from first chunk
        try:
            dummy_embedding = np.zeros(self.embedder.model.get_sentence_embedding_dimension())
            _, _, _, metadatas = self.vector_store.query(
                dummy_embedding,
                n_results=1,
                where={"doc_id": doc_id}
            )
            
            if metadatas:
                return {
                    "doc_id": doc_id,
                    "total_chunks": len(chunks),
                    "metadata": metadatas[0],
                }
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
        
        return None
    
    def list_all_documents(self) -> List[Dict]:
        """List all documents in the vector store"""
        try:
            # Get all chunks
            count = self.vector_store.count()
            if count == 0:
                return []
            
            dummy_embedding = np.zeros(self.embedder.model.get_sentence_embedding_dimension())
            _, _, _, metadatas = self.vector_store.query(
                dummy_embedding,
                n_results=min(count, 10000)
            )
            
            # Extract unique documents
            docs_dict = {}
            for meta in metadatas:
                doc_id = meta.get("doc_id", "unknown")
                if doc_id not in docs_dict:
                    docs_dict[doc_id] = {
                        "doc_id": doc_id,
                        "filename": meta.get("filename", "unknown"),
                        "student_id": meta.get("student_id"),
                        "chunk_count": 0,
                    }
                docs_dict[doc_id]["chunk_count"] += 1
            
            return list(docs_dict.values())
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

_query_system = None

def get_query_system() -> DocumentQuerySystem:
    """Get or create global query system instance"""
    global _query_system
    if _query_system is None:
        _query_system = DocumentQuerySystem()
    return _query_system


def query_document(doc_id: str, query: str, top_k: int = 5) -> DocumentQueryResult:
    """Convenience function to query a single document"""
    system = get_query_system()
    return system.query_single_document(doc_id, query, top_k)


def query_documents(doc_ids: List[str], query: str, top_k_per_doc: int = 5) -> List[DocumentQueryResult]:
    """Convenience function to query multiple documents"""
    system = get_query_system()
    return system.query_multiple_documents(doc_ids, query, top_k_per_doc)


def compare_documents(doc_ids: List[str], query: str) -> Dict:
    """Convenience function to compare documents"""
    system = get_query_system()
    return system.compare_documents(doc_ids, query)
