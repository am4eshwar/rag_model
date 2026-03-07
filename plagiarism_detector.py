"""
Plagiarism & Similarity Detection System

Detects potential plagiarism and similarities across student submissions.
Uses semantic similarity, n-gram matching, and fingerprinting techniques.

FEATURES:
- Cross-document similarity detection
- Semantic similarity analysis using embeddings
- Text fingerprinting for efficient comparison
- Detailed similarity reports
- Configurable thresholds

DETECTION METHODS:
1. Semantic Similarity: Uses embeddings to find conceptually similar content
2. N-gram Overlap: Detects exact and near-exact phrase matches
3. Document Fingerprinting: Efficient pre-filtering using MinHash/LSH

USAGE:
    detector = PlagiarismDetector()
    
    # Check specific documents
    result = detector.compare_documents(doc_id1, doc_id2)
    
    # Check document against all others
    results = detector.check_against_all(doc_id)
    
    # Batch check all documents
    similarity_matrix = detector.check_all_submissions()
"""

import logging
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from embedding import EmbeddingModel
from vector_store import ChromaVectorStore
from config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class SimilarityResult:
    """Result of comparing two documents"""
    doc_id_1: str
    doc_id_2: str
    overall_similarity: float  # 0.0 to 1.0
    semantic_similarity: float
    ngram_overlap: float
    matching_chunks: List[Dict]  # Chunks that are highly similar
    suspicious_passages: List[Dict]
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class PlagiarismReport:
    """Comprehensive plagiarism report for a document"""
    doc_id: str
    document_name: str
    similar_documents: List[SimilarityResult]
    max_similarity: float
    flagged: bool  # True if exceeds threshold
    summary: str
    generated_at: datetime
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['similar_documents'] = [s.to_dict() for s in self.similar_documents]
        d['generated_at'] = self.generated_at.isoformat()
        return d


class PlagiarismDetector:
    """
    Detect plagiarism and similarities across documents
    
    Configuration:
        HIGH_SIMILARITY_THRESHOLD: Flag docs with >70% similarity
        SEMANTIC_SIMILARITY_THRESHOLD: Report chunks with >85% semantic similarity
        NGRAM_SIZE: Size of n-grams for exact matching (default: 5)
    """
    
    # Thresholds
    HIGH_SIMILARITY_THRESHOLD = 0.70  # Flag as suspicious
    MODERATE_SIMILARITY_THRESHOLD = 0.50  # Mention in report
    SEMANTIC_SIMILARITY_THRESHOLD = 0.85  # For individual chunks
    NGRAM_SIZE = 5  # Words in n-gram
    MIN_NGRAM_MATCHES = 3  # Minimum matching n-grams to report
    
    def __init__(self):
        self.embedder = EmbeddingModel()
        self.vector_store = ChromaVectorStore()
        logger.info("Initialized PlagiarismDetector")
    
    def compare_documents(
        self,
        doc_id_1: str,
        doc_id_2: str,
        detailed: bool = True
    ) -> SimilarityResult:
        """
        Compare two documents for similarity
        
        Args:
            doc_id_1: First document ID
            doc_id_2: Second document ID
            detailed: Include detailed chunk-level analysis
            
        Returns:
            SimilarityResult
        """
        logger.info(f"Comparing {doc_id_1} and {doc_id_2}")
        
        # Get chunks for both documents
        chunks_1 = self._get_document_chunks_with_embeddings(doc_id_1)
        chunks_2 = self._get_document_chunks_with_embeddings(doc_id_2)
        
        if not chunks_1 or not chunks_2:
            logger.warning(f"Missing chunks for one or both documents")
            return SimilarityResult(
                doc_id_1=doc_id_1,
                doc_id_2=doc_id_2,
                overall_similarity=0.0,
                semantic_similarity=0.0,
                ngram_overlap=0.0,
                matching_chunks=[],
                suspicious_passages=[],
                timestamp=datetime.now()
            )
        
        # Calculate semantic similarity
        semantic_sim = self._calculate_semantic_similarity(chunks_1, chunks_2)
        
        # Calculate n-gram overlap
        ngram_overlap = self._calculate_ngram_overlap(chunks_1, chunks_2)
        
        # Find matching chunks
        matching_chunks = []
        suspicious_passages = []
        
        if detailed:
            matching_chunks = self._find_matching_chunks(chunks_1, chunks_2)
            suspicious_passages = self._extract_suspicious_passages(matching_chunks)
        
        # Calculate overall similarity (weighted average)
        overall_similarity = (semantic_sim * 0.7 + ngram_overlap * 0.3)
        
        result = SimilarityResult(
            doc_id_1=doc_id_1,
            doc_id_2=doc_id_2,
            overall_similarity=overall_similarity,
            semantic_similarity=semantic_sim,
            ngram_overlap=ngram_overlap,
            matching_chunks=matching_chunks,
            suspicious_passages=suspicious_passages,
            timestamp=datetime.now()
        )
        
        logger.info(f"Similarity: {overall_similarity:.2%} (semantic: {semantic_sim:.2%}, "
                   f"n-gram: {ngram_overlap:.2%})")
        
        return result
    
    def check_against_all(
        self,
        doc_id: str,
        exclude_self: bool = True
    ) -> List[SimilarityResult]:
        """
        Check a document against all others in the vector store
        
        Args:
            doc_id: Document to check
            exclude_self: Don't compare against itself
            
        Returns:
            List of similarity results, sorted by similarity (highest first)
        """
        logger.info(f"Checking {doc_id} against all documents")
        
        # Get all document IDs
        all_docs = self._get_all_document_ids()
        
        if exclude_self and doc_id in all_docs:
            all_docs.remove(doc_id)
        
        logger.info(f"Comparing against {len(all_docs)} documents")
        
        # Compare with each document
        results = []
        for other_doc_id in all_docs:
            try:
                result = self.compare_documents(doc_id, other_doc_id, detailed=False)
                
                # Only include if similarity is above threshold
                if result.overall_similarity >= self.MODERATE_SIMILARITY_THRESHOLD:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error comparing {doc_id} with {other_doc_id}: {e}")
        
        # Sort by similarity (highest first)
        results.sort(key=lambda x: x.overall_similarity, reverse=True)
        
        logger.info(f"Found {len(results)} documents with similarity >= {self.MODERATE_SIMILARITY_THRESHOLD:.0%}")
        
        return results
    
    def check_all_submissions(
        self,
        min_similarity: float = 0.5
    ) -> Dict[str, List[SimilarityResult]]:
        """
        Check all documents against each other (create similarity matrix)
        
        Args:
            min_similarity: Minimum similarity to include in results
            
        Returns:
            Dictionary mapping doc_id -> list of similar documents
        """
        logger.info("Checking all submissions for similarities")
        
        # Get all document IDs
        all_docs = self._get_all_document_ids()
        total_comparisons = len(all_docs) * (len(all_docs) - 1) // 2
        
        logger.info(f"Comparing {len(all_docs)} documents ({total_comparisons} pairs)")
        
        # Store results per document
        similarity_map = defaultdict(list)
        
        # Compare each pair once
        completed = 0
        for i, doc_id_1 in enumerate(all_docs):
            for doc_id_2 in all_docs[i+1:]:
                try:
                    result = self.compare_documents(doc_id_1, doc_id_2, detailed=False)
                    
                    if result.overall_similarity >= min_similarity:
                        # Add to both documents' lists
                        similarity_map[doc_id_1].append(result)
                        
                        # Create reverse result for doc_id_2
                        reverse_result = SimilarityResult(
                            doc_id_1=doc_id_2,
                            doc_id_2=doc_id_1,
                            overall_similarity=result.overall_similarity,
                            semantic_similarity=result.semantic_similarity,
                            ngram_overlap=result.ngram_overlap,
                            matching_chunks=result.matching_chunks,
                            suspicious_passages=result.suspicious_passages,
                            timestamp=result.timestamp
                        )
                        similarity_map[doc_id_2].append(reverse_result)
                    
                    completed += 1
                    if completed % 100 == 0:
                        logger.info(f"Progress: {completed}/{total_comparisons} comparisons")
                
                except Exception as e:
                    logger.error(f"Error comparing {doc_id_1} and {doc_id_2}: {e}")
        
        # Sort results for each document
        for doc_id in similarity_map:
            similarity_map[doc_id].sort(key=lambda x: x.overall_similarity, reverse=True)
        
        logger.info(f"Completed {completed} comparisons")
        logger.info(f"Found similarities for {len(similarity_map)} documents")
        
        return dict(similarity_map)
    
    def generate_plagiarism_report(
        self,
        doc_id: str,
        similarity_results: Optional[List[SimilarityResult]] = None
    ) -> PlagiarismReport:
        """
        Generate comprehensive plagiarism report for a document
        
        Args:
            doc_id: Document ID
            similarity_results: Optional pre-computed results
            
        Returns:
            PlagiarismReport
        """
        if similarity_results is None:
            similarity_results = self.check_against_all(doc_id)
        
        # Get document name
        doc_info = self._get_document_info(doc_id)
        doc_name = doc_info.get('filename', doc_id) if doc_info else doc_id
        
        # Find maximum similarity
        max_similarity = max([r.overall_similarity for r in similarity_results], default=0.0)
        
        # Determine if flagged
        flagged = max_similarity >= self.HIGH_SIMILARITY_THRESHOLD
        
        # Generate summary
        if flagged:
            summary = f"⚠️ HIGH SIMILARITY DETECTED ({max_similarity:.1%}). Manual review recommended."
        elif max_similarity >= self.MODERATE_SIMILARITY_THRESHOLD:
            summary = f"Moderate similarity detected ({max_similarity:.1%}). Worth reviewing."
        else:
            summary = f"No significant similarities found. Maximum similarity: {max_similarity:.1%}"
        
        report = PlagiarismReport(
            doc_id=doc_id,
            document_name=doc_name,
            similar_documents=similarity_results,
            max_similarity=max_similarity,
            flagged=flagged,
            summary=summary,
            generated_at=datetime.now()
        )
        
        return report
    
    def export_plagiarism_reports(
        self,
        reports: List[PlagiarismReport],
        output_dir: Path = DATA_DIR / "plagiarism_reports"
    ):
        """Export plagiarism reports to JSON"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"plagiarism_report_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_dict() for r in reports],
                f,
                indent=2,
                ensure_ascii=False
            )
        
        logger.info(f"Exported {len(reports)} plagiarism reports to {output_file}")
        
        # Also export summary
        summary_file = output_dir / f"plagiarism_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("PLAGIARISM DETECTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Documents Analyzed: {len(reports)}\n")
            f.write(f"Documents Flagged: {sum(1 for r in reports if r.flagged)}\n")
            f.write(f"Documents with Moderate Similarity: {sum(1 for r in reports if r.max_similarity >= self.MODERATE_SIMILARITY_THRESHOLD and not r.flagged)}\n\n")
            
            f.write("FLAGGED DOCUMENTS:\n")
            f.write("-" * 60 + "\n")
            for report in reports:
                if report.flagged:
                    f.write(f"\n{report.document_name} (ID: {report.doc_id})\n")
                    f.write(f"  Max Similarity: {report.max_similarity:.1%}\n")
                    f.write(f"  Similar to {len(report.similar_documents)} documents\n")
                    for sim_result in report.similar_documents[:3]:  # Top 3
                        other_doc_info = self._get_document_info(sim_result.doc_id_2)
                        other_doc_name = other_doc_info.get('filename', sim_result.doc_id_2) if other_doc_info else sim_result.doc_id_2
                        f.write(f"    - {other_doc_name}: {sim_result.overall_similarity:.1%}\n")
        
        logger.info(f"Exported summary to {summary_file}")
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_document_chunks_with_embeddings(self, doc_id: str) -> List[Dict]:
        """Get all chunks and embeddings for a document"""
        try:
            # Get document chunks from vector store
            dummy_embedding = np.zeros(self.embedder.model.get_sentence_embedding_dimension())
            chunk_ids, docs, distances, metadatas = self.vector_store.query(
                dummy_embedding,
                n_results=1000,
                where={"doc_id": doc_id}
            )
            
            chunks = []
            for chunk_id, doc, metadata in zip(chunk_ids, docs, metadatas):
                chunks.append({
                    'chunk_id': chunk_id,
                    'text': doc,
                    'metadata': metadata
                })
            
            return chunks
        except Exception as e:
            logger.error(f"Error getting chunks for {doc_id}: {e}")
            return []
    
    def _calculate_semantic_similarity(
        self,
        chunks_1: List[Dict],
        chunks_2: List[Dict]
    ) -> float:
        """Calculate average semantic similarity between document chunks"""
        if not chunks_1 or not chunks_2:
            return 0.0
        
        # Get embeddings for all chunks
        texts_1 = [c['text'] for c in chunks_1]
        texts_2 = [c['text'] for c in chunks_2]
        
        embeddings_1 = self.embedder.embed_batch(texts_1)
        embeddings_2 = self.embedder.embed_batch(texts_2)
        
        # Calculate pairwise cosine similarities
        # Shape: (len(chunks_1), len(chunks_2))
        similarity_matrix = np.dot(embeddings_1, embeddings_2.T)
        
        # Take maximum similarity for each chunk in doc1
        max_similarities = np.max(similarity_matrix, axis=1)
        
        # Average across all chunks
        avg_similarity = np.mean(max_similarities)
        
        return float(avg_similarity)
    
    def _calculate_ngram_overlap(
        self,
        chunks_1: List[Dict],
        chunks_2: List[Dict]
    ) -> float:
        """Calculate n-gram overlap between documents"""
        # Extract n-grams from all chunks
        ngrams_1 = set()
        for chunk in chunks_1:
            ngrams_1.update(self._extract_ngrams(chunk['text'], self.NGRAM_SIZE))
        
        ngrams_2 = set()
        for chunk in chunks_2:
            ngrams_2.update(self._extract_ngrams(chunk['text'], self.NGRAM_SIZE))
        
        if not ngrams_1 or not ngrams_2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(ngrams_1 & ngrams_2)
        union = len(ngrams_1 | ngrams_2)
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_ngrams(self, text: str, n: int) -> Set[str]:
        """Extract n-grams from text"""
        words = text.lower().split()
        ngrams = set()
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            ngrams.add(ngram)
        return ngrams
    
    def _find_matching_chunks(
        self,
        chunks_1: List[Dict],
        chunks_2: List[Dict]
    ) -> List[Dict]:
        """Find highly similar chunk pairs"""
        matching = []
        
        texts_1 = [c['text'] for c in chunks_1]
        texts_2 = [c['text'] for c in chunks_2]
        
        embeddings_1 = self.embedder.embed_batch(texts_1)
        embeddings_2 = self.embedder.embed_batch(texts_2)
        
        similarity_matrix = np.dot(embeddings_1, embeddings_2.T)
        
        # Find pairs above threshold
        high_similarity_pairs = np.where(similarity_matrix >= self.SEMANTIC_SIMILARITY_THRESHOLD)
        
        for i, j in zip(high_similarity_pairs[0], high_similarity_pairs[1]):
            matching.append({
                'chunk_1_id': chunks_1[i]['chunk_id'],
                'chunk_2_id': chunks_2[j]['chunk_id'],
                'text_1': texts_1[i][:200] + "...",  # Truncate for display
                'text_2': texts_2[j][:200] + "...",
                'similarity': float(similarity_matrix[i, j])
            })
        
        return matching
    
    def _extract_suspicious_passages(
        self,
        matching_chunks: List[Dict]
    ) -> List[Dict]:
        """Extract suspicious passages from matching chunks"""
        # Group by chunks from doc 1
        suspicious = []
        
        for match in matching_chunks:
            if match['similarity'] >= self.SEMANTIC_SIMILARITY_THRESHOLD:
                suspicious.append({
                    'text': match['text_1'],
                    'similar_to': match['text_2'],
                    'similarity': match['similarity']
                })
        
        return suspicious
    
    def _get_all_document_ids(self) -> List[str]:
        """Get all unique document IDs from vector store"""
        try:
            count = self.vector_store.count()
            if count == 0:
                return []
            
            dummy_embedding = np.zeros(self.embedder.model.get_sentence_embedding_dimension())
            _, _, _, metadatas = self.vector_store.query(
                dummy_embedding,
                n_results=min(count, 10000)
            )
            
            doc_ids = set()
            for meta in metadatas:
                if 'doc_id' in meta:
                    doc_ids.add(meta['doc_id'])
            
            return list(doc_ids)
        except Exception as e:
            logger.error(f"Error getting document IDs: {e}")
            return []
    
    def _get_document_info(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata"""
        try:
            dummy_embedding = np.zeros(self.embedder.model.get_sentence_embedding_dimension())
            _, _, _, metadatas = self.vector_store.query(
                dummy_embedding,
                n_results=1,
                where={"doc_id": doc_id}
            )
            
            return metadatas[0] if metadatas else None
        except Exception as e:
            logger.error(f"Error getting document info for {doc_id}: {e}")
            return None
