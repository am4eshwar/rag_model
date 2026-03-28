"""
Main Entry Point for RAG Pipeline

This script demonstrates how to use the RAG pipeline for document analysis.

SETUP STEPS:
1. Install dependencies: pip install -r requirements.txt
2. Set GROQ_API_KEY in .env file
3. Place PDF/DOCX files in data/raw_submissions/
4. Run this script: python main.py

USAGE:
- Build index: python main.py --build
- Query: python main.py --query "Main conclusion?"
- Interactive: python main.py --interactive

NEW: For advanced features (batch processing, analytics, similarity detection), use:
    python rag_orchestrator.py --help
"""

import logging
from pathlib import Path
from dotenv import load_dotenv
from ingestion import ingest_submission
from chunking import chunk_submission
from embedding import embed_chunks, EmbeddingModel
from vector_store import ChromaVectorStore, add_chunks_to_store
from indexing import VectorIndex
from generation import get_generator
from config import RAW_DOCS_DIR, SUPPORTED_FORMATS, INDEX_FILE_PATH, METADATA_FILE_PATH

# Load environment variables from .env file
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_index_from_directory(directory: Path = RAW_DOCS_DIR):
    """
    Build index from all documents in a directory
    
    Args:
        directory: Directory containing PDF/DOCX files
    """
    logger.info(f"Building index from documents in: {directory}")
    
    # Initialize components
    vector_store = ChromaVectorStore()
    index = VectorIndex()
    
    # Find all supported documents
    files = []
    for ext in SUPPORTED_FORMATS:
        files.extend(directory.glob(f"*{ext}"))
    
    if not files:
        logger.warning(f"No documents found in {directory}")
        logger.info(f"Supported formats: {SUPPORTED_FORMATS}")
        logger.info(f"Please place PDF or DOCX files in: {directory}")
        return
    
    logger.info(f"Found {len(files)} documents to process")
    
    all_chunks = []
    all_embeddings = []
    
    # Process each document
    for file_path in files:
        try:
            logger.info(f"\nProcessing: {file_path.name}")
            
            # Step 1: Ingest document
            doc = ingest_submission(str(file_path), save_to_disk=True)
            logger.info(f"  ✓ Ingested: {len(doc.text)} chars, {len(doc.pages)} pages")
            
            # Step 2: Chunk document
            chunks = chunk_submission(
                doc_id=doc.doc_id,
                text=doc.text,
                pages=doc.pages,
                metadata=doc.metadata
            )
            logger.info(f"  ✓ Chunked: {len(chunks)} chunks")
            
            # Step 3: Embed chunks
            chunk_ids, embeddings = embed_chunks(chunks, use_cache=True)
            logger.info(f"  ✓ Embedded: {embeddings.shape}")
            
            all_chunks.extend(chunks)
            all_embeddings.append(embeddings)
            
        except Exception as e:
            logger.error(f"  ✗ Failed to process {file_path.name}: {e}")
            continue
    
    if not all_chunks:
        logger.error("No documents were successfully processed")
        return
    
    # Combine all embeddings
    import numpy as np
    combined_embeddings = np.vstack(all_embeddings)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Total: {len(all_chunks)} chunks from {len(files)} documents")
    logger.info(f"{'='*60}\n")
    
    # Step 4: Store in ChromaDB
    logger.info("Storing in ChromaDB...")
    add_chunks_to_store(vector_store, all_chunks, combined_embeddings)
    logger.info(f"  ✓ ChromaDB: {vector_store.count()} vectors stored")
    
    # Step 5: Build FAISS index
    logger.info("Building FAISS index...")
    
    # Extract chunk IDs and use the combined embeddings
    chunk_ids = [chunk.chunk_id for chunk in all_chunks]
    chunk_metadata = [chunk.to_dict() for chunk in all_chunks]
    
    index.add_items(combined_embeddings, chunk_ids, chunk_metadata)
    logger.info(f"  ✓ Added {len(all_chunks)} chunks to index")
    
    # Step 6: Save FAISS index
    logger.info("Saving FAISS index...")
    index.save()
    logger.info(f"  ✓ FAISS index saved to {INDEX_FILE_PATH}")
    logger.info(f"  ✓ Metadata saved to {METADATA_FILE_PATH}")
    
    logger.info(f"\n{'='*60}")
    logger.info("✓ Index building complete!")
    logger.info(f"{'='*60}\n")


def query_documents(query: str):
    """
    Query the indexed documents and generate a report
    
    Args:
        query: User's question about the documents
    """
    logger.info(f"Query: {query}")
    logger.info("="*60)
    
    # Initialize components
    embedder = EmbeddingModel()
    vector_store = ChromaVectorStore()
    
    # Check if we have any documents
    count = vector_store.count()
    if count == 0:
        logger.error("No documents indexed. Run with --build first.")
        return
    
    logger.info(f"Searching {count} chunks...")
    
    # Step 1: Embed query
    query_embedding = embedder.embed_query(query)
    logger.info(f"✓ Query embedded: {query_embedding.shape}")
    
    # Step 2: Retrieve from ChromaDB
    chunk_ids, docs, distances, metadatas = vector_store.query(
        query_embedding,
        n_results=10
    )
    logger.info(f"✓ Retrieved {len(chunk_ids)} chunks")
    
    # Create retrieval result objects
    from dataclasses import dataclass
    
    @dataclass
    class RetrievalResult:
        chunk_id: str
        text: str
        score: float
        rank: int
        metadata: dict
        
        def to_dict(self):
            return {
                "chunk_id": self.chunk_id,
                "text": self.text,
                "score": self.score,
                "rank": self.rank,
                "metadata": self.metadata,
            }
    
    results = []
    for i, (cid, doc, dist, meta) in enumerate(zip(chunk_ids, docs, distances, metadatas)):
        # Convert distance to similarity score (for cosine: similarity = 1 - distance)
        score = 1 - dist if dist <= 1 else 0.0
        results.append(RetrievalResult(
            chunk_id=cid,
            text=doc,
            score=score,
            rank=i,
            metadata=meta
        ))
    
    # Display top results
    logger.info("\nTop 3 results:")
    for i, result in enumerate(results[:3]):
        logger.info(f"\n[{i+1}] {result.chunk_id} (score: {result.score:.3f})")
        logger.info(f"    {result.text[:150]}...")
    
    # Step 3: Generate report with LLM
    logger.info(f"\n{'='*60}")
    logger.info("Generating report with Groq...")
    logger.info(f"{'='*60}\n")
    
    try:
        generator = get_generator()  # Uses Groq by default
        
        # Prepare document metadata
        doc_metadata = {
            "doc_id": metadatas[0].get("doc_id", "unknown") if metadatas else "unknown",
            "filename": "documents",
            "page_count": "multiple"
        }
        
        report = generator.generate(query, results[:5], doc_metadata)
        
        # Display report
        print("\n" + "="*60)
        print("VERIFICATION REPORT")
        print("="*60)
        print(report.to_markdown())
        
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        logger.error("Make sure GROQ_API_KEY is set in .env file")


def interactive_mode():
    """Interactive query mode"""
    logger.info("="*60)
    logger.info("RAG Pipeline - Interactive Mode")
    logger.info("="*60)
    logger.info("Type your questions (or 'quit' to exit)\n")
    
    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                continue
            if query.lower() in ['quit', 'exit', 'q']:
                logger.info("Goodbye!")
                break
            
            query_documents(query)
            
        except KeyboardInterrupt:
            logger.info("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Pipeline for Document Analysis")
    parser.add_argument("--build", action="store_true", help="Build index from documents in data/raw_submissions/")
    parser.add_argument("--query", type=str, help="Query the documents")
    parser.add_argument("--interactive", action="store_true", help="Interactive query mode")
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.build:
        build_index_from_directory()
    elif args.query:
        query_documents(args.query)
    elif args.interactive:
        interactive_mode()
    else:
        # Default: show help
        parser.print_help()
        print("\nQuick start:")
        print("  1. Place PDF/DOCX files in: data/documents/")
        print("  2. Build index: python main.py --build")
        print("  3. Query: python main.py --query \"What is the main topic?\"")
        print("  4. For advanced features (reports, similarity, analytics), use: python rag_orchestrator.py")


if __name__ == "__main__":
    main()
