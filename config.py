"""
Configuration for RAG Pipeline
All hyperparameters, model choices, and tunable settings in one place.

RATIONALE FOR EACH PARAMETER:
- Embedding model: all-MiniLM-L6-v2 chosen for 384-dim (low memory), fast inference
- Chunk size: 1000 chars ≈ 250 tokens, optimal for general documents
- Overlap: 20% (200/1000) prevents semantic boundary cuts
- HNSW M: 16 = good balance (higher = more accuracy but slower + more memory)
- HNSW ef_construction: 200 = build quality (higher = better recall at indexing time)
- HNSW ef_search: 50 = query-time recall (increase if recall@20 < 0.85)
- Top-K retrieval: 20 for reranking, 5 for final generation
- Reranker: cross-encoder improves precision@5 by ~15-20% but adds 2-3s latency
- Generation temp: 0.1 for deterministic, reproducible reports
"""

from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a single .env file at the root
load_dotenv()

# ==============================================================================
# DIRECTORY PATHS
# ==============================================================================
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "documents"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DOCS_DIR, PROCESSED_DIR, INDEX_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# DOCUMENT INGESTION
# ==============================================================================
# Simplified: Only PDF and DOCX support using LangChain document loaders
SUPPORTED_FORMATS = [".pdf", ".docx"]

# OCR Configuration for scanned/image-based PDFs
USE_OCR = True  # Enable OCR for scanned documents
OCR_LANGUAGE = "eng"  # Tesseract language (eng, hin, spa, etc.)
OCR_MIN_TEXT_THRESHOLD = 100  # If PDF has < 100 chars, treat as scanned
USE_EASYOCR_FOR_HANDWRITING = True  # Use EasyOCR for handwritten text detection
EASYOCR_LANGUAGES = ["en"]  # EasyOCR languages (can add: 'hi', 'es', etc.)
OCR_DPI = 300  # DPI for converting PDF to images (higher = better quality, slower)

# ==============================================================================
# CHUNKING PARAMETERS
# ==============================================================================
# JUSTIFICATION:
# - 1000 chars ≈ 250 tokens for typical English text (1 token ≈ 4 chars)
# - Larger chunks capture more context, better for modern embedding models
# - 20% overlap (200 chars) ensures no concept is split across boundaries
# - LangChain RecursiveCharacterTextSplitter automatically handles separators
#   (tries: \n\n, \n, space, character in that order)

CHUNK_SIZE = 1000  # characters (not tokens)
CHUNK_OVERLAP = 200  # 20% overlap to preserve context at boundaries

# When to adjust:
# - CHUNK_SIZE: Increase to 1500 if docs are technical with long explanations
#               Decrease to 500 if docs are short-answer assignments
# - CHUNK_OVERLAP: Increase to 300 (30%) if retrieval misses boundary concepts
#                  Decrease to 100 (10%) if storage is limited

# ==============================================================================
# EMBEDDING MODEL
# ==============================================================================
# JUSTIFICATION:
# - all-MiniLM-L6-v2: 384 dimensions, 80MB model, fast CPU inference (<100ms/chunk)
# - Alternative: e5-base-v2 (768-dim, better quality, 2x slower, 220MB)
# - CRITICAL: Must use SAME model for both indexing and retrieval (no mismatch!)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # Output dimension (all-MiniLM-L6-v2)
EMBEDDING_BATCH_SIZE = 32  # Process chunks in batches for speed
EMBEDDING_DEVICE = "cpu"  # Change to "cuda" if GPU available

# Alternative models (comment/uncomment to switch):
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # 768-dim, slower, better
# EMBEDDING_DIM = 768
# EMBEDDING_MODEL_NAME = "intfloat/e5-base-v2"  # 768-dim, requires "query: " prefix
# EMBEDDING_DIM = 768

# Normalization: Always True for cosine similarity
NORMALIZE_EMBEDDINGS = True

# Embedding cache
CACHE_EMBEDDINGS = True  # Save computed embeddings to disk

# ==============================================================================
# CHROMADB CONFIGURATION
# ==============================================================================
# JUSTIFICATION:
# - ChromaDB: Modern vector database with built-in persistence
# - Simplified storage: No need for separate metadata files
# - Integration: Works seamlessly with embeddings
# - Distance metric: Cosine for normalized embeddings

CHROMA_PERSIST_DIR = INDEX_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "document_chunks"
CHROMA_DISTANCE_METRIC = "cosine"  # Options: "cosine", "l2", "ip" (inner product)

# ==============================================================================
# VECTOR INDEX (HNSW)
# ==============================================================================
# JUSTIFICATION:
# - HNSW: Hierarchical Navigable Small World graph algorithm
# - LIBRARY: FAISS HNSW (Facebook AI Similarity Search implementation)
# - M: Number of bi-directional links per node (higher = better recall, more memory)
#   * M=16: Good default (16 links × 2 directions = 32 edges per node)
#   * M=32: Use if you need >95% recall and have memory
#   * M=8: Use if memory is very limited (<2GB available)
# - ef_construction: Size of dynamic candidate list during index build
#   * 200: Good balance (higher = slower indexing but better graph quality)
#   * Increase to 400 if recall@20 < 0.85 in validation
# - ef_search: Size of dynamic candidate list during query
#   * 50: Fast queries (~50-100ms) with good recall
#   * Increase to 100 if recall drops below 0.80

HNSW_M = 16  # Number of bidirectional links per element
HNSW_EF_CONSTRUCTION = 200  # Quality of index construction
HNSW_EF_SEARCH = 50  # Quality of search (can be tuned per query)
HNSW_MAX_ELEMENTS = 100000  # Pre-allocate space (resize if needed)

# Index persistence
INDEX_FILE_PATH = INDEX_DIR / "faiss_index.bin"
METADATA_FILE_PATH = INDEX_DIR / "metadata.json"

# When to adjust:
# - Increase M to 32 if recall@10 < 0.90 on validation set
# - Increase ef_construction to 400 if index quality is poor
# - Increase ef_search to 100-200 if retrieval misses obvious matches

# ==============================================================================
# RETRIEVAL PARAMETERS
# ==============================================================================
# JUSTIFICATION:
# - RETRIEVAL_TOP_K=20: Fetch more candidates for reranking
# - FINAL_TOP_K=5: Send top-5 to generation (fits in LLM context, high precision)
# - MIN_SIMILARITY_THRESHOLD: Filter out irrelevant chunks (cosine similarity)

RETRIEVAL_TOP_K = 20  # Number of candidates from ANN search
FINAL_TOP_K = 3  # Reduced to 3 chunks to prevent input overflow with small local models
MIN_SIMILARITY_THRESHOLD = 0.3  # Cosine similarity cutoff (0.0-1.0)

# When to adjust:
# - Increase RETRIEVAL_TOP_K to 50 if reranker needs more candidates
# - Increase FINAL_TOP_K to 10 if generation needs more context
# - Adjust MIN_SIMILARITY_THRESHOLD based on domain (lower for broad queries)

# ==============================================================================
# RERANKING (OPTIONAL CROSS-ENCODER)
# ==============================================================================
# JUSTIFICATION:
# - Cross-encoders compute pairwise (query, document) relevance scores
# - More accurate than bi-encoders (embeddings) but slower (cannot pre-compute)
# - Used to refine top-20 candidates → top-5 with higher precision
# - ms-marco-MiniLM-L-6-v2: Trained on Microsoft MARCO dataset, 80MB, fast

USE_RERANKER = True  # Set False to skip reranking (faster but lower precision)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_BATCH_SIZE = 8  # Process query-doc pairs in batches
RERANKER_DEVICE = "cpu"  # Change to "cuda" if GPU available

# Alternative reranker models:
# RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"  # Larger, slower, better

# When to adjust:
# - Set USE_RERANKER=False if latency > 10s is unacceptable
# - Use larger cross-encoder if precision@5 < 0.80 on validation

# ==============================================================================
# GENERATION (LLM)
# ==============================================================================
# JUSTIFICATION:
# - Low temperature (0.1) for deterministic, factual reports
# - max_tokens=1024 sufficient for structured verification report
# - top_p=0.9 for slight diversity while staying deterministic
# - presence_penalty=0.0 to allow repeated citations
# - frequency_penalty=0.0 to avoid penalizing technical terms

# Generator model (modular - can swap for local or API-based LLM)
# Using Groq for fast LLM inference via LangChain
GENERATOR_TYPE = "groq"  # Options: "groq", "openai", "local", "mock"
GENERATOR_MODEL_NAME = "llama-3.3-70b-versatile"  # Groq model (recommended)

# Groq API configuration (requires GROQ_API_KEY environment variable)
# Get your API key from: https://console.groq.com/

# Alternative Groq models (as of Jan 2026):
# GENERATOR_MODEL_NAME = "llama-3.3-70b-versatile"  # Llama 3.3 70B (recommended)
# GENERATOR_MODEL_NAME = "llama-3.1-70b-versatile"  # Llama 3.1 70B
# GENERATOR_MODEL_NAME = "mixtral-8x7b-32768"  # DEPRECATED - no longer supported
# GENERATOR_MODEL_NAME = "gemma2-9b-it"  # Google Gemma 2 9B

# Generation hyperparameters (deterministic settings)
GENERATION_TEMPERATURE = 0.1  # Low for consistency (0.0-1.0)
GENERATION_MAX_TOKENS = 1024  # Max output length
GENERATION_TOP_P = 0.9  # Nucleus sampling threshold
GENERATION_FREQUENCY_PENALTY = 0.0  # Don't penalize term reuse
GENERATION_PRESENCE_PENALTY = 0.0  # Allow repeated concepts

# When to adjust:
# - Increase temperature to 0.3 if reports are too rigid
# - Increase max_tokens to 2048 if reports are truncated
# - Set temperature=0.0 for maximum determinism (may be too repetitive)

# ==============================================================================
# CACHING & PERFORMANCE
# ==============================================================================
CACHE_EMBEDDINGS = True  # Save computed embeddings to disk
CACHE_PROCESSED_CHUNKS = True  # Save chunked text to JSONL
ENABLE_BATCH_PROCESSING = True  # Process multiple docs in parallel

# ==============================================================================
# LOGGING & MONITORING
# ==============================================================================
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR
LOG_FILE_PATH = BASE_DIR / "pipeline.log"

# ==============================================================================
# FUTURE MODULE HOOKS
# ==============================================================================
# Placeholders for modules to be added later

# Plagiarism detection (to be implemented)
ENABLE_PLAGIARISM_CHECK = False
PLAGIARISM_SIMILARITY_THRESHOLD = 0.85  # Cosine similarity for flagging
PLAGIARISM_NGRAM_SIZE = 5  # For exact match detection

# Analytics (to be implemented)
ENABLE_ANALYTICS = False
ANALYTICS_DB_PATH = DATA_DIR / "analytics.db"

# ==============================================================================
# VALIDATION & TESTING
# ==============================================================================
TEST_DATA_DIR = BASE_DIR / "tests" / "test_data"
EVAL_METRICS_OUTPUT = BASE_DIR / "tests" / "eval_results.json"

# Validation dataset size for testing
VALIDATION_SET_SIZE = 50  # Number of query-doc pairs for evaluation
MIN_RECALL_AT_5 = 0.80  # Target recall@5 threshold
MIN_PRECISION_AT_5 = 0.60  # Target precision@5 threshold

# ==============================================================================
# SUMMARY OF TUNING RULES
# ==============================================================================
"""
QUICK TUNING GUIDE:

1. LOW RECALL (missing relevant chunks):
   - Increase HNSW_EF_SEARCH (50 → 100)
   - Increase RETRIEVAL_TOP_K (20 → 50)
   - Lower MIN_SIMILARITY_THRESHOLD (0.3 → 0.2)

2. LOW PRECISION (too many irrelevant chunks):
   - Enable USE_RERANKER = True
   - Increase MIN_SIMILARITY_THRESHOLD (0.3 → 0.5)
   - Decrease FINAL_TOP_K (5 → 3)

3. SLOW INDEXING:
   - Decrease HNSW_EF_CONSTRUCTION (200 → 100)
   - Decrease HNSW_M (16 → 8)
   - Reduce EMBEDDING_BATCH_SIZE (32 → 16)

4. SLOW RETRIEVAL:
   - Decrease HNSW_EF_SEARCH (50 → 30)
   - Disable USE_RERANKER
   - Decrease RETRIEVAL_TOP_K (20 → 10)

5. POOR GENERATION QUALITY:
   - Increase GENERATION_TEMPERATURE (0.1 → 0.3)
   - Increase FINAL_TOP_K (5 → 10) for more context
   - Switch to larger LLM model

6. MEMORY ISSUES:
   - Use smaller embedding model (all-MiniLM-L6-v2 is smallest)
   - Decrease HNSW_M (16 → 8)
   - Decrease EMBEDDING_BATCH_SIZE (32 → 8)
"""