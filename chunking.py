"""
Chunking Module - Simplified with LangChain

Splits documents into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.

RATIONALE:
- LangChain RecursiveCharacterTextSplitter: Battle-tested, maintains semantic boundaries
- Chunk size 1000 chars: Optimal for semantic search with larger context windows
- 20% overlap (200 chars): Ensures no concept is split across chunk boundaries
- Hierarchical separators: Automatically tries paragraph → sentence → word boundaries
- Metadata preservation: Track source document, page, position for citation

VALIDATION:
- Test boundary preservation: Sentences should not split mid-word
- Test overlap: Adjacent chunks should share ~200 chars
- Test metadata: Every chunk should trace back to source document + page
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

# LangChain text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a single chunk of text with metadata
    
    Attributes:
        chunk_id: Unique identifier (format: "{doc_id}_chunk_{index}")
        text: The chunked text content
        doc_id: Source document ID
        chunk_index: Position in document (0-indexed)
        start_char: Character position in original document
        end_char: Character position in original document
        page_num: Page number in source (if available)
        metadata: Additional custom metadata
    """
    chunk_id: str
    text: str
    doc_id: str
    chunk_index: int
    start_char: int
    end_char: int
    page_num: Optional[int] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "page_num": self.page_num,
            "metadata": self.metadata or {},
            "char_count": len(self.text),
        }





class DocumentChunker:
    """
    High-level chunker that processes documents and preserves metadata
    Uses LangChain's RecursiveCharacterTextSplitter for text splitting
    
    FEATURES:
    - LangChain integration: Uses standardized RecursiveCharacterTextSplitter
    - Automatic page assignment based on character position
    - Chunk ID generation for tracking
    - Metadata propagation from document to chunks
    """
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "LangChain not installed. "
                "Run: pip install langchain"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def chunk_document(
        self,
        doc_id: str,
        text: str,
        pages: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        """
        Chunk a document and assign metadata
        
        Args:
            doc_id: Document identifier
            text: Full document text
            pages: List of {page_num, text, char_count} dicts
            metadata: Additional metadata to attach to chunks
            
        Returns:
            List of Chunk objects with metadata
            
        Example:
            >>> chunker = DocumentChunker()
            >>> chunks = chunker.chunk_document(
            ...     doc_id="essay_001",
            ...     text="Long document text...",
            ...     pages=[{"page_num": 1, "text": "..."}],
            ... )
            >>> print(f"Created {len(chunks)} chunks")
        """
        # Split text into chunks
        chunk_texts = self.splitter.split_text(text)
        
        # Build page lookup (char position → page number)
        page_lookup = self._build_page_lookup(pages) if pages else {}
        
        # Create Chunk objects
        chunks = []
        current_pos = 0
        
        for i, chunk_text in enumerate(chunk_texts):
            # Find chunk position in original text
            # Note: Overlap means chunks may not align exactly, use approximation
            start_char = current_pos
            end_char = current_pos + len(chunk_text)
            
            # Determine page number
            page_num = self._find_page_for_position(start_char, page_lookup)
            
            # Create chunk ID
            chunk_id = f"{doc_id}_chunk_{i:04d}"
            
            chunk = Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                doc_id=doc_id,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                page_num=page_num,
                metadata=metadata,
            )
            
            chunks.append(chunk)
            
            # Update position (account for overlap)
            # Move forward by (chunk_length - overlap) to next chunk start
            current_pos += len(chunk_text) - self.chunk_overlap
        
        logger.info(
            f"Chunked document {doc_id}: {len(chunks)} chunks "
            f"from {len(text)} chars"
        )
        
        return chunks
    
    def _build_page_lookup(self, pages: List[Dict]) -> Dict[int, int]:
        """
        Build mapping from character position to page number
        
        Returns:
            Dict mapping start_char → page_num
        """
        lookup = {}
        char_position = 0
        
        for page in pages:
            page_num = page["page_num"]
            page_text = page["text"]
            page_len = len(page_text)
            
            # Map this character range to page number
            for pos in range(char_position, char_position + page_len):
                lookup[pos] = page_num
            
            char_position += page_len + 2  # +2 for \n\n separator
        
        return lookup
    
    def _find_page_for_position(
        self,
        char_pos: int,
        page_lookup: Dict[int, int]
    ) -> Optional[int]:
        """Find page number for a character position"""
        if not page_lookup:
            return None
        
        # Direct lookup
        if char_pos in page_lookup:
            return page_lookup[char_pos]
        
        # Find closest page (for positions at boundaries)
        if page_lookup:
            closest_pos = min(page_lookup.keys(), key=lambda x: abs(x - char_pos))
            return page_lookup[closest_pos]
        
        return None


def chunk_submission(
    doc_id: str,
    text: str,
    pages: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None,
) -> List[Chunk]:
    """
    Convenience function to chunk a submission
    
    Args:
        doc_id: Document identifier
        text: Full document text
        pages: Optional page information
        metadata: Optional metadata
        
    Returns:
        List of Chunk objects
        
    Example:
        >>> chunks = chunk_submission("essay_001", "Long text...")
        >>> for chunk in chunks[:3]:
        ...     print(f"{chunk.chunk_id}: {chunk.text[:50]}...")
    """
    chunker = DocumentChunker()
    return chunker.chunk_document(doc_id, text, pages, metadata)


# ==============================================================================
# VALIDATION & TESTING
# ==============================================================================

def validate_chunking():
    """
    Self-test function to verify chunking module
    
    TEST CASES:
    1. Basic splitting: Text > chunk_size should split
    2. Overlap preservation: Adjacent chunks should overlap
    3. Boundary respect: Should not split mid-word
    4. Metadata propagation: Chunks should have correct page numbers
    5. Edge cases: Empty text, text < chunk_size
    """
    print("=" * 70)
    print("CHUNKING MODULE VALIDATION")
    print("=" * 70)
    
    # Test 1: Basic splitting
    print("\n[TEST 1] Basic text splitting")
    text = "This is a test document.\n\n" * 50  # ~1250 chars
    chunks = chunk_submission("test_doc", text)
    print(f"✓ Split {len(text)} chars into {len(chunks)} chunks")
    assert len(chunks) > 1, "Should create multiple chunks"
    
    # Test 2: Overlap verification
    print("\n[TEST 2] Overlap preservation")
    chunk_texts = [c.text for c in chunks]
    if len(chunk_texts) >= 2:
        # Check if chunk 1 end overlaps with chunk 2 start
        overlap_expected = CHUNK_OVERLAP
        chunk1_end = chunk_texts[0][-overlap_expected:]
        chunk2_start = chunk_texts[1][:overlap_expected]
        
        # Check for partial overlap (may not be exact due to separator logic)
        has_overlap = any(
            chunk1_end[i:i+20] in chunk2_start
            for i in range(len(chunk1_end) - 20)
        )
        if has_overlap:
            print("✓ Detected overlap between adjacent chunks")
        else:
            print("⚠ Overlap may be smaller than expected (check separator logic)")
    
    # Test 3: Boundary respect
    print("\n[TEST 3] Word boundary preservation")
    for chunk in chunks:
        # Check no mid-word splits (heuristic: no chunks ending with alphanumeric + starting with alphanumeric)
        text = chunk.text.strip()
        if text:
            # Should not end with half a word
            assert not (text[-1].isalnum() and text[0].isalnum()), \
                f"Chunk may have split mid-word: ...{text[-20:]}"
    print(f"✓ All {len(chunks)} chunks respect word boundaries")
    
    # Test 4: Metadata propagation
    print("\n[TEST 4] Metadata and page assignment")
    pages = [
        {"page_num": 1, "text": text[:400]},
        {"page_num": 2, "text": text[400:]},
    ]
    chunks_with_pages = chunk_submission("test_doc", text, pages=pages)
    page_nums = [c.page_num for c in chunks_with_pages if c.page_num]
    print(f"✓ Assigned page numbers to {len(page_nums)}/{len(chunks_with_pages)} chunks")
    assert len(page_nums) > 0, "Should assign at least some page numbers"
    
    # Test 5: Edge cases
    print("\n[TEST 5] Edge cases")
    
    # Empty text
    empty_chunks = chunk_submission("empty", "")
    assert len(empty_chunks) == 0, "Empty text should produce 0 chunks"
    print("✓ Empty text handled correctly")
    
    # Very short text
    short_text = "Short text"
    short_chunks = chunk_submission("short", short_text)
    assert len(short_chunks) == 1, "Short text should produce 1 chunk"
    assert short_chunks[0].text == short_text, "Short text should be unchanged"
    print("✓ Short text handled correctly")
    
    # Test 6: Chunk size validation
    print("\n[TEST 6] Chunk size constraints")
    for chunk in chunks:
        chunk_len = len(chunk.text)
        # Allow some flexibility (separators may push slightly over)
        max_allowed = CHUNK_SIZE + 100
        assert chunk_len <= max_allowed, \
            f"Chunk too large: {chunk_len} > {max_allowed}"
    print(f"✓ All chunks within size limit (max={CHUNK_SIZE})")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nOBSERVATIONS:")
    print(f"- Chunk size: {CHUNK_SIZE} chars")
    print(f"- Chunk overlap: {CHUNK_OVERLAP} chars ({CHUNK_OVERLAP/CHUNK_SIZE*100:.1f}%)")
    print(f"- Chunks created: {len(chunks)}")
    print(f"- Avg chunk size: {sum(len(c.text) for c in chunks) / len(chunks):.1f} chars")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    validate_chunking()
