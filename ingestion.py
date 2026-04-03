"""
Document Ingestion Module - Simplified with LangChain + OCR Support

Handles extraction of text from student submissions using LangChain document loaders:
- PDF files (using PyPDFLoader)
- DOCX files (using Docx2txtLoader)
- Scanned/Image-based PDFs (using OCR)

OCR SUPPORT:
- Detects scanned PDFs (minimal extractable text)
- EasyOCR: Deep learning OCR for handwritten & printed text
- Automatic fallback: Tries normal extraction → EasyOCR

RATIONALE:
- LangChain document loaders: Unified interface, built-in metadata extraction
- PyPDFLoader: Simple PDF parsing with page-level granularity
- Docx2txtLoader: Reliable DOCX text extraction
- OCR integration: Handles photographed/scanned assignments
- Simplified architecture: Less custom code, easier to maintain

VALIDATION:
- Test with multi-page PDF
- Test with complex DOCX documents
- Test with scanned PDF (photographed assignments)
- Test with handwritten assignments
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# LangChain document loaders
try:
    from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
except ImportError:
    PyPDFLoader = None
    Docx2txtLoader = None

# OCR dependencies
try:
    from pdf2image import convert_from_path
    from PIL import Image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    convert_from_path = None
    Image = None

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

from config import (
    SUPPORTED_FORMATS,
    RAW_DOCS_DIR,
    PROCESSED_DIR,
    USE_OCR,
    OCR_LANGUAGE,
    OCR_MIN_TEXT_THRESHOLD,
    USE_EASYOCR_FOR_HANDWRITING,
    EASYOCR_LANGUAGES,
    OCR_DPI,
)

logger = logging.getLogger(__name__)


class DocumentIngestionError(Exception):
    """Raised when document ingestion fails"""
    pass


class Document:
    """
    Represents an ingested document with metadata
    """
    def __init__(
        self,
        doc_id: str,
        text: str,
        metadata: Dict,
        pages: Optional[List[Dict]] = None
    ):
        self.doc_id = doc_id
        self.text = text
        self.metadata = metadata
        self.pages = pages or []  # List of {page_num: int, text: str, metadata: dict}
        
    def to_dict(self) -> Dict:
        """Serialize to dictionary for storage"""
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "pages": self.pages,
            "ingestion_timestamp": datetime.utcnow().isoformat(),
        }


class DocumentIngester:
    """
    Extracts text from PDF and DOCX formats using LangChain document loaders
    with OCR support for scanned/image-based documents
    
    DESIGN CHOICES:
    - Simplified: Only PDF and DOCX support (most common submission formats)
    - LangChain integration: Uses standardized LangChain Document objects
    - Page-level granularity: Preserved for citation purposes
    - OCR support: Automatic detection and processing of scanned documents
    
    OCR WORKFLOW:
    1. Try normal text extraction
    2. If text < threshold → detect as scanned
    3. Apply EasyOCR (deep learning model)
    """
    
    def __init__(self):
        self.supported_formats = SUPPORTED_FORMATS
        
        if PyPDFLoader is None or Docx2txtLoader is None:
            raise ImportError(
                "LangChain document loaders not installed. "
                "Run: pip install langchain langchain-community pypdf python-docx"
            )
        
        # Initialize EasyOCR reader lazily (expensive to load)
        self._easyocr_reader = None
    
    @property
    def easyocr_reader(self):
        """Lazy initialization of EasyOCR reader"""
        if self._easyocr_reader is None and EASYOCR_AVAILABLE and USE_EASYOCR_FOR_HANDWRITING:
            logger.info(f"Initializing EasyOCR with languages: {EASYOCR_LANGUAGES}")
            self._easyocr_reader = easyocr.Reader(EASYOCR_LANGUAGES, gpu=False)
        return self._easyocr_reader
        
    def ingest(self, file_path: Path, doc_id: Optional[str] = None) -> Document:
        """
        Main ingestion entrypoint using LangChain document loaders
        
        Args:
            file_path: Path to document file
            doc_id: Optional custom ID (defaults to filename stem)
            
        Returns:
            Document object with extracted text and metadata
            
        Raises:
            DocumentIngestionError: If extraction fails
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise DocumentIngestionError(f"File not found: {file_path}")
            
        file_ext = file_path.suffix.lower()
        if file_ext not in self.supported_formats:
            raise DocumentIngestionError(
                f"Unsupported format: {file_ext}. "
                f"Supported: {self.supported_formats}"
            )
        
        # Generate doc_id if not provided
        if doc_id is None:
            doc_id = file_path.stem
        
        # Base metadata (common to all formats)
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "format": file_ext,
            "file_size_bytes": file_path.stat().st_size,
            "ingestion_timestamp": datetime.utcnow().isoformat(),
        }
        
        # Route to appropriate loader
        try:
            if file_ext == ".pdf":
                text, pages = self._load_pdf(file_path)
            elif file_ext == ".docx":
                text, pages = self._load_docx(file_path)
            else:
                raise DocumentIngestionError(f"No loader for {file_ext}")
                
            metadata["page_count"] = len(pages)
            metadata["char_count"] = len(text)
            metadata["extraction_success"] = True
            
            logger.info(
                f"Ingested {file_path.name}: "
                f"{len(pages)} pages, {len(text)} chars"
            )
            
            return Document(doc_id, text, metadata, pages)
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            metadata["extraction_success"] = False
            metadata["error"] = str(e)
            raise DocumentIngestionError(f"Extraction failed: {e}") from e
    
    def _load_pdf(self, file_path: Path) -> tuple[str, List[Dict]]:
        """
        Load PDF using LangChain PyPDFLoader with OCR fallback
        
        DESIGN NOTE:
        - PyPDFLoader automatically extracts text page-by-page
        - Each page becomes a LangChain Document with metadata
        - If text extraction yields < threshold chars → apply OCR
        - OCR uses EasyOCR (for both handwriting and printed text)
        """
        try:
            # Step 1: Try normal text extraction
            loader = PyPDFLoader(str(file_path))
            langchain_docs = loader.load()
            
            pages = []
            full_text = []
            
            for idx, doc in enumerate(langchain_docs):
                page_num = idx + 1
                page_text = doc.page_content
                
                pages.append({
                    "page_num": page_num,
                    "text": page_text,
                    "char_count": len(page_text),
                    "metadata": doc.metadata,
                })
                
                full_text.append(page_text)
            
            combined_text = "\n\n".join(full_text)
            
            # Step 2: Check if document is scanned (minimal text)
            if USE_OCR and len(combined_text.strip()) < OCR_MIN_TEXT_THRESHOLD:
                logger.info(
                    f"Detected scanned PDF ({len(combined_text)} chars < {OCR_MIN_TEXT_THRESHOLD}). "
                    "Applying OCR..."
                )
                
                # Apply OCR to extract text
                ocr_text, ocr_pages = self._apply_ocr_to_pdf(file_path)
                
                if len(ocr_text) > len(combined_text):
                    logger.info(f"OCR extracted {len(ocr_text)} chars (better than {len(combined_text)})")
                    return ocr_text, ocr_pages
                else:
                    logger.warning("OCR did not improve text extraction, using original")
            
            return combined_text, pages
            
        except Exception as e:
            raise DocumentIngestionError(f"PDF loading failed: {e}") from e
    
    def _apply_ocr_to_pdf(self, file_path: Path) -> Tuple[str, List[Dict]]:
        """
        Apply OCR to scanned PDF
        
        STRATEGY:
        1. Convert PDF pages to images
        2. Apply EasyOCR (robust for handwriting and text)
        
        Returns:
            Tuple of (full_text, pages_list)
        """
        if not PDF2IMAGE_AVAILABLE or not EASYOCR_AVAILABLE:
            logger.warning("pdf2image or easyocr not available, OCR disabled")
            return "", []
        
        try:
            # Convert PDF to images
            logger.info(f"Converting PDF to images at {OCR_DPI} DPI...")
            
            # Use local poppler if present
            poppler_path = r'C:\program RAG model\Release-25.12.0-0\Library\bin'
            if not Path(poppler_path).exists():
                poppler_path = None
                
            images = convert_from_path(str(file_path), dpi=OCR_DPI, poppler_path=poppler_path)
            
            pages = []
            full_text = []
            
            for page_num, image in enumerate(images, start=1):
                logger.info(f"Processing page {page_num}/{len(images)} with EasyOCR...")
                
                try:
                    # Apply EasyOCR for text extraction
                    page_text = self._apply_easyocr_to_image(image)
                    
                except Exception as e:
                    logger.error(f"EasyOCR failed on page {page_num}: {e}")
                    page_text = ""
                
                pages.append({
                    "page_num": page_num,
                    "text": page_text,
                    "char_count": len(page_text),
                    "metadata": {"ocr": True, "method": "easyocr"},
                })
                
                full_text.append(page_text)
            
            combined_text = "\n\n".join(full_text)
            logger.info(f"OCR completed: {len(combined_text)} chars from {len(images)} pages")
            
            return combined_text, pages
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return "", []
    
    def _apply_easyocr_to_image(self, image: "Image.Image") -> str:
        """
        Apply EasyOCR to image for handwriting recognition
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text
        """
        if not EASYOCR_AVAILABLE or not USE_EASYOCR_FOR_HANDWRITING:
            return ""
        
        try:
            # Convert PIL Image to numpy array
            import numpy as np
            image_array = np.array(image)
            
            # Run EasyOCR
            reader = self.easyocr_reader
            if reader is None:
                logger.warning("EasyOCR reader not initialized")
                return ""
            
            results = reader.readtext(image_array)
            
            # Extract text from results
            # EasyOCR returns: [(bbox, text, confidence), ...]
            text_lines = [text for (bbox, text, conf) in results if conf > 0.3]
            
            return " ".join(text_lines)
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return ""
    
    def _load_docx(self, file_path: Path) -> tuple[str, List[Dict]]:
        """
        Load DOCX using LangChain Docx2txtLoader
        
        DESIGN NOTE:
        - Docx2txtLoader extracts all text in one document
        - We simulate pages by chunking every ~500 characters
        - Simpler than custom python-docx paragraph iteration
        """
        try:
            loader = Docx2txtLoader(str(file_path))
            langchain_docs = loader.load()
            
            if not langchain_docs:
                raise DocumentIngestionError("No content extracted from DOCX")
            
            # Usually returns one document with all text
            full_text = "\n\n".join([doc.page_content for doc in langchain_docs])
            
            # Simulate pages (approximate 500 chars per page)
            CHARS_PER_PAGE = 500
            pages = []
            
            for i in range(0, len(full_text), CHARS_PER_PAGE):
                page_text = full_text[i:i+CHARS_PER_PAGE]
                page_num = (i // CHARS_PER_PAGE) + 1
                
                pages.append({
                    "page_num": page_num,
                    "text": page_text,
                    "char_count": len(page_text),
                    "metadata": {"simulated": True},
                })
            
            return full_text, pages
            
        except Exception as e:
            raise DocumentIngestionError(f"DOCX loading failed: {e}") from e


def ingest_submission(
    file_path: str,
    doc_id: Optional[str] = None,
    save_to_disk: bool = True
) -> Document:
    """
    Convenience function to ingest a single submission using LangChain loaders
    
    Args:
        file_path: Path to document (PDF or DOCX)
        doc_id: Optional custom ID
        save_to_disk: Save processed document to JSON
        
    Returns:
        Document object
        
    Example:
        >>> doc = ingest_submission("student_essay.pdf")
        >>> print(f"Extracted {len(doc.text)} characters")
        >>> print(f"Pages: {len(doc.pages)}")
    """
    ingester = DocumentIngester()
    doc = ingester.ingest(Path(file_path), doc_id)
    
    if save_to_disk:
        output_path = PROCESSED_DIR / f"{doc.doc_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Saved processed document to {output_path}")
    
    return doc


# ==============================================================================
# VALIDATION & TESTING
# ==============================================================================

def validate_ingestion():
    """
    Self-test function to verify ingestion module with LangChain loaders
    
    TEST CASES:
    1. Error handling for missing file
    2. Error handling for unsupported format
    
    RUN THIS TO VALIDATE:
        python ingestion.py
    """
    print("=" * 70)
    print("INGESTION MODULE VALIDATION (LangChain)")
    print("=" * 70)
    
    # Test 1: Error handling for missing file
    print("\n[TEST 1] Missing file error handling")
    try:
        ingest_submission("nonexistent_file.pdf")
        print("✗ Should have raised error")
    except DocumentIngestionError as e:
        print(f"✓ Correctly raised error: {e}")
    
    # Test 2: Unsupported format
    print("\n[TEST 2] Unsupported format error handling")
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        xyz_path = f.name
    
    try:
        ingest_submission(xyz_path)
        print("✗ Should have raised error")
    except DocumentIngestionError as e:
        print(f"✓ Correctly raised error: {e}")
    finally:
        os.unlink(xyz_path)
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("\nNEXT STEPS:")
    print("1. Install dependencies: pip install langchain langchain-community pypdf python-docx")
    print("2. Test with real PDF and DOCX submissions")
    print("3. Place documents in:", RAW_DOCS_DIR)


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    validate_ingestion()
