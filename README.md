# General Purpose RAG Pipeline System

A high-performance, scalable Retrieval-Augmented Generation (RAG) pipeline designed for document analysis, evaluation, and similarity detection. This system is optimized for handling 100+ documents with advanced retrieval and automated reporting.

---

## 🚀 Key Features

*   **Batch Processing**: Efficiently handle 100+ documents with parallel workers and persistent progress tracking.
*   **Intelligent Retrieval**: Hybrid search using ChromaDB and FAISS with HNSW indexing and Cross-Encoder reranking.
*   **OCR Support**: Automatically extract text from scanned PDFs and handwritten documents using Tesseract and EasyOCR.
*   **Automated Reporting**: Generate detailed evaluation reports based on customizable rubrics (Essay, Short Answer, etc.).
*   **Similarity Detection**: Identify similar documents using both conceptual (semantic) and exact (n-gram) matching.
*   **Entity Analytics**: Track performance and trends across metadata categories (e.g., Author, Student, Department).

---

## 📂 System Architecture

The pipeline consists of modular components for each stage of the RAG process:

1.  **Ingestion (`ingestion.py`)**: Loads PDF and DOCX files. Includes an automatic OCR layer.
2.  **Chunking (`chunking.py`)**: Splits text into semantic chunks (default 1000 chars, 20% overlap).
3.  **Embedding (`embedding.py`)**: Vectorizes text using `sentence-transformers` (all-MiniLM-L6-v2).
4.  **Vector Store (`vector_store.py`)**: Manages persistent storage via ChromaDB.
5.  **Indexing (`indexing.py`)**: High-speed ANN search using FAISS (HNSW).
6.  **Retrieval (`retrieval.py`)**: Retrieves context and applies Cross-Encoder reranking for precision.
7.  **Generation (`generation.py`)**: Orchestrates LLM responses using Groq (Mixtral 8x7b).
8.  **Management Layers**: 
    - `batch_processor.py`: SQLite-backed progress tracking for large batches.
    - `entity_analytics.py`: Long-term trend and performance analysis.

---

## 🛠️ Tech Stack & Tools

*   **Framework**: LangChain
*   **Vector DB**: ChromaDB
*   **Indexing**: FAISS
*   **Embeddings**: `sentence-transformers`
*   **LLM**: Groq (Mixtral 8x7b)
*   **OCR Engines**: Tesseract (Printed), EasyOCR (Handwritten)
*   **Utilities**: Scipy (Analytics), pdf2image, Pillow, python-docx

---

## ⚙️ Setup & Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Install OCR Engines (Optional for OCR)
If you need to process scanned documents or images, you must install:

**Tesseract OCR**:
- [Download Installer](https://github.com/UB-Mannheim/tesseract/wiki)
- Install to `C:\Program Files\Tesseract-OCR` and add to PATH.

**Poppler**:
- [Download Release](https://github.com/oschwartz10612/poppler-windows/releases)
- Extract and add `Library\bin` to PATH.

---

## 🛠️ Usage Guide

The `rag_orchestrator.py` script is the primary command-line interface.

### Processing Documents
Place your files in a directory and run:
```bash
python rag_orchestrator.py batch-process --dir ./your_docs --name "Project Name"
```

### Generating Evaluation Reports
```bash
python rag_orchestrator.py generate-reports --batch-id batch_TIMESTAMP --rubric essay --format csv
```

### Finding Similar Documents
```bash
python rag_orchestrator.py find-similar --batch-id batch_TIMESTAMP
```

### Student/Entity Analytics
```bash
python rag_orchestrator.py analytics --id "author_123" --key "author_id"
```

---

## 🔧 Maintenance & Logs
- **Logs**: All activities are logged to `pipeline.log`.
- **Reset Data**: Delete the `data/` directory (excluding original documents) to clear the index and database.
