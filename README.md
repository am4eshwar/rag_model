# General Purpose Retrieval-Augmented Generation (RAG) Pipeline

## Project Description and Objective
The primary objective of this project was to build a foundational Retrieval-Augmented Generation (RAG) model entirely from scratch. It was designed with adaptability in mind, serving as a flexible core that can be seamlessly integrated with virtually any system or application. Its main purpose is to ingest uploaded documents, extract the underlying text, and intelligently retrieve relevant data based on user queries.

## Key Features
*   **Batch Processing**: Engineered to process large volumes of documents simultaneously with persistent progress tracking.
*   **Intelligent Retrieval**: Employs a hybrid search approach, combining different indexing methods and reranking algorithms to ensure highly accurate results.
*   **Optical Character Recognition (OCR) Support**: Capable of reading and extracting text from scanned PDFs, printed images, and handwritten documents.
*   **Automated Reporting**: Automatically generates detailed evaluation reports based on customizable criteria.
*   **Similarity Detection**: Analyzes documents to identify both semantic (conceptual) and exact text matches to find similarities across a dataset.
*   **Entity Analytics**: Monitors and analyzes long-term trends and performance metrics across specific categories or authors.

## Tools and Technologies Used
*   **LangChain**: The core framework used to orchestrate the language model operations.
*   **ChromaDB**: A vector database used for the persistent storage of document embeddings.
*   **FAISS**: Utilized for high-speed approximate nearest neighbor searches.
*   **Sentence-Transformers**: Specifically the all-MiniLM-L6-v2 model, used to vectorize text.
*   **Groq (Mixtral 8x7b)**: The Large Language Model (LLM) responsible for generating responses and insights.
*   **OCR Engines**: Tesseract for printed text and EasyOCR for handwritten text extraction.
*   **Utilities**: Scipy for analytics, pdf2image and Pillow for image processing, and python-docx for Word document parsing.
*   **Streamlit**: Used to power the interactive user interface.

## System Architecture
The application is built on a modular architecture where each component is responsible for a specific stage of the RAG pipeline:

1.  **Ingestion**: The system first loads files (PDFs, DOCX, images). If a document is scanned or an image, the automatic OCR layer extracts the readable text.
2.  **Chunking**: The extracted text is split into smaller, overlapping chunks. This ensures that the context is preserved while keeping the text segments manageable for the model.
3.  **Embedding**: Each text chunk is converted into a mathematical vector (embedding) using the Sentence-Transformers model.
4.  **Storage and Indexing**: The vectors are stored in ChromaDB and indexed using FAISS to allow for rapid, high-speed similarity searches.
5.  **Retrieval**: When a query is made, the system finds the most relevant document chunks and applies a reranking process to prioritize the best matches.
6.  **Generation**: The retrieved context is sent to the LLM (Mixtral 8x7b via Groq), which formulates a precise, context-aware response.
7.  **Management Layers**: Additional modules handle the orchestration of these steps, manage SQLite-backed progress tracking for large batches, and perform data analytics.

## Designed Scale
This system is robustly designed and optimized to handle batch processing at a scale of 100+ documents simultaneously. The integration of parallel workers and efficient indexing mechanisms ensures that performance remains stable and retrieval remains fast even when dealing with large document repositories.
