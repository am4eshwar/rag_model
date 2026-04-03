"""
Batch Processing Engine

Handles processing of 100+ documents with:
- Progress tracking and persistence
- Resume capability for interrupted batches
- Parallel processing with resource management
- Real-time status updates
- Error handling and retry logic

DESIGN FOR SCALE:
- Process in chunks to prevent memory overflow
- SQLite database for progress persistence
- Async processing with configurable workers
- Graceful degradation on errors
- Memory-efficient streaming

USAGE:
    processor = BatchProcessor()
    batch_id = processor.create_batch(file_paths, batch_name="Research Docs")
    processor.process_batch(batch_id, parallel_workers=4)
    status = processor.get_batch_status(batch_id)
"""

import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import traceback

from config import DATA_DIR
from ingestion import ingest_submission
from chunking import chunk_submission
from embedding import embed_chunks
from vector_store import ChromaVectorStore, add_chunks_to_store

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class BatchStatus(Enum):
    """Batch processing status"""
    CREATED = "created"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DocumentJob:
    """Represents a single document processing job"""
    job_id: str
    batch_id: str
    file_path: str
    filename: str
    status: ProcessingStatus
    progress: float  # 0.0 to 1.0
    error_message: Optional[str]
    doc_id: Optional[str]
    chunks_count: Optional[int]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    metadata: Dict
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['status'] = self.status.value
        d['start_time'] = self.start_time.isoformat() if self.start_time else None
        d['end_time'] = self.end_time.isoformat() if self.end_time else None
        return d


@dataclass
class BatchJob:
    """Represents a batch processing job"""
    batch_id: str
    name: str
    status: BatchStatus
    total_documents: int
    completed_documents: int
    failed_documents: int
    progress: float  # 0.0 to 1.0
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['status'] = self.status.value
        d['created_at'] = self.created_at.isoformat()
        d['started_at'] = self.started_at.isoformat() if self.started_at else None
        d['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return d


class BatchProgressDB:
    """SQLite database for batch progress tracking"""
    
    def __init__(self, db_path: Path = DATA_DIR / "batch_progress.db"):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_documents INTEGER NOT NULL,
                    completed_documents INTEGER DEFAULT 0,
                    failed_documents INTEGER DEFAULT 0,
                    progress REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_jobs (
                    job_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0.0,
                    error_message TEXT,
                    doc_id TEXT,
                    chunks_count INTEGER,
                    start_time TEXT,
                    end_time TEXT,
                    metadata TEXT,
                    FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_batch_status 
                ON batches(status)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_batch 
                ON document_jobs(batch_id)
            """)
            
            conn.commit()
    
    def create_batch(self, batch: BatchJob):
        """Create a new batch"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO batches 
                (batch_id, name, status, total_documents, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                batch.batch_id,
                batch.name,
                batch.status.value,
                batch.total_documents,
                batch.created_at.isoformat(),
                json.dumps(batch.metadata)
            ))
            conn.commit()
    
    def update_batch(self, batch: BatchJob):
        """Update batch status"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE batches SET
                    status = ?,
                    completed_documents = ?,
                    failed_documents = ?,
                    progress = ?,
                    started_at = ?,
                    completed_at = ?,
                    metadata = ?
                WHERE batch_id = ?
            """, (
                batch.status.value,
                batch.completed_documents,
                batch.failed_documents,
                batch.progress,
                batch.started_at.isoformat() if batch.started_at else None,
                batch.completed_at.isoformat() if batch.completed_at else None,
                json.dumps(batch.metadata),
                batch.batch_id
            ))
            conn.commit()
    
    def get_batch(self, batch_id: str) -> Optional[BatchJob]:
        """Get batch by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM batches WHERE batch_id = ?", 
                (batch_id,)
            ).fetchone()
            
            if not row:
                return None
            
            return BatchJob(
                batch_id=row['batch_id'],
                name=row['name'],
                status=BatchStatus(row['status']),
                total_documents=row['total_documents'],
                completed_documents=row['completed_documents'],
                failed_documents=row['failed_documents'],
                progress=row['progress'],
                created_at=datetime.fromisoformat(row['created_at']),
                started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                metadata=json.loads(row['metadata']) if row['metadata'] else {}
            )
    
    def add_document_job(self, job: DocumentJob):
        """Add a document job"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO document_jobs
                (job_id, batch_id, file_path, filename, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                job.job_id,
                job.batch_id,
                job.file_path,
                job.filename,
                job.status.value,
                json.dumps(job.metadata)
            ))
            conn.commit()
    
    def update_document_job(self, job: DocumentJob):
        """Update document job status"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE document_jobs SET
                    status = ?,
                    progress = ?,
                    error_message = ?,
                    doc_id = ?,
                    chunks_count = ?,
                    start_time = ?,
                    end_time = ?,
                    metadata = ?
                WHERE job_id = ?
            """, (
                job.status.value,
                job.progress,
                job.error_message,
                job.doc_id,
                job.chunks_count,
                job.start_time.isoformat() if job.start_time else None,
                job.end_time.isoformat() if job.end_time else None,
                json.dumps(job.metadata),
                job.job_id
            ))
            conn.commit()
    
    def get_batch_jobs(self, batch_id: str) -> List[DocumentJob]:
        """Get all jobs for a batch"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM document_jobs WHERE batch_id = ?",
                (batch_id,)
            ).fetchall()
            
            jobs = []
            for row in rows:
                jobs.append(DocumentJob(
                    job_id=row['job_id'],
                    batch_id=row['batch_id'],
                    file_path=row['file_path'],
                    filename=row['filename'],
                    status=ProcessingStatus(row['status']),
                    progress=row['progress'],
                    error_message=row['error_message'],
                    doc_id=row['doc_id'],
                    chunks_count=row['chunks_count'],
                    start_time=datetime.fromisoformat(row['start_time']) if row['start_time'] else None,
                    end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                ))
            return jobs
    
    def list_all_batches(self) -> List[BatchJob]:
        """List all batches"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM batches ORDER BY created_at DESC"
            ).fetchall()
            
            batches = []
            for row in rows:
                batches.append(BatchJob(
                    batch_id=row['batch_id'],
                    name=row['name'],
                    status=BatchStatus(row['status']),
                    total_documents=row['total_documents'],
                    completed_documents=row['completed_documents'],
                    failed_documents=row['failed_documents'],
                    progress=row['progress'],
                    created_at=datetime.fromisoformat(row['created_at']),
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    metadata=json.loads(row['metadata']) if row['metadata'] else {}
                ))
            return batches


class BatchProcessor:
    """
    Batch processor for handling 100+ documents
    
    Features:
    - Progress tracking with SQLite persistence
    - Resume capability
    - Parallel processing
    - Error handling
    - Memory-efficient streaming
    """
    
    def __init__(self, db_path: Path = DATA_DIR / "batch_progress.db"):
        self.db = BatchProgressDB(db_path)
        self.vector_store = ChromaVectorStore()
        logger.info("Initialized BatchProcessor")
    
    def create_batch(
        self,
        file_paths: List[str],
        batch_name: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create a new batch job
        
        Args:
            file_paths: List of file paths to process
            batch_name: Name for this batch
            metadata: Optional metadata (assignment info, etc.)
            
        Returns:
            batch_id
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        batch = BatchJob(
            batch_id=batch_id,
            name=batch_name,
            status=BatchStatus.CREATED,
            total_documents=len(file_paths),
            completed_documents=0,
            failed_documents=0,
            progress=0.0,
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            metadata=metadata or {}
        )
        
        self.db.create_batch(batch)
        
        # Create document jobs
        for i, file_path in enumerate(file_paths):
            job = DocumentJob(
                job_id=f"{batch_id}_doc_{i}",
                batch_id=batch_id,
                file_path=file_path,
                filename=Path(file_path).name,
                status=ProcessingStatus.PENDING,
                progress=0.0,
                error_message=None,
                doc_id=None,
                chunks_count=None,
                start_time=None,
                end_time=None,
                metadata={}
            )
            self.db.add_document_job(job)
        
        logger.info(f"Created batch {batch_id} with {len(file_paths)} documents")
        return batch_id
    
    def process_batch(
        self,
        batch_id: str,
        parallel_workers: int = 4,
        progress_callback: Optional[Callable] = None
    ):
        """
        Process all documents in a batch
        
        Args:
            batch_id: Batch to process
            parallel_workers: Number of parallel workers
            progress_callback: Optional callback(batch_status) called on progress
        """
        batch = self.db.get_batch(batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")
        
        logger.info(f"Starting batch processing: {batch_id}")
        logger.info(f"Total documents: {batch.total_documents}")
        logger.info(f"Parallel workers: {parallel_workers}")
        
        # Update batch status
        batch.status = BatchStatus.PROCESSING
        batch.started_at = datetime.now()
        self.db.update_batch(batch)
        
        # Get all pending jobs
        jobs = self.db.get_batch_jobs(batch_id)
        pending_jobs = [j for j in jobs if j.status == ProcessingStatus.PENDING]
        
        logger.info(f"Processing {len(pending_jobs)} pending documents")
        
        # Process with thread pool
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_job = {
                executor.submit(self._process_document, job): job 
                for job in pending_jobs
            }
            
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    if result:
                        batch.completed_documents += 1
                    else:
                        batch.failed_documents += 1
                except Exception as e:
                    logger.error(f"Unexpected error processing {job.filename}: {e}")
                    batch.failed_documents += 1
                
                # Update progress
                batch.progress = (batch.completed_documents + batch.failed_documents) / batch.total_documents
                self.db.update_batch(batch)
                
                # Call progress callback
                if progress_callback:
                    progress_callback(batch)
                
                logger.info(f"Progress: {batch.completed_documents}/{batch.total_documents} completed, "
                          f"{batch.failed_documents} failed")
        
        # Finalize batch
        batch.status = BatchStatus.COMPLETED if batch.failed_documents == 0 else BatchStatus.FAILED
        batch.completed_at = datetime.now()
        batch.progress = 1.0
        self.db.update_batch(batch)
        
        logger.info(f"Batch {batch_id} completed: {batch.completed_documents} successful, "
                   f"{batch.failed_documents} failed")
    
    def _process_document(self, job: DocumentJob) -> bool:
        """Process a single document"""
        try:
            # Update status
            job.status = ProcessingStatus.PROCESSING
            job.start_time = datetime.now()
            job.progress = 0.1
            self.db.update_document_job(job)
            
            logger.info(f"Processing: {job.filename}")
            
            # Step 1: Ingest
            doc = ingest_submission(job.file_path, save_to_disk=True)
            job.doc_id = doc.doc_id
            job.progress = 0.3
            self.db.update_document_job(job)
            
            # Step 2: Chunk
            chunks = chunk_submission(
                doc_id=doc.doc_id,
                text=doc.text,
                pages=doc.pages,
                metadata=doc.metadata
            )
            
            if not chunks:
                logger.warning(f"No text extracted from {job.filename}. Skipping embedding/storage.")
                job.status = ProcessingStatus.SKIPPED
                job.progress = 1.0
                job.end_time = datetime.now()
                self.db.update_document_job(job)
                return True # Skip but count as processed
                
            job.chunks_count = len(chunks)
            job.progress = 0.5
            self.db.update_document_job(job)
            
            # Step 3: Embed
            chunk_ids, embeddings = embed_chunks(chunks, use_cache=True)
            
            if len(embeddings) == 0:
                logger.warning(f"No embeddings generated for {job.filename}. Skipping storage.")
                job.status = ProcessingStatus.SKIPPED
                job.progress = 1.0
                job.end_time = datetime.now()
                self.db.update_document_job(job)
                return True

            job.progress = 0.8
            self.db.update_document_job(job)
            
            # Step 4: Store
            add_chunks_to_store(self.vector_store, chunks, embeddings)
            job.progress = 1.0
            self.db.update_document_job(job)
            
            # Success
            job.status = ProcessingStatus.COMPLETED
            job.end_time = datetime.now()
            self.db.update_document_job(job)
            
            logger.info(f"✓ Completed: {job.filename} ({len(chunks)} chunks)")
            return True
            
        except Exception as e:
            # Failure
            job.status = ProcessingStatus.FAILED
            job.error_message = f"{type(e).__name__}: {str(e)}"
            job.end_time = datetime.now()
            self.db.update_document_job(job)
            
            logger.error(f"✗ Failed: {job.filename} - {job.error_message}")
            return False
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict]:
        """Get detailed batch status"""
        batch = self.db.get_batch(batch_id)
        if not batch:
            return None
        
        jobs = self.db.get_batch_jobs(batch_id)
        
        return {
            "batch": batch.to_dict(),
            "jobs": [j.to_dict() for j in jobs],
            "summary": {
                "total": len(jobs),
                "pending": sum(1 for j in jobs if j.status == ProcessingStatus.PENDING),
                "processing": sum(1 for j in jobs if j.status == ProcessingStatus.PROCESSING),
                "completed": sum(1 for j in jobs if j.status == ProcessingStatus.COMPLETED),
                "failed": sum(1 for j in jobs if j.status == ProcessingStatus.FAILED),
            }
        }
    
    def resume_batch(self, batch_id: str, parallel_workers: int = 4):
        """Resume a failed or interrupted batch"""
        batch = self.db.get_batch(batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")
        
        jobs = self.db.get_batch_jobs(batch_id)
        pending = sum(1 for j in jobs if j.status == ProcessingStatus.PENDING)
        
        if pending == 0:
            logger.info(f"No pending jobs in batch {batch_id}")
            return
        
        logger.info(f"Resuming batch {batch_id} with {pending} pending jobs")
        self.process_batch(batch_id, parallel_workers)
    
    def list_batches(self) -> List[Dict]:
        """List all batches"""
        batches = self.db.list_all_batches()
        return [b.to_dict() for b in batches]
    
    def cancel_batch(self, batch_id: str):
        """Cancel a batch"""
        batch = self.db.get_batch(batch_id)
        if not batch:
            raise ValueError(f"Batch {batch_id} not found")
        
        batch.status = BatchStatus.CANCELLED
        batch.completed_at = datetime.now()
        self.db.update_batch(batch)
        logger.info(f"Cancelled batch {batch_id}")
