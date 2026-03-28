"""
General Purpose RAG Pipeline Orchestrator

Unified command-line interface for RAG operations:
- Batch processing of documents
- Metadata-driven queries
- Similarity detection ("Similar documents")
- Performance analytics
- Report generation
- Progress tracking

USAGE:
    # Process batch of documents
    python rag_orchestrator.py batch-process --dir ./docs --name "Research Project"
    
    # Generate reports for a batch
    python rag_orchestrator.py generate-reports --batch-id batch_ID
    
    # Find similar documents
    python rag_orchestrator.py find-similar --batch-id batch_ID
    
    # Entity analytics
    python rag_orchestrator.py analytics --id author_123 --key author_id
    
    # Query specific document
    python rag_orchestrator.py query --doc-id doc_123 --query "Main conclusion?"
"""

import logging
import argparse
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables (API keys) from .env
load_dotenv()

from batch_processor import BatchProcessor, BatchStatus
from batch_reports import BatchReportGenerator, ESSAY_RUBRIC, SHORT_ANSWER_RUBRIC
from plagiarism_detector import PlagiarismDetector
from entity_analytics import EntityAnalytics
from document_queries import DocumentQuerySystem
from config import RAW_DOCS_DIR, DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGOrchestrator:
    """
    Main orchestrator for General RAG Pipeline
    """
    
    def __init__(self):
        self.batch_processor = BatchProcessor()
        self.report_generator = BatchReportGenerator()
        self.similarity_detector = PlagiarismDetector() # Logic remains same, rename in CLI
        self.analytics = EntityAnalytics()
        self.query_system = DocumentQuerySystem()
        logger.info("Initialized RAGOrchestrator")
    
    def batch_process(self, directory: Path, name: str, workers: int = 4) -> str:
        logger.info(f"Batch processing: {name} in {directory}")
        from config import SUPPORTED_FORMATS
        file_paths = []
        for ext in SUPPORTED_FORMATS:
            file_paths.extend(directory.glob(f"*{ext}"))
        
        if not file_paths:
            logger.error(f"No supported files found in {directory}")
            return None
            
        batch_id = self.batch_processor.create_batch(
            file_paths=[str(p) for p in file_paths],
            batch_name=name,
            metadata={"batch_name": name}
        )
        
        self.batch_processor.process_batch(batch_id, parallel_workers=workers)
        logger.info(f"Batch processing completed: {batch_id}")
        return batch_id

    def generate_reports(self, batch_id: str, rubric: str = "essay", format: str = "json", workers: int = 4):
        logger.info(f"Generating reports for batch: {batch_id}")
        status = self.batch_processor.get_batch_status(batch_id)
        if not status: return
        
        doc_ids = [j['doc_id'] for j in status['jobs'] if j['status'] == 'completed' and j['doc_id']]
        if not doc_ids: return
        
        rubric_obj = ESSAY_RUBRIC if rubric == "essay" else SHORT_ANSWER_RUBRIC
        reports = self.report_generator.generate_batch_reports(doc_ids, rubric=rubric_obj, parallel_workers=workers)
        
        output_dir = DATA_DIR / "reports" / batch_id
        self.report_generator.export_reports(reports, output_dir=output_dir, format=format)
        logger.info(f"Reports exported to {output_dir}")

    def find_similar(self, batch_id: Optional[str] = None, all: bool = False):
        """Check for similar documents"""
        logger.info("Starting similarity detection (Similar documents)")
        if all:
            similarity_matrix = self.similarity_detector.check_all_submissions(min_similarity=0.5)
            reports = []
            for doc_id, similarities in similarity_matrix.items():
                report = self.similarity_detector.generate_plagiarism_report(doc_id, similarity_results=similarities)
                if report.flagged: reports.append(report)
        elif batch_id:
            status = self.batch_processor.get_batch_status(batch_id)
            doc_ids = [j['doc_id'] for j in status['jobs'] if j['status'] == 'completed' and j['doc_id']]
            reports = [self.similarity_detector.generate_plagiarism_report(doc_id) for doc_id in doc_ids]
        else:
            logger.error("Must specify batch_id or --all")
            return
            
        self.similarity_detector.export_plagiarism_reports(reports)
        flagged = sum(1 for r in reports if r.flagged)
        logger.info(f"Similarity check complete: {flagged}/{len(reports)} documents with high similarity detected.")

    def run_analytics(self, entity_id: str, key: str = "author_id"):
        logger.info(f"Analyzing {key}: {entity_id}")
        profile = self.analytics.get_entity_profile(entity_id, metadata_key=key)
        progress = self.analytics.analyze_progress(entity_id, metadata_key=key)
        
        logger.info(f"\nEntity: {entity_id}")
        logger.info(f"Total Documents: {profile.total_documents}")
        logger.info(f"Overall Average: {profile.overall_average:.1f}%")
        logger.info(f"Trend: {profile.trend}")
        
        self.analytics.export_report(entity_id, metadata_key=key)

    def query_doc(self, doc_id: str, query: str):
        logger.info(f"Querying {doc_id}: {query}")
        result = self.query_system.query_single_document(doc_id, query)
        for i, res in enumerate(result.results, 1):
            logger.info(f"[{i}] Score: {res.score:.3f} | {res.text[:200]}...")

def main():
    parser = argparse.ArgumentParser(description="General Purpose RAG Pipeline")
    subparsers = parser.add_subparsers(dest='command')
    
    # batch-process
    bp = subparsers.add_parser('batch-process', help='Process documents')
    bp.add_argument('--dir', required=True)
    bp.add_argument('--name', required=True)
    bp.add_argument('--workers', type=int, default=4)
    
    # generate-reports
    gr = subparsers.add_parser('generate-reports', help='Generate evaluation reports')
    gr.add_argument('--batch-id', required=True)
    gr.add_argument('--rubric', default='essay', choices=['essay', 'short_answer'])
    gr.add_argument('--format', default='json', choices=['json', 'txt', 'csv'])
    
    # find-similar
    fs = subparsers.add_parser('find-similar', help='Find similar documents')
    fs.add_argument('--batch-id')
    fs.add_argument('--all', action='store_true')
    
    # analytics
    an = subparsers.add_parser('analytics', help='Generate entity analytics')
    an.add_argument('--id', required=True)
    an.add_argument('--key', default='author_id')
    
    # query
    q = subparsers.add_parser('query', help='Query a document')
    q.add_argument('--doc-id', required=True)
    q.add_argument('--query', required=True)
    
    # Internal list
    subparsers.add_parser('list-batches', help='List all batches')
    st = subparsers.add_parser('batch-status', help='Get batch status')
    st.add_argument('--batch-id', required=True)

    args = parser.parse_args()
    orch = RAGOrchestrator()
    
    if args.command == 'batch-process':
        orch.batch_process(Path(args.dir), args.name, args.workers)
    elif args.command == 'generate-reports':
        orch.generate_reports(args.batch_id, args.rubric, args.format)
    elif args.command == 'find-similar':
        orch.find_similar(args.batch_id, args.all)
    elif args.command == 'analytics':
        orch.run_analytics(args.id, args.key)
    elif args.command == 'query':
        orch.query_doc(args.doc_id, args.query)
    elif args.command == 'list-batches':
        batches = orch.batch_processor.list_batches()
        for b in batches: print(f"ID: {b['batch_id']} | Name: {b['name']} | Status: {b['status']}")
    elif args.command == 'batch-status':
        status = orch.batch_processor.get_batch_status(args.batch_id)
        print(json.dumps(status, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
