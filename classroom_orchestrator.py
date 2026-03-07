"""
Google Classroom Integration Orchestrator

Unified command-line interface for all Google Classroom features:
- Batch processing of submissions
- Document-specific queries
- Plagiarism detection
- Student analytics
- Report generation
- Progress tracking

USAGE:
    # Process batch of submissions
    python classroom_orchestrator.py batch-process --dir ./submissions --assignment "Essay 1"
    
    # Generate reports for all students
    python classroom_orchestrator.py generate-reports --batch-id batch_20260130_120000
    
    # Check plagiarism
    python classroom_orchestrator.py check-plagiarism --batch-id batch_20260130_120000
    
    # Student analytics
    python classroom_orchestrator.py student-analytics --student-id student_123
    
    # Query specific document
    python classroom_orchestrator.py query --doc-id doc_123 --query "What is the thesis?"
"""

import logging
import argparse
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

from batch_processor import BatchProcessor, BatchStatus
from batch_reports import BatchReportGenerator, ESSAY_RUBRIC, SHORT_ANSWER_RUBRIC
from plagiarism_detector import PlagiarismDetector
from student_analytics import StudentAnalytics
from document_queries import DocumentQuerySystem
from config import RAW_DOCS_DIR, DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClassroomOrchestrator:
    """
    Main orchestrator for Google Classroom integration
    
    Coordinates all classroom-related operations:
    - Batch processing
    - Report generation
    - Plagiarism detection
    - Analytics generation
    """
    
    def __init__(self):
        self.batch_processor = BatchProcessor()
        self.report_generator = BatchReportGenerator()
        self.plagiarism_detector = PlagiarismDetector()
        self.student_analytics = StudentAnalytics()
        self.query_system = DocumentQuerySystem()
        logger.info("Initialized ClassroomOrchestrator")
    
    def batch_process_submissions(
        self,
        directory: Path,
        assignment_name: str,
        parallel_workers: int = 4
    ) -> str:
        """
        Process a batch of student submissions
        
        Args:
            directory: Directory containing submission files
            assignment_name: Name of the assignment
            parallel_workers: Number of parallel workers
            
        Returns:
            batch_id for tracking
        """
        logger.info(f"Starting batch processing for: {assignment_name}")
        logger.info(f"Directory: {directory}")
        
        # Find all supported files
        from config import SUPPORTED_FORMATS
        file_paths = []
        for ext in SUPPORTED_FORMATS:
            file_paths.extend(directory.glob(f"*{ext}"))
        
        if not file_paths:
            logger.error(f"No supported files found in {directory}")
            return None
        
        logger.info(f"Found {len(file_paths)} submissions")
        
        # Create batch
        batch_id = self.batch_processor.create_batch(
            file_paths=[str(p) for p in file_paths],
            batch_name=assignment_name,
            metadata={"assignment_name": assignment_name}
        )
        
        logger.info(f"Created batch: {batch_id}")
        
        # Process batch with progress callback
        def progress_callback(batch):
            logger.info(f"Progress: {batch.completed_documents}/{batch.total_documents} "
                       f"({batch.progress*100:.1f}%)")
        
        self.batch_processor.process_batch(
            batch_id,
            parallel_workers=parallel_workers,
            progress_callback=progress_callback
        )
        
        logger.info(f"Batch processing completed: {batch_id}")
        return batch_id
    
    def generate_batch_reports(
        self,
        batch_id: str,
        rubric_type: str = "essay",
        parallel_workers: int = 4,
        export_format: str = "json"
    ):
        """
        Generate evaluation reports for all documents in a batch
        
        Args:
            batch_id: Batch identifier
            rubric_type: "essay" or "short_answer"
            parallel_workers: Number of parallel workers
            export_format: Export format ("json", "txt", "csv")
        """
        logger.info(f"Generating reports for batch: {batch_id}")
        
        # Get batch status
        status = self.batch_processor.get_batch_status(batch_id)
        if not status:
            logger.error(f"Batch {batch_id} not found")
            return
        
        # Get successfully processed doc IDs
        doc_ids = []
        for job in status['jobs']:
            if job['status'] == 'completed' and job['doc_id']:
                doc_ids.append(job['doc_id'])
        
        if not doc_ids:
            logger.error("No successfully processed documents found")
            return
        
        logger.info(f"Generating reports for {len(doc_ids)} documents")
        
        # Select rubric
        rubric = ESSAY_RUBRIC if rubric_type == "essay" else SHORT_ANSWER_RUBRIC
        
        # Generate reports
        def progress_callback(completed, total):
            logger.info(f"Report generation: {completed}/{total}")
        
        reports = self.report_generator.generate_batch_reports(
            doc_ids=doc_ids,
            rubric=rubric,
            parallel_workers=parallel_workers,
            progress_callback=progress_callback
        )
        
        # Export reports
        output_dir = DATA_DIR / "reports" / batch_id
        self.report_generator.export_reports(
            reports,
            output_dir=output_dir,
            format=export_format
        )
        
        # Generate comparative report
        comparative_stats = self.report_generator.generate_comparative_report(
            reports,
            output_file=output_dir / "comparative_stats.json"
        )
        
        logger.info(f"Reports generated and exported to {output_dir}")
        logger.info(f"Average score: {comparative_stats['average_score']:.1f}/{reports[0].max_score if reports else 0}")
        
        return reports
    
    def check_plagiarism(
        self,
        batch_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        check_all: bool = False
    ):
        """
        Check for plagiarism across submissions
        
        Args:
            batch_id: Check all docs in this batch
            doc_ids: Specific document IDs to check
            check_all: Check all documents in system
        """
        logger.info("Starting plagiarism detection")
        
        if check_all:
            logger.info("Checking all submissions for similarities")
            similarity_matrix = self.plagiarism_detector.check_all_submissions(
                min_similarity=0.5
            )
            
            # Generate reports for flagged documents
            reports = []
            for doc_id, similarities in similarity_matrix.items():
                report = self.plagiarism_detector.generate_plagiarism_report(
                    doc_id,
                    similarity_results=similarities
                )
                if report.flagged:
                    reports.append(report)
            
            logger.info(f"Found {len(reports)} documents with high similarity")
            
        elif batch_id:
            # Get docs from batch
            status = self.batch_processor.get_batch_status(batch_id)
            if not status:
                logger.error(f"Batch {batch_id} not found")
                return
            
            doc_ids = [j['doc_id'] for j in status['jobs'] 
                      if j['status'] == 'completed' and j['doc_id']]
            
            logger.info(f"Checking {len(doc_ids)} documents from batch")
            
            reports = []
            for doc_id in doc_ids:
                report = self.plagiarism_detector.generate_plagiarism_report(doc_id)
                reports.append(report)
        
        elif doc_ids:
            reports = []
            for doc_id in doc_ids:
                report = self.plagiarism_detector.generate_plagiarism_report(doc_id)
                reports.append(report)
        
        else:
            logger.error("Must specify batch_id, doc_ids, or check_all=True")
            return
        
        # Export reports
        self.plagiarism_detector.export_plagiarism_reports(reports)
        
        # Summary
        flagged_count = sum(1 for r in reports if r.flagged)
        logger.info(f"Plagiarism check complete: {flagged_count}/{len(reports)} documents flagged")
        
        return reports
    
    def generate_student_analytics(
        self,
        student_id: str,
        export: bool = True
    ):
        """
        Generate comprehensive analytics for a student
        
        Args:
            student_id: Student identifier
            export: Export report to file
        """
        logger.info(f"Generating analytics for student: {student_id}")
        
        # Generate profile
        profile = self.student_analytics.get_student_profile(student_id)
        
        # Generate progress analysis
        progress = self.student_analytics.analyze_progress(student_id)
        
        # Generate insights
        insights = self.student_analytics.generate_insights(student_id)
        
        # Print summary
        logger.info(f"\nStudent: {student_id}")
        logger.info(f"Total Assignments: {profile.total_assignments}")
        logger.info(f"Overall Average: {profile.overall_average:.1f}%")
        logger.info(f"Trend: {profile.trend}")
        logger.info(f"\nInsights:")
        for insight in insights:
            logger.info(f"  {insight}")
        
        # Export if requested
        if export:
            self.student_analytics.export_student_report(student_id)
        
        return profile, progress, insights
    
    def query_document(
        self,
        doc_id: str,
        query: str,
        top_k: int = 5
    ):
        """
        Query a specific document
        
        Args:
            doc_id: Document ID
            query: Query text
            top_k: Number of results
        """
        logger.info(f"Querying document: {doc_id}")
        logger.info(f"Query: {query}")
        
        result = self.query_system.query_single_document(doc_id, query, top_k)
        
        logger.info(f"\nResults ({result.retrieved_chunks} chunks):")
        for i, res in enumerate(result.results, 1):
            logger.info(f"\n[{i}] Score: {res.score:.3f}")
            logger.info(f"Text: {res.text[:200]}...")
        
        return result
    
    def list_batches(self):
        """List all batches"""
        batches = self.batch_processor.list_batches()
        
        logger.info(f"\nTotal Batches: {len(batches)}\n")
        for batch in batches:
            logger.info(f"Batch ID: {batch['batch_id']}")
            logger.info(f"  Name: {batch['name']}")
            logger.info(f"  Status: {batch['status']}")
            logger.info(f"  Documents: {batch['completed_documents']}/{batch['total_documents']}")
            logger.info(f"  Created: {batch['created_at']}\n")
        
        return batches
    
    def get_batch_status(self, batch_id: str):
        """Get detailed batch status"""
        status = self.batch_processor.get_batch_status(batch_id)
        
        if not status:
            logger.error(f"Batch {batch_id} not found")
            return None
        
        logger.info(f"\nBatch: {batch_id}")
        logger.info(f"Name: {status['batch']['name']}")
        logger.info(f"Status: {status['batch']['status']}")
        logger.info(f"Progress: {status['batch']['progress']*100:.1f}%")
        logger.info(f"\nSummary:")
        logger.info(f"  Total: {status['summary']['total']}")
        logger.info(f"  Completed: {status['summary']['completed']}")
        logger.info(f"  Failed: {status['summary']['failed']}")
        logger.info(f"  Pending: {status['summary']['pending']}")
        
        return status


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Google Classroom Integration - RAG Pipeline Orchestrator"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Batch processing
    batch_parser = subparsers.add_parser('batch-process', help='Process batch of submissions')
    batch_parser.add_argument('--dir', type=str, required=True, help='Directory with submissions')
    batch_parser.add_argument('--assignment', type=str, required=True, help='Assignment name')
    batch_parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    
    # Generate reports
    reports_parser = subparsers.add_parser('generate-reports', help='Generate evaluation reports')
    reports_parser.add_argument('--batch-id', type=str, required=True, help='Batch ID')
    reports_parser.add_argument('--rubric', type=str, default='essay', choices=['essay', 'short_answer'])
    reports_parser.add_argument('--format', type=str, default='json', choices=['json', 'txt', 'csv'])
    reports_parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    
    # Plagiarism check
    plag_parser = subparsers.add_parser('check-plagiarism', help='Check for plagiarism')
    plag_parser.add_argument('--batch-id', type=str, help='Batch ID')
    plag_parser.add_argument('--all', action='store_true', help='Check all submissions')
    
    # Student analytics
    analytics_parser = subparsers.add_parser('student-analytics', help='Generate student analytics')
    analytics_parser.add_argument('--student-id', type=str, required=True, help='Student ID')
    
    # Query document
    query_parser = subparsers.add_parser('query', help='Query a document')
    query_parser.add_argument('--doc-id', type=str, required=True, help='Document ID')
    query_parser.add_argument('--query', type=str, required=True, help='Query text')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of results')
    
    # List batches
    subparsers.add_parser('list-batches', help='List all batches')
    
    # Batch status
    status_parser = subparsers.add_parser('batch-status', help='Get batch status')
    status_parser.add_argument('--batch-id', type=str, required=True, help='Batch ID')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = ClassroomOrchestrator()
    
    # Execute command
    if args.command == 'batch-process':
        batch_id = orchestrator.batch_process_submissions(
            directory=Path(args.dir),
            assignment_name=args.assignment,
            parallel_workers=args.workers
        )
        print(f"\n✓ Batch processing complete: {batch_id}")
    
    elif args.command == 'generate-reports':
        orchestrator.generate_batch_reports(
            batch_id=args.batch_id,
            rubric_type=args.rubric,
            parallel_workers=args.workers,
            export_format=args.format
        )
        print(f"\n✓ Reports generated for batch: {args.batch_id}")
    
    elif args.command == 'check-plagiarism':
        orchestrator.check_plagiarism(
            batch_id=args.batch_id,
            check_all=args.all
        )
        print("\n✓ Plagiarism check complete")
    
    elif args.command == 'student-analytics':
        orchestrator.generate_student_analytics(
            student_id=args.student_id,
            export=True
        )
        print(f"\n✓ Analytics generated for student: {args.student_id}")
    
    elif args.command == 'query':
        orchestrator.query_document(
            doc_id=args.doc_id,
            query=args.query,
            top_k=args.top_k
        )
    
    elif args.command == 'list-batches':
        orchestrator.list_batches()
    
    elif args.command == 'batch-status':
        orchestrator.get_batch_status(args.batch_id)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
