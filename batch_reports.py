"""
Batch Report Generation System

Generates comprehensive reports for multiple documents efficiently.
Optimized for grading 100+ submissions with:
- Bulk report generation
- Template-based evaluation
- Automated scoring
- Feedback standardization

USAGE:
    generator = BatchReportGenerator()
    
    # Generate reports for all docs in batch
    reports = generator.generate_batch_reports(
        doc_ids=["doc1", "doc2", ...],
        evaluation_criteria=["thesis", "evidence", "structure"],
        rubric=custom_rubric
    )
    
    # Export reports
    generator.export_reports(reports, format="json")
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from document_queries import DocumentQuerySystem
from generation import get_generator
from config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class EvaluationCriterion:
    """Single evaluation criterion"""
    name: str
    description: str
    weight: float  # 0.0 to 1.0
    max_score: int  # Maximum points
    query_template: str  # Query to extract info for this criterion


@dataclass
class DocumentReport:
    """Comprehensive report for a single document"""
    doc_id: str
    document_name: str
    student_id: Optional[str]
    generated_at: datetime
    overall_score: float
    max_score: float
    criteria_scores: Dict[str, Dict]  # criterion_name -> {score, feedback, evidence}
    summary: str
    strengths: List[str]
    areas_for_improvement: List[str]
    detailed_feedback: str
    metadata: Dict
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['generated_at'] = self.generated_at.isoformat()
        return d


# ==============================================================================
# DEFAULT RUBRICS
# ==============================================================================

ESSAY_RUBRIC = [
    EvaluationCriterion(
        name="thesis_statement",
        description="Clear, focused thesis statement",
        weight=0.2,
        max_score=20,
        query_template="What is the main thesis or argument? Is it clear and focused?"
    ),
    EvaluationCriterion(
        name="evidence_support",
        description="Quality and relevance of supporting evidence",
        weight=0.3,
        max_score=30,
        query_template="What evidence is provided? Is it relevant and well-integrated?"
    ),
    EvaluationCriterion(
        name="organization",
        description="Logical structure and flow",
        weight=0.2,
        max_score=20,
        query_template="How is the document organized? Is there clear structure and transitions?"
    ),
    EvaluationCriterion(
        name="analysis_depth",
        description="Critical thinking and analysis",
        weight=0.2,
        max_score=20,
        query_template="What level of analysis and critical thinking is demonstrated?"
    ),
    EvaluationCriterion(
        name="writing_quality",
        description="Grammar, style, and clarity",
        weight=0.1,
        max_score=10,
        query_template="Assess the writing quality, grammar, and clarity of expression."
    )
]

SHORT_ANSWER_RUBRIC = [
    EvaluationCriterion(
        name="accuracy",
        description="Correctness of answer",
        weight=0.5,
        max_score=50,
        query_template="Is the answer correct and accurate? What information is provided?"
    ),
    EvaluationCriterion(
        name="completeness",
        description="Thoroughness of response",
        weight=0.3,
        max_score=30,
        query_template="Is the answer complete? Are all parts of the question addressed?"
    ),
    EvaluationCriterion(
        name="clarity",
        description="Clear communication",
        weight=0.2,
        max_score=20,
        query_template="Is the answer clearly explained and easy to understand?"
    )
]


class BatchReportGenerator:
    """
    Generate reports for multiple documents efficiently
    
    Features:
    - Parallel report generation
    - Template-based evaluation
    - Customizable rubrics
    - Automated scoring
    - Export in multiple formats
    """
    
    def __init__(self):
        self.query_system = DocumentQuerySystem()
        self.generator = get_generator()
        logger.info("Initialized BatchReportGenerator")
    
    def generate_batch_reports(
        self,
        doc_ids: List[str],
        rubric: Optional[List[EvaluationCriterion]] = None,
        parallel_workers: int = 4,
        progress_callback: Optional[callable] = None
    ) -> List[DocumentReport]:
        """
        Generate reports for multiple documents
        
        Args:
            doc_ids: List of document IDs
            rubric: Evaluation rubric (defaults to essay rubric)
            parallel_workers: Number of parallel workers
            progress_callback: Optional callback(completed, total)
            
        Returns:
            List of DocumentReport objects
        """
        if rubric is None:
            rubric = ESSAY_RUBRIC
        
        logger.info(f"Generating reports for {len(doc_ids)} documents")
        logger.info(f"Using rubric with {len(rubric)} criteria")
        
        reports = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_doc = {
                executor.submit(self._generate_single_report, doc_id, rubric): doc_id
                for doc_id in doc_ids
            }
            
            for future in as_completed(future_to_doc):
                doc_id = future_to_doc[future]
                try:
                    report = future.result()
                    reports.append(report)
                    completed += 1
                    
                    if progress_callback:
                        progress_callback(completed, len(doc_ids))
                    
                    logger.info(f"Generated report {completed}/{len(doc_ids)}: {doc_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate report for {doc_id}: {e}")
                    completed += 1
        
        logger.info(f"Completed {len(reports)}/{len(doc_ids)} reports")
        return reports
    
    def _generate_single_report(
        self,
        doc_id: str,
        rubric: List[EvaluationCriterion]
    ) -> DocumentReport:
        """Generate report for a single document"""
        
        # Get document info
        doc_info = self.query_system.get_document_info(doc_id)
        if not doc_info:
            raise ValueError(f"Document {doc_id} not found")
        
        metadata = doc_info.get('metadata', {})
        
        # Evaluate each criterion
        criteria_scores = {}
        total_score = 0.0
        total_max = sum(c.max_score for c in rubric)
        
        for criterion in rubric:
            logger.debug(f"Evaluating {criterion.name} for {doc_id}")
            
            # Query document for this criterion
            query_result = self.query_system.query_single_document(
                doc_id,
                criterion.query_template,
                top_k=5
            )
            
            # Generate evaluation using LLM
            eval_result = self._evaluate_criterion(
                criterion,
                query_result.results,
                metadata
            )
            
            criteria_scores[criterion.name] = eval_result
            total_score += eval_result['score']
        
        # Generate overall summary
        summary = self._generate_summary(criteria_scores, rubric)
        strengths = self._extract_strengths(criteria_scores, rubric)
        improvements = self._extract_improvements(criteria_scores, rubric)
        detailed_feedback = self._generate_detailed_feedback(criteria_scores, rubric)
        
        report = DocumentReport(
            doc_id=doc_id,
            document_name=metadata.get('filename', doc_id),
            student_id=metadata.get('student_id'),
            generated_at=datetime.now(),
            overall_score=total_score,
            max_score=total_max,
            criteria_scores=criteria_scores,
            summary=summary,
            strengths=strengths,
            areas_for_improvement=improvements,
            detailed_feedback=detailed_feedback,
            metadata=metadata
        )
        
        return report
    
    def _evaluate_criterion(
        self,
        criterion: EvaluationCriterion,
        retrieved_chunks: List,
        doc_metadata: Dict
    ) -> Dict:
        """Evaluate a single criterion using LLM"""
        
        # Format context from retrieved chunks
        context = "\n\n".join([
            f"[{i+1}] {chunk.text}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        # Create evaluation prompt
        prompt = f"""Evaluate the following criterion for a student submission:

Criterion: {criterion.name}
Description: {criterion.description}
Maximum Score: {criterion.max_score}

Retrieved Text from Submission:
{context}

Provide your evaluation in JSON format:
{{
  "score": <number 0-{criterion.max_score}>,
  "feedback": "<specific feedback on this criterion>",
  "evidence": ["<quote 1>", "<quote 2>"],
  "justification": "<explanation of score>"
}}

Be specific and reference the text. Give partial credit where appropriate."""
        
        try:
            # Generate evaluation
            response = self.generator.generate(
                prompt=prompt,
                temperature=0.2,
                max_tokens=500
            )
            
            # Parse JSON response
            result = json.loads(response)
            
            # Validate and clamp score
            score = max(0, min(criterion.max_score, result.get('score', 0)))
            
            return {
                'score': score,
                'feedback': result.get('feedback', ''),
                'evidence': result.get('evidence', []),
                'justification': result.get('justification', ''),
                'weight': criterion.weight
            }
            
        except Exception as e:
            logger.error(f"Error evaluating criterion {criterion.name}: {e}")
            return {
                'score': 0,
                'feedback': f'Error during evaluation: {e}',
                'evidence': [],
                'justification': 'Evaluation failed',
                'weight': criterion.weight
            }
    
    def _generate_summary(
        self,
        criteria_scores: Dict,
        rubric: List[EvaluationCriterion]
    ) -> str:
        """Generate overall summary"""
        total_score = sum(s['score'] for s in criteria_scores.values())
        max_score = sum(c.max_score for c in rubric)
        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        
        return f"Overall Score: {total_score:.1f}/{max_score} ({percentage:.1f}%)"
    
    def _extract_strengths(
        self,
        criteria_scores: Dict,
        rubric: List[EvaluationCriterion]
    ) -> List[str]:
        """Extract strengths (criteria with high scores)"""
        strengths = []
        for criterion in rubric:
            score_info = criteria_scores.get(criterion.name, {})
            score = score_info.get('score', 0)
            # Consider strength if score is >= 80% of max
            if score >= criterion.max_score * 0.8:
                strengths.append(f"{criterion.description}: {score_info.get('feedback', '')}")
        
        return strengths if strengths else ["Good effort shown"]
    
    def _extract_improvements(
        self,
        criteria_scores: Dict,
        rubric: List[EvaluationCriterion]
    ) -> List[str]:
        """Extract areas for improvement (criteria with low scores)"""
        improvements = []
        for criterion in rubric:
            score_info = criteria_scores.get(criterion.name, {})
            score = score_info.get('score', 0)
            # Consider improvement area if score is < 70% of max
            if score < criterion.max_score * 0.7:
                improvements.append(f"{criterion.description}: {score_info.get('feedback', '')}")
        
        return improvements if improvements else ["Continue developing all areas"]
    
    def _generate_detailed_feedback(
        self,
        criteria_scores: Dict,
        rubric: List[EvaluationCriterion]
    ) -> str:
        """Generate detailed feedback text"""
        feedback_parts = []
        
        for criterion in rubric:
            score_info = criteria_scores.get(criterion.name, {})
            feedback_parts.append(
                f"**{criterion.description}** ({score_info['score']}/{criterion.max_score}): "
                f"{score_info.get('feedback', 'No feedback')}"
            )
        
        return "\n\n".join(feedback_parts)
    
    def export_reports(
        self,
        reports: List[DocumentReport],
        output_dir: Path = DATA_DIR / "reports",
        format: str = "json"
    ):
        """
        Export reports to files
        
        Args:
            reports: List of reports
            output_dir: Output directory
            format: Export format ("json", "txt", "csv")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            # Export as JSON
            output_file = output_dir / f"reports_{timestamp}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(
                    [r.to_dict() for r in reports],
                    f,
                    indent=2,
                    ensure_ascii=False
                )
            logger.info(f"Exported {len(reports)} reports to {output_file}")
        
        elif format == "txt":
            # Export as text files (one per report)
            for report in reports:
                filename = f"report_{report.doc_id}_{timestamp}.txt"
                output_file = output_dir / filename
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"EVALUATION REPORT\n")
                    f.write(f"=" * 60 + "\n\n")
                    f.write(f"Document: {report.document_name}\n")
                    f.write(f"Student ID: {report.student_id or 'N/A'}\n")
                    f.write(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"{report.summary}\n\n")
                    f.write(f"STRENGTHS:\n")
                    for strength in report.strengths:
                        f.write(f"  • {strength}\n")
                    f.write(f"\nAREAS FOR IMPROVEMENT:\n")
                    for area in report.areas_for_improvement:
                        f.write(f"  • {area}\n")
                    f.write(f"\n{'-' * 60}\n")
                    f.write(f"DETAILED FEEDBACK:\n\n")
                    f.write(report.detailed_feedback)
                
            logger.info(f"Exported {len(reports)} reports to {output_dir}")
        
        elif format == "csv":
            # Export as CSV
            import csv
            output_file = output_dir / f"reports_{timestamp}.csv"
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'Document ID', 'Document Name', 'Student ID',
                    'Overall Score', 'Max Score', 'Percentage',
                    'Summary', 'Strengths', 'Areas for Improvement'
                ])
                
                # Data
                for report in reports:
                    percentage = (report.overall_score / report.max_score * 100) if report.max_score > 0 else 0
                    writer.writerow([
                        report.doc_id,
                        report.document_name,
                        report.student_id or 'N/A',
                        f"{report.overall_score:.1f}",
                        report.max_score,
                        f"{percentage:.1f}%",
                        report.summary,
                        "; ".join(report.strengths),
                        "; ".join(report.areas_for_improvement)
                    ])
            
            logger.info(f"Exported {len(reports)} reports to {output_file}")
    
    def generate_comparative_report(
        self,
        reports: List[DocumentReport],
        output_file: Optional[Path] = None
    ) -> Dict:
        """
        Generate comparative statistics across all reports
        
        Args:
            reports: List of reports to analyze
            output_file: Optional file to save report
            
        Returns:
            Comparative statistics
        """
        if not reports:
            return {}
        
        total_max = reports[0].max_score if reports else 0
        scores = [r.overall_score for r in reports]
        percentages = [(s / total_max * 100) if total_max > 0 else 0 for s in scores]
        
        # Calculate statistics
        stats = {
            "total_documents": len(reports),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "average_percentage": sum(percentages) / len(percentages) if percentages else 0,
            "score_distribution": {
                "90-100%": sum(1 for p in percentages if p >= 90),
                "80-89%": sum(1 for p in percentages if 80 <= p < 90),
                "70-79%": sum(1 for p in percentages if 70 <= p < 80),
                "60-69%": sum(1 for p in percentages if 60 <= p < 70),
                "below_60%": sum(1 for p in percentages if p < 60),
            },
            "criteria_averages": {}
        }
        
        # Calculate per-criterion averages
        if reports:
            all_criteria = reports[0].criteria_scores.keys()
            for criterion in all_criteria:
                criterion_scores = [
                    r.criteria_scores.get(criterion, {}).get('score', 0)
                    for r in reports
                ]
                stats["criteria_averages"][criterion] = (
                    sum(criterion_scores) / len(criterion_scores)
                )
        
        # Save if requested
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved comparative report to {output_file}")
        
        return stats
