"""
Per-Student Analytics System

Provides comprehensive analytics and insights for individual students:
- Performance tracking across assignments
- Writing style analysis
- Progress over time
- Strengths and weaknesses identification
- Personalized recommendations

FEATURES:
- Historical performance tracking
- Comparative analysis within cohort
- Skill progression mapping
- Automated insights generation

USAGE:
    analytics = StudentAnalytics()
    
    # Get student profile
    profile = analytics.get_student_profile(student_id="student_123")
    
    # Track progress
    progress = analytics.analyze_progress(student_id, assignments=["hw1", "hw2", "hw3"])
    
    # Generate insights
    insights = analytics.generate_insights(student_id)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import numpy as np

from document_queries import DocumentQuerySystem
from batch_reports import BatchReportGenerator, DocumentReport
from config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class AssignmentPerformance:
    """Performance on a single assignment"""
    assignment_id: str
    assignment_name: str
    doc_id: str
    score: float
    max_score: float
    percentage: float
    submitted_date: Optional[datetime]
    criteria_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['submitted_date'] = self.submitted_date.isoformat() if self.submitted_date else None
        return d


@dataclass
class StudentProfile:
    """Comprehensive student profile"""
    student_id: str
    student_name: Optional[str]
    total_assignments: int
    assignments: List[AssignmentPerformance]
    overall_average: float
    trend: str  # "improving", "declining", "stable"
    strongest_skills: List[str]
    areas_for_improvement: List[str]
    recommendations: List[str]
    generated_at: datetime
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['assignments'] = [a.to_dict() for a in self.assignments]
        d['generated_at'] = self.generated_at.isoformat()
        return d


@dataclass
class ProgressAnalysis:
    """Analysis of student progress over time"""
    student_id: str
    time_period: str
    score_progression: List[float]
    assignment_names: List[str]
    improvement_rate: float  # Percentage improvement from first to last
    consistency_score: float  # 0-1, how consistent performance is
    skill_progression: Dict[str, List[float]]  # Skill -> scores over time
    insights: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class StudentAnalytics:
    """
    Generate analytics and insights for individual students
    
    Tracks:
    - Performance across assignments
    - Skill development over time
    - Comparative performance
    - Personalized recommendations
    """
    
    def __init__(self):
        self.query_system = DocumentQuerySystem()
        self.report_generator = BatchReportGenerator()
        self.db_path = DATA_DIR / "student_analytics.json"
        self._load_analytics_db()
        logger.info("Initialized StudentAnalytics")
    
    def _load_analytics_db(self):
        """Load historical analytics data"""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                self.analytics_db = json.load(f)
        else:
            self.analytics_db = {
                "students": {},
                "assignments": {},
                "cohort_stats": {}
            }
    
    def _save_analytics_db(self):
        """Save analytics data"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.analytics_db, f, indent=2, ensure_ascii=False)
    
    def get_student_profile(
        self,
        student_id: str,
        include_recommendations: bool = True
    ) -> StudentProfile:
        """
        Get comprehensive student profile
        
        Args:
            student_id: Student identifier
            include_recommendations: Generate recommendations
            
        Returns:
            StudentProfile
        """
        logger.info(f"Generating profile for student: {student_id}")
        
        # Get all student's submissions
        doc_results = self.query_system.query_by_student(student_id, query="", top_k=1)
        
        if not doc_results:
            logger.warning(f"No submissions found for student {student_id}")
            return StudentProfile(
                student_id=student_id,
                student_name=None,
                total_assignments=0,
                assignments=[],
                overall_average=0.0,
                trend="stable",
                strongest_skills=[],
                areas_for_improvement=[],
                recommendations=[],
                generated_at=datetime.now()
            )
        
        # Get doc IDs
        doc_ids = [r.doc_id for r in doc_results]
        
        # Generate reports for all assignments
        reports = self.report_generator.generate_batch_reports(
            doc_ids,
            parallel_workers=2
        )
        
        # Build assignment performances
        assignments = []
        for report in reports:
            performance = self._report_to_performance(report)
            assignments.append(performance)
        
        # Sort by submission date
        assignments.sort(
            key=lambda x: x.submitted_date if x.submitted_date else datetime.min
        )
        
        # Calculate overall statistics
        overall_average = np.mean([a.percentage for a in assignments]) if assignments else 0.0
        
        # Analyze trend
        trend = self._analyze_trend([a.percentage for a in assignments])
        
        # Identify strongest skills
        strongest_skills = self._identify_strongest_skills(assignments)
        
        # Identify areas for improvement
        areas_for_improvement = self._identify_weak_areas(assignments)
        
        # Generate recommendations
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations(
                student_id,
                assignments,
                strongest_skills,
                areas_for_improvement,
                trend
            )
        
        profile = StudentProfile(
            student_id=student_id,
            student_name=self._get_student_name(student_id),
            total_assignments=len(assignments),
            assignments=assignments,
            overall_average=overall_average,
            trend=trend,
            strongest_skills=strongest_skills,
            areas_for_improvement=areas_for_improvement,
            recommendations=recommendations,
            generated_at=datetime.now()
        )
        
        # Save to analytics DB
        self.analytics_db["students"][student_id] = profile.to_dict()
        self._save_analytics_db()
        
        return profile
    
    def analyze_progress(
        self,
        student_id: str,
        assignment_ids: Optional[List[str]] = None
    ) -> ProgressAnalysis:
        """
        Analyze student's progress over time
        
        Args:
            student_id: Student identifier
            assignment_ids: Optional list of specific assignments to analyze
            
        Returns:
            ProgressAnalysis
        """
        logger.info(f"Analyzing progress for student: {student_id}")
        
        # Get student profile
        profile = self.get_student_profile(student_id, include_recommendations=False)
        
        if not profile.assignments:
            return ProgressAnalysis(
                student_id=student_id,
                time_period="N/A",
                score_progression=[],
                assignment_names=[],
                improvement_rate=0.0,
                consistency_score=0.0,
                skill_progression={},
                insights=["No assignments available for analysis"]
            )
        
        # Filter assignments if specified
        assignments = profile.assignments
        if assignment_ids:
            assignments = [a for a in assignments if a.assignment_id in assignment_ids]
        
        # Extract progression data
        score_progression = [a.percentage for a in assignments]
        assignment_names = [a.assignment_name for a in assignments]
        
        # Calculate improvement rate
        if len(score_progression) >= 2:
            first_score = score_progression[0]
            last_score = score_progression[-1]
            improvement_rate = ((last_score - first_score) / first_score * 100) if first_score > 0 else 0.0
        else:
            improvement_rate = 0.0
        
        # Calculate consistency
        consistency_score = self._calculate_consistency(score_progression)
        
        # Track skill progression
        skill_progression = self._track_skill_progression(assignments)
        
        # Generate insights
        insights = self._generate_progress_insights(
            score_progression,
            improvement_rate,
            consistency_score,
            skill_progression
        )
        
        # Determine time period
        if assignments:
            first_date = assignments[0].submitted_date
            last_date = assignments[-1].submitted_date
            if first_date and last_date:
                time_period = f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}"
            else:
                time_period = "Unknown"
        else:
            time_period = "N/A"
        
        return ProgressAnalysis(
            student_id=student_id,
            time_period=time_period,
            score_progression=score_progression,
            assignment_names=assignment_names,
            improvement_rate=improvement_rate,
            consistency_score=consistency_score,
            skill_progression=skill_progression,
            insights=insights
        )
    
    def compare_with_cohort(
        self,
        student_id: str,
        cohort_stats: Optional[Dict] = None
    ) -> Dict:
        """
        Compare student performance with cohort averages
        
        Args:
            student_id: Student to compare
            cohort_stats: Optional pre-computed cohort statistics
            
        Returns:
            Comparison dictionary
        """
        profile = self.get_student_profile(student_id, include_recommendations=False)
        
        if cohort_stats is None:
            cohort_stats = self.analytics_db.get("cohort_stats", {})
        
        cohort_avg = cohort_stats.get("average_score", 70.0)
        cohort_std = cohort_stats.get("std_deviation", 15.0)
        
        # Calculate z-score
        if cohort_std > 0:
            z_score = (profile.overall_average - cohort_avg) / cohort_std
        else:
            z_score = 0.0
        
        # Determine percentile (approximate)
        from scipy import stats
        try:
            percentile = stats.norm.cdf(z_score) * 100
        except:
            # Fallback if scipy not available
            percentile = 50.0
        
        # Performance category
        if profile.overall_average >= cohort_avg + cohort_std:
            category = "Above Average"
        elif profile.overall_average >= cohort_avg - cohort_std:
            category = "Average"
        else:
            category = "Below Average"
        
        return {
            "student_id": student_id,
            "student_average": profile.overall_average,
            "cohort_average": cohort_avg,
            "difference": profile.overall_average - cohort_avg,
            "z_score": z_score,
            "percentile": percentile,
            "category": category,
            "total_assignments": profile.total_assignments
        }
    
    def generate_insights(
        self,
        student_id: str
    ) -> List[str]:
        """Generate automated insights for a student"""
        profile = self.get_student_profile(student_id)
        progress = self.analyze_progress(student_id)
        
        insights = []
        
        # Performance insights
        if profile.overall_average >= 90:
            insights.append("🌟 Excellent overall performance across all assignments")
        elif profile.overall_average >= 80:
            insights.append("✅ Strong performance with room for growth")
        elif profile.overall_average >= 70:
            insights.append("👍 Satisfactory performance, focus on improvement areas")
        else:
            insights.append("⚠️ Performance below expectations, additional support recommended")
        
        # Trend insights
        if progress.improvement_rate > 10:
            insights.append(f"📈 Significant improvement shown ({progress.improvement_rate:.1f}% increase)")
        elif progress.improvement_rate < -10:
            insights.append(f"📉 Declining performance ({progress.improvement_rate:.1f}% decrease) - intervention may be needed")
        
        # Consistency insights
        if progress.consistency_score > 0.8:
            insights.append("🎯 Highly consistent performance")
        elif progress.consistency_score < 0.5:
            insights.append("⚡ Inconsistent performance - may benefit from study routine")
        
        # Skill-specific insights
        for skill, scores in progress.skill_progression.items():
            if len(scores) >= 2:
                improvement = scores[-1] - scores[0]
                if improvement > 10:
                    insights.append(f"✨ Strong improvement in {skill}")
                elif improvement < -10:
                    insights.append(f"⚠️ Declining performance in {skill}")
        
        return insights
    
    def export_student_report(
        self,
        student_id: str,
        output_dir: Path = DATA_DIR / "student_reports"
    ):
        """Export comprehensive student report"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all components
        profile = self.get_student_profile(student_id)
        progress = self.analyze_progress(student_id)
        insights = self.generate_insights(student_id)
        
        # Create report
        report = {
            "student_profile": profile.to_dict(),
            "progress_analysis": progress.to_dict(),
            "insights": insights,
            "generated_at": datetime.now().isoformat()
        }
        
        # Save as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_dir / f"student_{student_id}_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save as text report
        txt_file = output_dir / f"student_{student_id}_{timestamp}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"STUDENT ANALYTICS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Student ID: {student_id}\n")
            f.write(f"Total Assignments: {profile.total_assignments}\n")
            f.write(f"Overall Average: {profile.overall_average:.1f}%\n")
            f.write(f"Trend: {profile.trend}\n\n")
            
            f.write(f"STRONGEST SKILLS:\n")
            for skill in profile.strongest_skills:
                f.write(f"  ✓ {skill}\n")
            
            f.write(f"\nAREAS FOR IMPROVEMENT:\n")
            for area in profile.areas_for_improvement:
                f.write(f"  • {area}\n")
            
            f.write(f"\nINSIGHTS:\n")
            for insight in insights:
                f.write(f"  {insight}\n")
            
            f.write(f"\nRECOMMENDATIONS:\n")
            for rec in profile.recommendations:
                f.write(f"  → {rec}\n")
            
            f.write(f"\n" + "-" * 80 + "\n")
            f.write(f"PROGRESS ANALYSIS:\n")
            f.write(f"  Time Period: {progress.time_period}\n")
            f.write(f"  Improvement Rate: {progress.improvement_rate:.1f}%\n")
            f.write(f"  Consistency Score: {progress.consistency_score:.2f}\n")
        
        logger.info(f"Exported student report to {output_dir}")
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _report_to_performance(self, report: DocumentReport) -> AssignmentPerformance:
        """Convert DocumentReport to AssignmentPerformance"""
        return AssignmentPerformance(
            assignment_id=report.metadata.get("assignment_id", report.doc_id),
            assignment_name=report.metadata.get("assignment_name", report.document_name),
            doc_id=report.doc_id,
            score=report.overall_score,
            max_score=report.max_score,
            percentage=(report.overall_score / report.max_score * 100) if report.max_score > 0 else 0,
            submitted_date=None,  # Would come from metadata
            criteria_scores={k: v['score'] for k, v in report.criteria_scores.items()},
            strengths=report.strengths,
            weaknesses=report.areas_for_improvement
        )
    
    def _analyze_trend(self, scores: List[float]) -> str:
        """Analyze score trend"""
        if len(scores) < 2:
            return "stable"
        
        # Simple linear regression
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        
        if slope > 2:
            return "improving"
        elif slope < -2:
            return "declining"
        else:
            return "stable"
    
    def _identify_strongest_skills(self, assignments: List[AssignmentPerformance]) -> List[str]:
        """Identify consistently strong skills"""
        if not assignments:
            return []
        
        skill_scores = defaultdict(list)
        for assignment in assignments:
            for skill, score in assignment.criteria_scores.items():
                skill_scores[skill].append(score)
        
        # Find skills with average >= 80%
        strong_skills = []
        for skill, scores in skill_scores.items():
            if np.mean(scores) >= 0.8 * 100:  # Assuming scores are out of 100
                strong_skills.append(skill)
        
        return strong_skills[:3]  # Top 3
    
    def _identify_weak_areas(self, assignments: List[AssignmentPerformance]) -> List[str]:
        """Identify areas needing improvement"""
        if not assignments:
            return []
        
        from collections import defaultdict
        skill_scores = defaultdict(list)
        for assignment in assignments:
            for skill, score in assignment.criteria_scores.items():
                skill_scores[skill].append(score)
        
        # Find skills with average < 70%
        weak_areas = []
        for skill, scores in skill_scores.items():
            if np.mean(scores) < 0.7 * 100:
                weak_areas.append(skill)
        
        return weak_areas[:3]  # Top 3
    
    def _generate_recommendations(
        self,
        student_id: str,
        assignments: List[AssignmentPerformance],
        strengths: List[str],
        weaknesses: List[str],
        trend: str
    ) -> List[str]:
        """Generate personalized recommendations"""
        recommendations = []
        
        if trend == "declining":
            recommendations.append("Schedule one-on-one meeting to discuss recent challenges")
        
        for weakness in weaknesses:
            recommendations.append(f"Provide additional resources/practice for {weakness}")
        
        if not strengths:
            recommendations.append("Focus on building foundational skills")
        
        if len(assignments) < 3:
            recommendations.append("Encourage consistent submission of assignments")
        
        return recommendations
    
    def _calculate_consistency(self, scores: List[float]) -> float:
        """Calculate consistency score (0-1)"""
        if len(scores) < 2:
            return 1.0
        
        # Use coefficient of variation (inverted and normalized)
        std = np.std(scores)
        mean = np.mean(scores)
        
        if mean == 0:
            return 0.0
        
        cv = std / mean
        # Normalize: CV of 0 = perfect consistency (1.0), CV of 0.5+ = low consistency (0.0)
        consistency = max(0, 1 - (cv / 0.5))
        
        return consistency
    
    def _track_skill_progression(
        self,
        assignments: List[AssignmentPerformance]
    ) -> Dict[str, List[float]]:
        """Track progression of individual skills"""
        skill_progression = defaultdict(list)
        
        for assignment in assignments:
            for skill, score in assignment.criteria_scores.items():
                skill_progression[skill].append(score)
        
        return dict(skill_progression)
    
    def _generate_progress_insights(
        self,
        scores: List[float],
        improvement_rate: float,
        consistency: float,
        skill_progression: Dict[str, List[float]]
    ) -> List[str]:
        """Generate insights from progress analysis"""
        insights = []
        
        if improvement_rate > 15:
            insights.append("Excellent progress shown over time")
        elif improvement_rate < -15:
            insights.append("Performance has declined - may need intervention")
        
        if consistency > 0.8:
            insights.append("Performance is highly consistent")
        elif consistency < 0.5:
            insights.append("Performance varies significantly between assignments")
        
        return insights
    
    def _get_student_name(self, student_id: str) -> Optional[str]:
        """Get student name from metadata (if available)"""
        # Would retrieve from metadata in real implementation
        return None
