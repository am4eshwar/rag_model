"""
General Metadata-Based Analytics System

Provides comprehensive analytics and insights for any entity (Author, Source, Group, etc.):
- Performance tracking across batches
- Metadata-driven analysis
- Progress over time
- Strengths and weaknesses identification
- Automated insights generation

FEATURES:
- Historical performance tracking
- Comparative analysis within cohort/group
- Skill progression mapping
- Automated insights generation

USAGE:
    analytics = EntityAnalytics()
    
    # Get profile for an entity (e.g., an author)
    profile = analytics.get_entity_profile(entity_id="author_123", metadata_key="author_id")
    
    # Track progress
    progress = analytics.analyze_progress(entity_id, metadata_key="author_id")
    
    # Generate insights
    insights = analytics.generate_insights(entity_id)
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json
import numpy as np
from collections import defaultdict

from document_queries import DocumentQuerySystem
from batch_reports import BatchReportGenerator, DocumentReport
from config import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class PerformancePoint:
    """Performance on a single document/assignment"""
    batch_id: str
    batch_name: str
    doc_id: str
    score: float
    max_score: float
    percentage: float
    date: Optional[datetime]
    criteria_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['date'] = self.date.isoformat() if self.date else None
        return d


@dataclass
class EntityProfile:
    """Comprehensive entity profile (Author, Source, etc.)"""
    entity_id: str
    metadata_key: str
    total_documents: int
    performances: List[PerformancePoint]
    overall_average: float
    trend: str  # "improving", "declining", "stable"
    strongest_skills: List[str]
    areas_for_improvement: List[str]
    recommendations: List[str]
    generated_at: datetime
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['performances'] = [p.to_dict() for p in self.performances]
        d['generated_at'] = self.generated_at.isoformat()
        return d


@dataclass
class ProgressAnalysis:
    """Analysis of progress over time"""
    entity_id: str
    time_period: str
    score_progression: List[float]
    document_names: List[str]
    improvement_rate: float
    consistency_score: float
    skill_progression: Dict[str, List[float]]
    insights: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class EntityAnalytics:
    """
    Generate analytics and insights for any entity identified by metadata
    """
    
    def __init__(self):
        self.query_system = DocumentQuerySystem()
        self.report_generator = BatchReportGenerator()
        self.db_path = DATA_DIR / "entity_analytics.json"
        self._load_analytics_db()
        logger.info("Initialized EntityAnalytics")
    
    def _load_analytics_db(self):
        """Load historical analytics data"""
        if self.db_path.exists():
            with open(self.db_path, 'r', encoding='utf-8') as f:
                self.analytics_db = json.load(f)
        else:
            self.analytics_db = {
                "entities": {},
                "cohort_stats": {}
            }
    
    def _save_analytics_db(self):
        """Save analytics data"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.analytics_db, f, indent=2, ensure_ascii=False)
    
    def get_entity_profile(
        self,
        entity_id: str,
        metadata_key: str = "author_id",
        include_recommendations: bool = True
    ) -> EntityProfile:
        """
        Get comprehensive profile for an entity
        """
        logger.info(f"Generating profile for {metadata_key}: {entity_id}")
        
        # Get all entity's documents
        doc_results = self.query_system.query_by_metadata(metadata_key, entity_id, query="", top_k=1)
        
        if not doc_results:
            logger.warning(f"No documents found for {metadata_key} {entity_id}")
            return EntityProfile(
                entity_id=entity_id,
                metadata_key=metadata_key,
                total_documents=0,
                performances=[],
                overall_average=0.0,
                trend="stable",
                strongest_skills=[],
                areas_for_improvement=[],
                recommendations=[],
                generated_at=datetime.now()
            )
        
        doc_ids = [r.doc_id for r in doc_results]
        reports = self.report_generator.generate_batch_reports(doc_ids, parallel_workers=2)
        
        performances = []
        for report in reports:
            perf = self._report_to_performance(report)
            performances.append(perf)
        
        # Sort by date if available, else by doc_id
        performances.sort(key=lambda x: x.date if x.date else datetime.min)
        
        overall_average = np.mean([p.percentage for p in performances]) if performances else 0.0
        trend = self._analyze_trend([p.percentage for p in performances])
        strongest_skills = self._identify_strongest_skills(performances)
        weak_areas = self._identify_weak_areas(performances)
        
        recommendations = []
        if include_recommendations:
            recommendations = self._generate_recommendations(performances, strongest_skills, weak_areas, trend)
        
        profile = EntityProfile(
            entity_id=entity_id,
            metadata_key=metadata_key,
            total_documents=len(performances),
            performances=performances,
            overall_average=overall_average,
            trend=trend,
            strongest_skills=strongest_skills,
            areas_for_improvement=weak_areas,
            recommendations=recommendations,
            generated_at=datetime.now()
        )
        
        # Save to DB
        self.analytics_db["entities"][f"{metadata_key}:{entity_id}"] = profile.to_dict()
        self._save_analytics_db()
        
        return profile

    def analyze_progress(
        self,
        entity_id: str,
        metadata_key: str = "author_id"
    ) -> ProgressAnalysis:
        """Analyze progress over time"""
        profile = self.get_entity_profile(entity_id, metadata_key, include_recommendations=False)
        
        if not profile.performances:
            return ProgressAnalysis(entity_id=entity_id, time_period="N/A", score_progression=[], 
                                    document_names=[], improvement_rate=0.0, consistency_score=0.0, 
                                    skill_progression={}, insights=["No data available"])
        
        perfs = profile.performances
        score_progression = [p.percentage for p in perfs]
        document_names = [p.batch_name for p in perfs] # Or doc name
        
        improvement_rate = 0.0
        if len(score_progression) >= 2 and score_progression[0] > 0:
            improvement_rate = ((score_progression[-1] - score_progression[0]) / score_progression[0] * 100)
            
        consistency_score = self._calculate_consistency(score_progression)
        skill_progression = self._track_skill_progression(perfs)
        
        insights = self._generate_insights_list(score_progression, improvement_rate, consistency_score)
        
        time_period = "N/A"
        if perfs[0].date and perfs[-1].date:
            time_period = f"{perfs[0].date.strftime('%Y-%m-%d')} to {perfs[-1].date.strftime('%Y-%m-%d')}"
            
        return ProgressAnalysis(
            entity_id=entity_id,
            time_period=time_period,
            score_progression=score_progression,
            document_names=document_names,
            improvement_rate=improvement_rate,
            consistency_score=consistency_score,
            skill_progression=skill_progression,
            insights=insights
        )

    def _report_to_performance(self, report: DocumentReport) -> PerformancePoint:
        return PerformancePoint(
            batch_id=report.metadata.get("batch_id", "unknown"),
            batch_name=report.metadata.get("batch_name", report.document_name),
            doc_id=report.doc_id,
            score=report.overall_score,
            max_score=report.max_score,
            percentage=(report.overall_score / report.max_score * 100) if report.max_score > 0 else 0,
            date=None, # Extract from metadata if exists
            criteria_scores={k: v['score'] for k, v in report.criteria_scores.items()},
            strengths=report.strengths,
            weaknesses=report.areas_for_improvement
        )

    def _analyze_trend(self, scores: List[float]) -> str:
        if len(scores) < 2: return "stable"
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        if slope > 2: return "improving"
        elif slope < -2: return "declining"
        return "stable"

    def _identify_strongest_skills(self, perfs: List[PerformancePoint]) -> List[str]:
        if not perfs: return []
        skill_scores = defaultdict(list)
        for p in perfs:
            for skill, score in p.criteria_scores.items():
                skill_scores[skill].append(score)
        strong = [s for s, scores in skill_scores.items() if np.mean(scores) >= 80]
        return strong[:3]

    def _identify_weak_areas(self, perfs: List[PerformancePoint]) -> List[str]:
        if not perfs: return []
        skill_scores = defaultdict(list)
        for p in perfs:
            for skill, score in p.criteria_scores.items():
                skill_scores[skill].append(score)
        weak = [s for s, scores in skill_scores.items() if np.mean(scores) < 70]
        return weak[:3]

    def _calculate_consistency(self, scores: List[float]) -> float:
        if len(scores) < 2: return 1.0
        std = np.std(scores)
        mean = np.mean(scores)
        if mean == 0: return 0.0
        return max(0, 1 - ((std / mean) / 0.5))

    def _track_skill_progression(self, perfs: List[PerformancePoint]) -> Dict[str, List[float]]:
        prog = defaultdict(list)
        for p in perfs:
            for s, score in p.criteria_scores.items():
                prog[s].append(score)
        return dict(prog)

    def _generate_insights_list(self, scores, improvement, consistency) -> List[str]:
        insights = []
        if improvement > 10: insights.append("Significant improvement detected.")
        if consistency > 0.8: insights.append("High consistency in results.")
        elif consistency < 0.5: insights.append("Results vary significantly.")
        return insights

    def _generate_recommendations(self, perfs, strong, weak, trend) -> List[str]:
        recs = []
        if trend == "declining": recs.append("Review recent drop in performance.")
        for w in weak: recs.append(f"Focus on improving {w}.")
        return recs

    def export_report(self, entity_id: str, metadata_key: str = "author_id", output_dir: Path = DATA_DIR / "entity_reports"):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        profile = self.get_entity_profile(entity_id, metadata_key)
        progress = self.analyze_progress(entity_id, metadata_key)
        
        report = {"profile": profile.to_dict(), "progress": progress.to_dict(), "generated_at": datetime.now().isoformat()}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(output_dir / f"{entity_id}_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported report for {entity_id} to {output_dir}")
