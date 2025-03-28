"""Export service for the LMS analyzer."""

import pandas as pd
from typing import Dict, Any, List
from pathlib import Path
from io import BytesIO
from datetime import datetime

from ..models.data_models import AnalysisResults
from ..config.analysis_params import EXPORT_CONFIG

class ExportService:
    """Service for exporting analysis results."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_to_excel(self, results: AnalysisResults) -> bytes:
        """Export analysis results to Excel format."""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Overview sheet
            self._write_overview_sheet(writer, results)
            
            # Quality metrics sheet
            if results.quality_metrics:
                self._write_quality_sheet(writer, results.quality_metrics)
            
            # Activity metrics sheet
            if results.activity_metrics:
                self._write_activity_sheet(writer, results.activity_metrics)
            
            # Similarity metrics sheet
            if results.similarity_metrics:
                self._write_similarity_sheet(writer, results.similarity_metrics)
            
            # Recommendations sheet
            if results.recommendations:
                self._write_recommendations_sheet(writer, results.recommendations)
            
            # Validation results sheet
            if results.validation_results:
                self._write_validation_sheet(writer, results.validation_results)
        
        return output.getvalue()
    
    def _write_overview_sheet(self, writer: pd.ExcelWriter, results: AnalysisResults) -> None:
        """Write overview metrics to Excel sheet."""
        overview_data = {
            "Metric": [
                "Total Courses",
                "Active Courses",
                "Total Learners",
                "Regions Covered",
                "Data Sources",
                "Cross-Referenced Courses"
            ],
            "Value": [
                results.total_courses,
                results.active_courses,
                results.total_learners or "N/A",
                results.regions_covered or "N/A",
                results.data_sources or "N/A",
                results.cross_referenced_courses or "N/A"
            ]
        }
        df = pd.DataFrame(overview_data)
        df.to_excel(writer, sheet_name="Overview", index=False)
        self._format_sheet(writer.sheets["Overview"])
    
    def _write_quality_sheet(self, writer: pd.ExcelWriter, metrics: Any) -> None:
        """Write quality metrics to Excel sheet."""
        quality_data = {
            "Metric": [
                "Completeness Score",
                "Metadata Score",
                "Content Score",
                "Overall Score"
            ],
            "Value": [
                metrics.completeness_score,
                metrics.metadata_score,
                metrics.content_score,
                metrics.overall_score
            ]
        }
        df = pd.DataFrame(quality_data)
        df.to_excel(writer, sheet_name="Quality Metrics", index=False)
        self._format_sheet(writer.sheets["Quality Metrics"])
    
    def _write_activity_sheet(self, writer: pd.ExcelWriter, metrics: Any) -> None:
        """Write activity metrics to Excel sheet."""
        activity_data = {
            "Metric": [
                "Active Courses",
                "Recent Completions",
                "Average Completion Rate",
                "Last Activity Date"
            ],
            "Value": [
                metrics.active_courses,
                metrics.recent_completions,
                metrics.average_completion_rate,
                metrics.last_activity_date
            ]
        }
        df = pd.DataFrame(activity_data)
        df.to_excel(writer, sheet_name="Activity Metrics", index=False)
        self._format_sheet(writer.sheets["Activity Metrics"])
    
    def _write_similarity_sheet(self, writer: pd.ExcelWriter, metrics: Any) -> None:
        """Write similarity metrics to Excel sheet."""
        similarity_data = {
            "Metric": [
                "Total Pairs",
                "High Similarity Pairs",
                "Cross-Department Pairs",
                "Average Similarity"
            ],
            "Value": [
                metrics.total_pairs,
                metrics.high_similarity_pairs,
                metrics.cross_department_pairs,
                metrics.average_similarity
            ]
        }
        df = pd.DataFrame(similarity_data)
        df.to_excel(writer, sheet_name="Similarity Metrics", index=False)
        self._format_sheet(writer.sheets["Similarity Metrics"])
    
    def _write_recommendations_sheet(self, writer: pd.ExcelWriter, recommendations: List[Any]) -> None:
        """Write recommendations to Excel sheet."""
        data = []
        for rec in recommendations:
            data.append({
                "Category": rec.category,
                "Title": rec.title,
                "Description": rec.description,
                "Priority": rec.priority,
                "Impact": rec.impact,
                "Effort": rec.effort,
                "Action Items": "\n".join(rec.action_items) if rec.action_items else "",
                "Related Courses": "\n".join(rec.related_courses) if rec.related_courses else ""
            })
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name="Recommendations", index=False)
        self._format_sheet(writer.sheets["Recommendations"])
    
    def _write_validation_sheet(self, writer: pd.ExcelWriter, validations: List[Any]) -> None:
        """Write validation results to Excel sheet."""
        data = []
        for val in validations:
            data.append({
                "Severity": val.severity,
                "Message": val.message
            })
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name="Validation Results", index=False)
        self._format_sheet(writer.sheets["Validation Results"])
    
    def _format_sheet(self, worksheet: Any) -> None:
        """Format Excel worksheet."""
        # Auto-adjust column widths
        for idx, col in enumerate(worksheet.columns):
            max_length = 0
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2 