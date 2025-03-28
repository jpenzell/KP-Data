"""Recommendations component for the LMS analyzer."""

import streamlit as st
from typing import List, Dict, Any
from src.models.data_models import AnalysisResults, Recommendation

def display_recommendations(recommendations: List[Recommendation]) -> None:
    """Display recommendations."""
    if not recommendations:
        st.info("No recommendations available")
        return
    
    # Group recommendations by category
    categories = {}
    for rec in recommendations:
        category = rec.category if hasattr(rec, 'category') and rec.category else "General"
        if category not in categories:
            categories[category] = []
        categories[category].append(rec)
    
    # Display recommendations by category
    for category, cat_recs in categories.items():
        with st.expander(f"{category} Recommendations", expanded=True):
            for i, rec in enumerate(cat_recs):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.markdown(f"**{rec.title}**")
                    st.markdown(rec.description)
                
                with col2:
                    impact = rec.impact if hasattr(rec, 'impact') else "Medium"
                    effort = rec.effort if hasattr(rec, 'effort') else "Medium"
                    
                    # Color-code the impact/effort
                    impact_color = {
                        "High": "green",
                        "Medium": "orange",
                        "Low": "red"
                    }.get(impact, "gray")
                    
                    effort_color = {
                        "High": "red",
                        "Medium": "orange",
                        "Low": "green"
                    }.get(effort, "gray")
                    
                    st.markdown(f"**Impact:** <span style='color:{impact_color}'>{impact}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Effort:** <span style='color:{effort_color}'>{effort}</span>", unsafe_allow_html=True)
                
                # Show implementation details if available
                if hasattr(rec, 'implementation') and rec.implementation:
                    with st.expander("Implementation Details"):
                        st.markdown(rec.implementation)
                
                # Show action items if available
                if hasattr(rec, 'action_items') and rec.action_items:
                    with st.expander("Action Items"):
                        for item in rec.action_items:
                            st.markdown(f"- {item}")
                
                # Show related courses if available
                if hasattr(rec, 'related_courses') and rec.related_courses:
                    with st.expander("Related Courses"):
                        for course in rec.related_courses:
                            st.markdown(f"- {course}")
                
                if i < len(cat_recs) - 1:
                    st.markdown("---")
    
    # Display summary metrics
    st.subheader("Recommendation Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Recommendations", len(recommendations))
    with col2:
        # Priority is now integer (1-10), consider 1-3 as high priority
        high_priority = sum(1 for r in recommendations if hasattr(r, 'priority') and r.priority <= 3)
        st.metric("High Priority", high_priority)
    with col3:
        categories_count = len(categories)
        st.metric("Categories", categories_count)
    
    # Export recommendations
    if st.button("Export Recommendations"):
        st.download_button(
            label="Download as Excel",
            data=export_recommendations_to_excel(recommendations),
            file_name="lms_recommendations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def export_recommendations_to_excel(recommendations: List[Recommendation]) -> bytes:
    """Export recommendations to Excel format."""
    import pandas as pd
    from io import BytesIO
    
    # Convert recommendations to DataFrame
    data = []
    for rec in recommendations:
        # Safe access to all fields with defaults
        rec_data = {
            "Category": getattr(rec, 'category', 'General'),
            "Title": rec.title,
            "Description": rec.description,
            "Priority": getattr(rec, 'priority', 5),  # Default medium priority
            "Impact": getattr(rec, 'impact', 'Medium'),
            "Effort": getattr(rec, 'effort', 'Medium'),
            "Implementation": getattr(rec, 'implementation', ''),
            "Action Items": "\n".join(getattr(rec, 'action_items', [])) if hasattr(rec, 'action_items') else "",
            "Related Courses": "\n".join(getattr(rec, 'related_courses', [])) if hasattr(rec, 'related_courses') else ""
        }
        data.append(rec_data)
    
    # Create DataFrame - handle empty case
    if not data:
        data = [{"Category": "", "Title": "No recommendations", "Description": "", "Priority": "", 
                "Impact": "", "Effort": "", "Implementation": "", "Action Items": "", "Related Courses": ""}]
    
    df = pd.DataFrame(data)
    
    # Create Excel writer
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Recommendations')
        
        # Auto-adjust column widths
        worksheet = writer.sheets['Recommendations']
        for idx, col in enumerate(df.columns):
            try:
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                )
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
            except:
                # Default width if calculation fails
                worksheet.column_dimensions[chr(65 + idx)].width = 15
    
    return output.getvalue() 