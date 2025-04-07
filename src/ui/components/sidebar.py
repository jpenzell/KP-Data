"""Sidebar component for the application."""
import streamlit as st

def render_sidebar():
    """Render the sidebar with application information and controls."""
    with st.sidebar:
        st.title("LMS Course Analyzer")
        
        st.markdown("### About")
        st.markdown("""
        This tool analyzes LMS course data to identify potential duplicates 
        and provide insights into your course catalog.
        
        **Features:**
        - Identify similar courses across departments
        - Detect potential duplicates
        - Analyze course quality and usage metrics
        - Generate recommendations for course consolidation
        """)
        
        st.markdown("### Enhanced with LLM")
        st.markdown("""
        This application uses local LLM technology to provide enhanced semantic similarity detection:

        - More accurate similarity detection
        - Understands meaning beyond just text matching
        - Identifies conceptually similar courses
        - Better handles different terminology for same concepts
        """)
        
        st.markdown("### Analysis Settings")
        if st.session_state.get('data_processed', False):
            st.write("You can adjust these settings to refine the analysis:")
            
            # Add settings controls
            st.checkbox("Show cross-department matches only", value=False, key="show_cross_dept")
            
            # Add help text
            st.info("These settings will apply to the similarity analysis tab.")
        
        # Add separator
        st.markdown("---")
        
        # Add credits
        st.markdown("**Developed by KP Data Team**")
        st.markdown("Version 2.1 with Semantic Analysis") 