"""
Annotation Tool V2 - Legacy Compatibility Module

This module provides backward compatibility for Docker environments
that may still reference the old annotation_tool_v2 module.

For current functionality, please use the enhanced annotation viewer
or the main annotation tool in streamlit_app.py
"""

import streamlit as st


def show_annotation_tool_v2():
    """
    Legacy annotation tool function for backward compatibility.

    This function redirects users to the current annotation interfaces
    and prevents UnboundLocalError in environments running old code.
    """

    st.header("🔄 Annotation Tool V2 - Legacy Redirect")

    st.warning(
        """
        **⚠️ You are accessing a legacy annotation tool interface.**
        
        This version has been replaced with improved tools. Please use:
        - **📝 Annotation Tool** - Main annotation interface with drawing canvas
        - **🎨 Enhanced Annotation Viewer** - Multi-source annotation visualization
        """
    )

    st.markdown("---")

    # Provide navigation options
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Main Annotation Tool")
        st.markdown(
            """
            **Features:**
            - Interactive drawing canvas
            - Bounding box creation
            - Tag assignment interface
            - Save annotations to database
            """
        )

        if st.button("🎯 Go to Annotation Tool", use_container_width=True):
            st.info("Please navigate to '📝 Annotation Tool' in the sidebar.")

    with col2:
        st.subheader("🎨 Enhanced Annotation Viewer")
        st.markdown(
            """
            **Features:**
            - Multi-source annotation display
            - AI predictions visualization
            - Comparison tools
            - Statistical analysis
            """
        )

        if st.button("🔍 Go to Enhanced Viewer", use_container_width=True):
            st.info(
                "Please navigate to '🎨 Enhanced Annotation Viewer' in the sidebar."
            )

    st.markdown("---")

    # Technical information
    with st.expander("🔧 Technical Information"):
        st.markdown(
            """
            **Why am I seeing this page?**
            
            This legacy compatibility module was created to handle references to 
            `annotation_tool_v2.py` that may exist in cached Docker containers or 
            older configurations.
            
            **To fix this permanently:**
            1. Rebuild your Docker containers: `docker-compose build --no-cache`
            2. Restart the services: `docker-compose up -d`
            3. Clear any cached configurations
            
            **Current working tools:**
            - Main Annotation Tool (streamlit_app.py)
            - Enhanced Annotation Viewer (pages/enhanced_annotation_viewer.py)
            """
        )


if __name__ == "__main__":
    show_annotation_tool_v2()
