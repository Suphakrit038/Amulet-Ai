# ===== LAYOUT COMPONENTS =====

import streamlit as st
from typing import List, Dict, Any, Optional
from .mystical_components import mystical_card

def create_sidebar_navigation() -> Dict[str, Any]:
    """
    Create a mystical sidebar navigation with enhanced styling
    
    Returns:
        Dict containing navigation selections and states
    """
    with st.sidebar:
        # Sidebar Header
        st.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">
                <span class="logo-icon">🏺</span>
                <div class="logo-text">
                    <div class="logo-title">AMULET AI</div>
                    <div class="logo-subtitle">古董佛牌分析系统</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation Menu
        st.markdown('<div class="nav-section">', unsafe_allow_html=True)
        st.markdown("### 🔍 **Analysis Tools**")
        
        analysis_mode = st.selectbox(
            "Select Analysis Mode",
            ["Single Image Analysis", "Dual Image Comparison", "Batch Processing"],
            key="analysis_mode"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Model Selection
        st.markdown('<div class="nav-section">', unsafe_allow_html=True)
        st.markdown("### 🧠 **AI Model**")
        
        model_option = st.selectbox(
            "Choose Model",
            ["Somdej-FatherGuay Model", "General Amulet Model", "Advanced Ensemble"],
            key="model_selection"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            key="confidence_threshold"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Advanced Settings
        with st.expander("⚙️ **Advanced Settings**"):
            show_debug = st.checkbox("Show Debug Info", key="show_debug")
            auto_enhance = st.checkbox("Auto Image Enhancement", value=True, key="auto_enhance")
            save_results = st.checkbox("Save Analysis Results", key="save_results")
            
            # Processing Options
            st.markdown("**Processing Options**")
            preprocessing = st.multiselect(
                "Image Preprocessing",
                ["Noise Reduction", "Contrast Enhancement", "Edge Detection", "Color Normalization"],
                default=["Contrast Enhancement"],
                key="preprocessing_options"
            )
        
        # Information Cards
        st.markdown('<div class="nav-section">', unsafe_allow_html=True)
        st.markdown("### 📊 **System Status**")
        
        # System metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", "94.2%", "↑2.1%")
        with col2:
            st.metric("Analysis Speed", "1.2s", "↓0.3s")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Recent Activity
        with st.expander("📈 **Recent Activity**"):
            recent_activities = [
                {"time": "2 min ago", "action": "Analyzed Somdej amulet", "result": "98.5% confidence"},
                {"time": "5 min ago", "action": "Model update completed", "result": "Success"},
                {"time": "1 hour ago", "action": "Batch analysis finished", "result": "25 items processed"},
            ]
            
            for activity in recent_activities:
                st.markdown(f"""
                <div class="activity-item">
                    <div class="activity-time">{activity['time']}</div>
                    <div class="activity-action">{activity['action']}</div>
                    <div class="activity-result">{activity['result']}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Help & Support
        st.markdown('<div class="nav-section">', unsafe_allow_html=True)
        st.markdown("### ❓ **Help & Support**")
        
        help_options = st.selectbox(
            "Need Help?",
            ["Getting Started", "Analysis Tips", "Model Information", "Technical Support"],
            key="help_selection"
        )
        
        if st.button("📚 View Documentation", key="view_docs"):
            st.info("Documentation will open in a new tab")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div class="sidebar-footer">
            <div class="footer-info">
                <div class="version">Version 2.1.0</div>
                <div class="build">Build: 20241228</div>
            </div>
            <div class="footer-links">
                <a href="#" class="footer-link">About</a> • 
                <a href="#" class="footer-link">Privacy</a> • 
                <a href="#" class="footer-link">Terms</a>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    return {
        "analysis_mode": analysis_mode,
        "model_option": model_option,
        "confidence_threshold": confidence_threshold,
        "show_debug": show_debug,
        "auto_enhance": auto_enhance,
        "save_results": save_results,
        "preprocessing": preprocessing,
        "help_selection": help_options
    }

def create_main_layout(content_sections: List[Dict[str, Any]]) -> None:
    """
    Create the main content layout with mystical styling
    
    Args:
        content_sections: List of content sections with type, title, and content
    """
    for section in content_sections:
        section_type = section.get("type", "default")
        title = section.get("title")
        content = section.get("content")
        
        if section_type == "hero":
            create_hero_section(title, content)
        elif section_type == "upload":
            create_upload_section(title, content)
        elif section_type == "results":
            create_results_section(title, content)
        elif section_type == "comparison":
            create_comparison_section(title, content)
        else:
            mystical_card(content, title)

def create_hero_section(title: str, content: Dict[str, Any]) -> None:
    """Create a mystical hero section"""
    subtitle = content.get("subtitle", "")
    description = content.get("description", "")
    features = content.get("features", [])
    
    hero_html = f"""
    <div class="hero-section">
        <div class="hero-background">
            <div class="floating-particles"></div>
        </div>
        <div class="hero-content">
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
            <p class="hero-description">{description}</p>
        </div>
    </div>
    """
    st.markdown(hero_html, unsafe_allow_html=True)
    
    if features:
        cols = st.columns(len(features))
        for i, feature in enumerate(features):
            with cols[i]:
                feature_html = f"""
                <div class="feature-card">
                    <div class="feature-icon">{feature.get('icon', '✨')}</div>
                    <h3 class="feature-title">{feature.get('title', '')}</h3>
                    <p class="feature-description">{feature.get('description', '')}</p>
                </div>
                """
                st.markdown(feature_html, unsafe_allow_html=True)

def create_upload_section(title: str, content: Dict[str, Any]) -> None:
    """Create upload section with dual zones"""
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
    
    upload_type = content.get("type", "single")
    
    if upload_type == "dual":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="upload-container">
                <h3>📸 First Image</h3>
            </div>
            """, unsafe_allow_html=True)
            
            file1 = st.file_uploader(
                "Upload first image",
                type=["png", "jpg", "jpeg"],
                key="upload_1",
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("""
            <div class="upload-container">
                <h3>📷 Second Image</h3>
            </div>
            """, unsafe_allow_html=True)
            
            file2 = st.file_uploader(
                "Upload second image",
                type=["png", "jpg", "jpeg"],
                key="upload_2",
                label_visibility="collapsed"
            )
        
        return {"file1": file1, "file2": file2}
    
    else:
        # Single upload
        st.markdown("""
        <div class="upload-container-single">
            <div class="upload-zone">
                <div class="upload-icon">🏺</div>
                <h3>Drop your Buddhist amulet image here</h3>
                <p>or click to browse files</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        return st.file_uploader(
            "Upload amulet image",
            type=["png", "jpg", "jpeg"],
            key="upload_single",
            label_visibility="collapsed"
        )

def create_results_section(title: str, results: List[Dict[str, Any]]) -> None:
    """Create results display section"""
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
    
    if not results:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🔍</div>
            <h3>No Results Yet</h3>
            <p>Upload an image to begin analysis</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    for i, result in enumerate(results):
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if result.get("image"):
                    st.image(result["image"], caption=f"Analysis {i+1}", use_column_width=True)
            
            with col2:
                result_html = f"""
                <div class="result-summary">
                    <h3 class="result-type">{result.get('type', 'Unknown')}</h3>
                    <div class="confidence-display">
                        <span class="confidence-label">Confidence:</span>
                        <span class="confidence-value">{result.get('confidence', 0):.1%}</span>
                    </div>
                    <div class="result-details">
                        <div class="detail-grid">
                """
                
                details = result.get("details", {})
                for key, value in details.items():
                    result_html += f"""
                        <div class="detail-item">
                            <span class="detail-key">{key}:</span>
                            <span class="detail-value">{value}</span>
                        </div>
                    """
                
                result_html += """
                        </div>
                    </div>
                </div>
                """
                st.markdown(result_html, unsafe_allow_html=True)

def create_comparison_section(title: str, comparison_data: Dict[str, Any]) -> None:
    """Create comparison results section"""
    st.markdown(f"<h2 class='section-title'>{title}</h2>", unsafe_allow_html=True)
    
    similarity_score = comparison_data.get("similarity", 0)
    image1_data = comparison_data.get("image1", {})
    image2_data = comparison_data.get("image2", {})
    
    # Similarity Score Display
    st.markdown(f"""
    <div class="comparison-header">
        <div class="similarity-meter">
            <h3>Similarity Score</h3>
            <div class="similarity-score">{similarity_score:.1%}</div>
            <div class="similarity-bar">
                <div class="similarity-fill" style="width: {similarity_score*100}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Side by side comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>🖼️ First Image</h4>", unsafe_allow_html=True)
        if image1_data.get("image"):
            st.image(image1_data["image"], use_column_width=True)
        
        details1 = image1_data.get("analysis", {})
        details_html1 = "<div class='comparison-details'>"
        for key, value in details1.items():
            details_html1 += f"<div><strong>{key}:</strong> {value}</div>"
        details_html1 += "</div>"
        st.markdown(details_html1, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4>🖼️ Second Image</h4>", unsafe_allow_html=True)
        if image2_data.get("image"):
            st.image(image2_data["image"], use_column_width=True)
        
        details2 = image2_data.get("analysis", {})
        details_html2 = "<div class='comparison-details'>"
        for key, value in details2.items():
            details_html2 += f"<div><strong>{key}:</strong> {value}</div>"
        details_html2 += "</div>"
        st.markdown(details_html2, unsafe_allow_html=True)

def create_footer() -> None:
    """Create mystical footer"""
    footer_html = """
    <div class="mystical-footer">
        <div class="footer-content">
            <div class="footer-section">
                <h4>Amulet AI</h4>
                <p>Advanced Buddhist amulet analysis powered by artificial intelligence</p>
            </div>
            <div class="footer-section">
                <h4>Technology</h4>
                <ul>
                    <li>Deep Learning Models</li>
                    <li>Computer Vision</li>
                    <li>Image Recognition</li>
                </ul>
            </div>
            <div class="footer-section">
                <h4>Support</h4>
                <ul>
                    <li>Documentation</li>
                    <li>API Reference</li>
                    <li>Community</li>
                </ul>
            </div>
        </div>
        <div class="footer-bottom">
            <p>&copy; 2024 Amulet AI. All rights reserved.</p>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
