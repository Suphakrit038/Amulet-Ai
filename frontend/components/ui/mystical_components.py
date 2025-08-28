# ===== UI COMPONENTS =====

import streamlit as st
from typing import Optional, List, Dict, Any

def mystical_header(title: str, subtitle: str = None, icon: str = "🏺") -> None:
    """
    Create a mystical header with golden glow effect
    """
    header_html = f"""
    <div class="mystical-header">
        <div class="header-glow"></div>
        <h1 class="header-title">
            <span class="header-icon">{icon}</span>
            {title}
        </h1>
        {f'<p class="header-subtitle">{subtitle}</p>' if subtitle else ''}
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def mystical_card(content: str, title: str = None, variant: str = "default") -> None:
    """
    Create a mystical glassmorphic card
    
    Args:
        content: HTML or markdown content
        title: Optional card title
        variant: card style variant (default, primary, warning, success)
    """
    title_html = f'<h3 class="card-title">{title}</h3>' if title else ''
    
    card_html = f"""
    <div class="glass-card glass-card-{variant}">
        {title_html}
        <div class="card-content">
            {content}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def mystical_button(
    label: str, 
    key: str = None, 
    variant: str = "mystical",
    size: str = "base",
    icon: str = None
) -> bool:
    """
    Create a mystical styled button
    
    Args:
        label: Button text
        key: Unique key for Streamlit
        variant: Button style (mystical, primary, secondary, ghost)
        size: Button size (sm, base, lg, xl)
        icon: Optional icon
    
    Returns:
        bool: True if button was clicked
    """
    icon_html = f'<span class="btn-icon">{icon}</span>' if icon else ''
    size_class = f"btn-{size}" if size != "base" else ""
    
    button_html = f"""
    <button class="btn-base btn-{variant} {size_class}" data-key="{key or label}">
        {icon_html}
        <span>{label}</span>
    </button>
    """
    
    # Use Streamlit's button with custom styling
    return st.button(
        label, 
        key=key,
        help=None,
        use_container_width=False
    )

def mystical_progress(
    value: float, 
    text: str = None,
    show_percentage: bool = True,
    color: str = "primary"
) -> None:
    """
    Create a mystical progress bar
    
    Args:
        value: Progress value between 0 and 1
        text: Optional progress text
        show_percentage: Whether to show percentage
        color: Progress bar color theme
    """
    percentage = int(value * 100)
    progress_text = text or f"Processing... {percentage}%"
    
    progress_html = f"""
    <div class="progress-container">
        <div class="progress-label">{progress_text}</div>
        <div class="progress-mystical">
            <div class="progress-fill progress-{color}" style="width: {percentage}%"></div>
        </div>
        {f'<div class="progress-percentage">{percentage}%</div>' if show_percentage else ''}
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)

def mystical_alert(
    message: str,
    type: str = "info",
    title: str = None,
    dismissible: bool = False
) -> None:
    """
    Create a mystical alert/notification
    
    Args:
        message: Alert message
        type: Alert type (info, success, warning, error)
        title: Optional alert title
        dismissible: Whether alert can be dismissed
    """
    icons = {
        "info": "ℹ️",
        "success": "✅", 
        "warning": "⚠️",
        "error": "❌"
    }
    
    icon = icons.get(type, "ℹ️")
    title_html = f'<div class="alert-title">{icon} {title}</div>' if title else ''
    dismiss_btn = '<button class="alert-dismiss">×</button>' if dismissible else ''
    
    alert_html = f"""
    <div class="alert-base alert-{type}">
        {dismiss_btn}
        {title_html}
        <div class="alert-message">{message}</div>
    </div>
    """
    st.markdown(alert_html, unsafe_allow_html=True)

def mystical_upload_zone(
    accept: List[str] = None,
    max_size: int = 200,
    multiple: bool = False,
    key: str = None,
    help_text: str = "Drag and drop images here or click to browse"
) -> Any:
    """
    Create a mystical file upload zone
    
    Args:
        accept: List of accepted file types
        max_size: Maximum file size in MB
        multiple: Allow multiple files
        key: Unique key for Streamlit
        help_text: Help text to display
    
    Returns:
        Uploaded file(s) or None
    """
    accept_types = accept or ["png", "jpg", "jpeg"]
    
    upload_html = f"""
    <div class="upload-zone">
        <div class="upload-icon">📸</div>
        <div class="upload-text">
            <strong>Upload Buddhist Amulet Images</strong>
            <br>
            <span class="upload-help">{help_text}</span>
        </div>
        <div class="upload-specs">
            Max size: {max_size}MB • Types: {', '.join(accept_types).upper()}
        </div>
    </div>
    """
    st.markdown(upload_html, unsafe_allow_html=True)
    
    return st.file_uploader(
        "Upload Images",
        type=accept_types,
        accept_multiple_files=multiple,
        key=key,
        label_visibility="collapsed"
    )

def mystical_metrics(metrics: List[Dict[str, Any]]) -> None:
    """
    Display metrics in mystical cards
    
    Args:
        metrics: List of metric dictionaries with keys: label, value, delta, delta_color
    """
    cols = st.columns(len(metrics))
    
    for i, metric in enumerate(metrics):
        with cols[i]:
            label = metric.get('label', 'Metric')
            value = metric.get('value', 0)
            delta = metric.get('delta')
            delta_color = metric.get('delta_color', 'normal')
            
            delta_html = f'<div class="metric-delta metric-delta-{delta_color}">{delta}</div>' if delta else ''
            
            metric_html = f"""
            <div class="glass-card metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                {delta_html}
            </div>
            """
            st.markdown(metric_html, unsafe_allow_html=True)

def mystical_tabs(tabs: List[str], default: int = 0) -> str:
    """
    Create mystical styled tabs
    
    Args:
        tabs: List of tab labels
        default: Default active tab index
    
    Returns:
        str: Selected tab label
    """
    return st.tabs(tabs)

def confidence_indicator(confidence: float, label: str = "Confidence") -> None:
    """
    Display a mystical confidence indicator
    
    Args:
        confidence: Confidence value between 0 and 1
        label: Label for the indicator
    """
    percentage = int(confidence * 100)
    
    # Determine confidence level
    if confidence >= 0.9:
        level = "excellent"
        color = "#22c55e"
        icon = "🌟"
    elif confidence >= 0.8:
        level = "very-good"
        color = "#84cc16"
        icon = "⭐"
    elif confidence >= 0.7:
        level = "good"
        color = "#eab308"
        icon = "👍"
    elif confidence >= 0.6:
        level = "fair"
        color = "#f97316"
        icon = "⚡"
    else:
        level = "low"
        color = "#ef4444"
        icon = "⚠️"
    
    confidence_html = f"""
    <div class="confidence-indicator confidence-{level}">
        <div class="confidence-header">
            <span class="confidence-icon">{icon}</span>
            <span class="confidence-label">{label}</span>
            <span class="confidence-value">{percentage}%</span>
        </div>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {percentage}%; background: {color};"></div>
        </div>
        <div class="confidence-level">Level: {level.replace('-', ' ').title()}</div>
    </div>
    """
    st.markdown(confidence_html, unsafe_allow_html=True)

def result_display_card(
    title: str,
    confidence: float,
    details: Dict[str, Any],
    image_url: str = None
) -> None:
    """
    Display analysis results in a mystical card
    
    Args:
        title: Result title
        confidence: Confidence score
        details: Additional details dictionary
        image_url: Optional image URL
    """
    image_html = f'<img src="{image_url}" class="result-image" alt="Analysis Result">' if image_url else ''
    
    details_html = ""
    for key, value in details.items():
        details_html += f'<div class="result-detail"><span class="detail-key">{key}:</span> <span class="detail-value">{value}</span></div>'
    
    result_html = f"""
    <div class="glass-card result-card">
        <div class="result-header">
            <h3 class="result-title">{title}</h3>
            <div class="result-confidence">
                <div class="confidence-score">{int(confidence * 100)}%</div>
                <div class="confidence-bar-mini">
                    <div class="confidence-fill-mini" style="width: {int(confidence * 100)}%"></div>
                </div>
            </div>
        </div>
        {image_html}
        <div class="result-details">
            {details_html}
        </div>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
