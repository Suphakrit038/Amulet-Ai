"""
Enhanced UI Components for Amulet-AI Application
Modern, responsive UI components with Thai language support
"""

import streamlit as st
from typing import Dict, List, Optional, Any
import base64
from pathlib import Path
import logging

class UIComponents:
    """Enhanced UI components with modern design"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_css()
    
    def _load_css(self):
        """Load custom CSS styles"""
        css_path = Path(__file__).parent.parent / "assets" / "css" / "main.css"
        if css_path.exists():
            with open(css_path, 'r', encoding='utf-8') as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    def render_header(self, title: str, subtitle: str = "", logo_path: Optional[str] = None):
        """Render application header with logo and title"""
        header_html = f"""
        <div class="app-header fade-in">
            <h1>{title}</h1>
            {f'<p class="subtitle">{subtitle}</p>' if subtitle else ''}
        </div>
        """
        st.markdown(header_html, unsafe_allow_html=True)
    
    def render_card(self, title: str, content: str, icon: str = "📱", 
                   card_type: str = "default", actions: List[Dict] = None):
        """Render content card with modern styling"""
        
        type_classes = {
            "success": "card-success",
            "warning": "card-warning", 
            "error": "card-error",
            "info": "card-info",
            "default": ""
        }
        
        actions_html = ""
        if actions:
            actions_html = "<div class='card-actions'>"
            for action in actions:
                btn_class = f"btn btn-{action.get('type', 'primary')}"
                actions_html += f"""
                <button class="{btn_class}" onclick="{action.get('onclick', '')}">
                    {action.get('icon', '')} {action.get('label', '')}
                </button>
                """
            actions_html += "</div>"
        
        card_html = f"""
        <div class="card {type_classes.get(card_type, '')} slide-up">
            <div class="card-header">
                <h3 class="card-title">{icon} {title}</h3>
            </div>
            <div class="card-content">
                {content}
            </div>
            {actions_html}
        </div>
        """
        
        st.markdown(card_html, unsafe_allow_html=True)
    
    def render_upload_area(self, 
                          title: str = "อัปโหลดรูปภาพ",
                          subtitle: str = "ลากไฟล์มาวางหรือคลิกเพื่อเลือกไฟล์",
                          accepted_types: List[str] = None,
                          max_size: str = "10MB"):
        """Render enhanced file upload area"""
        
        if accepted_types is None:
            accepted_types = ["jpg", "jpeg", "png", "webp"]
        
        accepted_types_str = ', '.join(accepted_types).upper()
        upload_html = f"""
        <div class="upload-area" id="upload-area">
            <div class="upload-icon">📸</div>
            <div class="upload-text">{title}</div>
            <div class="upload-hint">{subtitle}</div>
            <div class="upload-info">
                รองรับไฟล์: {accepted_types_str} | ขนาดสูงสุด: {max_size}
            </div>
        </div>
        """
        
        st.markdown(upload_html, unsafe_allow_html=True)
    
    def render_progress_bar(self, progress: float, label: str = "", 
                          show_percentage: bool = True, color: str = "primary"):
        """Render animated progress bar"""
        
        percentage = min(100, max(0, progress * 100))
        
        progress_html = f"""
        <div class="progress-wrapper">
            {f'<div class="progress-label">{label}</div>' if label else ''}
            <div class="progress-container">
                <div class="progress-bar progress-{color}" 
                     style="width: {percentage}%"></div>
            </div>
            {f'<div class="progress-percentage">{percentage:.1f}%</div>' if show_percentage else ''}
        </div>
        """
        
        st.markdown(progress_html, unsafe_allow_html=True)
    
    def render_confidence_badge(self, confidence: float, 
                              labels: Dict[str, str] = None):
        """Render confidence level badge"""
        
        if labels is None:
            labels = {
                "high": "ความมั่นใจสูง",
                "medium": "ความมั่นใจปานกลาง", 
                "low": "ความมั่นใจต่ำ"
            }
        
        if confidence >= 0.8:
            badge_type = "high"
            icon = "🟢"
        elif confidence >= 0.6:
            badge_type = "medium"
            icon = "🟡"
        else:
            badge_type = "low"
            icon = "🔴"
        
        badge_html = f"""
        <div class="confidence-badge confidence-{badge_type}">
            {icon} {labels[badge_type]}: {confidence*100:.1f}%
        </div>
        """
        
        st.markdown(badge_html, unsafe_allow_html=True)
    
    def render_image_comparison(self, 
                              original_image: Any,
                              reference_images: List[Dict],
                              predictions: List[Dict]):
        """Render image comparison results"""
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔍 รูปภาพที่วิเคราะห์")
            st.image(original_image, use_column_width=True)
        
        with col2:
            st.subheader("🎯 ผลการเปรียบเทียบ")
            
            for i, pred in enumerate(predictions[:3]):  # Top 3 results
                with st.expander(f"อันดับ {i+1}: {pred['class']} ({pred['confidence']:.1f}%)", 
                               expanded=(i == 0)):
                    
                    col_ref, col_info = st.columns([1, 1])
                    
                    with col_ref:
                        if 'reference_image' in pred:
                            st.image(pred['reference_image'], 
                                   caption="รูปอ้างอิง",
                                   use_column_width=True)
                    
                    with col_info:
                        self.render_confidence_badge(pred['confidence'] / 100)
                        
                        if 'details' in pred:
                            st.write("**รายละเอียด:**")
                            for key, value in pred['details'].items():
                                st.write(f"• {key}: {value}")
    
    def render_analytics_dashboard(self, metrics: Dict):
        """Render analytics dashboard"""
        
        st.subheader("📊 ข้อมูลการใช้งาน")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.render_metric_card(
                "การเรียกใช้ API",
                metrics.get('total_api_calls', 0),
                metrics.get('api_calls_change', 0),
                "📡"
            )
        
        with col2:
            self.render_metric_card(
                "การทำนาย",
                metrics.get('total_predictions', 0),
                metrics.get('predictions_change', 0),
                "🔮"
            )
        
        with col3:
            self.render_metric_card(
                "ความแม่นยำเฉลี่ย",
                f"{metrics.get('avg_accuracy', 0):.1f}%",
                metrics.get('accuracy_change', 0),
                "🎯"
            )
        
        with col4:
            self.render_metric_card(
                "เวลาตอบสนองเฉลี่ย",
                f"{metrics.get('avg_response_time', 0):.2f}s",
                metrics.get('response_time_change', 0),
                "⚡"
            )
    
    def render_metric_card(self, title: str, value: Any, 
                          change: float, icon: str):
        """Render metric card with change indicator"""
        
        change_color = "success" if change >= 0 else "error"
        change_icon = "↗️" if change >= 0 else "↘️"
        
        metric_html = f"""
        <div class="metric-card card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-change status-{change_color}">
                {change_icon} {abs(change):.1f}%
            </div>
        </div>
        """
        
        st.markdown(metric_html, unsafe_allow_html=True)
    
    def render_notification(self, message: str, 
                          notification_type: str = "info",
                          auto_hide: bool = True,
                          duration: int = 5000):
        """Render notification toast"""
        
        type_icons = {
            "success": "✅",
            "warning": "⚠️",
            "error": "❌", 
            "info": "ℹ️"
        }
        
        notification_html = f"""
        <div class="notification notification-{notification_type} fade-in" 
             id="notification-{hash(message)}">
            <div class="notification-icon">{type_icons.get(notification_type, 'ℹ️')}</div>
            <div class="notification-message">{message}</div>
            <button class="notification-close" onclick="closeNotification('{hash(message)}')">&times;</button>
        </div>
        
        <script>
        function closeNotification(id) {{
            const notification = document.getElementById('notification-' + id);
            if (notification) {{
                notification.style.opacity = '0';
                setTimeout(() => notification.remove(), 300);
            }}
        }}
        
        {f'''
        setTimeout(() => {{
            closeNotification('{hash(message)}');
        }}, {duration});
        ''' if auto_hide else ''}
        </script>
        """
        
        st.markdown(notification_html, unsafe_allow_html=True)
    
    def render_search_interface(self, placeholder: str = "ค้นหาพระเครื่อง...",
                               filters: List[Dict] = None):
        """Render advanced search interface"""
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            search_query = st.text_input(
                "🔍 ค้นหา",
                placeholder=placeholder,
                help="ป้อนชื่อหรือคำอธิบายพระเครื่องที่ต้องการค้นหา"
            )
        
        with col2:
            search_button = st.button("ค้นหา", type="primary", use_container_width=True)
        
        if filters:
            with st.expander("🔧 ตัวกรองการค้นหา"):
                filter_cols = st.columns(len(filters))
                
                for i, filter_config in enumerate(filters):
                    with filter_cols[i]:
                        if filter_config['type'] == 'selectbox':
                            st.selectbox(
                                filter_config['label'],
                                filter_config['options'],
                                key=f"filter_{i}"
                            )
                        elif filter_config['type'] == 'slider':
                            st.slider(
                                filter_config['label'],
                                filter_config['min'],
                                filter_config['max'],
                                filter_config['default'],
                                key=f"filter_{i}"
                            )
        
        return search_query, search_button
    
    def render_loading_spinner(self, message: str = "กำลังประมวลผล..."):
        """Render loading spinner with message"""
        
        loading_html = f"""
        <div class="loading-container">
            <div class="loading-spinner pulse"></div>
            <div class="loading-message">{message}</div>
        </div>
        """
        
        return st.markdown(loading_html, unsafe_allow_html=True)


class ThaiUIHelpers:
    """Thai language specific UI helpers"""
    
    @staticmethod
    def format_thai_number(number: float, decimal_places: int = 2) -> str:
        """Format number with Thai locale"""
        thai_digits = "๐๑๒๓๔๕๖๗๘๙"
        english_digits = "0123456789"
        
        formatted = f"{number:.{decimal_places}f}"
        
        for eng, thai in zip(english_digits, thai_digits):
            formatted = formatted.replace(eng, thai)
        
        return formatted
    
    @staticmethod
    def get_thai_confidence_text(confidence: float) -> str:
        """Get Thai confidence level text"""
        if confidence >= 0.9:
            return "ความมั่นใจสูงมาก"
        elif confidence >= 0.8:
            return "ความมั่นใจสูง"
        elif confidence >= 0.7:
            return "ความมั่นใจดี"
        elif confidence >= 0.6:
            return "ความมั่นใจปานกลาง"
        elif confidence >= 0.5:
            return "ความมั่นใจพอใช้"
        else:
            return "ความมั่นใจต่ำ"
    
    @staticmethod
    def get_amulet_categories() -> Dict[str, str]:
        """Get Thai amulet categories"""
        return {
            "somdej": "สมเด็จ",
            "phra_rod": "พระรอด", 
            "phra_nang_phaya": "พระนางพญา",
            "phra_pidta": "พระปิดตา",
            "phra_kong": "พระกลม",
            "phra_chinnarat": "พระพุทธชินราช",
            "lp_tuad": "หลวงปู่ทวด",
            "wat_rakang": "วัดระฆัง",
            "wat_mahathat": "วัดมหาธาตุ",
            "other": "อื่นๆ"
        }
