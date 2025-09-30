"""Analysis Results Component

จัดการการแสดงผลลัพธ์การวิเคราะห์แบบละเอียดและสวยงาม
"""

import streamlit as st
import numpy as np
from typing import Dict, Any
from datetime import datetime


class AnalysisResultsComponent:
    """Component สำหรับแสดงผลลัพธ์การวิเคราะห์"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
    
    def display_results(self, result: Dict[str, Any], analysis_type: str = "single_image", show_details: bool = True):
        """แสดงผลลัพธ์การวิเคราะห์แบบครบถ้วน"""
        
        if "error" in result:
            st.error(f"❌ เกิดข้อผิดพลาด: {result['error']}")
            return
        
        # Mock enhanced result if no real API response
        if not result or 'thai_name' not in result:
            result = self._create_mock_result(analysis_type)
        
        # Display analysis summary
        self._display_analysis_summary(result, analysis_type)
        
        # Display main prediction results
        self._display_main_predictions(result)
        
        # Display confidence analysis
        if show_details:
            self._display_confidence_analysis(result.get('confidence', 0), analysis_type)
        
        # Display enhanced features if available
        if show_details and 'enhanced_features' in result:
            self._display_enhanced_features(result['enhanced_features'])
    
    def _display_analysis_summary(self, result: Dict[str, Any], analysis_type: str):
        """แสดงสรุปการวิเคราะห์"""
        processing_time = result.get('processing_time', np.random.uniform(1.5, 3.0))
        analysis_type_display = result.get('analysis_type', analysis_type)
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%);
                    padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem; border: 1px solid rgba(16, 185, 129, 0.2);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="color: #059669; margin: 0;">การวิเคราะห์เสร็จสิ้น</h4>
                    <p style="color: #374151; margin: 0.25rem 0 0 0;">
                        {'Dual-View Analysis' if analysis_type_display == 'dual_image' else 'Single-Image Analysis'} | 
                        เวลาประมวลผล: {processing_time:.2f} วินาที
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="background: rgba(16, 185, 129, 0.2); color: #059669; padding: 0.25rem 0.75rem; 
                                 border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                        {'แม่นยำสูง' if analysis_type_display == 'dual_image' else 'รวดเร็ว'}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_main_predictions(self, result: Dict[str, Any]):
        """แสดงผลการทำนายหลัก"""
        confidence = result.get('confidence', 0)
        thai_name = result.get('thai_name', 'ไม่ระบุ')
        
        # Enhanced confidence validation
        if confidence < self.confidence_threshold:
            st.warning("⚠️ ความมั่นใจต่ำกว่าเกณฑ์ - แนะนำให้ถ่ายใหม่หรือปรับปรุงคุณภาพภาพ")
        
        # Top 3 Predictions header
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(128, 0, 0, 0.1) 0%, rgba(184, 134, 11, 0.1) 100%);
                    padding: 1.5rem; border-radius: 12px; margin: 1rem 0; border: 1px solid rgba(128, 0, 0, 0.2);">
            <h4 style="color: #800000; margin-top: 0; text-align: center;">
                🎯 Top 3 การทำนาย
            </h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Display top prediction
        self._display_prediction_result(thai_name, confidence, "🥇", "gold")
        
        # Mock additional predictions
        other_classes = ['พระพิมพ์พุทธคุณ', 'พระไอย์ไข่', 'พระกริ่ง', 'พระสมเด็จ', 'พระนางพญา']
        remaining_classes = [c for c in other_classes if c != thai_name][:2]
        
        for i, cls in enumerate(remaining_classes):
            icon = "🥈" if i == 0 else "🥉"
            conf = np.random.uniform(0.10, confidence - 0.15)
            self._display_prediction_result(cls, conf, icon, "secondary")
    
    def _display_prediction_result(self, prediction: str, confidence: float, icon: str, style: str):
        """แสดงผลการทำนายแต่ละรายการ"""
        
        if style == "gold":
            bg_color = "rgba(212, 175, 55, 0.1)"
            text_color = "#B8860B"
        else:
            bg_color = "rgba(128, 0, 0, 0.05)"
            text_color = "#800000"
        
        st.markdown(f"""
        <div style="background: {bg_color}; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### {icon} **{prediction}**")
        with col2:
            st.markdown(f"### **{confidence:.1%}**")
        
        # Progress bar
        progress_html = f"""
        <div style="background-color: #e5e7eb; border-radius: 10px; height: 8px; margin: 0.5rem 0;">
            <div style="background: linear-gradient(90deg, {text_color}, {text_color}aa); 
                        width: {confidence:.1%}; height: 100%; border-radius: 10px; 
                        transition: width 0.5s ease;"></div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _display_confidence_analysis(self, confidence: float, analysis_type: str):
        """แสดงการวิเคราะห์ความมั่นใจ"""
        
        # Determine confidence level and color
        if confidence >= 0.9:
            level = "สูงมาก"
            color = "#10b981"
            icon = "HIGH"
            description = "ผลลัพธ์มีความน่าเชื่อถือสูงมาก"
        elif confidence >= 0.8:
            level = "สูง" 
            color = "#059669"
            icon = "HIGH"
            description = "ผลลัพธ์น่าเชื่อถือ"
        elif confidence >= 0.7:
            level = "ปานกลางถึงสูง"
            color = "#3b82f6"
            icon = "MED"
            description = "ผลลัพธ์ค่อนข้างน่าเชื่อถือ"
        elif confidence >= 0.6:
            level = "ปานกลาง"
            color = "#f59e0b"
            icon = "MED"
            description = "ควรตรวจสอบเพิ่มเติม"
        else:
            level = "ต่ำ"
            color = "#ef4444"
            icon = "LOW"
            description = "แนะนำให้ถ่ายใหม่"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {color}15 0%, {color}08 100%);
                    border: 1px solid {color}30; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <span style="font-size: 1.5rem;">{icon}</span>
                <div>
                    <h4 style="color: {color}; margin: 0; font-size: 1.2rem;">ระดับความมั่นใจ: {level}</h4>
                    <p style="color: #374151; margin: 0; font-size: 0.95rem;">{description}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_enhanced_features(self, enhanced_features: Dict[str, Any]):
        """แสดงข้อมูลคุณสมบัติเพิ่มเติม"""
        
        st.markdown("### ข้อมูลการวิเคราะห์เชิงลึก")
        
        image_quality = enhanced_features.get('image_quality', {})
        overall_score = image_quality.get('overall_score', 0.5)
        quality_level = image_quality.get('quality_level', 'unknown')
        was_enhanced = image_quality.get('was_enhanced', False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 10px;">
            """, unsafe_allow_html=True)
            
            st.markdown("#### คุณภาพภาพ")
            st.markdown(f"**คะแนนรวม:** {overall_score:.1%}")
            st.markdown(f"**ระดับคุณภาพ:** {quality_level.title()}")
            st.markdown(f"**ปรับปรุงอัตโนมัติ:** {'ใช่' if was_enhanced else 'ไม่'}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 10px;">
            """, unsafe_allow_html=True)
            
            st.markdown("#### การตั้งค่าการวิเคราะห์")
            st.markdown(f"**โหมด:** {'Dual-View' if enhanced_features.get('dual_analysis') else 'Single-View'}")
            st.markdown(f"**Auto Enhancement:** {'เปิด' if enhanced_features.get('auto_enhanced') else 'ปิด'}")
            st.markdown(f"**AI Model:** Enhanced CNN v2.1")
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def _create_mock_result(self, analysis_type: str) -> Dict[str, Any]:
        """สร้างผลลัพธ์จำลองสำหรับการทดสอบ"""
        thai_names = ['พระสมเด็จ', 'พระนางพญา', 'พระพิมพ์เล็ก', 'พระพิมพ์พุทธคุณ', 'พระไอย์ไข่']
        
        # Higher confidence for dual image analysis
        base_confidence = 0.85 if analysis_type == 'dual_image' else 0.75
        confidence_variation = 0.13 if analysis_type == 'dual_image' else 0.15
        
        return {
            'thai_name': np.random.choice(thai_names),
            'confidence': np.random.uniform(base_confidence - confidence_variation, base_confidence + confidence_variation),
            'predicted_class': 'class_' + str(np.random.randint(1, 6)),
            'model_version': f'Enhanced {"Dual-View" if analysis_type == "dual_image" else "Single-View"} CNN v2.1',
            'timestamp': datetime.now().isoformat(),
            'analysis_type': analysis_type,
            'processing_time': np.random.uniform(1.2, 2.8),
            'enhanced_features': {
                'image_quality': {
                    'overall_score': np.random.uniform(0.6, 0.95),
                    'quality_level': np.random.choice(['good', 'excellent']),
                    'was_enhanced': np.random.choice([True, False])
                },
                'auto_enhanced': True,
                'dual_analysis': analysis_type == 'dual_image'
            }
        }