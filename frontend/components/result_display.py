"""
คอมโพเนนต์สำหรับการแสดงผลลัพธ์
"""

import streamlit as st
import base64
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from io import BytesIO
from PIL import Image

# Add frontend to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import UI_SETTINGS
except ImportError:
    # Fallback configuration
    UI_SETTINGS = {
        "PRIMARY_COLOR": "#2563EB",
        "SECONDARY_COLOR": "#1E3A8A",
        "SUCCESS_COLOR": "#10B981",
        "WARNING_COLOR": "#F59E0B",
        "ERROR_COLOR": "#EF4444"
    }

class ResultDisplayer:
    """คลาสสำหรับแสดงผลลัพธ์การวิเคราะห์"""
    
    def __init__(self):
        self.colors = UI_SETTINGS
    
    def display_prediction_results(self, results: Dict[str, Any], front_image: Optional[Image.Image] = None, 
                                 back_image: Optional[Image.Image] = None):
        """
        แสดงผลลัพธ์การทำนาย
        
        Args:
            results: ผลลัพธ์จาก API
            front_image: รูปภาพด้านหน้า
            back_image: รูปภาพด้านหลัง
        """
        if results.get("error"):
            st.error(f"เกิดข้อผิดพลาด: {results.get('error_message', 'ไม่ทราบสาเหตุ')}")
            return
        
        # แสดงรูปภาพที่อัปโหลด
        self._display_uploaded_images(front_image, back_image)
        
        # แสดงผลการทำนาย
        self._display_prediction_summary(results)
        
        # แสดงรายละเอียดการทำนาย
        self._display_detailed_results(results)
        
        # แสดงการเปรียบเทียบ
        self._display_comparison_results(results)
        
        # แสดงข้อมูลเพิ่มเติม
        self._display_additional_info(results)
    
    def _display_uploaded_images(self, front_image: Optional[Image.Image], back_image: Optional[Image.Image]):
        """แสดงรูปภาพที่อัปโหลด"""
        st.markdown("### รูปภาพที่อัปโหลด")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if front_image:
                st.markdown("**ด้านหน้า**")
                st.image(front_image, use_column_width=True)
            else:
                st.info("ไม่มีรูปภาพด้านหน้า")
        
        with col2:
            if back_image:
                st.markdown("**ด้านหลัง**")
                st.image(back_image, use_column_width=True)
            else:
                st.info("ไม่มีรูปภาพด้านหลัง")
    
    def _display_prediction_summary(self, results: Dict[str, Any]):
        """แสดงสรุปผลการทำนาย"""
        st.markdown("### ผลการวิเคราะห์")
        
        top1 = results.get("top1", {})
        class_name = top1.get("class_name", "ไม่ทราบ")
        confidence = top1.get("confidence", 0.0)
        
        # แสดงผลแบบ metric
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="ประเภทพระเครื่อง",
                value=class_name,
                help="ประเภทที่โมเดล AI ทำนายได้"
            )
        
        with col2:
            confidence_percent = confidence * 100
            color = self._get_confidence_color(confidence)
            st.metric(
                label="ความเชื่อมั่น",
                value=f"{confidence_percent:.1f}%",
                help="ระดับความมั่นใจของโมเดล AI"
            )
        
        with col3:
            processing_time = results.get("processing_time", 0)
            st.metric(
                label="เวลาประมวลผล",
                value=f"{processing_time:.2f} วินาที",
                help="เวลาที่ใช้ในการประมวลผล"
            )
    
    def _display_detailed_results(self, results: Dict[str, Any]):
        """แสดงรายละเอียดผลการทำนาย"""
        st.markdown("### รายละเอียดการทำนาย")
        
        topk = results.get("topk", [])
        if topk:
            with st.expander("ผลการทำนาย Top-K", expanded=False):
                for i, result in enumerate(topk[:5]):  # แสดงแค่ 5 อันดับแรก
                    class_name = result.get("class_name", "ไม่ทราบ")
                    confidence = result.get("confidence", 0.0)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{i+1}. {class_name}")
                    with col2:
                        st.write(f"{confidence*100:.1f}%")
                    
                    # แสดง progress bar
                    st.progress(confidence)
    
    def _display_comparison_results(self, results: Dict[str, Any]):
        """แสดงผลการเปรียบเทียบ"""
        reference_images = results.get("reference_images", {})
        comparison_data = results.get("comparison_data", [])
        
        if reference_images:
            st.markdown("### การเปรียบเทียบกับรูปภาพอ้างอิง")
            
            # แสดงค่าความเหมือนโดยรวม
            overall_similarity = results.get("overall_similarity", 0.0)
            st.metric(
                label="ความเหมือนโดยรวม",
                value=f"{overall_similarity*100:.1f}%",
                help="ค่าเฉลี่ยความเหมือนกับรูปภาพอ้างอิงทั้งหมด"
            )
            
            # แสดงรูปภาพอ้างอิง
            cols = st.columns(min(len(reference_images), 3))
            for i, (key, ref_data) in enumerate(reference_images.items()):
                if i >= 3:  # แสดงแค่ 3 รูปแรก
                    break
                
                with cols[i]:
                    # หาข้อมูลความเหมือนสำหรับรูปนี้
                    similarity = 0.0
                    for comp in comparison_data:
                        if comp.get("ref_key") == key:
                            similarity = comp.get("similarity", 0.0)
                            break
                    
                    st.markdown(f"**{ref_data.get('view_type', 'Unknown').title()}**")
                    
                    # แสดงรูปภาพจาก base64
                    if "image_b64" in ref_data:
                        try:
                            img_data = base64.b64decode(ref_data["image_b64"])
                            img = Image.open(BytesIO(img_data))
                            st.image(img, use_column_width=True)
                        except Exception as e:
                            st.error(f"ไม่สามารถแสดงรูปภาพได้: {str(e)}")
                    
                    # แสดงค่าความเหมือน
                    st.metric(
                        label="ความเหมือน",
                        value=f"{similarity*100:.1f}%"
                    )
    
    def _display_additional_info(self, results: Dict[str, Any]):
        """แสดงข้อมูลเพิ่มเติม"""
        # แสดงข้อมูลการประเมินราคา (ถ้ามี)
        valuation = results.get("valuation")
        if valuation:
            st.markdown("### การประเมินราคา")
            with st.expander("รายละเอียดการประเมินราคา", expanded=False):
                for key, value in valuation.items():
                    if isinstance(value, (int, float)):
                        st.metric(label=key, value=f"{value:,.0f} บาท")
                    else:
                        st.write(f"**{key}**: {value}")
        
        # แสดงคำแนะนำ (ถ้ามี)
        recommendations = results.get("recommendations")
        if recommendations:
            st.markdown("### คำแนะนำ")
            with st.expander("คำแนะนำจากระบบ", expanded=False):
                for i, rec in enumerate(recommendations):
                    st.write(f"{i+1}. {rec}")
    
    def _get_confidence_color(self, confidence: float) -> str:
        """เลือกสีตามระดับความเชื่อมั่น"""
        if confidence >= 0.8:
            return self.colors["SUCCESS_COLOR"]
        elif confidence >= 0.6:
            return self.colors["WARNING_COLOR"]
        else:
            return self.colors["ERROR_COLOR"]

# สร้าง instance สำหรับใช้งาน
result_displayer = ResultDisplayer()

# Export ฟังก์ชันสำหรับใช้งาน
def display_results(results: Dict[str, Any], front_image: Optional[Image.Image] = None, 
                   back_image: Optional[Image.Image] = None):
    """ฟังก์ชันสำหรับแสดงผลลัพธ์"""
    return result_displayer.display_prediction_results(results, front_image, back_image)
