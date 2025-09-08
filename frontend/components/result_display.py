"""
คอมโพเนนต์สำหรับการแสดงผลลัพธ์ - Professional UI
"""

import streamlit as st
import base64
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from io import BytesIO
from PIL import Image
from datetime import datetime

# Add frontend to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ..config import UI_SETTINGS
except ImportError:
    try:
        import config
        UI_SETTINGS = config.UI_SETTINGS
    except ImportError:
        # Professional color scheme
        UI_SETTINGS = {
            "PRIMARY_COLOR": "#1F2937",      # Dark gray for text
            "SECONDARY_COLOR": "#6B7280",    # Medium gray
            "SUCCESS_COLOR": "#10B981",      # Green for high confidence
            "WARNING_COLOR": "#F59E0B",      # Yellow/Orange for medium
            "ERROR_COLOR": "#EF4444",        # Red for low confidence
            "CARD_BG": "#FFFFFF",            # White background
            "CARD_SHADOW": "rgba(0,0,0,0.1)", # Subtle shadow
            "BORDER_RADIUS": "12px"          # Rounded corners
        }

class ResultDisplayer:
    """คลาสสำหรับแสดงผลลัพธ์การวิเคราะห์ - Professional UI"""

    def __init__(self):
        self.colors = UI_SETTINGS
        self._inject_custom_css()

    def _inject_custom_css(self):
        """Inject custom CSS for professional card-style layout"""
        st.markdown("""
        <style>
        /* Professional Card Styling */
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin: 16px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07), 0 1px 3px rgba(0, 0, 0, 0.1);
            border: 1px solid #E5E7EB;
            transition: all 0.3s ease;
        }

        .result-card:hover {
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12), 0 4px 10px rgba(0, 0, 0, 0.08);
            transform: translateY(-2px);
        }

        .card-header {
            font-size: 18px;
            font-weight: 600;
            color: #1F2937;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .card-header::before {
            content: '';
            width: 4px;
            height: 20px;
            background: #3B82F6;
            border-radius: 2px;
        }

        .metric-card {
            background: #F9FAFB;
            border-radius: 8px;
            padding: 16px;
            margin: 8px 0;
            border-left: 4px solid #3B82F6;
        }

        .confidence-high { border-left-color: #10B981; }
        .confidence-medium { border-left-color: #F59E0B; }
        .confidence-low { border-left-color: #EF4444; }

        .progress-container {
            background: #E5E7EB;
            border-radius: 4px;
            height: 8px;
            margin: 8px 0;
            overflow: hidden;
        }

        .progress-bar {
            height: 100%;
            border-radius: 4px;
            transition: width 0.8s ease;
        }

        .timestamp {
            font-size: 12px;
            color: #6B7280;
            margin-top: 8px;
        }

        .section-icon {
            width: 20px;
            height: 20px;
            opacity: 0.8;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .result-card {
                padding: 16px;
                margin: 8px 0;
            }
        }
        </style>
        """, unsafe_allow_html=True)
    
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

        # 1. ผลการวิเคราะห์หลัก
        self._display_main_analysis(results)

        # 2. การเปรียบเทียบกับภาพในฐานข้อมูล
        self._display_comparison_section(results, front_image, back_image)

        # 3. สรุปการวิเคราะห์
        self._display_analysis_summary(results)

        # 4. การประเมินราคา
        self._display_price_assessment(results)

        # 5. แนะนำตลาดและช่องทางการขาย
        self._display_market_recommendations(results)
    
    
    def _display_main_analysis(self, results: Dict[str, Any]):
        """แสดงผลการวิเคราะห์หลัก - Professional Card Layout"""
        st.markdown("""
        <div class="result-card">
            <div class="card-header">
                <svg class="section-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
                </svg>
                ผลการวิเคราะห์หลัก
            </div>
        </div>
        """, unsafe_allow_html=True)

        # แสดงโหมด AI และเวลาประมวลผล - Simplified Professional Design
        col1, col2 = st.columns(2)

        with col1:
            ai_mode = results.get("ai_mode", "ไม่ระบุ")
            st.markdown(f"""
            <div style="background: #F9FAFB; padding: 16px; border-radius: 8px; border-left: 4px solid #3B82F6;">
                <div style="font-size: 14px; color: #6B7280; margin-bottom: 4px;">โหมด AI ที่ใช้</div>
                <div style="font-size: 16px; font-weight: 600; color: #1F2937;">{ai_mode}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            processing_time = results.get("processing_time", 0)
            st.markdown(f"""
            <div style="background: #F9FAFB; padding: 16px; border-radius: 8px; border-left: 4px solid #3B82F6;">
                <div style="font-size: 14px; color: #6B7280; margin-bottom: 4px;">เวลาประมวลผล</div>
                <div style="font-size: 16px; font-weight: 600; color: #1F2937;">{processing_time:.2f} วินาที</div>
            </div>
            """, unsafe_allow_html=True)

        # แสดงผลการจำแนกประเภท TOP 3 - Professional Simplified Design
        st.markdown("""
        <div style="margin-top: 24px;">
            <div style="font-size: 18px; font-weight: 600; color: #1F2937; margin-bottom: 16px;">
                ผลการจำแนกประเภท TOP 3
            </div>
        </div>
        """, unsafe_allow_html=True)

        topk = results.get("topk", [])

        if topk:
            for i, result in enumerate(topk[:3]):  # แสดงแค่ 3 อันดับแรก
                class_name = result.get("class_name", "ไม่ทราบ")
                class_name_thai = result.get("class_name_thai", class_name)
                confidence = result.get("confidence", 0.0)

                # Use Streamlit's native components for better reliability
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**{i+1}. {class_name_thai}**")
                    if class_name_thai != class_name:
                        st.markdown(f"<small style='color: #6B7280;'>({class_name})</small>", unsafe_allow_html=True)

                with col2:
                    st.markdown(f"**{confidence*100:.1f}%**")

                # Use Streamlit's progress bar
                st.progress(confidence)
        else:
            st.markdown("""
            <div style="background: #F9FAFB; padding: 20px; border-radius: 8px; margin: 8px 0; text-align: center; border: 1px solid #E5E7EB;">
                <div style="color: #6B7280; font-size: 14px;">ไม่มีข้อมูลการจำแนกประเภท</div>
            </div>
            """, unsafe_allow_html=True)
    
    def _display_analysis_summary(self, results: Dict[str, Any]):
        """แสดงสรุปการวิเคราะห์ - Professional Card Layout"""
        st.markdown("""
        <div class="result-card">
            <div class="card-header">
                <svg class="section-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M9 11H7v2h2v-2zm4 0h-2v2h2v-2zm4 0h-2v2h2v-2zm2-7h-1V2h-2v2H8V2H6v2H5c-1.1 0-1.99.9-1.99 2L3 20c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 16H5V9h14v11z"/>
                </svg>
                สรุปการวิเคราะห์
            </div>
        </div>
        """, unsafe_allow_html=True)

        summary = results.get("summary", "")
        if summary:
            st.markdown(f"""
            <div style="background: #F9FAFB; padding: 16px; border-radius: 8px; border-left: 4px solid #3B82F6;">
                <div style="font-size: 16px; line-height: 1.6; color: #1F2937;">{summary}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # สร้างสรุปอัตโนมัติจากข้อมูลที่มี
            top1 = results.get("top1", {})
            class_name = top1.get("class_name", "ไม่ทราบ")
            class_name_thai = top1.get("class_name_thai", class_name)
            confidence = top1.get("confidence", 0.0)
            overall_similarity = results.get("overall_similarity", 0.0)

            summary_text = f"จากการวิเคราะห์ พระเครื่องนี้ถูกจำแนกเป็น '{class_name_thai}' ด้วยความเชื่อมั่น {confidence*100:.1f}% และมีความเหมือนกับภาพในฐานข้อมูล {overall_similarity*100:.1f}%"

            st.markdown(f"""
            <div style="background: #F9FAFB; padding: 16px; border-radius: 8px; border-left: 4px solid #3B82F6;">
                <div style="font-size: 16px; line-height: 1.6; color: #1F2937;">{summary_text}</div>
                {f"<div style='font-size: 14px; color: #6B7280; margin-top: 8px;'>({class_name})</div>" if class_name_thai != class_name else ""}
            </div>
            """, unsafe_allow_html=True)
    
    def _display_comparison_section(self, results: Dict[str, Any], front_image: Optional[Image.Image] = None,
                                   back_image: Optional[Image.Image] = None):
        """แสดงการเปรียบเทียบกับภาพในฐานข้อมูล - Professional Card Layout"""
        st.markdown("""
        <div class="result-card">
            <div class="card-header">
                <svg class="section-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-1 9H9V9h10v2zm-4 4H9v-2h6v2zm4-8H9V5h10v2z"/>
                </svg>
                การเปรียบเทียบกับภาพในฐานข้อมูล
            </div>
        </div>
        """, unsafe_allow_html=True)

        # แสดงภาพผู้ใช้
        st.markdown("""
        <div style="font-size: 16px; font-weight: 600; color: #1F2937; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
            </svg>
            ภาพผู้ใช้
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            if front_image:
                st.markdown("""
                <div style="background: #F9FAFB; padding: 12px; border-radius: 8px; text-align: center; border-left: 4px solid #3B82F6;">
                    <div style="font-weight: 600; color: #1F2937;">ด้านหน้า</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(front_image, use_container_width=True, width=200)
            else:
                st.markdown("""
                <div style="background: #F9FAFB; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #E5E7EB;">
                    <div style="color: #6B7280;">ไม่มีรูปภาพด้านหน้า</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            if back_image:
                st.markdown("""
                <div style="background: #F9FAFB; padding: 12px; border-radius: 8px; text-align: center; border-left: 4px solid #3B82F6;">
                    <div style="font-weight: 600; color: #1F2937;">ด้านหลัง</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(back_image, use_container_width=True, width=200)
            else:
                st.markdown("""
                <div style="background: #F9FAFB; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #E5E7EB;">
                    <div style="color: #6B7280;">ไม่มีรูปภาพด้านหลัง</div>
                </div>
                """, unsafe_allow_html=True)

        # แสดงภาพอ้างอิงจากฐานข้อมูล
        reference_images = results.get("reference_images", {})
        comparison_data = results.get("comparison_data", [])

        if reference_images:
            st.markdown("""
            <div style="font-size: 16px; font-weight: 600; color: #1F2937; margin: 24px 0 16px 0; display: flex; align-items: center; gap: 8px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-1 9H9V9h10v2zm-4 4H9v-2h6v2zm4-8H9V5h10v2z"/>
                </svg>
                ภาพอ้างอิงจากฐานข้อมูล
            </div>
            """, unsafe_allow_html=True)

            # แสดงภาพอ้างอิงสูงสุด 4 ภาพ
            cols = st.columns(min(len(reference_images), 4))
            for i, (key, ref_data) in enumerate(list(reference_images.items())[:4]):
                with cols[i]:
                    # หาข้อมูลความเหมือนสำหรับรูปนี้
                    similarity = 0.0
                    for comp in comparison_data:
                        if comp.get("ref_key") == key:
                            similarity = comp.get("similarity", 0.0)
                            break

                    st.markdown(f"""
                    <div style="background: #F9FAFB; padding: 12px; border-radius: 8px; margin-bottom: 8px; text-align: center; border-left: 4px solid #3B82F6;">
                        <div style="font-weight: 600; color: #1F2937;">ภาพอ้างอิง {i+1}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # แสดงรูปภาพจาก base64
                    if "image_b64" in ref_data:
                        try:
                            img_data = base64.b64decode(ref_data["image_b64"])
                            img = Image.open(BytesIO(img_data))
                            st.image(img, use_container_width=True, width=150)
                        except Exception as e:
                            st.error(f"ไม่สามารถแสดงรูปภาพได้: {str(e)}")

                    # แสดงระดับความเหมือน
                    st.markdown(f"""
                    <div style="background: #F9FAFB; padding: 12px; border-radius: 8px; text-align: center; border-left: 4px solid #3B82F6;">
                        <div style="font-size: 14px; color: #6B7280; margin-bottom: 4px;">ระดับความเหมือน</div>
                        <div style="font-size: 16px; font-weight: 600; color: #3B82F6;">
                            {similarity*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="text-align: center; color: #6B7280; background: linear-gradient(135deg, #F9FAFB, #F3F4F6);">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor" style="opacity: 0.3; margin-bottom: 12px;">
                    <path d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-1 9H9V9h10v2zm-4 4H9v-2h6v2zm4-8H9V5h10v2z"/>
                </svg>
                <div style="font-size: 16px;">ไม่มีภาพอ้างอิงสำหรับเปรียบเทียบ</div>
            </div>
            """, unsafe_allow_html=True)
    
    def _display_price_assessment(self, results: Dict[str, Any]):
        """แสดงการประเมินราคา - Professional Card Layout"""
        st.markdown("""
        <div class="result-card">
            <div class="card-header">
                <svg class="section-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                </svg>
                การประเมินราคา
            </div>
        </div>
        """, unsafe_allow_html=True)

        valuation = results.get("valuation", {})

        if valuation:
            # แสดงช่วงราคาประเมิน
            price_min = valuation.get("price_min", 0)
            price_avg = valuation.get("price_avg", 0)
            price_max = valuation.get("price_max", 0)

            st.markdown("""
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 20px;">
            """, unsafe_allow_html=True)

            # ราคาต่ำสุด
            st.markdown(f"""
            <div style="background: #F9FAFB; padding: 16px; border-radius: 8px; text-align: center; border-left: 4px solid #3B82F6;">
                <div style="font-size: 14px; color: #6B7280; margin-bottom: 8px;">ราคาประเมินต่ำสุด</div>
                <div style="font-size: 20px; font-weight: 600; color: #1F2937;">{price_min:,.0f} บาท</div>
            </div>
            """, unsafe_allow_html=True)

            # ราคาเฉลี่ย
            st.markdown(f"""
            <div style="background: #F9FAFB; padding: 16px; border-radius: 8px; text-align: center; border-left: 4px solid #3B82F6;">
                <div style="font-size: 14px; color: #6B7280; margin-bottom: 8px;">ราคาประเมินเฉลี่ย</div>
                <div style="font-size: 24px; font-weight: 700; color: #1F2937;">{price_avg:,.0f} บาท</div>
            </div>
            """, unsafe_allow_html=True)

            # ราคาสูงสุด
            st.markdown(f"""
            <div style="background: #F9FAFB; padding: 16px; border-radius: 8px; text-align: center; border-left: 4px solid #3B82F6;">
                <div style="font-size: 14px; color: #6B7280; margin-bottom: 8px;">ราคาประเมินสูงสุด</div>
                <div style="font-size: 20px; font-weight: 600; color: #1F2937;">{price_max:,.0f} บาท</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # แสดงความเชื่อมั่นในการประเมิน
            confidence = valuation.get("confidence", 0.0)

            st.markdown(f"""
            <div style="background: #F9FAFB; padding: 16px; border-radius: 8px; margin-top: 16px; border-left: 4px solid #3B82F6;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-weight: 600; color: #1F2937;">ความเชื่อมั่นในการประเมิน</div>
                    <div style="font-size: 16px; font-weight: 600; color: #3B82F6;">
                        {confidence*100:.1f}%
                    </div>
                </div>
                <div style="background: #E5E7EB; height: 6px; border-radius: 3px; overflow: hidden;">
                    <div style="background: #3B82F6; height: 100%; width: {confidence*100:.1f}%; border-radius: 3px; transition: width 0.8s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # แสดงหมายเหตุ (ถ้ามี)
            notes = valuation.get("notes", "")
            if notes:
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #F3F4F6, #E5E7EB); border-left-color: #6B7280;">
                    <div style="font-weight: 600; color: #374151; margin-bottom: 8px; display: flex; align-items: center; gap: 6px;">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                            <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 1.99 2H18c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
                        </svg>
                        หมายเหตุ
                    </div>
                    <div style="color: #4B5563; line-height: 1.5;">{notes}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="text-align: center; color: #6B7280; background: linear-gradient(135deg, #F9FAFB, #F3F4F6);">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor" style="opacity: 0.3; margin-bottom: 12px;">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                </svg>
                <div style="font-size: 16px;">ไม่มีข้อมูลการประเมินราคา</div>
            </div>
            """, unsafe_allow_html=True)
    
    def _display_market_recommendations(self, results: Dict[str, Any]):
        """แสดงแนะนำตลาดและช่องทางการขาย - Professional Card Layout"""
        st.markdown("""
        <div class="result-card">
            <div class="card-header">
                <svg class="section-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M7 4V2C7 1.45 7.45 1 8 1h8c.55 0 1 .45 1 1v2h3c.55 0 1 .45 1 1s-.45 1-1 1h-1v13c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2V6H3c-.55 0-1-.45-1-1s.45-1 1-1h3zm2 2v12h8V6H9z"/>
                </svg>
                แนะนำตลาดและช่องทางการขาย
            </div>
        </div>
        """, unsafe_allow_html=True)

        market_channels = results.get("market_channels", [])
        recommendations = results.get("recommendations", [])

        if market_channels:
            st.markdown("""
            <div style="font-size: 16px; font-weight: 600; color: #1F2937; margin-bottom: 16px; display: flex; align-items: center; gap: 8px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
                </svg>
                รายชื่อช่องทางแนะนำ
            </div>
            """, unsafe_allow_html=True)

            for channel in market_channels:
                channel_name = channel.get("name", "ไม่ระบุ")
                channel_type = channel.get("type", "ไม่ระบุ")  # ออนไลน์/ออฟไลน์
                score = channel.get("score", 0.0)
                reason = channel.get("reason", "")

                # เลือกไอคอนตามประเภทช่องทาง
                icon_svg = """
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M7 4V2C7 1.45 7.45 1 8 1h8c.55 0 1 .45 1 1v2h3c.55 0 1 .45 1 1s-.45 1-1 1h-1v13c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2V6H3c-.55 0-1-.45-1-1s.45-1 1-1h3zm2 2v12h8V6H9z"/>
                </svg>
                """ if channel_type.lower() == "ออฟไลน์" else """
                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20 18c1.1 0 1.99-.9 1.99-2L22 6c0-1.1-.9-2-2-2H4c-1.1 0-2 .9-2 2v10c0 1.1.9 2 2 2H0v2h24v-2h-4zM4 6h16v10H4V6z"/>
                </svg>
                """

                with st.expander(f"{icon_svg} {channel_name} ({channel_type})", expanded=False):
                    st.markdown(f"""
                    <div style="padding: 16px; background: #F9FAFB; border-radius: 8px; margin: 8px 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <div style="font-weight: 600; color: #1F2937;">คะแนนแนะนำ</div>
                            <div style="font-size: 18px; font-weight: 600; color: #3B82F6;">
                                {score:.1f}/10
                            </div>
                        </div>
                        <div style="background: #E5E7EB; height: 6px; border-radius: 3px; margin-bottom: 12px; overflow: hidden;">
                            <div style="background: #3B82F6; height: 100%; width: {score*10:.1f}%; border-radius: 3px; transition: width 0.8s ease;"></div>
                        </div>
                        <div style="font-weight: 500; color: #374151; margin-bottom: 4px;">เหตุผล</div>
                        <div style="color: #4B5563; line-height: 1.5;">{reason}</div>
                    </div>
                    """, unsafe_allow_html=True)
        elif recommendations:
            # แสดงคำแนะนำแบบเก่า (ถ้ามี)
            st.markdown("""
            <div style="font-size: 16px; font-weight: 600; color: #1F2937; margin-bottom: 16px;">
                คำแนะนำจากระบบ
            </div>
            """, unsafe_allow_html=True)

            for i, rec in enumerate(recommendations):
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="background: #10B981; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600;">
                            {i+1}
                        </div>
                        <div style="color: #1F2937;">{rec}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-card" style="text-align: center; color: #6B7280; background: linear-gradient(135deg, #F9FAFB, #F3F4F6);">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor" style="opacity: 0.3; margin-bottom: 12px;">
                    <path d="M7 4V2C7 1.45 7.45 1 8 1h8c.55 0 1 .45 1 1v2h3c.55 0 1 .45 1 1s-.45 1-1 1h-1v13c0 1.1-.9 2-2 2H6c-1.1 0-2-.9-2-2V6H3c-.55 0-1-.45-1-1s.45-1 1-1h3zm2 2v12h8V6H9z"/>
                </svg>
                <div style="font-size: 16px;">ไม่มีข้อมูลแนะนำตลาดและช่องทางการขาย</div>
            </div>
            """, unsafe_allow_html=True)

    def _get_confidence_color(self, confidence: float) -> str:
        """เลือกสีตามระดับความเชื่อมั่น"""
        if confidence >= 0.8:
            return self.colors["SUCCESS_COLOR"]
        elif confidence >= 0.6:
            return self.colors["WARNING_COLOR"]
        else:
            return self.colors["ERROR_COLOR"]

    def _get_confidence_class(self, confidence: float) -> str:
        """เลือก CSS class ตามระดับความเชื่อมั่น"""
        if confidence >= 0.8:
            return "confidence-high"
        elif confidence >= 0.6:
            return "confidence-medium"
        else:
            return "confidence-low"

    def _get_confidence_level_text(self, confidence: float) -> str:
        """เลือกข้อความระดับความเชื่อมั่น"""
        if confidence >= 0.8:
            return "สูง"
        elif confidence >= 0.6:
            return "ปานกลาง"
        else:
            return "ต่ำ"

    def _get_score_color(self, score: float) -> str:
        """เลือกสีตามคะแนนแนะนำ"""
        if score >= 8.0:
            return "#10B981"  # Green for excellent
        elif score >= 6.0:
            return "#F59E0B"  # Yellow/Orange for good
        else:
            return "#EF4444"  # Red for poor

# สร้าง instance สำหรับใช้งาน
result_displayer = ResultDisplayer()

# Export ฟังก์ชันสำหรับใช้งาน
def display_results(results: Dict[str, Any], front_image: Optional[Image.Image] = None, 
                   back_image: Optional[Image.Image] = None):
    """ฟังก์ชันสำหรับแสดงผลลัพธ์"""
    return result_displayer.display_prediction_results(results, front_image, back_image)
