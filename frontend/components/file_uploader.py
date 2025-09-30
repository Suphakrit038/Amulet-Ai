"""File Uploader Component

Component สำหรับอัพโหลดไฟล์รูปภาพ
"""

import streamlit as st
from typing import List, Optional, Tuple
from ..utils.file_validator import FileValidator


class FileUploaderComponent:
    """Component สำหรับจัดการการอัพโหลดไฟล์"""
    
    def __init__(self, max_file_size: int = 10 * 1024 * 1024):
        self.max_file_size = max_file_size
        self.accepted_types = ['png', 'jpg', 'jpeg']
        self.validator = FileValidator(max_file_size)
    
    def dual_image_uploader(self) -> Tuple[Optional[st.runtime.uploaded_file_manager.UploadedFile], Optional[st.runtime.uploaded_file_manager.UploadedFile]]:
        """File uploader สำหรับรูปคู่ (หน้า-หลัง)"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### รูปด้านหน้า")
            front_image = st.file_uploader(
                "เลือกรูปด้านหน้า",
                type=self.accepted_types,
                key="front_image",
                help=f"รองรับไฟล์: {', '.join(self.accepted_types).upper()}"
            )
            
            front_valid = None
            if front_image:
                if self.validator.validate_file(front_image):
                    front_valid = front_image
                else:
                    st.error("ไฟล์ด้านหน้าไม่ผ่านการตรวจสอบ")
        
        with col2:
            st.markdown("#### รูปด้านหลัง")
            back_image = st.file_uploader(
                "เลือกรูปด้านหลัง",
                type=self.accepted_types,
                key="back_image", 
                help=f"รองรับไฟล์: {', '.join(self.accepted_types).upper()}"
            )
            
            back_valid = None
            if back_image:
                if self.validator.validate_file(back_image):
                    back_valid = back_image
                else:
                    st.error("ไฟล์ด้านหลังไม่ผ่านการตรวจสอบ")
        
        return front_valid, back_valid
    
    def display_upload_guidelines(self):
        """แสดงคำแนะนำการอัพโหลดไฟล์"""
        
        with st.expander("📋 คำแนะนำการอัพโหลดรูปภาพ", expanded=False):
            st.markdown("""
            ### 📷 เทคนิคการถ่ายรูปที่ดี
            
            #### ✅ ควรทำ:
            - **แสงสว่างเพียงพอ** - ใช้แสงธรรมชาติหรือแสงขาวนวล
            - **พื้นหลังเรียบ** - ใช้พื้นหลังสีขาวหรือสีเดียว
            - **ถ่ายตรงและชัดเจน** - หลีกเลี่ยงมุมเอียงหรือภาพเบลอ
            - **ระยะที่เหมาะสม** - ให้พระเครื่องเต็มเฟรมแต่ไม่ตัดขอบ
            
            #### ❌ ไม่ควรทำ:
            - **แสงสะท้อน** - หลีกเลี่ยงแสงแฟลชที่สะท้อนบนพระเครื่อง
            - **เงาบดบัง** - ไม่ให้เงามือหรือสิ่งของอื่นบดบังรายละเอียด
            - **ภาพเบลอ** - ตรวจสอบให้ภาพชัดก่อนอัพโหลด
            - **ขนาดไฟล์ใหญ่** - ไฟล์ไม่ควรเกิน 10MB
            
            ### 📐 ข้อกำหนดทางเทคนิค
            - **รูปแบบไฟล์:** PNG, JPG, JPEG
            - **ขนาดไฟล์:** สูงสุด 10MB
            - **ความละเอียด:** แนะนำ 800x600 ขึ้นไป
            - **อัตราส่วน:** ไม่จำกัด แต่แนะนำ 1:1 หรือ 4:3
            """)
    
    def display_upload_tips(self, mode: str):
        """แสดงเคล็ดลับเฉพาะสำหรับแต่ละโหมด"""
        
        if mode == 'dual':
            st.info("""
            **เคล็ดลับสำหรับ Dual Image Analysis:**
            - ถ่ายในสภาพแสงเดียวกันและพื้นหลังเดียวกัน
            - วางพระเครื่องในตำแหน่งเดียวกันสำหรับทั้งสองด้าน
            - ตรวจสอบให้ทั้งสองรูปมีความชัดเจนเท่าเทียมกัน
            - หากมีข้อความหรือตัวเลข ให้แน่ใจว่าเห็นได้ชัดในทั้งสองด้าน
            """)
    
    def get_upload_status_message(self, front_file, back_file=None, mode='dual'):
        """ส่งคืนข้อความสถานะการอัพโหลด"""
        
        if mode == 'dual':
            if front_file and back_file:
                return "อัพโหลดรูปทั้งสองด้านสำเร็จ - พร้อมวิเคราะห์แบบ Dual-View"
            elif front_file or back_file:
                return "กรุณาอัพโหลดรูปทั้งด้านหน้าและด้านหลัง"
            else:
                return "กรุณาเลือกรูปด้านหน้าและด้านหลัง"
        
        return ""