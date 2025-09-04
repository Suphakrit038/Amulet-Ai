"""
🔍 Amulet Comparison App
แอพพลิเคชันสำหรับเปรียบเทียบรูปภาพพระเครื่องของผู้ใช้กับฐานข้อมูล
"""
import os
import sys
import json
import logging
import numpy as np
import streamlit as st
import torch
from PIL import Image
from pathlib import Path
import time
import matplotlib.pyplot as plt
import io

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from image_comparison import FeatureExtractor, ImageComparer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("frontend/comparison_app.log")
    ]
)
logger = logging.getLogger("comparison_app")

# Configure Streamlit page
st.set_page_config(
    page_title="พระเครื่อง Image Comparison",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #2563EB;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .similarity-high {
        color: #10B981;
        font-weight: bold;
    }
    
    .similarity-medium {
        color: #F59E0B;
        font-weight: bold;
    }
    
    .similarity-low {
        color: #EF4444;
        font-weight: bold;
    }
    
    .img-container {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 0.5rem;
        background-color: #F9FAFB;
    }
    
    .info-box {
        background-color: #EFF6FF;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #BFDBFE;
    }
    
    .stButton button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_config():
    """Load configuration for the app"""
    config_path = Path("config.json")
    
    if not config_path.exists():
        # Default config
        config = {
            "model_path": "training_output_improved/models/best_model.pth",
            "database_dir": "dataset_organized",
            "top_k": 5
        }
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    # Ensure required keys exist
    required_keys = ["model_path", "database_dir", "top_k"]
    for key in required_keys:
        if key not in config:
            if key == "model_path":
                config[key] = "training_output_improved/models/best_model.pth"
            elif key == "database_dir":
                config[key] = "dataset_organized"
            elif key == "top_k":
                config[key] = 5
    
    return config

def get_similarity_class(similarity):
    """Get CSS class for similarity score"""
    if similarity >= 0.85:
        return "similarity-high"
    elif similarity >= 0.7:
        return "similarity-medium"
    else:
        return "similarity-low"

def plot_comparison(query_img, match_results):
    """Create a matplotlib figure for comparison"""
    n_matches = len(match_results)
    fig, axes = plt.subplots(1, n_matches + 1, figsize=(12, 4))
    
    # Show query image
    axes[0].imshow(query_img)
    axes[0].set_title("ภาพที่อัพโหลด")
    axes[0].axis('off')
    
    # Show matches
    for i, match in enumerate(match_results):
        img = Image.open(match["path"]).convert('RGB')
        similarity = match["similarity"]
        class_name = match["class"]
        
        axes[i+1].imshow(img)
        axes[i+1].set_title(f"{class_name}\nความเหมือน: {similarity:.2f}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def main():
    """Main function for the Streamlit app"""
    st.markdown('<h1 class="main-header">📸 ระบบเปรียบเทียบรูปภาพพระเครื่อง</h1>', unsafe_allow_html=True)
    
    # Load config
    config = load_config()
    
    # Sidebar
    st.sidebar.title("⚙️ การตั้งค่า")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "ที่อยู่ไฟล์โมเดล",
        value=config["model_path"]
    )
    
    # Database selection
    database_dir = st.sidebar.text_input(
        "ที่อยู่ฐานข้อมูลรูปภาพ",
        value=config["database_dir"]
    )
    
    # Top-k selection
    top_k = st.sidebar.slider(
        "จำนวนภาพที่เหมือนที่สุดที่ต้องการแสดง",
        min_value=1,
        max_value=10,
        value=config["top_k"]
    )
    
    # Save config button
    if st.sidebar.button("บันทึกการตั้งค่า"):
        new_config = {
            "model_path": model_path,
            "database_dir": database_dir,
            "top_k": top_k
        }
        
        with open("config.json", 'w', encoding='utf-8') as f:
            json.dump(new_config, f, indent=2, ensure_ascii=False)
        
        st.sidebar.success("บันทึกการตั้งค่าเรียบร้อยแล้ว!")
    
    # Instructions in sidebar
    st.sidebar.markdown("""
    <div class="info-box">
        <h3>วิธีใช้งาน</h3>
        <ol>
            <li>อัพโหลดรูปภาพพระเครื่องที่ต้องการเปรียบเทียบ</li>
            <li>รอระบบวิเคราะห์และค้นหาภาพที่คล้ายกัน</li>
            <li>ระบบจะแสดงผลการเปรียบเทียบและค่าความเหมือนของแต่ละภาพ</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Information about similarity score
    st.sidebar.markdown("""
    <div class="info-box">
        <h3>ค่าความเหมือน</h3>
        <p><span class="similarity-high">0.85 - 1.00</span>: เหมือนมาก</p>
        <p><span class="similarity-medium">0.70 - 0.84</span>: เหมือนปานกลาง</p>
        <p><span class="similarity-low">0.00 - 0.69</span>: เหมือนน้อย</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<h2 class="sub-header">อัพโหลดรูปภาพพระเครื่องที่ต้องการเปรียบเทียบ</h2>', unsafe_allow_html=True)
    
    # Upload image
    uploaded_file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Save image temporarily
        temp_path = Path("temp_upload.jpg")
        image.save(temp_path)
        
        st.markdown('<div class="img-container">', unsafe_allow_html=True)
        st.image(image, caption="รูปภาพที่อัพโหลด", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Compare button
        if st.button("เปรียบเทียบรูปภาพ"):
            with st.spinner("กำลังวิเคราะห์รูปภาพ..."):
                try:
                    # Initialize image comparer
                    comparer = ImageComparer(model_path, database_dir)
                    
                    # Compare image
                    start_time = time.time()
                    result = comparer.compare_image(temp_path, top_k=top_k)
                    elapsed_time = time.time() - start_time
                    
                    # Display results
                    st.markdown(f'<h2 class="sub-header">ผลการเปรียบเทียบ (ใช้เวลา {elapsed_time:.2f} วินาที)</h2>', unsafe_allow_html=True)
                    
                    # Plot comparison
                    comparison_img = plot_comparison(image, result["top_matches"])
                    st.image(comparison_img, use_column_width=True)
                    
                    # Display table of results
                    st.markdown('<h3 class="sub-header">รายละเอียดความเหมือน</h3>', unsafe_allow_html=True)
                    
                    # Create columns for results
                    for i, match in enumerate(result["top_matches"]):
                        similarity = match["similarity"]
                        similarity_class = get_similarity_class(similarity)
                        
                        st.markdown(f"""
                        <div class="info-box">
                            <h4>{i+1}. {match['class']}</h4>
                            <p>ค่าความเหมือน: <span class="{similarity_class}">{similarity:.4f}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการเปรียบเทียบรูปภาพ: {e}")
                    logger.error(f"Error in image comparison: {e}", exc_info=True)
                finally:
                    # Remove temporary file
                    if temp_path.exists():
                        temp_path.unlink()
    else:
        # Display sample or instructions
        st.markdown("""
        <div class="info-box">
            <h3>กรุณาอัพโหลดรูปภาพพระเครื่องที่ต้องการเปรียบเทียบ</h3>
            <p>ระบบจะวิเคราะห์และค้นหาภาพที่คล้ายกันจากฐานข้อมูล</p>
            <p>รองรับไฟล์ภาพนามสกุล JPG, JPEG และ PNG</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
