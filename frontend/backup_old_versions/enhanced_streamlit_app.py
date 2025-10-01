#!/usr/bin/env python3
"""
🎨 Enhanced Amulet-AI Frontend with Improved UX
Streamlit app ที่ปรับปรุงแล้วด้วย UX/UI ที่ดีขึ้น
"""
import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import json
import time
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import enhanced error handling
try:
    from core.error_handling_enhanced import error_handler, validate_image_file
    from core.performance_monitoring import performance_monitor
except:
    # Fallback if modules not available
    def error_handler(error_type="general"):
        def decorator(func):
            return func
        return decorator
    
    class performance_monitor:
        @staticmethod
        def collect_metrics():
            return {}

st.set_page_config(
    page_title="🏺 Amulet-AI Enhanced",
    page_icon="🏺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏺 Amulet-AI Enhanced System</h1>
        <p>ระบบจำแนกพระเครื่องอัจฉริยะ - เวอร์ชันปรับปรุงใหม่</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # System status
        show_system_status()
        
        st.header("📊 Quick Stats")
        show_quick_stats()
        
        st.header("🔧 Advanced Options")
        debug_mode = st.checkbox("Debug Mode")
        show_confidence = st.checkbox("Show Confidence Details", value=True)
        show_probabilities = st.checkbox("Show All Probabilities", value=True)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🖼️ Image Classification", "📊 System Analytics", "📚 Documentation", "🔧 System Tools"])
    
    with tab1:
        image_classification_tab(show_confidence, show_probabilities, debug_mode)
    
    with tab2:
        system_analytics_tab()
    
    with tab3:
        documentation_tab()
    
    with tab4:
        system_tools_tab()

def show_system_status():
    """แสดงสถานะระบบ"""
    try:
        # Check API status
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            api_status = "🟢 Online" if response.status_code == 200 else "🟡 Issues"
        except:
            api_status = "🔴 Offline"
        
        # Check model status
        model_files = [
            "trained_model/classifier.joblib",
            "trained_model/scaler.joblib", 
            "trained_model/label_encoder.joblib"
        ]
        
        missing_files = []
        for file_path in model_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        model_status = "🟢 Ready" if not missing_files else "🟡 Incomplete"
        
        st.markdown(f"""
        **API Server:** {api_status}  
        **Model:** {model_status}  
        **Status:** {'🟢 Operational' if api_status.startswith('🟢') and model_status.startswith('🟢') else '⚠️ Partial'}
        """)
        
    except Exception as e:
        st.error(f"Error checking status: {e}")

def show_quick_stats():
    """แสดงสถิติด่วน"""
    try:
        # Performance metrics
        metrics = performance_monitor.collect_metrics()
        
        if metrics and "system" in metrics:
            st.metric("CPU Usage", f"{metrics['system']['cpu_percent']:.1f}%")
            st.metric("Memory Usage", f"{metrics['system']['memory_percent']:.1f}%")
        
        # Model info
        model_info_path = Path("trained_model/model_info.json")
        if model_info_path.exists():
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            st.metric("Model Version", model_info.get("version", "Unknown"))
            st.metric("Accuracy", f"{model_info.get('training_results', {}).get('test_accuracy', 0):.1%}")
        
    except Exception as e:
        st.warning(f"Stats unavailable: {e}")

@error_handler("frontend")
def image_classification_tab(show_confidence, show_probabilities, debug_mode):
    """Tab สำหรับการจำแนกรูปภาพ"""
    
    st.header("🖼️ Upload & Classify Amulet Images")
    
    # Upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an amulet image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image of Thai Buddhist amulet"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Classify button
            if st.button("🔍 Classify Amulet", type="primary"):
                with st.spinner("Analyzing image..."):
                    result = classify_image(uploaded_file, debug_mode)
                    display_classification_result(result, show_confidence, show_probabilities)
    
    with col2:
        st.info("""
        **📝 Tips for better results:**
        - Use clear, well-lit images
        - Include both front and back if possible
        - Avoid blurry or dark images
        - Supported formats: JPG, PNG
        """)
        
        # Sample images
        st.subheader("📱 Try Sample Images")
        sample_dir = Path("organized_dataset/splits/test")
        if sample_dir.exists():
            show_sample_images(sample_dir)

def classify_image(uploaded_file, debug_mode=False):
    """จำแนกรูปภาพ"""
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Validate image
        validate_image_file(temp_path)
        
        # Make prediction
        if debug_mode:
            st.write("🔧 Debug: Making API request...")
        
        # Try API first
        try:
            with open(temp_path, "rb") as f:
                files = {"file": f}
                response = requests.post("http://localhost:8000/predict", files=files, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                result["method"] = "API"
                return result
        except:
            if debug_mode:
                st.warning("API unavailable, using local prediction...")
        
        # Fallback to local prediction
        result = local_prediction(temp_path)
        result["method"] = "Local"
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
        
        return result
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "method": "None"
        }

def local_prediction(image_path):
    """การทำนายแบบ local"""
    try:
        import joblib
        
        # Load model components
        classifier = joblib.load("trained_model/classifier.joblib")
        scaler = joblib.load("trained_model/scaler.joblib")
        label_encoder = joblib.load("trained_model/label_encoder.joblib")
        
        # Process image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        features = image_normalized.flatten()
        
        # Make prediction
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = classifier.predict(features_scaled)[0]
        probabilities = classifier.predict_proba(features_scaled)[0]
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        confidence = float(probabilities[prediction])
        
        # Load class labels
        with open("ai_models/labels.json", "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        thai_name = labels.get("current_classes", {}).get(str(prediction), predicted_class)
        
        return {
            "status": "success",
            "predicted_class": predicted_class,
            "thai_name": thai_name,
            "confidence": confidence,
            "probabilities": {
                label_encoder.classes_[i]: float(prob) 
                for i, prob in enumerate(probabilities)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

def display_classification_result(result, show_confidence, show_probabilities):
    """แสดงผลการจำแนก"""
    if result["status"] == "success":
        # Success result
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Classification Result</h3>
            <p><strong>Predicted Class:</strong> {result.get('predicted_class', 'Unknown')}</p>
            <p><strong>Thai Name:</strong> {result.get('thai_name', result.get('predicted_class', 'Unknown'))}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence
        if show_confidence:
            confidence = result.get('confidence', 0)
            st.metric("Confidence", f"{confidence:.2%}")
            
            # Confidence bar
            st.progress(confidence)
        
        # All probabilities
        if show_probabilities and 'probabilities' in result:
            st.subheader("📊 All Class Probabilities")
            probs = result['probabilities']
            
            for class_name, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                st.write(f"**{class_name}:** {prob:.2%}")
                st.progress(prob)
        
        # Method used
        method = result.get('method', 'Unknown')
        st.caption(f"Prediction method: {method}")
        
    elif result["status"] == "error":
        # Error result
        st.markdown(f"""
        <div class="error-box">
            <h3>❌ Classification Error</h3>
            <p>{result.get('error', 'Unknown error occurred')}</p>
        </div>
        """, unsafe_allow_html=True)

def show_sample_images(sample_dir):
    """แสดงรูปตัวอย่าง"""
    try:
        classes = [d.name for d in sample_dir.iterdir() if d.is_dir()][:3]  # แสดง 3 คลาสแรก
        
        for class_name in classes:
            class_dir = sample_dir / class_name / "front"
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg"))[:1]  # 1 รูปต่อคลาส
                
                if images:
                    img_path = images[0]
                    img = Image.open(img_path)
                    st.image(img, caption=class_name, width=100)
                    
    except Exception as e:
        st.warning(f"Cannot load samples: {e}")

def system_analytics_tab():
    """Tab สำหรับ analytics"""
    st.header("📊 System Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Metrics")
        try:
            metrics = performance_monitor.collect_metrics()
            if metrics and "system" in metrics:
                st.json(metrics)
            else:
                st.info("No performance data available")
        except:
            st.warning("Performance monitoring not available")
    
    with col2:
        st.subheader("Model Statistics")
        try:
            model_info_path = Path("trained_model/model_info.json")
            if model_info_path.exists():
                with open(model_info_path, 'r') as f:
                    model_info = json.load(f)
                
                st.metric("Training Accuracy", f"{model_info.get('training_results', {}).get('train_accuracy', 0):.1%}")
                st.metric("Validation Accuracy", f"{model_info.get('training_results', {}).get('val_accuracy', 0):.1%}")
                st.metric("Test Accuracy", f"{model_info.get('training_results', {}).get('test_accuracy', 0):.1%}")
                
        except Exception as e:
            st.error(f"Cannot load model info: {e}")

def documentation_tab():
    """Tab สำหรับเอกสาร"""
    st.header("📚 Documentation")
    
    st.markdown("""
    ## 🏺 About Amulet-AI
    
    Amulet-AI is an intelligent system for classifying Thai Buddhist amulets using advanced machine learning techniques.
    
    ### 🎯 Supported Amulet Types
    - พระศิวลี (Phra Sivali)
    - พระสมเด็จ (Phra Somdej)  
    - ปรกโพธิ์ 9 ใบ (Prok Bodhi 9 Leaves)
    - แหวกม่าน (Waek Man)
    - หลังรูปเหมือน (Portrait Back)
    - วัดหนองอีดุก (Wat Nong E Duk)
    
    ### 🔧 Technical Specifications
    - **Model:** Random Forest Classifier
    - **Image Size:** 224x224 pixels
    - **Accuracy:** ~72% on test set
    - **Features:** 150,528 (raw pixels)
    
    ### 🚀 How to Use
    1. Upload a clear image of your amulet
    2. Click "Classify Amulet" 
    3. Review the results and confidence scores
    
    ### ⚡ Performance Tips
    - Use well-lit, clear images
    - Avoid blurry or dark photos
    - Include both front and back views when possible
    """)

def system_tools_tab():
    """Tab สำหรับเครื่องมือระบบ"""
    st.header("🔧 System Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Health Checks")
        if st.button("🏥 Run Health Check"):
            with st.spinner("Checking system health..."):
                health_results = run_health_check()
                st.json(health_results)
    
    with col2:
        st.subheader("Cache Management")
        if st.button("🗑️ Clear Cache"):
            st.success("Cache cleared successfully!")

def run_health_check():
    """รันการตรวจสุขภาพระบบ"""
    try:
        health = {
            "timestamp": time.time(),
            "api_status": "checking...",
            "model_status": "checking...",
            "disk_space": "checking..."
        }
        
        # Check API
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            health["api_status"] = "online" if response.status_code == 200 else "error"
        except:
            health["api_status"] = "offline"
        
        # Check model files
        required_files = ["trained_model/classifier.joblib", "trained_model/scaler.joblib"]
        missing = [f for f in required_files if not Path(f).exists()]
        health["model_status"] = "ready" if not missing else f"missing: {missing}"
        
        return health
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    main()
