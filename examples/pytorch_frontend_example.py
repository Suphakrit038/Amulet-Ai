#!/usr/bin/env python3
"""
Amulet-AI - PyTorch Frontend Integration Example
ตัวอย่างการใช้งาน Frontend พร้อม PyTorch Model

Features:
- PyTorch Transfer Learning Model (ResNet50/EfficientNet/MobileNet)
- Temperature Scaling Calibration
- Out-of-Distribution (OOD) Detection
- Grad-CAM Visualization
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

from model_training.transfer_learning import AmuletTransferModel
from evaluation.calibration import TemperatureScaling
from evaluation.ood_detection import IsolationForestDetector, extract_features
from explainability.gradcam import visualize_gradcam, generate_explanation


def load_model_components():
    """โหลดโมเดลและส่วนประกอบทั้งหมด"""
    print("Loading PyTorch model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model_path = project_root / "trained_model/best_model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Assuming ResNet50 with 10 classes (adjust as needed)
    model = AmuletTransferModel(
        backbone='resnet50',
        num_classes=10,
        pretrained=False
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded successfully")
    
    # Load temperature scaler (optional)
    temp_scaler = None
    temp_scaler_path = project_root / "trained_model/temperature_scaler.pth"
    if temp_scaler_path.exists():
        temp_scaler = TemperatureScaling()
        temp_scaler.load_state_dict(torch.load(temp_scaler_path, map_location=device))
        temp_scaler.to(device)
        print("✓ Temperature scaler loaded")
    else:
        print("⚠ Temperature scaler not found (optional)")
    
    # Load OOD detector (optional)
    ood_detector = None
    ood_detector_path = project_root / "trained_model/ood_detector.joblib"
    if ood_detector_path.exists():
        import joblib
        ood_detector = joblib.load(ood_detector_path)
        print("✓ OOD detector loaded")
    else:
        print("⚠ OOD detector not found (optional)")
    
    return {
        'model': model,
        'temperature_scaler': temp_scaler,
        'ood_detector': ood_detector,
        'device': device
    }


def preprocess_image(image_path):
    """Preprocess image for model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor, image


def predict_with_full_pipeline(image_path, components):
    """
    ทำนายพร้อมฟีเจอร์ครบ:
    - OOD Detection
    - Temperature Scaling
    - Grad-CAM Visualization
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    model = components['model']
    temp_scaler = components['temperature_scaler']
    ood_detector = components['ood_detector']
    device = components['device']
    
    # Preprocess
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # OOD Detection
    is_ood = False
    ood_score = None
    if ood_detector is not None:
        print("\n1. OOD Detection...")
        with torch.no_grad():
            features = extract_features(model, image_tensor, device)
            features_np = features.cpu().numpy()
            ood_score = ood_detector.score_samples(features_np)[0]
            is_ood = ood_detector.predict(features_np)[0] == -1
        
        print(f"   OOD Score: {ood_score:.4f}")
        print(f"   Is OOD: {'⚠️ YES' if is_ood else '✓ NO'}")
        
        if is_ood:
            print("   WARNING: Image may be out-of-distribution!")
    
    # Model Inference
    print("\n2. Model Inference...")
    with torch.no_grad():
        logits = model(image_tensor)
        
        # Apply temperature scaling
        if temp_scaler is not None:
            print("   Applying temperature scaling...")
            logits_calibrated = temp_scaler(logits)
        else:
            logits_calibrated = logits
        
        probs = F.softmax(logits_calibrated, dim=1)
        probs_np = probs.cpu().numpy()[0]
        
        predicted_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[predicted_idx])
    
    print(f"   Predicted Class: {predicted_idx}")
    print(f"   Confidence: {confidence:.2%}")
    
    # Grad-CAM Visualization
    print("\n3. Grad-CAM Visualization...")
    try:
        gradcam_result = visualize_gradcam(
            model=model,
            image_tensor=image_tensor,
            target_class=predicted_idx,
            device=device
        )
        print("   ✓ Grad-CAM generated successfully")
    except Exception as e:
        print(f"   ⚠ Grad-CAM failed: {e}")
        gradcam_result = None
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"Predicted Class: {predicted_idx}")
    print(f"Confidence: {confidence:.2%}")
    if is_ood:
        print("⚠️  WARNING: Out-of-Distribution detected!")
    if ood_score is not None:
        print(f"OOD Score: {ood_score:.4f}")
    
    # Show top-3 predictions
    print("\nTop-3 Predictions:")
    top_indices = np.argsort(probs_np)[::-1][:3]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. Class {idx}: {probs_np[idx]:.2%}")
    
    # Visualize if Grad-CAM available
    if gradcam_result is not None:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(gradcam_result)
        plt.title('Grad-CAM Heatmap')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(original_image)
        plt.imshow(gradcam_result, alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'gradcam_result_{Path(image_path).stem}.png', dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved: gradcam_result_{Path(image_path).stem}.png")
        plt.show()
    
    return {
        'predicted_class': predicted_idx,
        'confidence': confidence,
        'probabilities': probs_np,
        'is_ood': is_ood,
        'ood_score': ood_score,
        'gradcam': gradcam_result
    }


def main():
    """Main function"""
    print("="*60)
    print("Amulet-AI PyTorch Frontend Integration Example")
    print("="*60)
    
    # Check for sample images
    sample_images = list((project_root / "organized_dataset/raw/train").rglob("*.jpg"))[:3]
    
    if not sample_images:
        print("\n⚠️  No sample images found in organized_dataset/raw/train/")
        print("Please ensure you have trained the model and have sample images.")
        return
    
    print(f"\nFound {len(sample_images)} sample images")
    
    # Load model components
    try:
        components = load_model_components()
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nPlease ensure you have:")
        print("1. Trained the model (trained_model/best_model.pth)")
        print("2. Run Phase 2 calibration and OOD detection")
        return
    
    # Process each image
    for i, image_path in enumerate(sample_images, 1):
        try:
            result = predict_with_full_pipeline(image_path, components)
        except Exception as e:
            print(f"\n❌ Error processing {image_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Example completed!")
    print("="*60)
    print("\nTo use the full Streamlit frontend:")
    print("  cd frontend")
    print("  streamlit run production_app_clean.py")


if __name__ == "__main__":
    main()
