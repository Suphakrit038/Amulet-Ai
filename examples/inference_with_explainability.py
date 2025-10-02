"""
Complete Inference Example with Explainability

Demonstrates:
1. Loading trained model
2. Loading calibration & OOD components
3. Preprocessing input image
4. Making prediction with confidence
5. OOD detection
6. Grad-CAM visualization

Run with:
    python -m examples.inference_with_explainability

Author: Amulet-AI Team
Date: October 2, 2025
"""

import sys
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules
from model_training.transfer_learning import create_transfer_model
from evaluation.calibration import TemperatureScaling
from explainability.gradcam import visualize_gradcam, generate_explanation, get_target_layer
from torchvision import transforms


def load_model_components(model_dir: str = 'trained_model'):
    """
    Load all model components.
    
    Returns:
    --------
    dict : {
        'model': trained model,
        'temp_scaler': temperature scaler,
        'ood_detector': OOD detector,
        'config': model configuration,
        'transform': preprocessing transform
    }
    """
    model_path = Path(model_dir)
    
    # Load config
    with open(model_path / 'model_config.json', 'r') as f:
        config = json.load(f)
    
    print("[INFO] Loading model components...")
    print(f"  - Backbone: {config['backbone']}")
    print(f"  - Classes: {config['num_classes']}")
    print(f"  - Temperature: {config['temperature']:.4f}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_transfer_model(
        backbone=config['backbone'],
        num_classes=config['num_classes'],
        pretrained=False,  # Load trained weights
        device=device
    )
    
    # Load weights
    model.load_state_dict(torch.load(
        model_path / 'best_model.pth',
        map_location=device
    ))
    model.eval()
    print(f"  ✓ Model loaded")
    
    # Load temperature scaler
    temp_scaler = TemperatureScaling()
    temp_scaler.load_state_dict(torch.load(
        model_path / 'temperature_scaler.pth',
        map_location=device
    ))
    temp_scaler.eval()
    print(f"  ✓ Temperature scaler loaded")
    
    # Load OOD detector
    try:
        import joblib
        ood_detector = joblib.load(model_path / 'ood_detector.joblib')
        print(f"  ✓ OOD detector loaded")
    except:
        ood_detector = None
        print(f"  ! OOD detector not found (optional)")
    
    # Create preprocessing transform
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return {
        'model': model,
        'temp_scaler': temp_scaler,
        'ood_detector': ood_detector,
        'config': config,
        'transform': transform,
        'device': device
    }


def predict_with_explanation(
    image_path: str,
    components: dict,
    show_top_k: int = 3,
    ood_threshold: float = -0.1,
    confidence_threshold: float = 0.6
):
    """
    Make prediction with comprehensive explanation.
    
    Parameters:
    -----------
    image_path : str
        Path to image
    components : dict
        Model components from load_model_components()
    show_top_k : int
        Number of top predictions to show
    ood_threshold : float
        OOD score threshold
    confidence_threshold : float
        Minimum confidence for accepting prediction
        
    Returns:
    --------
    dict : Prediction results with explanations
    """
    model = components['model']
    temp_scaler = components['temp_scaler']
    ood_detector = components['ood_detector']
    config = components['config']
    transform = components['transform']
    device = components['device']
    
    # Load and preprocess image
    print(f"\n[INFO] Processing image: {image_path}")
    
    try:
        original_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[ERROR] Could not load image: {e}")
        return None
    
    # Transform
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # -------------------------------------------------------------------------
    # STEP 1: OOD Detection
    # -------------------------------------------------------------------------
    
    is_ood = False
    ood_score = None
    
    if ood_detector is not None:
        print("\n[1/4] OOD Detection...")
        
        # Extract features
        with torch.no_grad():
            features = model.get_features(input_tensor)
            features_np = features.cpu().numpy()
        
        # Check OOD
        ood_score = ood_detector.score(features_np)[0]
        is_ood = ood_score < ood_threshold
        
        print(f"  - OOD Score: {ood_score:.4f}")
        
        if is_ood:
            print(f"  ⚠️  WARNING: Out-of-distribution input detected!")
            print(f"     This image may not be from the training distribution.")
            print(f"     Prediction confidence may be unreliable.")
        else:
            print(f"  ✓ Input is in-distribution")
    
    # -------------------------------------------------------------------------
    # STEP 2: Prediction with Calibration
    # -------------------------------------------------------------------------
    
    print("\n[2/4] Making Prediction...")
    
    with torch.no_grad():
        # Get logits
        logits = model(input_tensor)
        
        # Apply temperature scaling for calibration
        calibrated_probs = temp_scaler(logits)
        calibrated_probs = calibrated_probs[0].cpu().numpy()
        
        # Get top-k predictions
        top_indices = np.argsort(calibrated_probs)[::-1][:show_top_k]
        top_probs = calibrated_probs[top_indices]
    
    # Print predictions
    print(f"\n  Top {show_top_k} Predictions:")
    class_names = config['class_names']
    
    for i, (idx, prob) in enumerate(zip(top_indices, top_probs)):
        marker = "🏆" if i == 0 else f" {i+1}."
        print(f"  {marker} {class_names[idx]}: {prob:.2%}")
    
    # Check confidence
    top_confidence = top_probs[0]
    
    if top_confidence < confidence_threshold:
        print(f"\n  ⚠️  WARNING: Low confidence ({top_confidence:.2%} < {confidence_threshold:.0%})")
        print(f"     Consider manual review by expert.")
    
    # -------------------------------------------------------------------------
    # STEP 3: Generate Grad-CAM Explanations
    # -------------------------------------------------------------------------
    
    print("\n[3/4] Generating Visual Explanations (Grad-CAM)...")
    
    # Get target layer
    target_layer = get_target_layer(model, architecture=config['backbone'].split('_')[0])
    
    # Generate explanations for top-k
    explanations = generate_explanation(
        model=model,
        image=original_image,
        target_layer=target_layer,
        transform=transform,
        class_names=class_names,
        top_k=show_top_k,
        method='gradcam'
    )
    
    print(f"  ✓ Generated Grad-CAM for top {show_top_k} predictions")
    
    # -------------------------------------------------------------------------
    # STEP 4: Visualize Results
    # -------------------------------------------------------------------------
    
    print("\n[4/4] Creating Visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(16, 4))
    
    # Original image
    ax1 = plt.subplot(1, show_top_k + 1, 1)
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Grad-CAM for each top prediction
    for i, exp in enumerate(explanations['top_predictions']):
        ax = plt.subplot(1, show_top_k + 1, i + 2)
        ax.imshow(exp['overlay'])
        
        title = f"{exp['class']}\nConf: {exp['confidence']:.2%}"
        color = 'green' if i == 0 else 'gray'
        ax.set_title(title, fontsize=11, fontweight='bold', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    vis_path = Path('visualization.png')
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Visualization saved: {vis_path}")
    
    plt.show()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Predicted Class: {class_names[top_indices[0]]}")
    print(f"📈 Confidence: {top_confidence:.2%}")
    
    if is_ood:
        print(f"⚠️  OOD Warning: Input may be out-of-distribution (score: {ood_score:.4f})")
    
    if top_confidence < confidence_threshold:
        print(f"⚠️  Confidence Warning: Below threshold ({confidence_threshold:.0%})")
        print(f"   → Recommendation: Request expert review")
    else:
        print(f"✓ Confidence: Above threshold ({confidence_threshold:.0%})")
        print(f"   → Recommendation: Prediction is reliable")
    
    print("\n" + "=" * 80)
    
    # Return results
    results = {
        'predicted_class': class_names[top_indices[0]],
        'predicted_index': int(top_indices[0]),
        'confidence': float(top_confidence),
        'top_k_predictions': [
            {
                'class': class_names[idx],
                'confidence': float(prob)
            }
            for idx, prob in zip(top_indices, top_probs)
        ],
        'is_ood': is_ood,
        'ood_score': float(ood_score) if ood_score is not None else None,
        'should_review': (top_confidence < confidence_threshold) or is_ood,
        'explanations': explanations
    }
    
    return results


def main():
    """
    Example usage of complete inference pipeline.
    """
    print("=" * 80)
    print("AMULET-AI INFERENCE WITH EXPLAINABILITY")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    MODEL_DIR = 'trained_model'
    IMAGE_PATH = 'test_image.jpg'  # Replace with your test image
    
    # Thresholds
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to accept prediction
    OOD_THRESHOLD = -0.1  # OOD score threshold
    SHOW_TOP_K = 3  # Number of top predictions to show
    
    # -------------------------------------------------------------------------
    # Load Model
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("LOADING MODEL COMPONENTS")
    print("=" * 80)
    
    try:
        components = load_model_components(MODEL_DIR)
    except Exception as e:
        print(f"\n[ERROR] Could not load model: {e}")
        print("\n[INFO] Make sure you have:")
        print("  1. Trained a model (run complete_training_example.py)")
        print("  2. Model saved in 'trained_model/' directory")
        return
    
    # -------------------------------------------------------------------------
    # Make Prediction
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("MAKING PREDICTION")
    print("=" * 80)
    
    # Check if image exists
    if not Path(IMAGE_PATH).exists():
        print(f"\n[ERROR] Image not found: {IMAGE_PATH}")
        print("\n[INFO] Usage:")
        print(f"  python -m examples.inference_with_explainability")
        print(f"\n  Or provide custom image path:")
        print(f"  IMAGE_PATH = 'path/to/your/image.jpg'")
        return
    
    # Run inference
    results = predict_with_explanation(
        image_path=IMAGE_PATH,
        components=components,
        show_top_k=SHOW_TOP_K,
        ood_threshold=OOD_THRESHOLD,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    if results is None:
        return
    
    # -------------------------------------------------------------------------
    # Additional Examples
    # -------------------------------------------------------------------------
    
    print("\n" + "=" * 80)
    print("INTEGRATION EXAMPLES")
    print("=" * 80)
    
    print("\n[INFO] Example 1: API Integration")
    print("""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    components = load_model_components('trained_model')
    
    @app.route('/predict', methods=['POST'])
    def predict():
        image = request.files['image']
        results = predict_with_explanation(image, components)
        
        return jsonify({
            'class': results['predicted_class'],
            'confidence': results['confidence'],
            'should_review': results['should_review'],
            'top_predictions': results['top_k_predictions']
        })
    """)
    
    print("\n[INFO] Example 2: Batch Processing")
    print("""
    from pathlib import Path
    
    components = load_model_components('trained_model')
    
    for image_path in Path('test_images').glob('*.jpg'):
        results = predict_with_explanation(
            image_path=str(image_path),
            components=components,
            show_top_k=1
        )
        
        if results['should_review']:
            print(f"⚠️  {image_path.name} needs expert review")
    """)
    
    print("\n[INFO] Example 3: Streamlit UI")
    print("""
    import streamlit as st
    
    st.title('Amulet Classifier')
    
    uploaded = st.file_uploader('Upload Image', type=['jpg', 'png'])
    
    if uploaded:
        results = predict_with_explanation(uploaded, components)
        
        st.write(f"**Prediction:** {results['predicted_class']}")
        st.write(f"**Confidence:** {results['confidence']:.2%}")
        
        if results['should_review']:
            st.warning('⚠️ Expert review recommended')
        
        # Show Grad-CAM
        st.image(results['explanations']['top_predictions'][0]['overlay'])
    """)
    
    print("\n" + "=" * 80)
    print("INFERENCE COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
