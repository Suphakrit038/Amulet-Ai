# 📚 Amulet-AI Examples

Complete working examples demonstrating all features of the Amulet-AI system.

---

## 🎯 Available Examples

### 1. Complete Training Pipeline ⭐ RECOMMENDED

**File**: `complete_training_example.py`  
**Lines**: ~500  
**Time**: 30-60 minutes (with dataset)

**What it demonstrates**:
- ✅ Data loading with augmentation presets
- ✅ Class distribution analysis
- ✅ Balanced sampler for imbalanced data
- ✅ Two-stage transfer learning (freeze → fine-tune)
- ✅ Comprehensive evaluation (per-class F1, balanced accuracy)
- ✅ Model calibration with temperature scaling
- ✅ OOD detector training
- ✅ Complete model saving

**Run**:
```bash
python -m examples.complete_training_example
```

**Output**:
```
trained_model/
├── best_model.pth              # Trained model weights
├── temperature_scaler.pth      # Calibration component
├── ood_detector.joblib         # OOD detector
├── model_config.json           # Configuration
├── test_metrics.json           # Evaluation results
└── logs/                       # Training logs
    ├── experiment.csv
    └── experiment.json
```

---

### 2. Inference with Explainability

**File**: `inference_with_explainability.py`  
**Lines**: ~400  
**Time**: < 1 minute per image

**What it demonstrates**:
- ✅ Loading trained model + components
- ✅ Image preprocessing
- ✅ OOD detection check
- ✅ Calibrated prediction with confidence
- ✅ Top-K predictions
- ✅ Grad-CAM visualization
- ✅ Human-in-the-loop workflow

**Run**:
```bash
# Make sure you have trained a model first
python -m examples.complete_training_example

# Then run inference
python -m examples.inference_with_explainability
```

**Output**:
- Grad-CAM visualizations for top-3 predictions
- Confidence scores (calibrated)
- OOD warning if applicable
- Recommendation for expert review
- Saved visualization: `visualization.png`

---

### 3. Data Management Examples

**Files**: 
- `data_management/examples/example_usage.py` (All features)
- `data_management/examples/complete_examples.py` (Real workflows)

**What it demonstrates**:
- ✅ All augmentation presets (minimal → heavy)
- ✅ MixUp and CutMix in action
- ✅ Preprocessing pipelines
- ✅ Quality checking
- ✅ Balanced sampling strategies
- ✅ Dataset splitting

**Run**:
```bash
python -m data_management.examples.example_usage
```

---

## 🚀 Quick Start

### Step 1: Prepare Dataset

```
organized_dataset/
├── train/
│   ├── class_0/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   ├── class_1/
│   └── ...
├── val/
│   ├── class_0/
│   └── ...
└── test/
    ├── class_0/
    └── ...
```

**Tips**:
- Minimum 50 images per class for training
- Recommended 200+ images per class
- Use stratified split (70/15/15 or 80/10/10)
- Images should be clear, properly lit
- Run quality checker first: `QualityChecker().batch_check()`

---

### Step 2: Train Model

```bash
# Run complete training example
python -m examples.complete_training_example
```

**Expected output**:
```
================================================================================
AMULET-AI COMPLETE TRAINING EXAMPLE
================================================================================

[INFO] Using device: cuda

================================================================================
STEP 1: DATA LOADING & AUGMENTATION
================================================================================

[INFO] Creating augmentation pipeline (preset: medium)...
[INFO] Augmentation configuration:
  - image_size: 224
  - rand_augment_n: 2
  - rand_augment_m: 9
  - random_erasing_p: 0.5
  - mixup_alpha: 0.2
  - cutmix_alpha: 1.0

[INFO] Loading datasets...
  ✓ Train: 1200 images
  ✓ Val: 300 images
  ✓ Test: 300 images

================================================================================
STEP 2: CLASS BALANCE ANALYSIS
================================================================================

[INFO] Training set class distribution:
  Class 0: 250 images (20.83%)
  Class 1: 180 images (15.00%)
  Class 2: 200 images (16.67%)
  Class 3: 220 images (18.33%)
  Class 4: 150 images (12.50%)
  Class 5: 200 images (16.67%)
  
  Imbalance Ratio: 1.67
  Recommendation: Use weighted sampler

[INFO] Creating weighted sampler for balanced training...

... [training continues] ...

================================================================================
TRAINING COMPLETE!
================================================================================

✓ Model: resnet50
✓ Test Accuracy: 92.30%
✓ Macro F1: 0.9180
✓ Balanced Accuracy: 0.9210
✓ ECE (calibrated): 0.0520
✓ All artifacts saved to: E:\Amulet-Ai\trained_model
```

**Training time**: ~30 minutes (1,500 images, RTX 3090)

---

### Step 3: Test Inference

```bash
# Replace test_image.jpg with your image
python -m examples.inference_with_explainability
```

**Expected output**:
```
================================================================================
AMULET-AI INFERENCE WITH EXPLAINABILITY
================================================================================

[INFO] Loading model components...
  - Backbone: resnet50
  - Classes: 6
  - Temperature: 1.4523
  ✓ Model loaded
  ✓ Temperature scaler loaded
  ✓ OOD detector loaded

[INFO] Processing image: test_image.jpg

[1/4] OOD Detection...
  - OOD Score: 0.2341
  ✓ Input is in-distribution

[2/4] Making Prediction...
  Top 3 Predictions:
  🏆 Class A: 85.23%
   2. Class B: 8.45%
   3. Class C: 3.21%

[3/4] Generating Visual Explanations (Grad-CAM)...
  ✓ Generated Grad-CAM for top 3 predictions

[4/4] Creating Visualization...
  ✓ Visualization saved: visualization.png

================================================================================
PREDICTION SUMMARY
================================================================================

📊 Predicted Class: Class A
📈 Confidence: 85.23%
✓ Confidence: Above threshold (60%)
   → Recommendation: Prediction is reliable

================================================================================
```

---

## 📝 Customization

### Customize Training

Edit `examples/complete_training_example.py`:

```python
# Change configuration
BACKBONE = 'efficientnet_b0'  # or 'mobilenet_v2', 'resnet101'
AUG_PRESET = 'heavy'           # or 'light', 'minimal'
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 5e-4

# Change model architecture
model = create_transfer_model(
    backbone=BACKBONE,
    num_classes=NUM_CLASSES,
    hidden_dim=256,        # Default: 128
    dropout=0.5,           # Default: 0.3
    num_fc_layers=2        # Default: 1
)

# Change training strategy
history1 = trainer.train_stage1(
    train_loader, val_loader,
    epochs=10,             # Default: 5
    lr=5e-3,              # Default: 1e-3
    patience=5            # Default: 3
)

history2 = trainer.train_stage2(
    train_loader, val_loader,
    epochs=50,            # Default: 20
    lr=1e-5,             # Default: 1e-4
    unfreeze_layers=20,   # Default: 10
    patience=10          # Default: 5
)
```

---

### Customize Inference

Edit `examples/inference_with_explainability.py`:

```python
# Change thresholds
CONFIDENCE_THRESHOLD = 0.75  # Default: 0.6
OOD_THRESHOLD = 0.0          # Default: -0.1
SHOW_TOP_K = 5               # Default: 3

# Change Grad-CAM method
results = predict_with_explanation(
    image_path=IMAGE_PATH,
    components=components,
    method='gradcam++'  # Default: 'gradcam'
)

# Use different explainability
from explainability.saliency import generate_smoothgrad

saliency = generate_smoothgrad(
    model, image_tensor,
    n_samples=100,       # Default: 50
    noise_level=0.2     # Default: 0.15
)
```

---

## 🎓 Advanced Examples

### Example A: Custom Augmentation

```python
from data_management.augmentation import AugmentationPipeline

# Create custom pipeline
pipeline = AugmentationPipeline(
    image_size=224,
    batch_size=32,
    num_classes=6,
    
    # Customize RandAugment
    rand_augment_n=3,        # More operations
    rand_augment_m=12,       # Higher magnitude
    
    # Customize MixUp/CutMix
    mixup_alpha=0.4,         # More mixing
    cutmix_alpha=1.0,
    mixup_prob=0.5,          # 50% MixUp
    cutmix_prob=0.5,         # 50% CutMix
    
    # Customize RandomErasing
    random_erasing_p=0.7,    # Higher probability
    random_erasing_scale=(0.02, 0.4),
    random_erasing_ratio=(0.3, 3.3)
)

train_transform = pipeline.get_transform('train')
train_loader = pipeline.create_dataloader(dataset, mode='train')
```

---

### Example B: Advanced Evaluation

```python
from evaluation.metrics import compute_per_class_metrics
from evaluation.calibration import evaluate_calibration
from evaluation.ood_detection import compute_ood_auroc

# 1. Detailed per-class metrics
metrics = evaluate_model(model, test_loader, device, class_names)
metrics.print_report()

# Get per-class results
for i, class_name in enumerate(class_names):
    print(f"{class_name}:")
    print(f"  Precision: {metrics.per_class_precision[i]:.4f}")
    print(f"  Recall: {metrics.per_class_recall[i]:.4f}")
    print(f"  F1: {metrics.per_class_f1[i]:.4f}")

# 2. Calibration analysis
calib_results = evaluate_calibration(
    model, test_loader, device,
    temp_scaler=temp_scaler,
    n_bins=20  # More fine-grained
)

print(f"ECE: {calib_results['ece']:.4f}")
print(f"Brier: {calib_results['brier_score']:.4f}")

# 3. OOD evaluation
from sklearn.datasets import load_sample_images

# Get OOD images (e.g., natural images)
ood_images = load_sample_images().images
ood_features = extract_features_from_images(model, ood_images)

# Compute AUROC
auroc = compute_ood_auroc(
    in_distribution_features=train_features,
    out_distribution_features=ood_features,
    detector=ood_detector
)

print(f"OOD Detection AUROC: {auroc:.4f}")
```

---

### Example C: FID/KID Validation

```python
from evaluation.fid_kid import compute_fid, compute_kid

# Load real and synthetic images
real_images = load_images('organized_dataset/train/class_0')
synthetic_images = load_images('synthetic_data/class_0')

# Compute FID
fid = compute_fid(
    real_images, synthetic_images,
    device='cuda',
    batch_size=32
)

print(f"FID Score: {fid:.2f}")
print(f"Interpretation: {'Good' if fid < 50 else 'Fair' if fid < 100 else 'Poor'}")

# Compute KID (better for small datasets)
kid = compute_kid(
    real_images, synthetic_images,
    device='cuda',
    batch_size=32
)

print(f"KID Score: {kid:.6f}")

# Classwise FID
for class_idx in range(num_classes):
    real_cls = load_images(f'organized_dataset/train/class_{class_idx}')
    synth_cls = load_images(f'synthetic_data/class_{class_idx}')
    
    fid_cls = compute_fid(real_cls, synth_cls, device='cuda')
    print(f"Class {class_idx} FID: {fid_cls:.2f}")
```

---

## 🐛 Troubleshooting

### Problem 1: Dataset Not Found

**Error**:
```
[ERROR] Could not load dataset: FileNotFoundError
```

**Solution**:
```python
# Check dataset structure
import os
dataset_root = 'organized_dataset'

for split in ['train', 'val', 'test']:
    split_dir = os.path.join(dataset_root, split)
    if not os.path.exists(split_dir):
        print(f"Missing: {split_dir}")
        os.makedirs(split_dir, exist_ok=True)

# Dataset should have this structure:
# organized_dataset/
#   train/class_0/, train/class_1/, ...
#   val/class_0/, val/class_1/, ...
#   test/class_0/, test/class_1/, ...
```

---

### Problem 2: CUDA Out of Memory

**Error**:
```
RuntimeError: CUDA out of memory
```

**Solution**:
```python
# In complete_training_example.py

# Reduce batch size
BATCH_SIZE = 16  # Was 32

# Or use CPU
device = torch.device('cpu')

# Or enable gradient accumulation
config = TrainingConfig(
    batch_size=16,
    accumulation_steps=2  # Effective batch = 32
)
```

---

### Problem 3: Model Not Found (Inference)

**Error**:
```
[ERROR] Could not load model: FileNotFoundError
```

**Solution**:
```bash
# Make sure you trained the model first
python -m examples.complete_training_example

# This will create trained_model/ directory with:
# - best_model.pth
# - temperature_scaler.pth
# - ood_detector.joblib
# - model_config.json
```

---

### Problem 4: Import Errors

**Error**:
```
ModuleNotFoundError: No module named 'data_management'
```

**Solution**:
```bash
# Make sure you're in the project root
cd /path/to/Amulet-Ai

# Run with -m flag (module mode)
python -m examples.complete_training_example

# Not: python examples/complete_training_example.py
```

---

## 📚 Additional Resources

- **System Documentation**: [`README_ML_SYSTEM.md`](../README_ML_SYSTEM.md)
- **Quick Start**: [`QUICK_START.md`](../QUICK_START.md)
- **Complete Summary**: [`PHASE1_2_COMPLETE_SUMMARY.md`](../PHASE1_2_COMPLETE_SUMMARY.md)

---

## 🎯 Next Steps

1. ✅ Run `complete_training_example.py` to train your first model
2. ✅ Run `inference_with_explainability.py` to test predictions
3. ✅ Customize parameters for your specific dataset
4. ✅ Integrate with your application (API, Streamlit, etc.)
5. ✅ Deploy to production

---

**Happy Training! 🚀**

*Amulet-AI Team - October 2, 2025*
