# 🚀 Amulet-AI Quick Start Guide

**Train a production-ready Thai amulet classifier in 5 minutes!**

---

## ⚡ 30-Second Setup

```bash
# 1. Install dependencies
pip install torch torchvision numpy pillow scikit-learn scipy opencv-python

# 2. Prepare your dataset
# organized_dataset/
#   train/
#     class_0/
#     class_1/
#     ...
#   val/
#   test/

# 3. Train!
python -m examples.complete_training_example
```

That's it! 🎉

---

## 📖 What You Get

### ✅ Complete ML Pipeline (Phases 1 + 2)

**Phase 1: Data Management** (COMPLETE)
- ✅ Advanced Augmentation (MixUp, CutMix, RandAugment, RandomErasing)
- ✅ Preprocessing (CLAHE, Denoising, Quality Check)
- ✅ Balanced Sampling (Weighted, Stratified)
- ✅ Dataset Tools (Splitting, Analysis)

**Phase 2: Transfer Learning & Evaluation** (COMPLETE)
- ✅ Transfer Learning (ResNet, EfficientNet, MobileNet)
- ✅ Two-Stage Training (Freeze → Fine-tune)
- ✅ Complete Trainer (Mixed Precision, Callbacks)
- ✅ Per-Class Metrics (F1, Balanced Accuracy)
- ✅ Calibration (Temperature Scaling, ECE)
- ✅ OOD Detection (IsolationForest, Mahalanobis)
- ✅ FID/KID for Synthetic Validation
- ✅ Grad-CAM Explainability

**Total**: 30+ files, ~15,000+ lines of production code

---

## 🎯 Quick Examples

### Example 1: Complete Training (Recommended)

```python
from data_management.augmentation import create_pipeline_from_preset
from data_management.dataset import create_amulet_dataset, create_balanced_sampler
from model_training.transfer_learning import create_transfer_model, TwoStageTrainer

# 1. Setup augmentation
pipeline = create_pipeline_from_preset('medium', batch_size=32)

# 2. Load data
train_dataset = create_amulet_dataset('organized_dataset', 'train', 
                                      transform=pipeline.get_transform('train'))
val_dataset = create_amulet_dataset('organized_dataset', 'val',
                                    transform=pipeline.get_transform('val'))

# 3. Balanced sampling
labels = [label for _, label in train_dataset]
sampler = create_balanced_sampler(labels, strategy='weighted')
train_loader = pipeline.create_dataloader(train_dataset, mode='train', sampler=sampler)
val_loader = pipeline.create_dataloader(val_dataset, mode='val')

# 4. Create model
device = 'cuda'
model = create_transfer_model('resnet50', num_classes=6, device=device)

# 5. Two-stage training
criterion = torch.nn.CrossEntropyLoss()
trainer = TwoStageTrainer(model, criterion, device)

# Stage 1: Train head only (3-5 epochs)
trainer.train_stage1(train_loader, val_loader, epochs=5, lr=1e-3)

# Stage 2: Fine-tune (15-20 epochs)
trainer.train_stage2(train_loader, val_loader, epochs=20, lr=1e-4, unfreeze_layers=10)

# 6. Save
torch.save(model.state_dict(), 'trained_model/best_model.pth')
```

**Time**: ~30 minutes on 1,000 images with GPU

---

### Example 2: Quick Evaluation

```python
from evaluation.metrics import evaluate_model
from evaluation.calibration import calibrate_model

# Evaluate
metrics = evaluate_model(model, test_loader, device)
metrics.print_report()

# Output:
# Class 0: Precision=0.92, Recall=0.89, F1=0.90
# Class 1: Precision=0.88, Recall=0.91, F1=0.89
# ...
# Macro F1: 0.895
# Balanced Accuracy: 0.893

# Calibrate
temp_scaler = calibrate_model(model, val_loader, device)
print(f"Temperature: {temp_scaler.temperature.item():.4f}")

# Use calibrated predictions
calibrated_probs = temp_scaler(model(image))
```

---

### Example 3: Inference with Grad-CAM

```python
from explainability.gradcam import visualize_gradcam, get_target_layer

# Get target layer
target_layer = get_target_layer(model, architecture='resnet')

# Generate Grad-CAM
overlay = visualize_gradcam(
    model=model,
    image='test_amulet.jpg',
    target_layer=target_layer,
    transform=transform,
    class_names=['Class A', 'Class B', ...]
)

# Show
from PIL import Image
Image.fromarray(overlay).show()
```

---

### Example 4: OOD Detection

```python
from evaluation.ood_detection import IsolationForestDetector, extract_features

# Train detector on in-distribution data
train_features, _ = extract_features(model, train_loader, device)
detector = IsolationForestDetector(contamination=0.01)
detector.fit(train_features)

# Check test image
test_features = model.get_features(test_image)
is_ood = detector.predict(test_features.cpu().numpy())[0] == -1

if is_ood:
    print("⚠️ Out-of-distribution input! Prediction may be unreliable.")
```

---

## 📊 Performance Guide

### Dataset Size Strategy

| Dataset Size | Aug Preset | Training Time | Expected Accuracy |
|--------------|------------|---------------|-------------------|
| < 500 images | `light` | 10-15 min | 75-85% |
| 500-2000 | `medium` | 20-40 min | 85-92% |
| > 2000 | `heavy` | 1-2 hours | 90-95% |

*Time estimates for 50 epochs with GPU*

### Backbone Selection

| Backbone | Speed | Accuracy | When to Use |
|----------|-------|----------|-------------|
| **MobileNetV2** | ⚡⚡⚡ Fast | 😊 Good | Mobile, fast inference |
| **ResNet50** | ⚡⚡ Medium | 😄 Better | Balanced (recommended) |
| **EfficientNet-B0** | ⚡⚡ Medium | 😄 Better | Best accuracy/size |
| **EfficientNet-B3** | ⚡ Slow | 😍 Best | Maximum accuracy |

---

## 🎓 Best Practices Checklist

### ✅ Data Preparation
- [ ] Use stratified split (train/val/test)
- [ ] Analyze class distribution
- [ ] Use weighted sampler for imbalanced data
- [ ] Apply appropriate augmentation preset
- [ ] Run quality checks on images

### ✅ Training
- [ ] Use transfer learning (freeze → fine-tune)
- [ ] Enable mixed precision (AMP) for speed
- [ ] Use weighted loss for imbalanced classes
- [ ] Apply early stopping (patience=5-10)
- [ ] Monitor both loss and per-class F1

### ✅ Evaluation
- [ ] Report per-class metrics (not just accuracy)
- [ ] Compute balanced accuracy
- [ ] Check confusion matrix for class confusion
- [ ] Calibrate model (temperature scaling)
- [ ] Measure ECE (< 0.1 is good)

### ✅ Deployment
- [ ] Setup OOD detection
- [ ] Set confidence threshold (0.6-0.75)
- [ ] Enable Grad-CAM for explanations
- [ ] Add "review by expert" workflow
- [ ] Monitor drift over time

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size
```python
pipeline = create_pipeline_from_preset('medium', batch_size=16)  # Was 32
```

**Solution 2**: Disable mixed precision
```python
config = TrainingConfig(mixed_precision=False)
```

**Solution 3**: Use smaller backbone
```python
model = create_transfer_model('mobilenet_v2', num_classes=6)  # Was resnet50
```

---

### Issue: Low F1 for Minority Class

**Solution 1**: Weighted sampler (RECOMMENDED)
```python
sampler = create_balanced_sampler(labels, strategy='weighted')
train_loader = DataLoader(dataset, sampler=sampler, batch_size=32)
```

**Solution 2**: Class weights in loss
```python
class_weights = 1.0 / np.bincount(labels)
class_weights = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**Solution 3**: Both (most effective)
```python
# Use weighted sampler + weighted loss together
```

---

### Issue: Model Overconfident (High ECE)

**Solution**: Temperature scaling
```python
# After training
temp_scaler = calibrate_model(model, val_loader, device)

# At inference
calibrated_probs = temp_scaler(model(image))
```

Typical ECE reduction: 0.15 → 0.05

---

### Issue: Training Too Slow

**Speed-up tricks**:

1. **Enable AMP**: 1.5-2x faster
```python
config = TrainingConfig(mixed_precision=True)
```

2. **Reduce image size**: 224 → 192
```python
pipeline = create_pipeline_from_preset('medium', image_size=192)
```

3. **Use smaller backbone**:
```python
model = create_transfer_model('mobilenet_v2', num_classes=6)
# 3x faster than ResNet50
```

4. **Gradient accumulation** (simulate larger batch):
```python
config = TrainingConfig(
    batch_size=16,  # Actual batch
    accumulation_steps=2  # Effective batch = 32
)
```

---

## 📚 Full Documentation

See [`README_ML_SYSTEM.md`](README_ML_SYSTEM.md) for:
- Complete API reference
- Advanced usage examples
- Research paper references
- Deployment guide

---

## 🎯 Next Steps

1. **Train your first model**:
   ```bash
   python -m examples.complete_training_example
   ```

2. **Test inference**:
   ```bash
   python -m examples.inference_with_explainability
   ```

3. **Integrate with your app**:
   - See integration examples in `examples/`
   - Check Streamlit UI in `frontend/`

4. **Deploy to production**:
   - Export to ONNX for faster inference
   - Setup FastAPI endpoint
   - Add monitoring & logging

---

## 💡 Tips & Tricks

### Tip 1: Start Small, Scale Up
```python
# Phase 1: Quick baseline (5 min)
model = create_transfer_model('mobilenet_v2', num_classes=6)
trainer.train_stage1(train_loader, val_loader, epochs=3)

# Phase 2: Full training if baseline works
model = create_transfer_model('resnet50', num_classes=6)
# ... full two-stage training
```

### Tip 2: Use Presets
```python
# Don't manually tune augmentation first time
pipeline = create_pipeline_from_preset('medium')  # Works 80% of time

# Only customize if needed
custom_config = {
    'mixup_alpha': 0.3,  # Increase for more mixing
    'rand_augment_m': 12  # Stronger augmentation
}
pipeline = create_pipeline_from_preset('medium', **custom_config)
```

### Tip 3: Monitor What Matters
```python
# Not just accuracy!
callbacks = create_default_callbacks(
    monitor='val_f1',  # ← Better than 'val_loss'
    patience=10
)
```

### Tip 4: Save Everything
```python
# Save model + calibration + OOD detector together
torch.save({
    'model': model.state_dict(),
    'temp_scaler': temp_scaler.state_dict(),
    'config': config
}, 'complete_model.pth')
```

---

## 🎉 You're Ready!

Start training your production-ready amulet classifier now! 🚀

**Questions?** Check [`README_ML_SYSTEM.md`](README_ML_SYSTEM.md) for detailed documentation.

**Issues?** See Troubleshooting section above.

---

**Built with ❤️ by Amulet-AI Team**  
*October 2, 2025*
