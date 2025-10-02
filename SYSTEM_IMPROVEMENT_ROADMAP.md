# 🚀 Amulet-AI System Improvement Roadmap

**วันที่จัดทำ**: 2 ตุลาคม 2025  
**สถานะปัจจุบัน**: Phase 1+2 Complete (Data Management + Transfer Learning)  
**เป้าหมาย**: ปรับปรุงระบบให้พร้อม Production ตามหลักการ ML Best Practices

---

## 📊 สถานะปัจจุบัน (Existing Assets Analysis)

### ✅ สิ่งที่มีอยู่แล้ว (Good Foundation)

#### 1. Data Management (Phase 1) ✅
- **Augmentation**: 
  - ✅ MixUp (alpha=0.2) - `advanced_augmentation.py`
  - ✅ CutMix (alpha=1.0) - `advanced_augmentation.py`
  - ✅ RandAugment (n=2, m=9) - `advanced_augmentation.py`
  - ✅ RandomErasing (p=0.5) - `advanced_augmentation.py`
- **Preprocessing**: CLAHE, Denoising, Quality Check
- **Dataset Tools**: Splitter, Analyzer

#### 2. Model Training (Phase 2) ✅
- **Transfer Learning**:
  - ✅ 6 backbones (ResNet50/101, EfficientNet B0/B3, MobileNet V2/V3)
  - ✅ Freeze/Unfreeze strategy - `transfer_learning.py`
  - ✅ Two-stage training - `trainer.py`
- **Callbacks**: EarlyStopping, ModelCheckpoint, LRScheduler

#### 3. Evaluation (Phase 2) ✅
- **Calibration**:
  - ✅ Temperature Scaling - `calibration.py`
  - ✅ ECE (Expected Calibration Error)
  - ✅ Brier Score
- **OOD Detection**:
  - ✅ IsolationForest - `ood_detection.py`
  - ✅ Mahalanobis Distance
- **Metrics**: Per-class F1, Balanced Accuracy
- **Explainability**: Grad-CAM, Grad-CAM++, Saliency

#### 4. Synthetic Validation ⚠️
- ✅ FID/KID implementation exists - `fid_kid.py`
- ❌ **Missing**: Integration with training pipeline
- ❌ **Missing**: Expert review workflow

---

## 🎯 ปัญหาที่ต้องแก้ (Gap Analysis)

### 🔴 Critical Issues

1. **Stratified Sampling with Variance** ❌
   - มี `sampler.py` แต่ไม่มี WeightedRandomSampler + strong augmentation
   - ไม่มี variance strategy สำหรับ minority classes

2. **Synthetic Data Validation Pipeline** ⚠️
   - มี FID/KID แต่ไม่ integrate กับ training
   - ไม่มี expert review workflow
   - ไม่มี A/B testing framework

3. **Model Selection & Architecture Experiment** ⚠️
   - มี backbones แต่ไม่มี systematic comparison
   - ไม่มี ablation study workflow
   - ไม่มี per-class performance tracking

4. **Deployment Optimization** ❌
   - ไม่มี quantization (INT8)
   - ไม่มี TorchScript/ONNX export
   - ไม่มี inference benchmarking

5. **MLOps & Monitoring** ❌
   - ไม่มี data drift detection
   - ไม่มี model versioning system
   - ไม่มี A/B testing framework

---

## 📋 Implementation Checklist (Prioritized)

### Phase 3A: Core ML Improvements (Week 1-2) 🔥

#### Priority 1: Stratified Sampling + Variance
- [ ] **Task 1.1**: Enhance `sampler.py` with WeightedRandomSampler
  - Add variance-aware oversampling
  - Integrate with augmentation pipeline
  - Test on imbalanced dataset
  
- [ ] **Task 1.2**: Create balanced dataloader wrapper
  ```python
  # File: data_management/dataset/balanced_loader.py
  class BalancedDataLoader:
      """WeightedRandomSampler + strong augmentation"""
      def __init__(self, dataset, class_weights, augment_strength='strong')
      def __iter__(self): # apply per-sample augmentation
  ```

- [ ] **Task 1.3**: Add class weights to loss
  ```python
  # File: model_training/trainer.py
  # Modify AmuletTrainer to accept class_weights
  criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
  ```

**Expected Outcome**: 
- Minority class F1 improvement: +10-15%
- No memorization (verify with train/val gap)

---

#### Priority 2: Model Architecture Experiments
- [ ] **Task 2.1**: Create experiment framework
  ```python
  # File: experiments/model_comparison.py
  class ModelExperiment:
      def run_baseline(self): # current small CNN
      def run_tl_head_only(self, backbone): # freeze backbone
      def run_tl_finetune(self, backbone, num_layers):
      def compare_results(self): # per-class F1, latency
  ```

- [ ] **Task 2.2**: Run systematic comparison
  - Baseline: Current CNN
  - TL-head (ResNet50, EfficientNet-B0, MobileNet-V2)
  - TL-finetune (best from above, unfreeze last 10 layers)
  - Document: accuracy, latency (p95), model size

- [ ] **Task 2.3**: Select production model
  - Criteria: F1 > 0.85, latency < 2s (p95), size < 50MB
  - Write decision report

**Expected Outcome**:
- F1 improvement: +15-20% over baseline
- Optimal backbone selected with justification

---

#### Priority 3: Comprehensive Evaluation
- [ ] **Task 3.1**: Extend metrics tracking
  ```python
  # File: evaluation/metrics.py
  class ComprehensiveMetrics:
      def compute_all(self, y_true, y_pred, y_prob):
          return {
              'per_class_f1': ...,
              'balanced_accuracy': ...,
              'confusion_matrix': ...,
              'ece': ...,  # after temperature scaling
              'brier_score': ...,
              'auroc_ood': ...  # if OOD test set available
          }
  ```

- [ ] **Task 3.2**: Create ablation study framework
  ```python
  # File: experiments/ablation_studies.py
  def compare_augmentations():
      # Baseline vs +MixUp vs +CutMix vs +Both
      # Track per-class F1, calibration, OOD
  ```

- [ ] **Task 3.3**: Integrate temperature scaling into pipeline
  - Split validation → val_train + val_calib
  - Auto-fit temperature after training
  - Save temperature with model checkpoint

**Expected Outcome**:
- ECE < 0.05 (well-calibrated)
- Per-class F1 report for all experiments
- Ablation study insights documented

---

### Phase 3B: Synthetic Data Validation (Week 3) 🧪

#### Priority 4: FID/KID Integration
- [ ] **Task 4.1**: Create synthetic validation pipeline
  ```python
  # File: evaluation/synthetic_validator.py
  class SyntheticValidator:
      def compute_classwise_fid(self, real_dir, synthetic_dir):
      def compute_global_fid(self, real_dir, synthetic_dir):
      def validate_quality(self, threshold=200):
          # Accept if FID < threshold
  ```

- [ ] **Task 4.2**: A/B testing framework
  ```python
  # File: experiments/synthetic_ab_test.py
  def train_with_synthetic(real_data, synthetic_data):
      # Train model on real + synthetic
      # Evaluate on held-out real test set
      # Compare F1 vs real-only baseline
  ```

- [ ] **Task 4.3**: Expert review system
  ```python
  # File: utilities/expert_review_tool.py
  class ExpertReviewTool:
      def sample_images(self, synthetic_dir, n=100):
      def create_review_form(self):
          # Plausible / Not / Unsure
          # Shape OK? Texture OK? Artifacts?
      def compute_agreement(self, reviews):
          # Cohen's kappa, acceptance rate
  ```

**Expected Outcome**:
- FID/KID computed for all synthetic classes
- Expert review acceptance rate > 80%
- Decision: keep/discard synthetic per class

---

### Phase 3C: OOD & Robustness (Week 4) 🛡️

#### Priority 5: OOD Detection Pipeline
- [ ] **Task 5.1**: Two-stage OOD detector
  ```python
  # File: evaluation/ood_detector.py
  class TwoStageOODDetector:
      def stage_a_quick_checks(self, image):
          # Size, extreme blur/brightness
      def stage_b_learned(self, embeddings):
          # IsolationForest on backbone features
      def detect(self, image, threshold=0.6):
          # Return: in-domain / OOD / unsure
  ```

- [ ] **Task 5.2**: Collect OOD test set
  - Non-amulet images (coins, jewelry, random objects)
  - Target: 200-500 images
  - Evaluate AUROC of detector

- [ ] **Task 5.3**: Adversarial robustness augmentation
  ```python
  # File: data_management/augmentation/robust_augment.py
  class RobustAugmentation:
      def __init__(self):
          self.transforms = [
              JPEGCompression(quality=50-95),
              GaussianBlur(kernel_size=3-7),
              RandomOcclusion(size=0.1-0.3),
              MotionBlur(),
          ]
  ```

**Expected Outcome**:
- OOD AUROC > 0.85
- Reject option integrated (confidence < threshold)
- Model robust to realistic degradations

---

#### Priority 6: Explainability Integration
- [ ] **Task 6.1**: Grad-CAM UI integration
  ```python
  # File: frontend/production_app_clean.py
  # Modify display_classification_result()
  # Add Grad-CAM heatmap display
  # Show overlay + explanation text
  ```

- [ ] **Task 6.2**: Batch Grad-CAM generation
  ```python
  # File: explainability/batch_gradcam.py
  def generate_gradcam_batch(model, images, target_classes):
      # Generate heatmaps for multiple images
      # Save to cache for UI display
  ```

- [ ] **Task 6.3**: Expert review of Grad-CAM
  - Random sample 50 predictions
  - Expert checks: "Does heatmap focus on right area?"
  - Document failure cases

**Expected Outcome**:
- Grad-CAM displayed in frontend
- Expert validation of explanation quality
- Failure case analysis for model improvement

---

### Phase 3D: Deployment & MLOps (Week 5-6) 🚀

#### Priority 7: Model Optimization
- [ ] **Task 7.1**: Quantization (INT8)
  ```python
  # File: deployment/quantize_model.py
  def quantize_to_int8(model_path, calib_loader):
      # PyTorch dynamic quantization
      # Test accuracy degradation < 2%
  ```

- [ ] **Task 7.2**: Export to TorchScript/ONNX
  ```python
  # File: deployment/export_model.py
  def export_torchscript(model, sample_input):
  def export_onnx(model, sample_input):
  def benchmark_inference(model, device):
      # Measure p50, p95, p99 latency
  ```

- [ ] **Task 7.3**: Inference benchmarking
  - Target: p95 < 2s on CPU, p95 < 500ms on GPU
  - Profile bottlenecks (feature extraction vs head)

**Expected Outcome**:
- Model size reduced 2-4x
- Latency meets target
- Export formats ready for production

---

#### Priority 8: MLOps Foundation
- [ ] **Task 8.1**: Model versioning
  ```python
  # File: core/model_registry.py
  class ModelRegistry:
      def register(self, model, metrics, metadata):
          # Save: model.pth, metrics.json, metadata.yaml
          # Version: v{date}_{backbone}_{f1:.3f}
      def load_best(self, metric='f1'):
      def list_versions(self):
  ```

- [ ] **Task 8.2**: Monitoring system
  ```python
  # File: monitoring/metrics_tracker.py
  class MetricsTracker:
      def log_prediction(self, input_hash, pred, conf, latency):
      def compute_drift(self, window='7d'):
          # KS test on feature distributions
      def alert_if_drift(self, threshold=0.05):
  ```

- [ ] **Task 8.3**: Data collection loop
  ```python
  # File: monitoring/feedback_collector.py
  class FeedbackCollector:
      def log_low_confidence(self, image, pred, conf):
          # Queue for expert review
      def queue_for_labeling(self, image):
      def trigger_retraining(self, threshold=1000):
  ```

**Expected Outcome**:
- Model registry with version control
- Drift detection alerts
- Feedback loop for continuous improvement

---

#### Priority 9: Security & Privacy
- [ ] **Task 9.1**: API security hardening
  ```python
  # File: api/security_middleware.py
  - HTTPS/TLS enforcement
  - API key authentication
  - Rate limiting (100 req/min)
  - Input sanitization
  ```

- [ ] **Task 9.2**: Privacy compliance
  ```python
  # File: core/privacy_manager.py
  class PrivacyManager:
      def hash_image(self, image): # store hash not raw
      def get_user_consent(self):
      def delete_user_data(self, user_id):
      def generate_retention_report(self):
  ```

**Expected Outcome**:
- HTTPS + Auth enabled
- Privacy policy implemented
- GDPR/PDPA compliance ready

---

#### Priority 10: Domain & Ethics
- [ ] **Task 10.1**: UI disclaimers
  ```python
  # File: frontend/production_app_clean.py
  # Add prominent disclaimer:
  # "ผลประเมินเป็นเบื้องต้น ควรให้ผู้เชี่ยวชาญยืนยัน"
  ```

- [ ] **Task 10.2**: Human-in-the-loop workflow
  ```python
  # File: utilities/expert_review_queue.py
  class ExpertReviewQueue:
      def enqueue(self, image, pred, conf):
          # If conf < 0.75 → send to expert
      def expert_interface(self):
          # Show image, prediction, allow override
      def log_expert_decision(self, image_id, expert_label):
  ```

- [ ] **Task 10.3**: Audit trail
  ```python
  # File: monitoring/audit_log.py
  # Log: timestamp, input_hash, prediction, confidence, expert_override
  ```

**Expected Outcome**:
- Transparent AI system
- Human oversight for low confidence
- Full audit trail

---

## 📊 Success Metrics (Definition of Done)

### Model Performance
- [ ] Per-class F1 > 0.85 (all classes)
- [ ] Balanced Accuracy > 0.87
- [ ] ECE < 0.05 (calibrated)
- [ ] OOD AUROC > 0.85

### Deployment
- [ ] p95 Latency < 2s (CPU) or < 500ms (GPU)
- [ ] Model size < 50MB (quantized)
- [ ] TorchScript/ONNX export successful

### MLOps
- [ ] Model versioning operational
- [ ] Drift detection implemented
- [ ] Feedback loop active
- [ ] Security hardened (HTTPS + Auth)

### Validation
- [ ] Synthetic FID < 200 (per-class)
- [ ] Expert review acceptance > 80%
- [ ] A/B test confirms synthetic value

---

## 🗓️ Timeline Estimate

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| 3A: Core ML | 2 weeks | Balanced sampling, Model comparison, Ablation studies |
| 3B: Synthetic | 1 week | FID/KID pipeline, Expert review, A/B test |
| 3C: OOD | 1 week | Two-stage detector, Robustness augmentation, Grad-CAM UI |
| 3D: Deployment | 2 weeks | Quantization, Versioning, Monitoring, Security |
| **Total** | **6 weeks** | Production-ready system |

---

## 📝 Implementation Order (Step-by-Step)

### Week 1: Foundation
1. Implement `BalancedDataLoader` with WeightedRandomSampler
2. Add class weights to `AmuletTrainer`
3. Run experiment: baseline vs balanced sampling
4. Document per-class F1 improvement

### Week 2: Model Selection
1. Create `ModelExperiment` framework
2. Run TL experiments (head-only, finetune)
3. Compare: accuracy, latency, size
4. Select production model + document decision

### Week 3: Synthetic Validation
1. Implement `SyntheticValidator` (FID/KID)
2. Create expert review tool
3. Run A/B test: real vs real+synthetic
4. Decision: keep/discard synthetic per class

### Week 4: OOD & Explainability
1. Implement `TwoStageOODDetector`
2. Collect OOD test set (200-500 images)
3. Add Grad-CAM to frontend UI
4. Expert review of explanations

### Week 5: Optimization
1. Quantize model to INT8
2. Export TorchScript/ONNX
3. Benchmark inference latency
4. Profile and optimize bottlenecks

### Week 6: MLOps & Security
1. Implement `ModelRegistry`
2. Set up `MetricsTracker` + drift detection
3. Harden API security (HTTPS + Auth)
4. Create feedback collection loop

---

## 🎯 Quick Wins (Can Start Immediately)

1. **Add WeightedRandomSampler** (2 hours)
   - Modify `data_management/dataset/sampler.py`
   - Test on current dataset

2. **Run TL head-only experiment** (4 hours)
   - Use existing `AmuletTransferModel`
   - Train ResNet50 head-only
   - Compare F1 vs baseline

3. **Compute FID/KID** (2 hours)
   - Use existing `evaluation/fid_kid.py`
   - Run on current augmented data
   - Document scores

4. **Add temperature scaling** (3 hours)
   - Use existing `evaluation/calibration.py`
   - Fit after training
   - Save with model checkpoint

5. **UI disclaimer** (1 hour)
   - Add to `frontend/production_app_clean.py`
   - Prominent warning message

**Total Quick Wins**: 12 hours → Immediate visible improvements

---

## 📚 Code Templates (Ready to Use)

### 1. Balanced Loader
```python
# File: data_management/dataset/balanced_loader.py
from torch.utils.data import WeightedRandomSampler, DataLoader
import numpy as np

def create_balanced_loader(dataset, labels, batch_size=32, strong_augment=True):
    """
    Create balanced dataloader with WeightedRandomSampler
    
    Args:
        dataset: Dataset instance (with augmentation)
        labels: np.array of class labels
        batch_size: Batch size
        strong_augment: Use strong augmentation for minorities
    
    Returns:
        DataLoader with balanced sampling
    """
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
```

### 2. MixUp Collate Function
```python
# File: data_management/augmentation/mixup_collate.py
def mixup_collate_fn(batch, alpha=0.2):
    """MixUp collate function for DataLoader"""
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    
    if alpha > 0 and np.random.rand() < 0.5:  # 50% chance
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(len(images))
        mixed_images = lam * images + (1 - lam) * images[idx]
        labels_a, labels_b = labels, labels[idx]
        return mixed_images, (labels_a, labels_b, lam), True
    else:
        return images, labels, False
```

### 3. Model Experiment Framework
```python
# File: experiments/model_comparison.py
class ModelExperiment:
    def __init__(self, train_loader, val_loader, test_loader, device):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.results = {}
    
    def run_experiment(self, model_name, config):
        """Run single experiment and track metrics"""
        model = create_model(model_name, config)
        trainer = AmuletTrainer(model, ...)
        
        # Train
        history = trainer.fit(...)
        
        # Evaluate
        metrics = evaluate_comprehensive(model, self.test_loader)
        
        # Benchmark
        latency = benchmark_inference(model, n=100)
        
        self.results[model_name] = {
            'metrics': metrics,
            'latency': latency,
            'model_size': get_model_size(model)
        }
        
        return metrics
    
    def compare_all(self):
        """Generate comparison report"""
        import pandas as pd
        df = pd.DataFrame(self.results).T
        df.to_csv('model_comparison.csv')
        return df
```

### 4. Synthetic Validator
```python
# File: evaluation/synthetic_validator.py
from evaluation.fid_kid import compute_fid

class SyntheticValidator:
    def __init__(self, real_dir, synthetic_dir):
        self.real_dir = real_dir
        self.synthetic_dir = synthetic_dir
    
    def validate_classwise(self, class_name, threshold=200):
        """Validate synthetic data for one class"""
        real_path = self.real_dir / class_name
        synth_path = self.synthetic_dir / class_name
        
        fid_score = compute_fid(real_path, synth_path)
        
        return {
            'class': class_name,
            'fid': fid_score,
            'passed': fid_score < threshold
        }
    
    def validate_all(self):
        """Validate all classes"""
        results = []
        for class_name in os.listdir(self.real_dir):
            result = self.validate_classwise(class_name)
            results.append(result)
        return results
```

---

## 🔗 Next Steps

1. **Review this roadmap** with team
2. **Prioritize** based on business impact
3. **Start with Quick Wins** (Week 1)
4. **Track progress** in project board
5. **Document learnings** in each phase

---

**Author**: Amulet-AI Team  
**Date**: October 2, 2025  
**Status**: Ready for Implementation 🚀
