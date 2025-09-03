# 🎯 Advanced Amulet AI - Maximum Quality ML System

## 🌟 ภาพรวมระบบ (System Overview)

ระบบ AI สำหรับจดจำพระเครื่องที่ให้ความสำคัญกับคุณภาพภาพสูงสุด โดยใช้เทคนิค Self-supervised Learning และ Advanced Image Processing

### ✨ จุดเด่นหลัก (Key Features)
- **🖼️ Maximum Image Quality**: รักษาคุณภาพภาพสูงสุด (512x512) ไม่ทำลายรายละเอียด
- **🧠 Self-Supervised Learning**: ใช้ Contrastive Learning ไม่ต้องพึ่งข้อมูล label มาก
- **🔍 Advanced Similarity Search**: ระบบค้นหาความคล้ายคลึงด้วย FAISS
- **⚡ High-Performance Processing**: ใช้ EfficientNet-B4 และ GPU acceleration
- **📊 Comprehensive Analytics**: วิเคราะห์และ visualize ผลลัพธ์แบบละเอียด

## 🚀 การติดตั้ง (Installation)

### ขั้นตอนที่ 1: Setup ระบบ
```bash
# รันสคริปต์ติดตั้งอัตโนมัติ
cd ai_models
python setup_advanced.py
```

### ขั้นตอนที่ 2: ติดตั้ง Dependencies แบบ Manual (ถ้าจำเป็น)
```bash
# ติดตั้ง requirements
pip install -r requirements_advanced.txt

# สำหรับ GPU (ถ้ามี CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install faiss-gpu
```

### ขั้นตอนที่ 3: เตรียม Dataset
```
dataset/
├── somdej-fatherguay/
├── พระสมเด็จหลังรูปเหมือน/
├── สมเด็จพิมพ์ปรกโพธิ์ 9 ใบ/
├── สมเด็จแหวกม่าน/
├── ออกวัดหนองอีดุก/
└── วัดหนองอีดุก/
```

## 🎯 การใช้งาน (Usage)

### Quick Start - เทรนทดสอบ
```bash
cd ai_models
python train_advanced_amulet_ai.py --quick-start
```

### เทรนแบบเต็มรูปแบบ
```bash
cd ai_models
python train_advanced_amulet_ai.py --config config_advanced.json
```

### การใช้งานแบบกำหนดเอง
```bash
# เทรน 100 epochs
python train_advanced_amulet_ai.py --epochs 100 --batch-size 16

# ใช้ dataset อื่น
python train_advanced_amulet_ai.py --dataset-path /path/to/dataset

# ประเมินผลเท่านั้น (ไม่เทรน)
python train_advanced_amulet_ai.py --evaluate-only
```

## 🔧 การตั้งค่า (Configuration)

### ไฟล์ config_advanced.json
```json
{
  "model_name": "efficientnet-b4",
  "embedding_dim": 512,
  "batch_size": 16,
  "learning_rate": 0.0001,
  "num_epochs": 100,
  "temperature": 0.1,
  "min_quality_score": 0.8
}
```

### การตั้งค่าการประมวลผลภาพ
- **ขนาดเป้าหมาย**: 512x512 pixels (คุณภาพสูงสุด)
- **Super Resolution**: LANCZOS upscaling
- **Noise Reduction**: OpenCV fastNlMeansDenoising
- **Sharpening**: Unsharp mask filter
- **Contrast Enhancement**: CLAHE adaptive histogram

### การตั้งค่า Self-Supervised Learning
- **Architecture**: EfficientNet-B4 backbone
- **Loss Function**: NT-Xent Contrastive Loss
- **Temperature**: 0.1 (สำหรับ softmax)
- **Embedding Dimension**: 512

## 📊 โครงสร้างระบบ (System Architecture)

```
🎯 Master Training System
├── 📸 Advanced Image Processor
│   ├── Super Resolution Upscaling
│   ├── Advanced Noise Reduction
│   ├── Unsharp Mask Sharpening
│   └── CLAHE Contrast Enhancement
│
├── 🧠 Self-Supervised Learning
│   ├── ContrastiveLearningModel (EfficientNet-B4)
│   ├── NT-Xent Loss Function
│   └── Advanced Embedding System
│
├── 🔄 Data Pipeline
│   ├── Quality-based Filtering
│   ├── Dataset Organization
│   └── High-Quality DataLoader
│
├── 🗄️ Embedding Database
│   ├── SQLite Storage
│   ├── FAISS Similarity Index
│   └── Clustering Analysis
│
└── 📊 Evaluation & Visualization
    ├── Training Metrics
    ├── t-SNE Visualization
    └── Similarity Search Results
```

## 📈 ขั้นตอนการเทรน (Training Pipeline)

1. **📊 Dataset Organization**: จัดเรียง dataset เป็น train/validation/test (80/10/10)
2. **🔄 Data Pipeline**: สร้าง high-quality dataloader พร้อม quality filtering
3. **🧠 Model Initialization**: เตรียม ContrastiveLearningModel และ trainer
4. **🗄️ Database Setup**: สร้าง embedding database พร้อม FAISS index
5. **🎯 Model Training**: เทรนด้วย contrastive learning
6. **🔗 Embedding Creation**: สร้าง embeddings สำหรับทุกภาพ
7. **📊 Evaluation**: ประเมินผลและสร้าง visualizations

## 📁 ผลลัพธ์ (Output Structure)

```
training_output/
├── models/
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
├── embeddings/
│   ├── embeddings.db
│   └── faiss_index_ivfflat.bin
├── visualizations/
│   ├── training_curves.png
│   └── embedding_tsne.png
├── reports/
│   ├── final_report.json
│   └── dataset_analysis.json
└── logs/
    └── training.log
```

## 🔍 การใช้งาน Embedding Database

### การค้นหาภาพคล้ายคลึง
```python
from ai_models.dataset_organizer import EmbeddingDatabase

# เปิด database
db = EmbeddingDatabase('training_output/embeddings/embeddings.db')

# ค้นหาภาพคล้ายคลึง
query_embedding = model.extract_embedding(query_image)
similar_items = db.search_similar(query_embedding, k=5)

for item in similar_items:
    print(f"Category: {item['category']}")
    print(f"Similarity: {item['similarity']:.3f}")
    print(f"Path: {item['image_path']}")
```

### สถิติ Database
```python
stats = db.get_statistics()
print(f"Total embeddings: {stats['total_embeddings']}")
print(f"Categories: {list(stats['categories'].keys())}")
```

## ⚡ Performance Optimization

### GPU Acceleration
- CUDA support สำหรับ PyTorch
- Mixed precision training (FP16)
- FAISS GPU index (ถ้าติดตั้ง faiss-gpu)

### Memory Management
- Gradient accumulation สำหรับ batch size ใหญ่
- DataLoader กับ num_workers=4
- Pin memory สำหรับ GPU transfer

### Speed Optimizations
- EfficientNet architecture (speed vs accuracy balance)
- IVF FAISS index สำหรับ similarity search
- Batch processing สำหรับ embedding extraction

## 🔬 Advanced Features

### Image Quality Analysis
```python
from ai_models.advanced_image_processor import advanced_processor

# วิเคราะห์คุณภาพภาพ
image = Image.open('path/to/image.jpg')
quality_metrics = advanced_processor.get_image_quality_metrics(image)

print(f"Sharpness: {quality_metrics['sharpness']}")
print(f"Contrast: {quality_metrics['contrast']}")
print(f"SNR: {quality_metrics['snr']}")
```

### Embedding Clustering
```python
from ai_models.self_supervised_learning import AdvancedEmbeddingSystem

# สร้าง embedding system
embedding_system = AdvancedEmbeddingSystem(config)

# วิเคราะห์ clusters
cluster_analysis = embedding_system.analyze_clusters(embeddings)
print(f"Optimal clusters: {cluster_analysis['optimal_k']}")
```

## 📊 Monitoring & Logging

### TensorBoard
```bash
# เปิด TensorBoard
tensorboard --logdir training_output/tensorboard
```

### การดู Logs
```bash
# ดู training logs
tail -f training_output/logs/training.log
```

## 🐛 Troubleshooting

### ปัญหา CUDA
```bash
# ตรวจสอบ CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# ตรวจสอบ GPU memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')"
```

### ปัญหา Memory
- ลด batch_size ใน config
- ลด num_workers ใน DataLoader
- ใช้ gradient_accumulation_steps

### ปัญหา Dataset
```bash
# ตรวจสอบ dataset structure
python -c "
from pathlib import Path
dataset = Path('dataset')
for cat in dataset.iterdir():
    if cat.is_dir():
        count = len(list(cat.glob('*.jpg'))) + len(list(cat.glob('*.png')))
        print(f'{cat.name}: {count} images')
"
```

## 📚 API Reference

### MasterTrainingSystem
```python
from ai_models.master_training_system import create_master_training_system, MasterTrainingConfig

# สร้าง config
config = MasterTrainingConfig(
    dataset_path="dataset",
    num_epochs=50,
    batch_size=16
)

# สร้าง training system
system = create_master_training_system(config)

# รันการเทรน
results = system.run_complete_training_pipeline()
```

### AdvancedImageProcessor
```python
from ai_models.advanced_image_processor import process_image_max_quality

# ประมวลผลภาพคุณภาพสูง
processed_image, img_array, metrics = process_image_max_quality(image)
```

### EmbeddingDatabase
```python
from ai_models.dataset_organizer import EmbeddingDatabase, create_embedding_record

# สร้าง database
db = EmbeddingDatabase('embeddings.db', dimension=512)

# เพิ่ม embedding
record = create_embedding_record(
    image_path="path/to/image.jpg",
    category="somdej-fatherguay", 
    embedding=embedding_vector,
    metadata={"quality": 0.95},
    quality_score=0.95
)
db.add_embedding(record)
```

## 📝 License & Credits

- **Framework**: PyTorch, FastAPI, Streamlit
- **Computer Vision**: OpenCV, PIL, scikit-image
- **ML Libraries**: scikit-learn, FAISS, transformers
- **Architecture**: EfficientNet (Google Research)

## 🤝 Contributing

1. สร้าง feature branch
2. ทำการเปลี่ยนแปลง
3. เขียน tests
4. ส่ง pull request

## 📞 Support

หากมีปัญหาหรือข้อสงสัย:
1. ตรวจสอบ logs ใน `training_output/logs/`
2. ดู troubleshooting section
3. เปิด GitHub issue

---

**🎯 Advanced Amulet AI - Maximum Quality ML System**  
*Built for preserving image quality while achieving state-of-the-art recognition performance*
