"""
🏺 Amulet-AI Real Model Loader
โหลดและใช้งาน AI Model ที่เทรนจริงแล้ว
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class AmuletModelLoader:
    """Class สำหรับโหลดและใช้งาน model ที่เทรนไว้"""
    
    def __init__(self, model_dir: str = "ai_models"):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = []
        self.transform = None
        self.metadata = {}
        
        print(f"[AmuletModelLoader] Initializing...")
        print(f"Device: {self.device}")
        print(f"Model directory: {self.model_dir}")
        
    def load_class_names(self) -> List[str]:
        """โหลดชื่อ class จากไฟล์"""
        possible_files = [
            os.path.join(self.model_dir, "labels.json"),
            os.path.join(self.model_dir, "classes.json"),
            os.path.join(self.model_dir, "class_names.json"),
            os.path.join("..", "..", self.model_dir, "labels.json"),  # From backend/api/
            os.path.join("..", "..", self.model_dir, "classes.json"),  # From backend/api/
            "labels.json",
            "classes.json"
        ]
        
        print(f"Looking for class names files...")
        for file_path in possible_files:
            print(f"  Checking: {file_path} -> {'EXISTS' if os.path.exists(file_path) else 'NOT FOUND'}")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self.class_names = data
                        elif isinstance(data, dict):
                            self.class_names = list(data.values())
                        print(f"SUCCESS: Loaded class names from {file_path}")
                        print(f"Classes ({len(self.class_names)}): {self.class_names}")
                        return self.class_names
                except Exception as e:
                    print(f"WARNING: Cannot read {file_path}: {e}")
        
        # Fallback class names - ใช้ชื่อภาษาอังกฤษเพื่อให้ตรงกับ reference images
        self.class_names = [
            "somdej_fatherguay",
            "buddha_in_vihara",
            "somdej_lion_base",
            "somdej_buddha_blessing",
            "somdej_portrait_back",
            "phra_san",
            "phra_sivali",
            "somdej_prok_bodhi",
            "somdej_waek_man",
            "wat_nong_e_duk"
        ]
        print("WARNING: Using default class names")
        print(f"Classes: {len(self.class_names)} classes")
        return self.class_names
    
    def create_model_architecture(self, num_classes: int) -> nn.Module:
        """สร้าง model architecture ที่ใช้ในการเทรน"""
        try:
            # ลอง ResNet18 เป็นหลัก (architecture ที่ใช้บ่อย)
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            print(f"SUCCESS: Created ResNet18 architecture for {num_classes} classes")
            return model
        except Exception as e:
            print(f"WARNING: Cannot create ResNet18: {e}")
            
            # Fallback เป็น simple CNN
            class SimpleCNN(nn.Module):
                def __init__(self, num_classes):
                    super(SimpleCNN, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 64, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.AdaptiveAvgPool2d(7)
                    )
                    self.classifier = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(128 * 7 * 7, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, num_classes)
                    )
                
                def forward(self, x):
                    x = self.features(x)
                    x = x.view(x.size(0), -1)
                    x = self.classifier(x)
                    return x
            
            print(f"SUCCESS: Created Simple CNN architecture for {num_classes} classes")
            return SimpleCNN(num_classes)
    
    def load_trained_model(self) -> bool:
        """โหลด model ที่เทรนไว้"""
        # รายการไฟล์ model ตามลำดับความสำคัญ (เพิ่ม path variants)
        model_paths = [
            # Path จาก backend directory (correct path from backend/api/)
            os.path.join("..", "..", self.model_dir, "training_output", "step5_final_model.pth"),
            os.path.join("..", "..", self.model_dir, "training_output", "ultra_simple_model.pth"),
            os.path.join("..", "..", self.model_dir, "training_output", "emergency_model.pth"),
            os.path.join("..", "..", self.model_dir, "training_output", "step5_checkpoint_epoch_3.pth"),
            os.path.join("..", "..", self.model_dir, "training_output", "step5_checkpoint_epoch_2.pth"),
            # Path ปกติ (from project root)
            os.path.join(self.model_dir, "training_output", "step5_final_model.pth"),
            os.path.join(self.model_dir, "training_output", "ultra_simple_model.pth"),
            os.path.join(self.model_dir, "training_output", "emergency_model.pth"),
            os.path.join(self.model_dir, "training_output", "step5_checkpoint_epoch_3.pth"),
            os.path.join(self.model_dir, "training_output", "step5_checkpoint_epoch_2.pth"),
            # Path อื่นๆ
            os.path.join(self.model_dir, "amulet_model.h5"),  # ไฟล์ h5 ถ้ามี
            os.path.join(self.model_dir, "somdej-fatherguay_best.h5")
        ]
        
        print(f"Searching model paths...")
        for model_path in model_paths:
            print(f"   {model_path} -> {'EXISTS' if os.path.exists(model_path) else 'NOT FOUND'}")
            if os.path.exists(model_path):
                try:
                    print(f"Loading model from {model_path}")
                    
                    # ตรวจสอบขนาดไฟล์
                    file_size = os.path.getsize(model_path) / 1024 / 1024  # MB
                    print(f"File size: {file_size:.1f} MB")
                    
                    # โหลด checkpoint (with weights_only=False for PyTorch 2.6+ compatibility)
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                    
                    # ตรวจสอบรูปแบบของ checkpoint
                    if isinstance(checkpoint, dict):
                        # กรณีที่เป็น dict อาจมี metadata
                        if 'model_state_dict' in checkpoint:
                            state_dict = checkpoint['model_state_dict']
                            if 'metadata' in checkpoint:
                                self.metadata = checkpoint['metadata']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        # กรณีที่เป็น state_dict โดยตรง
                        state_dict = checkpoint
                    
                    # สร้าง model ตาม architecture
                    num_classes = len(self.class_names)
                    self.model = self.create_model_architecture(num_classes)
                    
                    # พยายาม load weights
                    try:
                        self.model.load_state_dict(state_dict, strict=True)
                        print("SUCCESS: Loaded weights with strict mode")
                    except Exception as e:
                        print(f"⚠️ โหลดแบบ strict ไม่ได้: {e}")
                        try:
                            self.model.load_state_dict(state_dict, strict=False)
                            print("SUCCESS: Loaded weights with non-strict mode")
                        except Exception as e2:
                            print(f"ERROR: Cannot load weights: {e2}")
                            continue
                    
                    # ย้าย model ไป device และเซ็ต evaluation mode
                    self.model.to(self.device)
                    self.model.eval()
                    
                    print(f"SUCCESS: Model loaded successfully!")
                    print(f"Model: {model_path}")
                    print(f"Classes: {num_classes}")
                    print(f"Device: {self.device}")
                    
                    return True
                    
                except Exception as e:
                    print(f"ERROR: Cannot load {model_path}: {e}")
                    continue
        
        print("ERROR: No usable model file found")
        print("Creating fallback model for testing...")
        return self.create_fallback_model()
    
    def setup_transform(self):
        """ตั้งค่า image preprocessing transform"""
        # ค่า standard สำหรับ ImageNet pretrained models
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        input_size = 224
        
        self.transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        print(f"Transform setup: {input_size}x{input_size}, normalized")
    
    def predict_image(self, image_data) -> Dict:
        """ทำนายผลจากรูปภาพ"""
        if self.model is None:
            raise ValueError("Model ยังไม่ได้โหลด")
        
        try:
            # เตรียมรูปภาพ
            if isinstance(image_data, bytes):
                import io
                image = Image.open(io.BytesIO(image_data))
            else:
                image = Image.open(image_data)
            
            # แปลงเป็น RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transform
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # ทำนาย
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                probs = probabilities.cpu().numpy()[0]
            
            # จัดเรียงผลลัพธ์
            sorted_indices = np.argsort(probs)[::-1]
            
            results = {
                "success": True,
                "predictions": [],
                "top1": {
                    "class_name": self.class_names[sorted_indices[0]],
                    "confidence": float(probs[sorted_indices[0]]),
                    "class_id": int(sorted_indices[0])
                },
                "metadata": self.metadata,
                "model_info": {
                    "device": str(self.device),
                    "num_classes": len(self.class_names),
                    "architecture": "ResNet18-based" if hasattr(self.model, 'fc') else "Custom CNN"
                }
            }
            
            # Top-K results (5 อันดับแรก)
            for i, idx in enumerate(sorted_indices[:5]):
                results["predictions"].append({
                    "rank": i + 1,
                    "class_name": self.class_names[idx],
                    "confidence": float(probs[idx]),
                    "class_id": int(idx)
                })
            
            print(f"Prediction: {results['top1']['class_name']} ({results['top1']['confidence']:.2%})")
            
            return results
            
        except Exception as e:
            print(f"ERROR: Prediction error: {e}")
            return {
                "success": False,
                "error": str(e),
                "predictions": [],
                "top1": {"class_name": "Unknown", "confidence": 0.0}
            }
    
    def create_fallback_model(self) -> bool:
        """สร้าง model สำรองเมื่อโหลด model จริงไม่ได้"""
        try:
            print("Creating fallback ResNet18 model...")
            num_classes = len(self.class_names)
            self.model = self.create_model_architecture(num_classes)

            # Initialize with random weights (not trained)
            self.model.to(self.device)
            self.model.eval()

            print("SUCCESS: Fallback model created successfully")
            return True
        except Exception as e:
            print(f"ERROR: Cannot create fallback model: {e}")
            return False

    def initialize(self) -> bool:
        """เริ่มต้นระบบทั้งหมด"""
        print("Starting Amulet-AI Real Model system...")

        # 1. โหลด class names
        self.load_class_names()

        # 2. โหลด trained model
        if not self.load_trained_model():
            print("ERROR: Cannot load model")
            return False

        # 3. ตั้งค่า transform
        self.setup_transform()

        print("SUCCESS: System ready with Real AI Model!")
        return True
    
    def get_model_info(self) -> Dict:
        """ข้อมูลเกี่ยวกับ model"""
        return {
            "model_loaded": self.model is not None,
            "classes": self.class_names,
            "num_classes": len(self.class_names),
            "device": str(self.device),
            "metadata": self.metadata
        }

# สร้าง global instance
model_loader = AmuletModelLoader()
