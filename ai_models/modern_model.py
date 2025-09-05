"""
การตั้งค่าและการจัดการโมเดล AI ที่ทันสมัย
"""

import torch
import torch.nn as nn
from torchvision import transforms
from typing import Dict, List, Tuple, Optional
import timm
import logging

logger = logging.getLogger(__name__)

class ModelConfig:
    """การตั้งค่าโมเดล AI"""
    
    # โมเดลที่รองรับ
    SUPPORTED_MODELS = {
        "efficientnetv2_rw_s": {
            "family": "efficientnet",
            "input_size": 384,
            "description": "EfficientNetV2 Small - เร็วและมีประสิทธิภาพดี"
        },
        "vit_base_patch16_224": {
            "family": "vision_transformer", 
            "input_size": 224,
            "description": "Vision Transformer Base - ดีสำหรับรายละเอียดเล็กๆ"
        },
        "convnext_small": {
            "family": "convnext",
            "input_size": 224, 
            "description": "ConvNeXt Small - สมดุลระหว่างความเร็วและประสิทธิภาพ"
        },
        "swin_small_patch4_window7_224": {
            "family": "swin_transformer",
            "input_size": 224,
            "description": "Swin Transformer - ดีสำหรับภาพที่มีรายละเอียดมาก"
        }
    }
    
    # การตั้งค่าการฝึกสอน
    TRAINING_CONFIG = {
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 100,
        "early_stopping_patience": 10,
        "lr_scheduler": "cosine",
        "optimizer": "adamw",
        "label_smoothing": 0.1,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0
    }
    
    # การตั้งค่าการประมวลผลภาพ
    IMAGE_PREPROCESSING = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "resize_method": "center_crop",
        "interpolation": "bicubic"
    }
    
    # การตั้งค่า Data Augmentation
    AUGMENTATION_CONFIG = {
        "random_resized_crop": {"scale": (0.8, 1.0), "ratio": (0.8, 1.2)},
        "random_horizontal_flip": {"p": 0.5},
        "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1, "hue": 0.05},
        "random_rotation": {"degrees": 15},
        "random_affine": {"degrees": 0, "translate": (0.1, 0.1), "scale": (0.9, 1.1)},
        "gaussian_blur": {"kernel_size": 3, "sigma": (0.1, 2.0)}
    }

class ModernAmuletModel(nn.Module):
    """
    โมเดล Amulet ที่ทันสมัย
    """
    def __init__(self, num_classes: int, model_name: str = "efficientnetv2_rw_s", 
                 pretrained: bool = True, dropout_rate: float = 0.3):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # โหลดโมเดลพื้นฐาน
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0,  # ไม่ใช้ head จาก timm
            global_pool='avg'
        )
        
        # ดึงขนาด feature
        self.feature_dim = self.backbone.num_features
        
        # สร้าง classifier head ที่ซับซ้อนขึ้น
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),  # Swish activation
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
        
        # สร้าง embedding projector สำหรับ similarity search
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )
        
        # เริ่มต้นน้ำหนัก
        self._initialize_weights()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        embeddings = self.projector(features)
        
        # Normalize embeddings สำหรับ cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return {
            "logits": logits,
            "embeddings": embeddings,
            "features": features
        }
    
    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """สกัด embedding สำหรับการค้นหาความคล้ายคลึง"""
        with torch.no_grad():
            features = self.backbone(x)
            embeddings = self.projector(features)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def _initialize_weights(self):
        """เริ่มต้นน้ำหนักของ head layers"""
        for module in [self.classifier, self.projector]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

def get_transforms(input_size: int, is_training: bool = False) -> transforms.Compose:
    """
    สร้าง transforms สำหรับการประมวลผลภาพ
    
    Args:
        input_size: ขนาดของรูปภาพ input
        is_training: โหมดการฝึกสอนหรือไม่
        
    Returns:
        transforms.Compose: ชุด transforms
    """
    config = ModelConfig.IMAGE_PREPROCESSING
    
    if is_training:
        # Transforms สำหรับการฝึกสอน (มี augmentation)
        aug_config = ModelConfig.AUGMENTATION_CONFIG
        
        transform_list = [
            transforms.RandomResizedCrop(
                input_size, 
                scale=aug_config["random_resized_crop"]["scale"],
                ratio=aug_config["random_resized_crop"]["ratio"],
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(p=aug_config["random_horizontal_flip"]["p"]),
            transforms.ColorJitter(**aug_config["color_jitter"]),
            transforms.RandomRotation(degrees=aug_config["random_rotation"]["degrees"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"])
        ]
    else:
        # Transforms สำหรับการทำนาย
        transform_list = [
            transforms.Resize(int(input_size * 1.125)),  # Resize เล็กน้อยแล้ว crop
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["mean"], std=config["std"])
        ]
    
    return transforms.Compose(transform_list)

def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> ModernAmuletModel:
    """
    สร้างโมเดลใหม่
    
    Args:
        model_name: ชื่อโมเดล
        num_classes: จำนวนคลาส
        pretrained: ใช้โมเดลที่ฝึกแล้วหรือไม่
        
    Returns:
        ModernAmuletModel: โมเดลที่สร้างขึ้น
    """
    if model_name not in ModelConfig.SUPPORTED_MODELS:
        raise ValueError(f"Model {model_name} ไม่รองรับ. รองรับ: {list(ModelConfig.SUPPORTED_MODELS.keys())}")
    
    logger.info(f"สร้างโมเดล {model_name} สำหรับ {num_classes} คลาส")
    
    model = ModernAmuletModel(
        num_classes=num_classes,
        model_name=model_name,
        pretrained=pretrained
    )
    
    return model

def get_model_info(model_name: str) -> Dict:
    """
    ดึงข้อมูลของโมเดล
    
    Args:
        model_name: ชื่อโมเดล
        
    Returns:
        Dict: ข้อมูลของโมเดล
    """
    if model_name not in ModelConfig.SUPPORTED_MODELS:
        return {"error": f"Model {model_name} ไม่รองรับ"}
    
    return ModelConfig.SUPPORTED_MODELS[model_name]

def calculate_model_size(model: nn.Module) -> Dict[str, float]:
    """
    คำนวณขนาดของโมเดล
    
    Args:
        model: โมเดล PyTorch
        
    Returns:
        Dict: ข้อมูลขนาดโมเดล
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = param_size + buffer_size
    
    return {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_size_mb": model_size / 1024 / 1024,
        "param_size_mb": param_size / 1024 / 1024,
        "buffer_size_mb": buffer_size / 1024 / 1024
    }
