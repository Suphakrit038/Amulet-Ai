#!/usr/bin/env python3
"""
🚀 Main Training Script for Amulet-AI
เรียกใช้ระบบเทรนโมเดลหลัก

Author: Amulet-AI Team
Date: October 2, 2025
"""

import sys
import logging
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set PYTHONPATH environment
os.environ['PYTHONPATH'] = str(project_root)

# Use direct import
try:
    from examples.complete_training_example import main as train_main
    TRAIN_FUNCTION = train_main
except ImportError:
    try:
        from ai_models.enhanced_production_system import main as prod_main  
        TRAIN_FUNCTION = prod_main
    except ImportError:
        # Fallback to manual training
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from torchvision import transforms, models, datasets
        TRAIN_FUNCTION = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('E:/Amulet-Ai/logs/main_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def manual_training():
    """Manual training implementation"""
    logger.info("Using manual training implementation")
    
    # Import required modules
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms, models, datasets
    from sklearn.metrics import classification_report
    import json
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dir = 'E:/Amulet-Ai/organized_dataset/splits/train'
    val_dir = 'E:/Amulet-Ai/organized_dataset/splits/val'
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    logger.info(f"Classes: {class_names}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    num_epochs = 25
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch+1}/{num_epochs}')
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        scheduler.step()
        
        logger.info(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'num_classes': num_classes,
                'best_acc': best_acc
            }, 'E:/Amulet-Ai/trained_model/best_model.pth')
            
            # Save class mapping
            class_mapping = {
                'num_classes': num_classes,
                'classes': class_names,
                'class_to_idx': train_dataset.class_to_idx,
                'thai_names': {
                    'phra_sivali': 'พระสีวลี',
                    'portrait_back': 'หลังรูปเหมือน',
                    'prok_bodhi_9_leaves': 'ปรกโพธิ์9ใบ',
                    'somdej_pratanporn_buddhagavak': 'พระสมเด็จประธานพรเนื้อพุทธกวัก',
                    'waek_man': 'แหวกม่าน',
                    'wat_nong_e_duk': 'วัดหนองอีดุก'
                }
            }
            
            with open('E:/Amulet-Ai/trained_model/class_mapping.json', 'w', encoding='utf-8') as f:
                json.dump(class_mapping, f, ensure_ascii=False, indent=2)
            
            logger.info(f'New best model saved! Val Acc: {val_acc:.2f}%')
    
    return {'best_val_acc': best_acc, 'class_names': class_names}


def main():
    """Main training function"""
    logger.info("🚀 Starting Amulet-AI Main Training Pipeline")
    
    try:
        if TRAIN_FUNCTION:
            # Use existing training function
            logger.info("Using existing training function")
            results = TRAIN_FUNCTION()
        else:
            # Use manual training
            results = manual_training()
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"Best validation accuracy: {results.get('best_val_acc', 'N/A')}")
        logger.info(f"Results saved to: E:/Amulet-Ai/trained_model")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()