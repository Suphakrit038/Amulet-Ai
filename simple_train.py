"""
Simple Training Script for Amulet-AI
เทรนโมเดลแบบง่าย
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Starting Simple Amulet-AI Training")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dir = r'E:\Amulet-Ai\organized_dataset\splits\train'
    val_dir = r'E:\Amulet-Ai\organized_dataset\splits\val'
    
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    logger.info(f"Classes: {class_names}")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Training loop
    num_epochs = 30
    best_acc = 0.0
    
    # Create output directory
    output_dir = Path(r'E:\Amulet-Ai\trained_model')
    output_dir.mkdir(exist_ok=True)
    
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
                'best_acc': best_acc,
                'epoch': epoch
            }, output_dir / 'best_model.pth')
            
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
            
            with open(output_dir / 'class_mapping.json', 'w', encoding='utf-8') as f:
                json.dump(class_mapping, f, ensure_ascii=False, indent=2)
            
            logger.info(f'✅ New best model saved! Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if epoch > 10 and val_acc < best_acc * 0.95:
            logger.info("Early stopping triggered")
            break
    
    logger.info(f"🎉 Training completed! Best accuracy: {best_acc:.2f}%")
    logger.info(f"Model saved to: {output_dir}")
    
    return {
        'best_val_acc': best_acc,
        'class_names': class_names,
        'output_dir': str(output_dir)
    }

if __name__ == "__main__":
    main()