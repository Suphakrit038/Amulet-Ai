"""
🔄 Transfer Learning Models
===========================

Transfer learning implementations with freeze/unfreeze strategy.

Supported Backbones:
- ResNet50, ResNet101
- Efficien    def forward(self, x: torch.Tensor) -> torch.Tensor:
        \"\"\"Forward pass\"\"\"
        features = self.backbone(x)
        
        # Ensure features are 4D for AdaptiveAvgPool2d
        # Some backbones might output 2D after removing classifier
        if len(features.shape) == 2:
            # [batch, features] → [batch, features, 1, 1]
            features = features.unsqueeze(-1).unsqueeze(-1)
        elif len(features.shape) == 4:
            # [batch, channels, height, width] - already correct
            pass
        else:
            raise ValueError(f\"Unexpected feature shape: {features.shape}\")7
- MobileNetV2, MobileNetV3
- Vision Transformer (ViT)

Strategy:
1. Freeze backbone → Train head only (3-10 epochs)
2. Unfreeze last        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience
        )layers → Fine-tune (1e-4 ~ 1e-5 LR)
3. Early stopping + LR scheduling

Author: Amulet-AI Team
Date: October 2025
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, List, Dict, Literal
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AmuletTransferModel(nn.Module):
    """
    🎯 Transfer Learning Model for Amulet Classification
    
    Architecture:
    - Pretrained backbone (frozen initially)
    - Global Average Pooling
    - 1-2 dense layers with dropout
    - Final classification layer
    
    Features:
    - Freeze/unfreeze control
    - Multi-stage training support
    - Efficient architecture (avoid large FC)
    
    Example:
        >>> model = AmuletTransferModel('resnet50', num_classes=6)
        >>> # Stage 1: Train head only
        >>> model.freeze_backbone()
        >>> # ... train ...
        >>> # Stage 2: Fine-tune last layers
        >>> model.unfreeze_layers(num_layers=10)
    """
    
    def __init__(
        self,
        backbone_name: Literal['resnet50', 'resnet101', 'efficientnet_b0', 
                               'efficientnet_b3', 'mobilenet_v2', 'mobilenet_v3'] = 'resnet50',
        num_classes: int = 6,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden_dim: int = 128,
        use_two_fc: bool = True
    ):
        """
        Initialize Transfer Learning Model
        
        Args:
            backbone_name: Name of pretrained backbone
            num_classes: Number of output classes
            pretrained: Use pretrained weights (ImageNet)
            dropout: Dropout rate for FC layers
            hidden_dim: Hidden dimension for FC layer
            use_two_fc: Use 2 FC layers instead of 1
        """
        super().__init__()
        
        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        
        # Create backbone
        self.backbone, self.feature_dim = self._create_backbone(
            backbone_name, pretrained
        )
        
        # Create head
        self.head = self._create_head(
            self.feature_dim, 
            num_classes, 
            hidden_dim, 
            dropout,
            use_two_fc
        )
        
        # Initialize weights
        self._init_head_weights()
        
        logger.info(f"AmuletTransferModel created: {backbone_name}, "
                   f"features={self.feature_dim}, classes={num_classes}")
    
    def _create_backbone(
        self, 
        name: str, 
        pretrained: bool
    ) -> tuple[nn.Module, int]:
        """Create and configure backbone"""
        
        if name == 'resnet50':
            backbone = models.resnet50(pretrained=pretrained)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()  # Remove original FC
            
        elif name == 'resnet101':
            backbone = models.resnet101(pretrained=pretrained)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            
        elif name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            
        elif name == 'efficientnet_b3':
            backbone = models.efficientnet_b3(pretrained=pretrained)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            
        elif name == 'mobilenet_v2':
            backbone = models.mobilenet_v2(pretrained=pretrained)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            
        elif name == 'mobilenet_v3':
            backbone = models.mobilenet_v3_small(pretrained=pretrained)
            feature_dim = backbone.classifier[3].in_features
            backbone.classifier = nn.Identity()
            
        else:
            raise ValueError(f"Unknown backbone: {name}")
        
        return backbone, feature_dim
    
    def _create_head(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int,
        dropout: float,
        use_two_fc: bool
    ) -> nn.Module:
        """Create classification head
        
        Note: Head expects 2D input [batch, features]
        Forward() method handles shape conversion from backbone
        """
        
        layers = []
        
        if use_two_fc:
            # Two FC layers: feature_dim → hidden_dim → num_classes
            layers.extend([
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ])
        else:
            # One FC layer: feature_dim → num_classes
            layers.extend([
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes)
            ])
        
        return nn.Sequential(*layers)
    
    def _init_head_weights(self):
        """Initialize head weights (Xavier initialization)"""
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        features = self.backbone(x)
        
        # Handle different backbone output shapes
        if len(features.shape) == 4:
            # Conv features: [batch, channels, height, width]
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.flatten(1)
        elif len(features.shape) == 2:
            # Already flattened: [batch, features]
            pass
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        logits = self.head(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features (for OOD detection, embeddings)"""
        features = self.backbone(x)
        # After backbone, before final FC
        if isinstance(self.head[0], nn.AdaptiveAvgPool2d):
            pooled = self.head[0](features)
            flattened = self.head[1](pooled)
            return flattened
        return features
    
    def freeze_backbone(self):
        """Freeze backbone parameters (train head only)"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        logger.info("Backbone frozen (train head only)")
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        logger.info("Backbone unfrozen (full fine-tuning)")
    
    def unfreeze_layers(self, num_layers: int):
        """
        Unfreeze last N layers of backbone
        
        Args:
            num_layers: Number of layers to unfreeze (from end)
        """
        # First freeze all
        self.freeze_backbone()
        
        # Get backbone layers
        if self.backbone_name.startswith('resnet'):
            # ResNet: unfreeze layer4, layer3, etc.
            layers = [self.backbone.layer4, self.backbone.layer3, 
                     self.backbone.layer2, self.backbone.layer1]
        elif self.backbone_name.startswith('efficientnet'):
            # EfficientNet: unfreeze blocks from end
            layers = list(self.backbone.features.children())[::-1]
        elif self.backbone_name.startswith('mobilenet'):
            # MobileNet: unfreeze from end
            layers = list(self.backbone.features.children())[::-1]
        else:
            logger.warning(f"Layer unfreezing not implemented for {self.backbone_name}")
            return
        
        # Unfreeze last N layers
        unfrozen = 0
        for layer in layers[:num_layers]:
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen += 1
        
        logger.info(f"Unfroze last {num_layers} layers ({unfrozen} parameters)")
    
    def get_trainable_params(self) -> Dict[str, int]:
        """Get trainable parameter statistics"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        head_params = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'backbone_trainable': backbone_params,
            'head_trainable': head_params,
            'frozen': total_params - trainable_params
        }


class TwoStageTrainer:
    """
    🎓 Two-Stage Training Strategy
    
    Stage 1: Freeze backbone → Train head (higher LR, 3-10 epochs)
    Stage 2: Unfreeze last N layers → Fine-tune (lower LR, with early stopping)
    
    This is the RECOMMENDED strategy for small datasets.
    
    Example:
        >>> trainer = TwoStageTrainer(model, train_loader, val_loader)
        >>> # Stage 1
        >>> trainer.train_stage1(epochs=5, lr=1e-3)
        >>> # Stage 2
        >>> trainer.train_stage2(epochs=20, lr=1e-4, unfreeze_layers=10)
    """
    
    def __init__(
        self,
        model: AmuletTransferModel,
        criterion: nn.Module,
        device: torch.device = None
    ):
        """
        Initialize Two-Stage Trainer
        
        Args:
            model: AmuletTransferModel instance
            criterion: Loss function
            device: Training device
        """
        self.model = model
        self.criterion = criterion
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"TwoStageTrainer initialized on {self.device}")
    
    def train_stage1(
        self,
        train_loader,
        val_loader,
        epochs: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 3
    ) -> Dict:
        """
        Stage 1: Train head only (frozen backbone)
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            lr: Learning rate (higher for head)
            weight_decay: Weight decay
            patience: Early stopping patience
            
        Returns:
            Training history dict
        """
        logger.info("="*60)
        logger.info("🎯 Stage 1: Training Head Only (Frozen Backbone)")
        logger.info("="*60)
        
        # Freeze backbone
        self.model.freeze_backbone()
        
        # Optimizer (only head parameters)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # LR scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2
        )
        
        # Training loop
        history = self._training_loop(
            train_loader, val_loader, optimizer, scheduler,
            epochs, patience, stage='stage1'
        )
        
        logger.info("✅ Stage 1 complete!")
        return history
    
    def train_stage2(
        self,
        train_loader,
        val_loader,
        epochs: int = 20,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        unfreeze_layers: int = 10,
        patience: int = 5
    ) -> Dict:
        """
        Stage 2: Fine-tune last N layers
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            epochs: Number of epochs
            lr: Learning rate (lower for fine-tuning)
            weight_decay: Weight decay
            unfreeze_layers: Number of layers to unfreeze
            patience: Early stopping patience
            
        Returns:
            Training history dict
        """
        logger.info("="*60)
        logger.info("🔥 Stage 2: Fine-Tuning Last Layers")
        logger.info("="*60)
        
        # Unfreeze last N layers
        self.model.unfreeze_layers(unfreeze_layers)
        
        # Log trainable params
        params_info = self.model.get_trainable_params()
        logger.info(f"Trainable parameters: {params_info['trainable']:,} / {params_info['total']:,}")
        
        # Optimizer (different LRs for backbone vs head)
        optimizer = torch.optim.Adam([
            {'params': self.model.backbone.parameters(), 'lr': lr},
            {'params': self.model.head.parameters(), 'lr': lr * 10}  # Higher LR for head
        ], weight_decay=weight_decay)
        
        # LR scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Training loop
        history = self._training_loop(
            train_loader, val_loader, optimizer, scheduler,
            epochs, patience, stage='stage2'
        )
        
        logger.info("✅ Stage 2 complete!")
        return history
    
    def _training_loop(
        self,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        epochs: int,
        patience: int,
        stage: str
    ) -> Dict:
        """Internal training loop"""
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader, optimizer)
            
            # Validate
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Log
            logger.info(f"[{stage.upper()}] Epoch {epoch+1}/{epochs}: "
                       f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # LR scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model (optional)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        return history
    
    def _train_epoch(self, train_loader, optimizer) -> tuple[float, float]:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_data in train_loader:
            # Handle MixUp/CutMix collate
            if len(batch_data) == 4:  # MixUp/CutMix
                images, labels_a, labels_b, lam = batch_data
                images = images.to(self.device)
                labels_a = labels_a.to(self.device)
                labels_b = labels_b.to(self.device)
                
                # Forward
                outputs = self.model(images)
                loss = lam * self.criterion(outputs, labels_a) + \
                       (1 - lam) * self.criterion(outputs, labels_b)
                
                # Accuracy (approximate)
                _, predicted = outputs.max(1)
                correct += (lam * predicted.eq(labels_a).sum().item() +
                           (1 - lam) * predicted.eq(labels_b).sum().item())
            else:
                images, labels = batch_data
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Accuracy
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total += labels.size(0) if len(batch_data) == 2 else labels_a.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader) -> tuple[float, float]:
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                
                total_loss += loss.item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy


# ============================================================================
# Helper Functions
# ============================================================================

def create_transfer_model(
    backbone: str = 'resnet50',
    num_classes: int = 6,
    pretrained: bool = True,
    device: str = 'auto'
) -> AmuletTransferModel:
    """
    Quick create transfer learning model
    
    Args:
        backbone: Backbone name ('resnet50', 'efficientnet_b0', etc.)
        num_classes: Number of classes
        pretrained: Use pretrained weights
        device: 'cuda', 'cpu', or 'auto'
        
    Returns:
        AmuletTransferModel instance
    """
    model = AmuletTransferModel(
        backbone_name=backbone,
        num_classes=num_classes,
        pretrained=pretrained
    )
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    
    logger.info(f"Transfer model created: {backbone} on {device}")
    return model


def freeze_backbone(model: AmuletTransferModel):
    """Freeze backbone (helper function)"""
    model.freeze_backbone()


def unfreeze_layers(model: AmuletTransferModel, num_layers: int):
    """Unfreeze last N layers (helper function)"""
    model.unfreeze_layers(num_layers)


if __name__ == "__main__":
    # Quick test
    print("🔄 Transfer Learning Module")
    print("=" * 60)
    
    # Create model
    model = create_transfer_model('resnet50', num_classes=6)
    
    print(f"\n✅ Model created:")
    params = model.get_trainable_params()
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")
    
    # Test freeze/unfreeze
    print(f"\n🔒 Freezing backbone...")
    model.freeze_backbone()
    params = model.get_trainable_params()
    print(f"  Trainable after freeze: {params['trainable']:,}")
    
    print(f"\n🔓 Unfreezing last 10 layers...")
    model.unfreeze_layers(10)
    params = model.get_trainable_params()
    print(f"  Trainable after unfreeze: {params['trainable']:,}")
    
    print("\n✅ Transfer learning module ready!")
