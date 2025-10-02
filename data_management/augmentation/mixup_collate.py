"""
🎨 MixUp & CutMix Collate Functions
===================================

Collate functions for DataLoader to apply MixUp/CutMix augmentation

Features:
- MixUp: Linear interpolation between images
- CutMix: Cut and paste patches
- Flexible alpha parameter
- Compatible with standard PyTorch DataLoader

Author: Amulet-AI Team
Date: October 2, 2025
"""

import torch
import numpy as np
from typing import Tuple, List, Optional, Union


def mixup_collate_fn(batch: List[Tuple[torch.Tensor, int]], alpha: float = 0.2, p: float = 0.5):
    """
    MixUp collate function for DataLoader
    
    Args:
        batch: List of (image, label) tuples
        alpha: Beta distribution parameter (recommended: 0.2-0.4)
        p: Probability of applying MixUp (0.5 = 50% of batches)
        
    Returns:
        If MixUp applied:
            images: Mixed images (B, C, H, W)
            (labels_a, labels_b, lam): Label mixing info
            True: MixUp applied flag
        Else:
            images: Original images
            labels: Original labels
            False: MixUp not applied
            
    Example:
        >>> from torch.utils.data import DataLoader
        >>> from functools import partial
        >>> 
        >>> # Create DataLoader with MixUp
        >>> collate_with_mixup = partial(mixup_collate_fn, alpha=0.2, p=0.5)
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_with_mixup)
        >>> 
        >>> for images, targets, is_mixed in loader:
        ...     if is_mixed:
        ...         labels_a, labels_b, lam = targets
        ...         loss = lam * criterion(pred, labels_a) + (1-lam) * criterion(pred, labels_b)
        ...     else:
        ...         loss = criterion(pred, targets)
    """
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    
    # Apply MixUp with probability p
    if alpha > 0 and np.random.rand() < p:
        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Random shuffle
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # Mix images: x_mixed = λ * x_i + (1-λ) * x_j
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Return mixed images and both labels
        labels_a = labels
        labels_b = labels[index]
        
        return mixed_images, (labels_a, labels_b, lam), True
    else:
        # No mixing
        return images, labels, False


def cutmix_collate_fn(batch: List[Tuple[torch.Tensor, int]], alpha: float = 1.0, p: float = 0.5):
    """
    CutMix collate function for DataLoader
    
    Args:
        batch: List of (image, label) tuples
        alpha: Beta distribution parameter (recommended: 1.0 for CutMix)
        p: Probability of applying CutMix
        
    Returns:
        If CutMix applied:
            images: Mixed images (B, C, H, W)
            (labels_a, labels_b, lam): Label mixing info
            True: CutMix applied flag
        Else:
            images: Original images
            labels: Original labels
            False: CutMix not applied
            
    Example:
        >>> from functools import partial
        >>> collate_with_cutmix = partial(cutmix_collate_fn, alpha=1.0, p=0.5)
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_with_cutmix)
    """
    images = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    
    # Apply CutMix with probability p
    if alpha > 0 and np.random.rand() < p:
        # Sample mixing coefficient
        lam = np.random.beta(alpha, alpha)
        
        # Random shuffle
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # Generate random bounding box
        _, _, H, W = images.size()
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Bounding box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix: paste patch from shuffled images
        images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        labels_a = labels
        labels_b = labels[index]
        
        return images, (labels_a, labels_b, lam), True
    else:
        return images, labels, False


def mixed_augmentation_collate_fn(
    batch: List[Tuple[torch.Tensor, int]], 
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    p_mixup: float = 0.25,
    p_cutmix: float = 0.25,
    p_none: float = 0.5
):
    """
    Combined MixUp + CutMix collate function
    
    Randomly chooses between:
    - MixUp (p_mixup)
    - CutMix (p_cutmix)
    - No augmentation (p_none)
    
    Args:
        batch: List of (image, label) tuples
        mixup_alpha: MixUp beta parameter
        cutmix_alpha: CutMix beta parameter
        p_mixup: Probability of MixUp
        p_cutmix: Probability of CutMix
        p_none: Probability of no augmentation
        
    Note: p_mixup + p_cutmix + p_none should = 1.0
    
    Example:
        >>> from functools import partial
        >>> collate_fn = partial(
        ...     mixed_augmentation_collate_fn,
        ...     p_mixup=0.3, p_cutmix=0.3, p_none=0.4
        ... )
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    # Normalize probabilities
    total = p_mixup + p_cutmix + p_none
    p_mixup /= total
    p_cutmix /= total
    p_none /= total
    
    # Choose augmentation
    choice = np.random.choice(['mixup', 'cutmix', 'none'], p=[p_mixup, p_cutmix, p_none])
    
    if choice == 'mixup':
        return mixup_collate_fn(batch, alpha=mixup_alpha, p=1.0)
    elif choice == 'cutmix':
        return cutmix_collate_fn(batch, alpha=cutmix_alpha, p=1.0)
    else:
        images = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return images, labels, False


def compute_mixed_loss(
    criterion,
    pred: torch.Tensor,
    targets: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, float]],
    is_mixed: bool
) -> torch.Tensor:
    """
    Compute loss for mixed or standard targets
    
    Args:
        criterion: Loss function (e.g., nn.CrossEntropyLoss)
        pred: Model predictions (B, num_classes)
        targets: Either labels or (labels_a, labels_b, lam)
        is_mixed: Whether mixing was applied
        
    Returns:
        Loss value
        
    Example:
        >>> criterion = nn.CrossEntropyLoss()
        >>> for images, targets, is_mixed in loader:
        ...     pred = model(images)
        ...     loss = compute_mixed_loss(criterion, pred, targets, is_mixed)
        ...     loss.backward()
    """
    if is_mixed:
        labels_a, labels_b, lam = targets
        loss = lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)
    else:
        loss = criterion(pred, targets)
    
    return loss


# ==================== Training Loop Example ====================

class MixedAugmentationTrainer:
    """
    🎯 Example trainer with MixUp/CutMix support
    
    Usage:
        >>> trainer = MixedAugmentationTrainer(
        ...     model, optimizer, criterion, device,
        ...     use_mixup=True, use_cutmix=True
        ... )
        >>> trainer.train_epoch(train_loader)
    """
    
    def __init__(
        self, 
        model, 
        optimizer, 
        criterion, 
        device,
        use_mixup: bool = True,
        use_cutmix: bool = True,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
    
    def train_epoch(self, loader, epoch: int):
        """Train one epoch with mixed augmentation"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch_data in enumerate(loader):
            if len(batch_data) == 3:  # Mixed format
                images, targets, is_mixed = batch_data
            else:  # Standard format
                images, targets = batch_data
                is_mixed = False
            
            images = images.to(self.device)
            if is_mixed:
                labels_a, labels_b, lam = targets
                labels_a = labels_a.to(self.device)
                labels_b = labels_b.to(self.device)
                targets = (labels_a, labels_b, lam)
            else:
                targets = targets.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            pred = self.model(images)
            loss = compute_mixed_loss(self.criterion, pred, targets, is_mixed)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Stats
            total_loss += loss.item()
            if not is_mixed:
                _, predicted = pred.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        return avg_loss, accuracy


# ==================== Usage Examples ====================

if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    from functools import partial
    import torch.nn as nn
    
    # Mock dataset
    images = torch.randn(100, 3, 224, 224)
    labels = torch.randint(0, 6, (100,))
    dataset = TensorDataset(images, labels)
    
    print("="*60)
    print("Testing MixUp Collate Function")
    print("="*60)
    
    # Create loader with MixUp
    mixup_collate = partial(mixup_collate_fn, alpha=0.2, p=0.5)
    loader = DataLoader(dataset, batch_size=16, collate_fn=mixup_collate)
    
    mixed_count = 0
    for images, targets, is_mixed in loader:
        if is_mixed:
            labels_a, labels_b, lam = targets
            print(f"MixUp batch: shape={images.shape}, lam={lam:.3f}")
            mixed_count += 1
        else:
            print(f"Normal batch: shape={images.shape}")
    
    print(f"\nMixed batches: {mixed_count}/{len(loader)} ({mixed_count/len(loader)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("Testing CutMix Collate Function")
    print("="*60)
    
    # Create loader with CutMix
    cutmix_collate = partial(cutmix_collate_fn, alpha=1.0, p=0.5)
    loader = DataLoader(dataset, batch_size=16, collate_fn=cutmix_collate)
    
    for images, targets, is_mixed in loader:
        if is_mixed:
            labels_a, labels_b, lam = targets
            print(f"CutMix batch: shape={images.shape}, lam={lam:.3f}")
            break
    
    print("\n✅ All collate functions working correctly!")
