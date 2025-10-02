"""
🎨 Advanced Augmentation Techniques
=================================

Implementation of state-of-the-art augmentation methods:
- MixUp (Zhang et al., 2017)
- CutMix (Yun et al., 2019)
- RandAugment (Cubuk et al., 2020)
- RandomErasing (Zhong et al., 2020)

These techniques help improve model generalization and robustness.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import random
from PIL import Image, ImageEnhance, ImageOps
import torchvision.transforms as transforms


class MixUpAugmentation:
    """
    MixUp: Linear interpolation between two images and their labels
    
    Paper: mixup: Beyond Empirical Risk Minimization (Zhang et al., 2017)
    
    Args:
        alpha: Beta distribution parameter (default: 0.2)
               Higher values = more aggressive mixing
               Recommended: 0.2-0.4 for most cases
    """
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        
    def __call__(self, batch_images: torch.Tensor, batch_labels: torch.Tensor):
        """
        Apply MixUp augmentation to a batch
        
        Args:
            batch_images: Tensor of shape (B, C, H, W)
            batch_labels: Tensor of shape (B,) or (B, num_classes)
            
        Returns:
            mixed_images: Mixed images
            labels_a: Original labels
            labels_b: Shuffled labels
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size).to(batch_images.device)
        
        mixed_images = lam * batch_images + (1 - lam) * batch_images[index]
        labels_a = batch_labels
        labels_b = batch_labels[index]
        
        return mixed_images, labels_a, labels_b, lam
    
    def compute_loss(self, criterion, pred, labels_a, labels_b, lam):
        """
        Compute MixUp loss
        
        Loss = λ * loss(pred, y_a) + (1-λ) * loss(pred, y_b)
        """
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


class CutMixAugmentation:
    """
    CutMix: Cut and paste patches between images
    
    Paper: CutMix: Regularization Strategy to Train Strong Classifiers 
           with Localizable Features (Yun et al., 2019)
    
    Args:
        alpha: Beta distribution parameter (default: 1.0)
               Recommended: 1.0 for CutMix
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        
    def rand_bbox(self, size, lam):
        """Generate random bounding box"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform random center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def __call__(self, batch_images: torch.Tensor, batch_labels: torch.Tensor):
        """
        Apply CutMix augmentation
        
        Returns:
            mixed_images: Images with cut patches
            labels_a: Original labels
            labels_b: Shuffled labels
            lam: Actual mixing ratio (by area)
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
            
        batch_size = batch_images.size(0)
        index = torch.randperm(batch_size).to(batch_images.device)
        
        # Generate bounding box
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(batch_images.size(), lam)
        
        # Apply CutMix
        batch_images[:, :, bbx1:bbx2, bby1:bby2] = batch_images[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (batch_images.size()[-1] * batch_images.size()[-2]))
        
        labels_a = batch_labels
        labels_b = batch_labels[index]
        
        return batch_images, labels_a, labels_b, lam
    
    def compute_loss(self, criterion, pred, labels_a, labels_b, lam):
        """Compute CutMix loss (same as MixUp)"""
        return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


class RandAugmentPipeline:
    """
    RandAugment: Practical automated data augmentation
    
    Paper: RandAugment: Practical automated data augmentation with a 
           reduced search space (Cubuk et al., 2020)
    
    Args:
        n: Number of augmentation transformations to apply (default: 2)
        m: Magnitude for all operations (default: 9, range: 0-10)
    """
    
    def __init__(self, n: int = 2, m: int = 9):
        self.n = n
        self.m = m
        self.augment_list = self._get_augment_list()
        
    def _get_augment_list(self):
        """Define augmentation operations"""
        return [
            self._auto_contrast,
            self._equalize,
            self._rotate,
            self._solarize,
            self._color,
            self._posterize,
            self._contrast,
            self._brightness,
            self._sharpness,
            self._shear_x,
            self._shear_y,
            self._translate_x,
            self._translate_y,
        ]
    
    def _auto_contrast(self, img, magnitude):
        """Auto contrast"""
        return ImageOps.autocontrast(img)
    
    def _equalize(self, img, magnitude):
        """Histogram equalization"""
        return ImageOps.equalize(img)
    
    def _rotate(self, img, magnitude):
        """Rotate image"""
        degree = (magnitude / 10) * 30  # Max 30 degrees
        degree = random.choice([-degree, degree])
        return img.rotate(degree, resample=Image.BILINEAR)
    
    def _solarize(self, img, magnitude):
        """Solarize (invert pixels above threshold)"""
        threshold = int((magnitude / 10) * 256)
        return ImageOps.solarize(img, threshold)
    
    def _color(self, img, magnitude):
        """Adjust color"""
        factor = 1 + (magnitude / 10) * 0.9
        factor = random.choice([factor, 1/factor])
        return ImageEnhance.Color(img).enhance(factor)
    
    def _posterize(self, img, magnitude):
        """Reduce bits per channel"""
        bits = 8 - int((magnitude / 10) * 4)
        return ImageOps.posterize(img, bits)
    
    def _contrast(self, img, magnitude):
        """Adjust contrast"""
        factor = 1 + (magnitude / 10) * 0.9
        factor = random.choice([factor, 1/factor])
        return ImageEnhance.Contrast(img).enhance(factor)
    
    def _brightness(self, img, magnitude):
        """Adjust brightness"""
        factor = 1 + (magnitude / 10) * 0.9
        factor = random.choice([factor, 1/factor])
        return ImageEnhance.Brightness(img).enhance(factor)
    
    def _sharpness(self, img, magnitude):
        """Adjust sharpness"""
        factor = 1 + (magnitude / 10) * 0.9
        factor = random.choice([factor, 1/factor])
        return ImageEnhance.Sharpness(img).enhance(factor)
    
    def _shear_x(self, img, magnitude):
        """Shear X"""
        shear = (magnitude / 10) * 0.3
        shear = random.choice([-shear, shear])
        return img.transform(img.size, Image.AFFINE, (1, shear, 0, 0, 1, 0), resample=Image.BILINEAR)
    
    def _shear_y(self, img, magnitude):
        """Shear Y"""
        shear = (magnitude / 10) * 0.3
        shear = random.choice([-shear, shear])
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, shear, 1, 0), resample=Image.BILINEAR)
    
    def _translate_x(self, img, magnitude):
        """Translate X"""
        pixels = (magnitude / 10) * (img.size[0] * 0.3)
        pixels = random.choice([-pixels, pixels])
        return img.transform(img.size, Image.AFFINE, (1, 0, pixels, 0, 1, 0), resample=Image.BILINEAR)
    
    def _translate_y(self, img, magnitude):
        """Translate Y"""
        pixels = (magnitude / 10) * (img.size[1] * 0.3)
        pixels = random.choice([-pixels, pixels])
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, pixels), resample=Image.BILINEAR)
    
    def __call__(self, img):
        """
        Apply n random augmentations with magnitude m
        
        Args:
            img: PIL Image
            
        Returns:
            Augmented PIL Image
        """
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(img, self.m)
        return img


class RandomErasingTransform:
    """
    Random Erasing: Randomly erase rectangular regions
    
    Paper: Random Erasing Data Augmentation (Zhong et al., 2020)
    
    Args:
        p: Probability of applying random erasing (default: 0.5)
        scale: Range of proportion of erased area (default: (0.02, 0.33))
        ratio: Range of aspect ratio (default: (0.3, 3.3))
        value: Erasing value (default: 0 for black, 'random' for random)
    """
    
    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: str = 'random'
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
        
    def __call__(self, img: torch.Tensor):
        """
        Apply random erasing
        
        Args:
            img: Tensor of shape (C, H, W)
            
        Returns:
            Erased image tensor
        """
        if random.random() > self.p:
            return img
            
        _, h, w = img.shape
        area = h * w
        
        for _ in range(100):  # Try 100 times
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
            w_erase = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if h_erase < h and w_erase < w:
                i = random.randint(0, h - h_erase)
                j = random.randint(0, w - w_erase)
                
                if self.value == 'random':
                    img[:, i:i+h_erase, j:j+w_erase] = torch.rand_like(img[:, i:i+h_erase, j:j+w_erase])
                else:
                    img[:, i:i+h_erase, j:j+w_erase] = self.value
                    
                return img
                
        return img


class MixUpCutMixCollator:
    """
    Collate function that applies MixUp or CutMix randomly
    
    Usage:
        collate_fn = MixUpCutMixCollator(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.5)
        dataloader = DataLoader(dataset, collate_fn=collate_fn, ...)
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        num_classes: int = 6
    ):
        self.mixup = MixUpAugmentation(alpha=mixup_alpha)
        self.cutmix = CutMixAugmentation(alpha=cutmix_alpha)
        self.prob = prob
        self.num_classes = num_classes
        
    def __call__(self, batch):
        """Apply MixUp or CutMix to batch"""
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch])
        
        # Randomly choose MixUp or CutMix
        if random.random() < self.prob:
            # Use MixUp
            mixed_images, labels_a, labels_b, lam = self.mixup(images, labels)
        else:
            # Use CutMix
            mixed_images, labels_a, labels_b, lam = self.cutmix(images, labels)
            
        return mixed_images, labels_a, labels_b, lam


# Convenience function for creating augmentation transform
def create_training_augmentation(
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    rand_augment_n: int = 2,
    rand_augment_m: int = 9,
    random_erasing_p: float = 0.5
):
    """
    Create complete training augmentation pipeline
    
    Recommended pipeline:
    1. Resize
    2. RandAugment
    3. ToTensor
    4. Normalize
    5. RandomErasing
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        rand_augment_n: Number of RandAugment ops
        rand_augment_m: Magnitude of RandAugment
        random_erasing_p: Probability of random erasing
        
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        RandAugmentPipeline(n=rand_augment_n, m=rand_augment_m),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        RandomErasingTransform(p=random_erasing_p)
    ])


def create_validation_transform(
    image_size: int = 224,
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
):
    """
    Create validation/test transform (no augmentation)
    
    Args:
        image_size: Target image size
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
