#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing & Augmentation - Phase 2
ประมวลผลและเพิ่มจำนวนข้อมูล
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pathlib import Path
import json
import random
import datetime
from collections import defaultdict
import shutil
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, source_dir="organized_dataset", target_size=(224, 224)):
        self.source_dir = Path(source_dir)
        self.target_size = target_size
        self.stats = defaultdict(int)
        
    def preprocess_image(self, image_path, target_folder):
        """ประมวลผลรูปภาพพื้นฐาน"""
        try:
            # Load image
            img = cv2.imread(str(image_path))
            if img is None:
                # Try with PIL
                img = Image.open(image_path)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            img_resized = cv2.resize(img_rgb, self.target_size)
            
            # Normalize (0-255 range maintained for saving)
            img_normalized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX)
            
            # Save processed image
            processed_filename = f"processed_{image_path.stem}.jpg"
            output_path = target_folder / processed_filename
            
            # Convert back to BGR for saving
            img_bgr = cv2.cvtColor(img_normalized, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_path), img_bgr)
            
            self.stats['processed'] += 1
            return output_path
            
        except Exception as e:
            print(f"      ❌ ข้อผิดพลาด {image_path.name}: {e}")
            self.stats['errors'] += 1
            return None
    
    def augment_image(self, image_path, target_folder, num_augments=5):
        """สร้างรูปภาพเสริม"""
        augmented_paths = []
        
        try:
            # Load image with PIL for easier manipulation
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to target size
            img = img.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Original processed image
            base_name = image_path.stem
            
            # Augmentation techniques
            augmentations = [
                ("rotate_15", lambda x: x.rotate(15, fillcolor='white')),
                ("rotate_m15", lambda x: x.rotate(-15, fillcolor='white')),
                ("bright_1p2", lambda x: ImageEnhance.Brightness(x).enhance(1.2)),
                ("bright_0p8", lambda x: ImageEnhance.Brightness(x).enhance(0.8)),
                ("contrast_1p2", lambda x: ImageEnhance.Contrast(x).enhance(1.2)),
                ("contrast_0p8", lambda x: ImageEnhance.Contrast(x).enhance(0.8)),
                ("flip_lr", lambda x: ImageOps.mirror(x)),
                ("blur_1", lambda x: x.filter(ImageFilter.GaussianBlur(radius=1))),
                ("sharp_1p5", lambda x: ImageEnhance.Sharpness(x).enhance(1.5)),
                ("color_1p2", lambda x: ImageEnhance.Color(x).enhance(1.2))
            ]
            
            # Select random augmentations
            selected_augs = random.sample(augmentations, min(num_augments, len(augmentations)))
            
            for i, (aug_name, aug_func) in enumerate(selected_augs):
                try:
                    augmented_img = aug_func(img)
                    
                    # Save augmented image
                    aug_filename = f"aug_{aug_name}_{base_name}.jpg"
                    aug_path = target_folder / aug_filename
                    augmented_img.save(aug_path, "JPEG", quality=95)
                    
                    augmented_paths.append(aug_path)
                    self.stats['augmented'] += 1
                    
                except Exception as e:
                    print(f"        ⚠️ ข้อผิดพลาด augmentation {aug_name}: {e}")
                    self.stats['aug_errors'] += 1
            
            return augmented_paths
            
        except Exception as e:
            print(f"      ❌ ข้อผิดพลาด augmentation {image_path.name}: {e}")
            self.stats['aug_errors'] += 1
            return []
    
    def process_class_folder(self, class_name, side="front"):
        """ประมวลผลโฟลเดอร์ของคลาสหนึ่ง"""
        print(f"    🔄 ประมวลผล {class_name}/{side}")
        
        # Source and target paths
        source_folder = self.source_dir / "raw" / class_name / side
        processed_folder = self.source_dir / "processed" / class_name / side
        augmented_folder = self.source_dir / "augmented" / class_name / side
        
        # Create target folders
        processed_folder.mkdir(parents=True, exist_ok=True)
        augmented_folder.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(source_folder.glob(ext))
        
        processed_count = 0
        augmented_count = 0
        
        for image_file in image_files:
            # Preprocess
            processed_path = self.preprocess_image(image_file, processed_folder)
            if processed_path:
                processed_count += 1
                
                # Augment (only if we have few images)
                if len(image_files) < 30:  # Augment if less than 30 images
                    aug_paths = self.augment_image(processed_path, augmented_folder, num_augments=3)
                    augmented_count += len(aug_paths)
                else:
                    # Copy processed to augmented folder without additional augmentation
                    shutil.copy2(processed_path, augmented_folder / processed_path.name)
                    augmented_count += 1
        
        print(f"      ✅ ประมวลผล: {processed_count} ไฟล์")
        print(f"      ✅ เสริมข้อมูล: {augmented_count} ไฟล์")
        
        return processed_count, augmented_count
    
    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """แบ่งข้อมูลเป็น train/validation/test"""
        print(f"\n📊 แบ่งข้อมูล (Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio})")
        
        # Ensure ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
        
        split_stats = defaultdict(lambda: defaultdict(int))
        
        # Process each class
        augmented_dir = self.source_dir / "augmented"
        splits_dir = self.source_dir / "splits"
        
        for class_folder in augmented_dir.iterdir():
            if not class_folder.is_dir():
                continue
                
            class_name = class_folder.name
            print(f"  📁 {class_name}")
            
            for side in ['front', 'back']:
                side_folder = class_folder / side
                if not side_folder.exists():
                    continue
                
                # Get all images
                images = list(side_folder.glob("*.jpg")) + list(side_folder.glob("*.png"))
                
                if len(images) == 0:
                    continue
                
                # Split images
                if len(images) < 3:
                    # Too few images, put all in train
                    train_images = images
                    val_images = []
                    test_images = []
                else:
                    # Split with stratification
                    train_images, temp_images = train_test_split(
                        images, test_size=(val_ratio + test_ratio), random_state=42
                    )
                    
                    if len(temp_images) < 2:
                        val_images = temp_images
                        test_images = []
                    else:
                        val_images, test_images = train_test_split(
                            temp_images, test_size=(test_ratio / (val_ratio + test_ratio)), 
                            random_state=42
                        )
                
                # Copy files to split folders
                for split_name, split_images in [
                    ('train', train_images), 
                    ('validation', val_images), 
                    ('test', test_images)
                ]:
                    split_folder = splits_dir / split_name / class_name / side
                    split_folder.mkdir(parents=True, exist_ok=True)
                    
                    for img in split_images:
                        shutil.copy2(img, split_folder / img.name)
                        split_stats[class_name][f"{split_name}_{side}"] += 1
                
                total = len(train_images) + len(val_images) + len(test_images)
                print(f"    {side}: {total} -> Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
        return split_stats
    
    def generate_metadata(self, processing_stats, split_stats):
        """สร้าง metadata สำหรับ Phase 2"""
        metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "phase": 2,
            "target_size": self.target_size,
            "processing_statistics": dict(self.stats),
            "split_statistics": dict(split_stats),
            "augmentation_methods": [
                "rotation (±15°)",
                "brightness adjustment (0.8x, 1.2x)",
                "contrast adjustment (0.8x, 1.2x)", 
                "horizontal flip",
                "gaussian blur",
                "sharpness enhancement",
                "color enhancement"
            ]
        }
        
        # Save metadata
        metadata_file = self.source_dir / "metadata" / "phase2_processing.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        return metadata
    
    def run_phase2(self):
        """รัน Phase 2"""
        print("🚀 เริ่ม Phase 2: Data Preprocessing & Augmentation")
        print("=" * 60)
        
        # Get all classes
        raw_dir = self.source_dir / "raw"
        classes = [d.name for d in raw_dir.iterdir() if d.is_dir()]
        
        print(f"📋 ประมวลผล {len(classes)} คลาส...")
        
        processing_stats = {}
        
        # Process each class
        for class_name in classes:
            print(f"\n📁 {class_name}")
            
            # Process front and back
            front_processed, front_augmented = self.process_class_folder(class_name, "front")
            back_processed, back_augmented = self.process_class_folder(class_name, "back")
            
            processing_stats[class_name] = {
                "front_processed": front_processed,
                "front_augmented": front_augmented,
                "back_processed": back_processed,
                "back_augmented": back_augmented
            }
        
        # Split dataset
        split_stats = self.split_dataset()
        
        # Generate metadata
        metadata = self.generate_metadata(processing_stats, split_stats)
        
        print(f"\n🎉 Phase 2 เสร็จสิ้น!")
        print(f"📊 สถิติการประมวลผล:")
        print(f"  ✅ ประมวลผลสำเร็จ: {self.stats['processed']} ไฟล์")
        print(f"  ✅ เสริมข้อมูล: {self.stats['augmented']} ไฟล์")
        print(f"  ❌ ข้อผิดพลาด: {self.stats['errors'] + self.stats['aug_errors']} ไฟล์")
        
        return metadata

def main():
    preprocessor = DataPreprocessor()
    metadata = preprocessor.run_phase2()
    
    print(f"\n📈 สรุปการแบ่งข้อมูล:")
    for class_name, stats in metadata['split_statistics'].items():
        total_train = stats.get('train_front', 0) + stats.get('train_back', 0)
        total_val = stats.get('validation_front', 0) + stats.get('validation_back', 0) 
        total_test = stats.get('test_front', 0) + stats.get('test_back', 0)
        total = total_train + total_val + total_test
        
        print(f"  📁 {class_name}: {total} ไฟล์ (Train: {total_train}, Val: {total_val}, Test: {total_test})")

if __name__ == "__main__":
    main()