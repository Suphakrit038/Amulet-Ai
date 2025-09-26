#!/usr/bin/env python3
"""
Dataset Optimization Tool for Amulet-AI
ปรับปรุงฐานข้อมูลให้เหมาะสำหรับการใช้งานจริง
- Training: 20 รูปต่อ class
- Validation: 5 รูปต่อ class  
- Test: 10 รูปต่อ class
"""

import os
import shutil
import random
from pathlib import Path
import json
from datetime import datetime

class DatasetOptimizer:
    def __init__(self, source_dir, target_dir):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.config = {
            'train_samples_per_class': 20,
            'val_samples_per_class': 5,
            'test_samples_per_class': 10,
            'classes': ['phra_nang_phya', 'phra_rod', 'phra_somdej']
        }
        
    def optimize_dataset(self):
        """ปรับปรุงและจัดระเบียบ dataset ใหม่"""
        print("🔄 Starting Dataset Optimization...")
        
        # สร้างโฟลเดอร์ใหม่
        self._create_target_structure()
        
        # คัดเลือกและย้ายรูปภาพ
        for class_name in self.config['classes']:
            self._process_class(class_name)
            
        # สร้างรายงาน
        self._generate_report()
        print("✅ Dataset optimization completed!")
        
    def _create_target_structure(self):
        """สร้างโครงสร้างโฟลเดอร์ใหม่"""
        structure = ['train', 'validation', 'test']
        
        for folder in structure:
            for class_name in self.config['classes']:
                target_path = self.target_dir / folder / class_name
                target_path.mkdir(parents=True, exist_ok=True)
                
    def _process_class(self, class_name):
        """ประมวลผลแต่ละ class"""
        print(f"📁 Processing class: {class_name}")
        
        # รวบรวมรูปภาพทั้งหมดจาก class นี้
        source_class_dir = self.source_dir / 'train' / class_name
        all_images = list(source_class_dir.glob('*.jpg')) + list(source_class_dir.glob('*.png'))
        
        if len(all_images) == 0:
            print(f"⚠️ No images found for class {class_name}")
            return
            
        print(f"   Found {len(all_images)} images")
        
        # สร้างสำเนาและสุ่มเลือก
        random.shuffle(all_images)
        
        train_count = self.config['train_samples_per_class']
        val_count = self.config['val_samples_per_class']
        test_count = self.config['test_samples_per_class']
        
        total_needed = train_count + val_count + test_count
        
        # ถ้ารูปไม่พอ ให้ใช้ทั้งหมดที่มี
        if len(all_images) < total_needed:
            print(f"⚠️ Only {len(all_images)} images available, using all")
            # ปรับสัดส่วน
            ratio = len(all_images) / total_needed
            train_count = int(train_count * ratio)
            val_count = int(val_count * ratio)
            test_count = len(all_images) - train_count - val_count
        
        # แบ่งรูปภาพ
        train_images = all_images[:train_count]
        val_images = all_images[train_count:train_count + val_count]
        test_images = all_images[train_count + val_count:train_count + val_count + test_count]
        
        # คัดลอกไฟล์
        self._copy_images(train_images, self.target_dir / 'train' / class_name)
        self._copy_images(val_images, self.target_dir / 'validation' / class_name)
        self._copy_images(test_images, self.target_dir / 'test' / class_name)
        
        print(f"   ✅ Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
    def _copy_images(self, image_list, target_dir):
        """คัดลอกรูปภาพไปยังโฟลเดอร์เป้าหมาย"""
        for i, img_path in enumerate(image_list, 1):
            target_path = target_dir / f"{target_dir.name}_{i:03d}{img_path.suffix}"
            shutil.copy2(img_path, target_path)
            
    def _generate_report(self):
        """สร้างรายงานการปรับปรุง dataset"""
        report = {
            'optimization_date': datetime.now().isoformat(),
            'source_directory': str(self.source_dir),
            'target_directory': str(self.target_dir),
            'configuration': self.config,
            'results': {}
        }
        
        # นับไฟล์ในแต่ละโฟลเดอร์
        for split in ['train', 'validation', 'test']:
            report['results'][split] = {}
            for class_name in self.config['classes']:
                class_dir = self.target_dir / split / class_name
                count = len(list(class_dir.glob('*')))
                report['results'][split][class_name] = count
                
        # บันทึกรายงาน
        report_path = self.target_dir / 'optimization_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        print(f"📄 Report saved to: {report_path}")

def main():
    """ฟังก์ชันหลัก"""
    source_dir = "c:/Users/Admin/Documents/GitHub/Amulet-Ai/dataset_realistic"
    target_dir = "c:/Users/Admin/Documents/GitHub/Amulet-Ai/dataset_optimized"
    
    optimizer = DatasetOptimizer(source_dir, target_dir)
    optimizer.optimize_dataset()

if __name__ == "__main__":
    main()