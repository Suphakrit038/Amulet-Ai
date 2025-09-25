#!/usr/bin/env python3
"""
สร้างไฟล์ภาพตัวอย่างสำหรับทดสอบระบบ
"""

import numpy as np
import cv2
from pathlib import Path

def create_sample_images():
    """สร้างภาพตัวอย่างสำหรับทดสอบระบบ"""
    
    # สร้างโครงสร้าง dataset
    dataset_dir = Path("dataset")
    classes = ["phra_somdej", "phra_nang_phya", "phra_rod"]
    
    # จำนวนภาพต่อ class (เพิ่มเป็น 300+ เพื่อให้ PCA ทำงานได้)
    n_images = {
        "phra_somdej": 150,  # มาก
        "phra_nang_phya": 100,  # กลาง
        "phra_rod": 50   # น้อย (เพื่อทดสอบ augmentation)
    }
    
    for class_name in classes:
        class_dir = dataset_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i in range(n_images[class_name]):
            # สร้างภาพตัวอย่าง (224x224 RGB)
            if class_name == "phra_somdej":
                # ภาพที่มีลักษณะเฉพาะ - สี่เหลี่ยมกับวงกลม
                img = np.ones((224, 224, 3), dtype=np.uint8) * 200
                cv2.rectangle(img, (50, 50), (174, 174), (100, 150, 200), -1)
                cv2.circle(img, (112, 112), 30, (50, 100, 150), -1)
                
            elif class_name == "phra_nang_phya":
                # ภาพที่มีลักษณะเฉพาะ - สามเหลี่ยมกับเส้น
                img = np.ones((224, 224, 3), dtype=np.uint8) * 180
                pts = np.array([[112, 50], [50, 174], [174, 174]], np.int32)
                cv2.fillPoly(img, [pts], (150, 100, 200))
                cv2.line(img, (50, 200), (174, 200), (100, 50, 150), 5)
                
            else:  # phra_rod
                # ภาพที่มีลักษณะเฉพาะ - วงรีกับจุด
                img = np.ones((224, 224, 3), dtype=np.uint8) * 160
                cv2.ellipse(img, (112, 112), (80, 50), 0, 0, 360, (200, 150, 100), -1)
                cv2.circle(img, (112, 112), 10, (50, 150, 100), -1)
            
            # เพิ่ม noise เล็กน้อยเพื่อความหลากหลาย
            noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # บันทึกภาพ
            img_path = class_dir / f"{class_name}_{i:03d}.jpg"
            cv2.imwrite(str(img_path), img)
        
        print(f"✅ Created {n_images[class_name]} images for {class_name}")
    
    print(f"\n📊 Dataset created with imbalanced classes:")
    for class_name, count in n_images.items():
        print(f"   {class_name}: {count} images")

if __name__ == "__main__":
    create_sample_images()