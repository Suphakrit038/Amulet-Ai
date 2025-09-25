#!/usr/bin/env python3
"""
สร้างฐานข้อมูลแบบสมจริงสำหรับการฝึกและทดสอบ ML Model
- สร้างภาพพระเครื่อง 3 ประเภท (Target Classes) 
- สร้างภาพวัตถุอื่นๆ (Out-of-Distribution Classes)
- เพิ่มความหลากหลาย: แสง, เงา, สี, ความสึกกรอน, อายุ
- แยกชุด Train และ Test อย่างเหมาะสม
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
import shutil
import random
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DatasetConfig:
    """การตั้งค่าสำหรับการสร้างฐานข้อมูล"""
    
    # Target Classes (In-Distribution) - พระเครื่อง
    target_classes = {
        'phra_somdej': {
            'samples_train': 120,
            'samples_test': 30,
            'description': 'พระสมเด็จ - รูปทรงสี่เหลี่ยม มียอดแหลม',
            'base_color': (180, 140, 100),  # สีน้ำตาลทอง
            'shape': 'rectangular_pointed'
        },
        'phra_nang_phya': {
            'samples_train': 100,
            'samples_test': 25,
            'description': 'พระนางพญา - รูปทรงโค้งมน มีลวดลาย',
            'base_color': (160, 120, 80),   # สีน้ำตาลเข้ม
            'shape': 'curved_ornate'
        },
        'phra_rod': {
            'samples_train': 80,
            'samples_test': 20,
            'description': 'พระรอด - รูปทรงเล็ก กลม',
            'base_color': (200, 160, 120),  # สีน้ำตาลอ่อน
            'shape': 'small_round'
        }
    }
    
    # Out-of-Distribution Classes - วัตถุอื่นๆ ที่ไม่ใช่พระเครื่อง
    ood_classes = {
        'coins': {
            'samples_test': 25,
            'description': 'เหรียญ - วัตถุกลมแบน คล้ายพระเครื่อง',
            'base_color': (200, 180, 120),  # สีทอง
            'shape': 'flat_round'
        },
        'jewelry': {
            'samples_test': 25,
            'description': 'เครื่องประดับ - จี้ กำไล',
            'base_color': (220, 200, 160),  # สีเงิน-ทอง
            'shape': 'ornamental'
        },
        'stones': {
            'samples_test': 20,
            'description': 'หิน - รูปทรงธรรมชาติ',
            'base_color': (120, 100, 80),   # สีเทา-น้ำตาล
            'shape': 'irregular'
        },
        'buttons': {
            'samples_test': 15,
            'description': 'กระดุม - วัตถุกลมเล็ก',
            'base_color': (100, 80, 60),    # สีน้ำตาลเข้ม
            'shape': 'small_flat'
        }
    }
    
    # การปรับแต่งภาพ
    variations = {
        'lighting': [0.3, 0.6, 0.9, 1.2, 1.5],  # ความสว่าง
        'wear_levels': [0.0, 0.2, 0.4, 0.6, 0.8],  # ระดับความสึกกรอน
        'age_effects': [0.0, 0.3, 0.6, 0.9],  # ผลของอายุ
        'shadow_intensity': [0.0, 0.2, 0.4, 0.6],  # ความเข้มของเงา
        'color_shift': [0.8, 0.9, 1.0, 1.1, 1.2]  # การเปลี่ยนสี
    }
    
    # ขนาดภาพ
    image_size = (224, 224)
    background_colors = [
        (240, 240, 240),  # ขาว
        (200, 200, 200),  # เทาอ่อน
        (180, 180, 180),  # เทา
        (50, 50, 50),     # เทาเข้ม
        (220, 200, 180),  # ครีม
    ]


class RealisticImageGenerator:
    """สร้างภาพจำลองแบบสมจริง"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
    def generate_amulet_image(self, class_info: Dict, variations: Dict[str, float]) -> np.ndarray:
        """สร้างภาพพระเครื่อง"""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # เลือกสีพื้นหลัง
        bg_color = random.choice(self.config.background_colors)
        img[:] = bg_color
        
        # กำหนดตำแหน่งและขนาดของวัตถุ
        center_x, center_y = 112, 112
        base_size = random.randint(60, 100)
        
        # สร้างรูปทรงตามประเภท
        if class_info['shape'] == 'rectangular_pointed':
            self._draw_somdej(img, center_x, center_y, base_size, class_info, variations)
        elif class_info['shape'] == 'curved_ornate':
            self._draw_nang_phya(img, center_x, center_y, base_size, class_info, variations)
        elif class_info['shape'] == 'small_round':
            self._draw_rod(img, center_x, center_y, base_size, class_info, variations)
            
        # ใส่เอฟเฟกต์ต่างๆ
        img = self._apply_lighting(img, variations['lighting'])
        img = self._apply_wear_effect(img, variations['wear_level'])
        img = self._apply_age_effect(img, variations['age_effect'])
        img = self._apply_shadow(img, variations['shadow_intensity'])
        img = self._apply_color_shift(img, variations['color_shift'])
        
        return img
    
    def generate_ood_image(self, class_info: Dict, variations: Dict[str, float]) -> np.ndarray:
        """สร้างภาพวัตถุ Out-of-Distribution"""
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # เลือกสีพื้นหลัง
        bg_color = random.choice(self.config.background_colors)
        img[:] = bg_color
        
        center_x, center_y = 112, 112
        base_size = random.randint(40, 120)
        
        # สร้างรูปทรงตามประเภท OOD
        if class_info['shape'] == 'flat_round':
            self._draw_coin(img, center_x, center_y, base_size, class_info, variations)
        elif class_info['shape'] == 'ornamental':
            self._draw_jewelry(img, center_x, center_y, base_size, class_info, variations)
        elif class_info['shape'] == 'irregular':
            self._draw_stone(img, center_x, center_y, base_size, class_info, variations)
        elif class_info['shape'] == 'small_flat':
            self._draw_button(img, center_x, center_y, base_size, class_info, variations)
            
        # ใส่เอฟเฟกต์
        img = self._apply_lighting(img, variations['lighting'])
        img = self._apply_wear_effect(img, variations['wear_level'])
        img = self._apply_shadow(img, variations['shadow_intensity'])
        
        return img
    
    def _draw_somdej(self, img: np.ndarray, cx: int, cy: int, size: int, 
                     class_info: Dict, variations: Dict[str, float]):
        """วาดพระสมเด็จ - รูปสี่เหลี่ยมมียอดแหลม"""
        color = self._adjust_color(class_info['base_color'], variations)
        
        # วาดตัวหลัก (สี่เหลี่ยม)
        width, height = int(size * 0.8), int(size * 1.2)
        x1, y1 = cx - width//2, cy - height//2
        x2, y2 = cx + width//2, cy + height//2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # วาดยอดแหลม
        points = np.array([[cx, y1-20], [cx-15, y1], [cx+15, y1]], np.int32)
        cv2.fillPoly(img, [points], color)
        
        # เพิ่มรายละเอียด
        self._add_texture(img, x1, y1, x2, y2, variations)
        
    def _draw_nang_phya(self, img: np.ndarray, cx: int, cy: int, size: int,
                        class_info: Dict, variations: Dict[str, float]):
        """วาดพระนางพญา - รูปโค้งมน"""
        color = self._adjust_color(class_info['base_color'], variations)
        
        # วาดรูปไข่
        axes = (int(size * 0.6), int(size * 0.9))
        cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, color, -1)
        
        # เพิ่มลวดลาย
        for i in range(3):
            y_offset = -30 + i * 30
            cv2.ellipse(img, (cx, cy + y_offset), (axes[0]//3, 8), 0, 0, 360, 
                       tuple(max(0, c-30) for c in color), -1)
        
        self._add_texture(img, cx-axes[0], cy-axes[1], cx+axes[0], cy+axes[1], variations)
    
    def _draw_rod(self, img: np.ndarray, cx: int, cy: int, size: int,
                  class_info: Dict, variations: Dict[str, float]):
        """วาดพระรอด - รูปกลมเล็ก"""
        color = self._adjust_color(class_info['base_color'], variations)
        
        radius = int(size * 0.5)
        cv2.circle(img, (cx, cy), radius, color, -1)
        
        # เพิ่มขอบ
        cv2.circle(img, (cx, cy), radius, tuple(max(0, c-40) for c in color), 3)
        
        self._add_texture(img, cx-radius, cy-radius, cx+radius, cy+radius, variations)
    
    def _draw_coin(self, img: np.ndarray, cx: int, cy: int, size: int,
                   class_info: Dict, variations: Dict[str, float]):
        """วาดเหรียญ"""
        color = self._adjust_color(class_info['base_color'], variations)
        
        radius = int(size * 0.6)
        cv2.circle(img, (cx, cy), radius, color, -1)
        
        # เพิ่มขอบเหรียญ
        cv2.circle(img, (cx, cy), radius, tuple(min(255, c+50) for c in color), 4)
        cv2.circle(img, (cx, cy), radius-8, tuple(max(0, c-30) for c in color), 2)
        
    def _draw_jewelry(self, img: np.ndarray, cx: int, cy: int, size: int,
                      class_info: Dict, variations: Dict[str, float]):
        """วาดเครื่องประดับ"""
        color = self._adjust_color(class_info['base_color'], variations)
        
        # วาดจี้รูปหัวใจ
        points = np.array([
            [cx, cy+size//2],
            [cx-size//3, cy],
            [cx-size//4, cy-size//3],
            [cx, cy-size//4],
            [cx+size//4, cy-size//3],
            [cx+size//3, cy]
        ], np.int32)
        cv2.fillPoly(img, [points], color)
        
    def _draw_stone(self, img: np.ndarray, cx: int, cy: int, size: int,
                    class_info: Dict, variations: Dict[str, float]):
        """วาดหิน"""
        color = self._adjust_color(class_info['base_color'], variations)
        
        # สร้างรูปทรงไม่แน่นอน
        points = []
        for i in range(8):
            angle = i * 2 * np.pi / 8 + random.uniform(-0.3, 0.3)
            radius = size * random.uniform(0.3, 0.7)
            x = cx + int(radius * np.cos(angle))
            y = cy + int(radius * np.sin(angle))
            points.append([x, y])
        
        points = np.array(points, np.int32)
        cv2.fillPoly(img, [points], color)
        
    def _draw_button(self, img: np.ndarray, cx: int, cy: int, size: int,
                     class_info: Dict, variations: Dict[str, float]):
        """วาดกระดุม"""
        color = self._adjust_color(class_info['base_color'], variations)
        
        radius = int(size * 0.4)
        cv2.circle(img, (cx, cy), radius, color, -1)
        
        # เพิ่มรูกระดุม
        hole_radius = 3
        cv2.circle(img, (cx-8, cy-8), hole_radius, (0, 0, 0), -1)
        cv2.circle(img, (cx+8, cy-8), hole_radius, (0, 0, 0), -1)
        cv2.circle(img, (cx-8, cy+8), hole_radius, (0, 0, 0), -1)
        cv2.circle(img, (cx+8, cy+8), hole_radius, (0, 0, 0), -1)
    
    def _adjust_color(self, base_color: Tuple[int, int, int], variations: Dict[str, float]) -> Tuple[int, int, int]:
        """ปรับสีตาม variations"""
        r, g, b = base_color
        
        # ปรับตามการเปลี่ยนสี
        color_shift = variations['color_shift']
        r = int(r * color_shift)
        g = int(g * color_shift)
        b = int(b * color_shift)
        
        # ปรับตามอายุ
        age_effect = variations['age_effect']
        darkening = int(50 * age_effect)
        r = max(0, r - darkening)
        g = max(0, g - darkening)
        b = max(0, b - darkening)
        
        return (max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
    
    def _add_texture(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int, variations: Dict[str, float]):
        """เพิ่ม texture และรายละเอียด"""
        # เพิ่มสัญญาณรบกวนเล็กน้อย
        noise = np.random.randint(-10, 10, (y2-y1, x2-x1, 3))
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = np.clip(roi.astype(int) + noise, 0, 255).astype(np.uint8)
    
    def _apply_lighting(self, img: np.ndarray, lighting: float) -> np.ndarray:
        """ปรับแสง"""
        return np.clip(img * lighting, 0, 255).astype(np.uint8)
    
    def _apply_wear_effect(self, img: np.ndarray, wear_level: float) -> np.ndarray:
        """เพิ่มเอฟเฟกต์ความสึกกรอน"""
        if wear_level > 0:
            # เพิ่มรอยขีดข่วน
            for _ in range(int(wear_level * 10)):
                x1, y1 = random.randint(0, 223), random.randint(0, 223)
                x2, y2 = x1 + random.randint(-20, 20), y1 + random.randint(-20, 20)
                x2, y2 = max(0, min(223, x2)), max(0, min(223, y2))
                cv2.line(img, (x1, y1), (x2, y2), (100, 100, 100), 1)
        return img
    
    def _apply_age_effect(self, img: np.ndarray, age_effect: float) -> np.ndarray:
        """เพิ่มเอฟเฟกต์อายุ"""
        if age_effect > 0:
            # เพิ่มจุดด่างดำ
            for _ in range(int(age_effect * 15)):
                x, y = random.randint(0, 223), random.randint(0, 223)
                radius = random.randint(1, 3)
                color = tuple(random.randint(50, 100) for _ in range(3))
                cv2.circle(img, (x, y), radius, color, -1)
        return img
    
    def _apply_shadow(self, img: np.ndarray, shadow_intensity: float) -> np.ndarray:
        """เพิ่มเงา"""
        if shadow_intensity > 0:
            # สร้างเงาด้านล่าง-ขวา
            shadow = np.zeros_like(img)
            shadow[10:, 10:] = img[:-10, :-10]
            shadow = (shadow * shadow_intensity * 0.3).astype(np.uint8)
            img = np.maximum(img, shadow)
        return img
    
    def _apply_color_shift(self, img: np.ndarray, color_shift: float) -> np.ndarray:
        """ปรับเปลี่ยนสี"""
        return img  # สีถูกปรับใน _adjust_color แล้ว


class RealisticDatasetCreator:
    """สร้างฐานข้อมูลแบบสมจริง"""
    
    def __init__(self, output_dir: str = "dataset_realistic"):
        self.output_dir = Path(output_dir)
        self.config = DatasetConfig()
        self.generator = RealisticImageGenerator(self.config)
        
        # สร้างโฟลเดอร์
        self.train_dir = self.output_dir / "train"
        self.test_dir = self.output_dir / "test"
        self.ood_test_dir = self.output_dir / "ood_test"
        
        for dir_path in [self.train_dir, self.test_dir, self.ood_test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_dataset(self):
        """สร้างฐานข้อมูลทั้งหมด"""
        logger.info("🚀 Starting realistic dataset creation...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'target_classes': len(self.config.target_classes),
                'ood_classes': len(self.config.ood_classes),
                'variations_applied': list(self.config.variations.keys())
            },
            'classes_created': {},
            'statistics': {}
        }
        
        # 1. สร้าง Target Classes (Train + Test)
        logger.info("📸 Creating target classes (In-Distribution)...")
        for class_name, class_info in self.config.target_classes.items():
            logger.info(f"  Creating {class_name}...")
            
            # สร้างโฟลเดอร์
            train_class_dir = self.train_dir / class_name
            test_class_dir = self.test_dir / class_name
            train_class_dir.mkdir(exist_ok=True)
            test_class_dir.mkdir(exist_ok=True)
            
            # สร้างภาพ Train
            train_count = self._create_class_images(
                class_info, class_name, train_class_dir, 
                class_info['samples_train'], is_target=True
            )
            
            # สร้างภาพ Test
            test_count = self._create_class_images(
                class_info, class_name, test_class_dir,
                class_info['samples_test'], is_target=True
            )
            
            report['classes_created'][class_name] = {
                'type': 'target',
                'train_samples': train_count,
                'test_samples': test_count,
                'description': class_info['description']
            }
        
        # 2. สร้าง OOD Classes (Test เท่านั้น)
        logger.info("🎭 Creating OOD classes (Out-of-Distribution)...")
        for class_name, class_info in self.config.ood_classes.items():
            logger.info(f"  Creating OOD: {class_name}...")
            
            # สร้างโฟลเดอร์
            ood_class_dir = self.ood_test_dir / class_name
            ood_class_dir.mkdir(exist_ok=True)
            
            # สร้างภาพ OOD Test
            ood_count = self._create_class_images(
                class_info, class_name, ood_class_dir,
                class_info['samples_test'], is_target=False
            )
            
            report['classes_created'][class_name] = {
                'type': 'ood',
                'test_samples': ood_count,
                'description': class_info['description']
            }
        
        # 3. สร้าง Mixed Test Set
        logger.info("🔀 Creating mixed test set...")
        mixed_test_dir = self.output_dir / "mixed_test"
        mixed_test_dir.mkdir(exist_ok=True)
        self._create_mixed_test_set(mixed_test_dir)
        
        # 4. คำนวณสถิติ
        report['statistics'] = self._calculate_statistics()
        
        # 5. บันทึก report
        report_file = self.output_dir / "dataset_creation_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ Realistic dataset creation completed!")
        self._print_summary(report)
        
        return report
    
    def _create_class_images(self, class_info: Dict, class_name: str, output_dir: Path, 
                           num_samples: int, is_target: bool) -> int:
        """สร้างภาพสำหรับ class หนึ่งๆ"""
        created_count = 0
        
        for i in range(num_samples):
            # สุ่ม variations
            variations = {
                'lighting': random.choice(self.config.variations['lighting']),
                'wear_level': random.choice(self.config.variations['wear_levels']),
                'age_effect': random.choice(self.config.variations['age_effects']),
                'shadow_intensity': random.choice(self.config.variations['shadow_intensity']),
                'color_shift': random.choice(self.config.variations['color_shift'])
            }
            
            # สร้างภาพ
            if is_target:
                img = self.generator.generate_amulet_image(class_info, variations)
            else:
                img = self.generator.generate_ood_image(class_info, variations)
            
            # บันทึกภาพ
            filename = f"{class_name}_{i:03d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            created_count += 1
            
            if (i + 1) % 20 == 0:
                logger.info(f"    Created {i + 1}/{num_samples} images")
        
        return created_count
    
    def _create_mixed_test_set(self, mixed_dir: Path):
        """สร้างชุดทดสอบแบบผสม (Target + OOD)"""
        # คัดลอกภาพจาก test set
        target_images = []
        for class_dir in self.test_dir.iterdir():
            if class_dir.is_dir():
                class_images = list(class_dir.glob("*.jpg"))
                # เลือกแค่ครึ่งหนึ่ง
                selected = random.sample(class_images, min(10, len(class_images)))
                for img_path in selected:
                    new_name = f"target_{class_dir.name}_{img_path.name}"
                    shutil.copy2(img_path, mixed_dir / new_name)
                    target_images.append(new_name)
        
        # คัดลอกภาพจาก OOD set
        ood_images = []
        for class_dir in self.ood_test_dir.iterdir():
            if class_dir.is_dir():
                class_images = list(class_dir.glob("*.jpg"))
                # เลือกแค่ครึ่งหนึ่ง
                selected = random.sample(class_images, min(8, len(class_images)))
                for img_path in selected:
                    new_name = f"ood_{class_dir.name}_{img_path.name}"
                    shutil.copy2(img_path, mixed_dir / new_name)
                    ood_images.append(new_name)
        
        # บันทึกรายการ
        mixed_info = {
            'target_images': target_images,
            'ood_images': ood_images,
            'total_target': len(target_images),
            'total_ood': len(ood_images)
        }
        
        with open(mixed_dir / "mixed_test_info.json", 'w') as f:
            json.dump(mixed_info, f, indent=2)
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """คำนวณสถิติของฐานข้อมูล"""
        stats = {
            'train_total': 0,
            'test_total': 0,
            'ood_total': 0,
            'class_distribution': {}
        }
        
        # นับภาพ Train
        if self.train_dir.exists():
            for class_dir in self.train_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.jpg")))
                    stats['train_total'] += count
                    stats['class_distribution'][f"{class_dir.name}_train"] = count
        
        # นับภาพ Test
        if self.test_dir.exists():
            for class_dir in self.test_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.jpg")))
                    stats['test_total'] += count
                    stats['class_distribution'][f"{class_dir.name}_test"] = count
        
        # นับภาพ OOD
        if self.ood_test_dir.exists():
            for class_dir in self.ood_test_dir.iterdir():
                if class_dir.is_dir():
                    count = len(list(class_dir.glob("*.jpg")))
                    stats['ood_total'] += count
                    stats['class_distribution'][f"{class_dir.name}_ood"] = count
        
        return stats
    
    def _print_summary(self, report: Dict[str, Any]):
        """พิมพ์สรุปผลลัพธ์"""
        print("\n" + "="*60)
        print("📊 REALISTIC DATASET CREATION SUMMARY")
        print("="*60)
        
        stats = report['statistics']
        print(f"🎯 Target Classes (Train): {stats['train_total']} images")
        print(f"🧪 Target Classes (Test): {stats['test_total']} images")
        print(f"🎭 OOD Classes (Test): {stats['ood_total']} images")
        print(f"📁 Total Images: {stats['train_total'] + stats['test_total'] + stats['ood_total']}")
        
        print(f"\n📋 Class Distribution:")
        for class_name, class_info in report['classes_created'].items():
            if class_info['type'] == 'target':
                print(f"  {class_name}: {class_info['train_samples']} train + {class_info['test_samples']} test")
            else:
                print(f"  {class_name} (OOD): {class_info['test_samples']} test")
        
        print(f"\n✨ Features Applied:")
        for feature in report['config']['variations_applied']:
            print(f"  ✓ {feature}")
        
        print(f"\n📄 Report saved to: {self.output_dir}/dataset_creation_report.json")


if __name__ == "__main__":
    # สร้างฐานข้อมูลใหม่
    creator = RealisticDatasetCreator("dataset_realistic")
    report = creator.create_dataset()
    
    print("\n🎉 Dataset creation completed successfully!")
    print("Ready for training and OOD testing! 🚀")