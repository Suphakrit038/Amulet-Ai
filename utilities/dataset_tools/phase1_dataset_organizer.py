#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Organizer - Phase 1
จัดระเบียบ dataset และสร้างโครงสร้างใหม่
"""

import os
import shutil
import json
from pathlib import Path
from PIL import Image
import re
from collections import defaultdict
import datetime

class DatasetOrganizer:
    def __init__(self, source_dir="Data set", target_dir="organized_dataset"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.stats = defaultdict(lambda: defaultdict(int))
        
        # Class mapping
        self.class_mapping = {
            "ปรกโพธิ์9ใบ": "prok_bodhi_9_leaves",
            "พระสมเด็จประธานพรเนื้อพุทธกวัก": "somdej_pratanporn_buddhagavak", 
            "พระสีวลี": "phra_sivali",
            "วัดหนองอีดุก": "wat_nong_e_duk",
            "หลังรูปเหมือน": "portrait_back",
            "แหวกม่าน": "waek_man"
        }
        
    def analyze_current_dataset(self):
        """วิเคราะห์ dataset ปัจจุบัน"""
        print("🔍 วิเคราะห์ Dataset ปัจจุบัน...")
        print("=" * 50)
        
        analysis = {}
        
        for class_folder in self.source_dir.iterdir():
            if not class_folder.is_dir():
                continue
                
            class_name = class_folder.name
            english_name = self.class_mapping.get(class_name, class_name)
            
            print(f"\n📁 {class_name}")
            print(f"   English: {english_name}")
            
            # Count files
            total_files = 0
            front_files = 0
            back_files = 0
            subfolders = 0
            
            # Count direct files
            for file in class_folder.iterdir():
                if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    total_files += 1
                    filename = file.name.lower()
                    if 'front' in filename:
                        front_files += 1
                    elif 'back' in filename:
                        back_files += 1
                elif file.is_dir():
                    subfolders += 1
                    # Count files in subfolders
                    for subfile in file.rglob("*"):
                        if subfile.is_file() and subfile.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            total_files += 1
                            filename = subfile.name.lower()
                            if 'front' in filename:
                                front_files += 1
                            elif 'back' in filename:
                                back_files += 1
            
            analysis[class_name] = {
                "english_name": english_name,
                "total_files": total_files,
                "front_files": front_files,
                "back_files": back_files,
                "subfolders": subfolders,
                "other_files": total_files - front_files - back_files
            }
            
            print(f"   📊 รวม: {total_files} ไฟล์")
            print(f"   🎭 หน้า: {front_files} ไฟล์")
            print(f"   🔙 หลัง: {back_files} ไฟล์")
            print(f"   📂 โฟลเดอร์ย่อย: {subfolders}")
            print(f"   ❓ อื่นๆ: {total_files - front_files - back_files} ไฟล์")
        
        return analysis
    
    def create_organized_structure(self):
        """สร้างโครงสร้างโฟลเดอร์ใหม่"""
        print(f"\n🏗️ สร้างโครงสร้างใหม่ใน {self.target_dir}")
        
        # Create main structure
        structure = {
            "raw": "ข้อมูลต้นฉบับ",
            "processed": "ข้อมูลที่ประมวลผลแล้ว", 
            "augmented": "ข้อมูลที่เพิ่มจำนวน",
            "splits": "แบ่งข้อมูล train/val/test",
            "metadata": "ข้อมูลเมตา"
        }
        
        for folder, description in structure.items():
            folder_path = self.target_dir / folder
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"   ✅ {folder}/ - {description}")
            
            if folder in ["raw", "processed", "augmented"]:
                # Create class subfolders
                for thai_name, eng_name in self.class_mapping.items():
                    class_folder = folder_path / eng_name
                    class_folder.mkdir(exist_ok=True)
                    
                    # Create front/back subfolders
                    (class_folder / "front").mkdir(exist_ok=True)
                    (class_folder / "back").mkdir(exist_ok=True)
                    
            elif folder == "splits":
                for split in ["train", "validation", "test"]:
                    split_folder = folder_path / split
                    split_folder.mkdir(exist_ok=True)
                    for thai_name, eng_name in self.class_mapping.items():
                        class_folder = split_folder / eng_name
                        class_folder.mkdir(exist_ok=True)
                        (class_folder / "front").mkdir(exist_ok=True)
                        (class_folder / "back").mkdir(exist_ok=True)
        
        print(f"   ✅ โครงสร้างพร้อมใช้งาน!")
        
    def copy_and_organize_files(self):
        """คัดลอกและจัดระเบียบไฟล์"""
        print(f"\n📋 จัดระเบียบและคัดลอกไฟล์...")
        
        copy_stats = defaultdict(lambda: defaultdict(int))
        
        for class_folder in self.source_dir.iterdir():
            if not class_folder.is_dir():
                continue
                
            class_name = class_folder.name
            english_name = self.class_mapping.get(class_name, class_name)
            
            if english_name not in self.class_mapping.values():
                print(f"   ⚠️ ข้าม: {class_name} (ไม่อยู่ในรายการ)")
                continue
                
            print(f"\n   📁 ประมวลผล: {class_name}")
            
            # Create target folders
            target_class = self.target_dir / "raw" / english_name
            
            # Process all files in class folder and subfolders
            all_files = []
            
            # Direct files
            for file in class_folder.iterdir():
                if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    all_files.append(file)
            
            # Files in subfolders  
            for file in class_folder.rglob("*"):
                if file.is_file() and file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    if file not in all_files:  # Avoid duplicates
                        all_files.append(file)
            
            # Copy and organize files
            for file in all_files:
                try:
                    # Determine if front or back
                    filename = file.name.lower()
                    
                    if 'front' in filename:
                        target_folder = target_class / "front"
                        copy_stats[english_name]["front"] += 1
                    elif 'back' in filename:
                        target_folder = target_class / "back"  
                        copy_stats[english_name]["back"] += 1
                    else:
                        # Try to guess from filename patterns
                        if any(pattern in filename for pattern in ['หน้า', 'front', 'f']):
                            target_folder = target_class / "front"
                            copy_stats[english_name]["front"] += 1
                        elif any(pattern in filename for pattern in ['หลัง', 'back', 'b']):
                            target_folder = target_class / "back"
                            copy_stats[english_name]["back"] += 1
                        else:
                            # Default to front if unclear
                            target_folder = target_class / "front"
                            copy_stats[english_name]["front"] += 1
                            copy_stats[english_name]["unclear"] += 1
                    
                    # Generate new filename
                    counter = copy_stats[english_name]["front"] + copy_stats[english_name]["back"]
                    side = "front" if "front" in str(target_folder) else "back"
                    new_filename = f"{english_name}_{side}_{counter:03d}{file.suffix}"
                    
                    # Copy file
                    target_file = target_folder / new_filename
                    shutil.copy2(file, target_file)
                    
                except Exception as e:
                    print(f"      ❌ ข้อผิดพลาด {file.name}: {e}")
                    copy_stats[english_name]["errors"] += 1
            
            print(f"      ✅ หน้า: {copy_stats[english_name]['front']} ไฟล์")
            print(f"      ✅ หลัง: {copy_stats[english_name]['back']} ไฟล์")
            if copy_stats[english_name]["unclear"] > 0:
                print(f"      ⚠️ ไม่ชัดเจน: {copy_stats[english_name]['unclear']} ไฟล์")
            if copy_stats[english_name]["errors"] > 0:
                print(f"      ❌ ข้อผิดพลาด: {copy_stats[english_name]['errors']} ไฟล์")
        
        return copy_stats
    
    def generate_metadata(self, analysis, copy_stats):
        """สร้างไฟล์ metadata"""
        print(f"\n📄 สร้างไฟล์ metadata...")
        
        metadata = {
            "created_at": datetime.datetime.now().isoformat(),
            "source_directory": str(self.source_dir),
            "target_directory": str(self.target_dir),
            "classes": self.class_mapping,
            "original_analysis": analysis,
            "copy_statistics": dict(copy_stats),
            "total_organized_files": sum(
                stats["front"] + stats["back"] 
                for stats in copy_stats.values()
            )
        }
        
        # Save metadata
        metadata_file = self.target_dir / "metadata" / "organization_log.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ บันทึก metadata ใน {metadata_file}")
        
        # Create summary
        summary_file = self.target_dir / "metadata" / "summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("🗂️ Dataset Organization Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"📅 Created: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("📊 Class Statistics:\n")
            for class_name, stats in copy_stats.items():
                total = stats["front"] + stats["back"]
                f.write(f"  {class_name}:\n")
                f.write(f"    - Total: {total} files\n")
                f.write(f"    - Front: {stats['front']} files\n") 
                f.write(f"    - Back: {stats['back']} files\n")
                if stats["unclear"] > 0:
                    f.write(f"    - Unclear: {stats['unclear']} files\n")
                f.write("\n")
        
        print(f"   ✅ บันทึก summary ใน {summary_file}")
        
        return metadata
    
    def run_phase1(self):
        """รันการจัดระเบียบ Phase 1"""
        print("🚀 เริ่ม Phase 1: จัดระเบียบ Dataset")
        print("=" * 60)
        
        # Step 1: Analyze
        analysis = self.analyze_current_dataset()
        
        # Step 2: Create structure
        self.create_organized_structure()
        
        # Step 3: Copy and organize
        copy_stats = self.copy_and_organize_files()
        
        # Step 4: Generate metadata
        metadata = self.generate_metadata(analysis, copy_stats)
        
        print(f"\n🎉 Phase 1 เสร็จสิ้น!")
        print(f"📁 ข้อมูลใหม่อยู่ที่: {self.target_dir}")
        print(f"📋 รวมไฟล์ที่จัดระเบียบ: {metadata['total_organized_files']} ไฟล์")
        
        return metadata

def main():
    organizer = DatasetOrganizer()
    metadata = organizer.run_phase1()
    
    print(f"\n📈 สถิติสรุป:")
    for class_name, stats in metadata['copy_statistics'].items():
        total = stats.get('front', 0) + stats.get('back', 0)
        print(f"  📁 {class_name}: {total} ไฟล์ ({stats.get('front', 0)} หน้า, {stats.get('back', 0)} หลัง)")

if __name__ == "__main__":
    main()