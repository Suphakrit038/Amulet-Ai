# เครื่องมือบำรุงรักษาระบบ Amulet-AI (Tools)

โฟลเดอร์นี้ประกอบด้วยเครื่องมือต่างๆ สำหรับการบำรุงรักษาและจัดการโปรเจค Amulet-AI

## เครื่องมือที่มี

| เครื่องมือ | คำอธิบาย | การใช้งาน |
|------------|----------|---------|
| `amulet_toolkit.py` | เครื่องมือรวมสำหรับการบำรุงรักษาระบบ | `python amulet_toolkit.py --menu` |
| `cleanup.py` | เครื่องมือลบไฟล์ที่ไม่จำเป็นและรวมไฟล์ซ้ำซ้อน | `python cleanup.py` |
| `cleanup_files_phase2.py` | ลบไฟล์ที่ไม่จำเป็นเพิ่มเติม (Phase 2) | `python cleanup_files_phase2.py` |
| `cleanup_root.py` | จัดระเบียบไฟล์ในโฟลเดอร์หลัก | `python cleanup_root.py [--dry-run]` |
| `comprehensive_file_test.py` | ทดสอบการทำงานของระบบไฟล์ต่างๆ | `python comprehensive_file_test.py` |
| `file_access_test.py` | ทดสอบการเข้าถึงและอ่านไฟล์สำคัญ | `python file_access_test.py [FILE]` |
| `maintenance.py` | เครื่องมือบำรุงรักษาระบบ | `python maintenance.py` |
| `organize_files.py` | จัดระเบียบไฟล์ในระบบ | `python organize_files.py` |
| `organize_internal_structure.py` | เครื่องมือจัดระเบียบภายในโฟลเดอร์ | `python organize_internal_structure.py [--dry-run] [--folder FOLDER]` |
| `repair_system.py` | ระบบซ่อมแซมอัตโนมัติ | `python repair_system.py` |
| `restructure_project.py` | เครื่องมือจัดโครงสร้างโปรเจค | `python restructure_project.py [--dry-run]` |
| `verify_system.py` | ระบบทดสอบและตรวจสอบความถูกต้อง | `python verify_system.py` |
| `show_project_structure.ps1` | สคริปต์ PowerShell แสดงโครงสร้างโปรเจค | `.\show_project_structure.ps1 [-Depth N] [-ExcludeFiles] [-OutputFile FILE]` |

## วิธีการใช้งาน

### เครื่องมือบำรุงรักษา (amulet_toolkit.py)

เครื่องมือนี้รวมฟังก์ชันการบำรุงรักษาหลายอย่างเข้าด้วยกัน:

```bash
# เปิดเมนูหลัก
python amulet_toolkit.py --menu

# ตรวจสอบระบบ
python amulet_toolkit.py --verify

# ซ่อมแซมระบบ
python amulet_toolkit.py --repair

# บำรุงรักษาระบบ
python amulet_toolkit.py --maintain

# ทดสอบไฟล์
python amulet_toolkit.py --test-file PATH
```

### เครื่องมือจัดโครงสร้างโปรเจค (restructure_project.py)

เครื่องมือนี้ช่วยในการจัดระเบียบโครงสร้างไฟล์ของโปรเจค:

```bash
# ทดสอบการจัดโครงสร้างโดยไม่ทำการเปลี่ยนแปลงจริง
python restructure_project.py --dry-run

# จัดโครงสร้างโปรเจค
python restructure_project.py

# สำรองข้อมูลเท่านั้น ไม่จัดโครงสร้าง
python restructure_project.py --backup-only
```

### เครื่องมือจัดระเบียบภายในโฟลเดอร์ (organize_internal_structure.py)

เครื่องมือนี้ช่วยในการจัดระเบียบไฟล์ภายในโฟลเดอร์หลัก:

```bash
# ทดสอบการจัดระเบียบโดยไม่ทำการเปลี่ยนแปลงจริง
python organize_internal_structure.py --dry-run

# จัดระเบียบภายในโฟลเดอร์ทั้งหมด
python organize_internal_structure.py

# จัดระเบียบเฉพาะโฟลเดอร์ ai_models
python organize_internal_structure.py --folder ai_models
```

### สคริปต์แสดงโครงสร้างโปรเจค (show_project_structure.ps1)

สคริปต์ PowerShell นี้ช่วยแสดงโครงสร้างไฟล์โปรเจคในรูปแบบที่อ่านง่าย:

```powershell
# แสดงโครงสร้างไฟล์ทั้งหมด (ความลึกเริ่มต้น: 4 ระดับ)
.\show_project_structure.ps1

# แสดงโครงสร้างไฟล์ลึกสุด 3 ระดับ
.\show_project_structure.ps1 -Depth 3

# แสดงเฉพาะโฟลเดอร์
.\show_project_structure.ps1 -ExcludeFiles

# บันทึกผลลัพธ์ลงในไฟล์
.\show_project_structure.ps1 -OutputFile structure.txt
```

## การปรับแต่งเครื่องมือ

หากต้องการปรับแต่งเครื่องมือ สามารถแก้ไขตัวแปรที่กำหนดไว้ในไฟล์:

- **restructure_project.py**: ปรับแต่ง `RESTRUCTURE_RULES` และ `INTERNAL_RESTRUCTURE_RULES`
- **organize_internal_structure.py**: ปรับแต่ง `ExcludePattern` หรือเพิ่มกฎสำหรับโฟลเดอร์เฉพาะ
- **show_project_structure.ps1**: ปรับแต่ง `ExcludePattern` หรือเพิ่มไอคอนสำหรับนามสกุลไฟล์ใหม่ใน `Get-FileIcon`

## ข้อแนะนำการใช้งาน

1. เมื่อเริ่มใช้เครื่องมือ แนะนำให้ใช้ `--dry-run` เพื่อดูการเปลี่ยนแปลงที่จะเกิดขึ้นก่อนเสมอ
2. ควรสำรองข้อมูลโปรเจคก่อนทำการจัดโครงสร้างหรือเปลี่ยนแปลงครั้งใหญ่
3. หลังจากจัดโครงสร้างใหม่ ควรตรวจสอบว่าระบบยังทำงานได้ตามปกติหรือไม่
