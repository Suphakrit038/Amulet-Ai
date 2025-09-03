"""
Amulet-AI Tools Package
เครื่องมือสำหรับระบบ Amulet-AI

โมดูลนี้รวมเครื่องมือต่างๆ สำหรับการทำงานกับ Amulet-AI:
- การตรวจสอบระบบ
- การบำรุงรักษา
- การทดสอบ
- การแก้ไขปัญหา
"""

from .amulet_toolkit import (
    # Utility functions
    read_file,
    read_json_file,
    print_colored,
    print_success,
    print_warning,
    print_error,
    print_info,
    print_header,
    get_file_hash,
    
    # Verification class
    SystemVerifier,
    
    # Repair class
    SystemRepair,
    
    # Maintenance class
    SystemMaintainer,
    
    # Testing functions
    print_file_info,
    
    # Direct access to main functions
    main as run_toolkit,
    main_menu as show_toolkit_menu
)

# Import constants
from .amulet_toolkit import (
    PROJECT_ROOT,
    DIRS,
    CRITICAL_FILES,
    BACKUP_DIR,
    TEMP_FILE_PATTERNS,
    Colors
)
