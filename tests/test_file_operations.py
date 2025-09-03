"""
File Operations Test Script
เครื่องมือทดสอบการเปิดและปิดไฟล์ในระบบ Amulet-AI
"""

import json
import os
from pathlib import Path
import sys

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

def print_success(message):
    print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")

def print_warning(message):
    print(f"{Colors.YELLOW}⚠️ {message}{Colors.ENDC}")

def print_error(message):
    print(f"{Colors.RED}❌ {message}{Colors.ENDC}")

def print_info(message):
    print(f"{Colors.BLUE}ℹ️ {message}{Colors.ENDC}")

def test_file_open_close(file_path, encoding='utf-8'):
    """
    Test opening and closing a file with specified encoding
    """
    file_path = Path(file_path)
    
    print_info(f"Testing file: {file_path}")
    print_info(f"Using encoding: {encoding}")
    
    if not file_path.exists():
        print_error(f"File does not exist: {file_path}")
        return False
    
    try:
        # Try to open the file
        with open(file_path, 'r', encoding=encoding) as f:
            # Read the first few characters to verify
            content = f.read(100)
            print_success(f"File opened successfully")
            print_info(f"First 100 characters: {content[:100]}...")
        
        # If we got here, the file was successfully closed
        print_success(f"File closed successfully")
        return True
    
    except UnicodeDecodeError as e:
        print_error(f"Encoding error: {e}")
        print_warning(f"Try a different encoding like 'utf-8-sig', 'latin-1', or 'cp1252'")
        return False
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def test_json_file(file_path, encoding='utf-8'):
    """
    Test opening a JSON file, parsing it, and closing it
    """
    file_path = Path(file_path)
    
    print_info(f"Testing JSON file: {file_path}")
    print_info(f"Using encoding: {encoding}")
    
    if not file_path.exists():
        print_error(f"File does not exist: {file_path}")
        return False
    
    try:
        # Try to open and parse the JSON file
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
            print_success(f"JSON file parsed successfully")
            
            # Display the top-level keys
            print_info(f"Top-level keys: {', '.join(data.keys())}")
        
        # If we got here, the file was successfully closed
        print_success(f"File closed successfully")
        return True
    
    except json.JSONDecodeError as e:
        print_error(f"JSON parsing error: {e}")
        return False
    
    except UnicodeDecodeError as e:
        print_error(f"Encoding error: {e}")
        print_warning(f"Try a different encoding like 'utf-8-sig', 'latin-1', or 'cp1252'")
        return False
    
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def fix_config_encoding(file_path, source_encoding='latin-1', target_encoding='utf-8'):
    """
    Fix encoding issues in the config file
    """
    file_path = Path(file_path)
    
    print_info(f"Attempting to fix encoding in: {file_path}")
    print_info(f"Converting from {source_encoding} to {target_encoding}")
    
    if not file_path.exists():
        print_error(f"File does not exist: {file_path}")
        return False
    
    # Create a backup first
    backup_path = file_path.with_suffix('.json.bak')
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        print_success(f"Created backup at: {backup_path}")
    except Exception as e:
        print_error(f"Failed to create backup: {e}")
        return False
    
    try:
        # Read with source encoding
        with open(file_path, 'r', encoding=source_encoding) as f:
            content = f.read()
        
        # Write with target encoding
        with open(file_path, 'w', encoding=target_encoding) as f:
            f.write(content)
        
        print_success(f"Successfully converted encoding from {source_encoding} to {target_encoding}")
        return True
    
    except Exception as e:
        print_error(f"Error fixing encoding: {e}")
        
        # Restore from backup if something went wrong
        try:
            import shutil
            shutil.copy2(backup_path, file_path)
            print_warning(f"Restored original file from backup")
        except Exception as restore_error:
            print_error(f"Failed to restore from backup: {restore_error}")
        
        return False

def main():
    print("\n" + "="*80)
    print("🔍 AMULET-AI FILE OPERATIONS TEST 🔍")
    print("="*80 + "\n")
    
    # Test opening config.json with different encodings
    config_path = Path("config.json")
    
    print("\n--- Testing with UTF-8 encoding ---")
    utf8_result = test_file_open_close(config_path, 'utf-8')
    
    if not utf8_result:
        print("\n--- Testing with Latin-1 encoding ---")
        latin1_result = test_file_open_close(config_path, 'latin-1')
        
        if latin1_result:
            # If Latin-1 works but UTF-8 doesn't, offer to fix
            print("\n--- Fixing encoding ---")
            should_fix = input("Would you like to fix the encoding to UTF-8? (y/n): ").lower() == 'y'
            
            if should_fix:
                fix_config_encoding(config_path, 'latin-1', 'utf-8')
                
                # Test again with UTF-8
                print("\n--- Testing with UTF-8 after fix ---")
                test_file_open_close(config_path, 'utf-8')
    
    # Test JSON parsing
    print("\n--- Testing JSON parsing ---")
    test_json_file(config_path, 'utf-8' if utf8_result else 'latin-1')
    
    # Test a few more files
    print("\n--- Testing other important files ---")
    test_file_open_close("README.md")
    test_file_open_close("docs/SYSTEM_GUIDE.md")
    
    print("\n" + "="*80)
    print("✅ File operations test completed")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
