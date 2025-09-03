"""
Amulet-AI System File Operations Test
ทดสอบการทำงานของระบบไฟล์ต่างๆ ใน Amulet-AI
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
import logging
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('file_test')

# Define color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'

def print_colored(color, message):
    """Print colored text to terminal"""
    print(f"{color}{message}{Colors.ENDC}")

def print_success(message):
    print_colored(Colors.GREEN, f"✅ {message}")

def print_warning(message):
    print_colored(Colors.YELLOW, f"⚠️ {message}")

def print_error(message):
    print_colored(Colors.RED, f"❌ {message}")

def print_info(message):
    print_colored(Colors.BLUE, f"ℹ️ {message}")

def print_header(message):
    print("\n" + "="*80)
    print_colored(Colors.BLUE, message)
    print("="*80)

def test_file_exists(file_path):
    """Test if a file exists"""
    file_path = Path(file_path)
    if file_path.exists():
        print_success(f"File exists: {file_path}")
        return True
    else:
        print_error(f"File does not exist: {file_path}")
        return False

def get_file_hash(file_path):
    """Get MD5 hash of a file"""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)
            return file_hash.hexdigest()
    except Exception as e:
        print_error(f"Error calculating hash for {file_path}: {e}")
        return None

def test_file_read(file_path, encoding='utf-8'):
    """Test reading a file"""
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read(1024)  # Read first 1KB
            print_success(f"File read successful: {file_path}")
            print_info(f"First 100 characters: {content[:100]}...")
            return True
    except UnicodeDecodeError:
        print_warning(f"Encoding error with {encoding}. Trying with latin-1...")
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read(1024)
                print_success(f"File read successful with latin-1: {file_path}")
                print_info(f"First 100 characters: {content[:100]}...")
                return True
        except Exception as e:
            print_error(f"Error reading file with latin-1: {e}")
            return False
    except Exception as e:
        print_error(f"Error reading file: {e}")
        return False

def test_file_write(file_path, content, encoding='utf-8'):
    """Test writing to a file"""
    # First backup the file if it exists
    file_path = Path(file_path)
    backup_path = None
    
    if file_path.exists():
        backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
        try:
            shutil.copy2(file_path, backup_path)
            print_info(f"Created backup at: {backup_path}")
        except Exception as e:
            print_error(f"Failed to create backup: {e}")
            return False
    
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        print_success(f"File write successful: {file_path}")
        
        # Verify write by reading back
        with open(file_path, 'r', encoding=encoding) as f:
            read_content = f.read()
            if read_content == content:
                print_success(f"Write verification successful")
            else:
                print_error(f"Write verification failed: content mismatch")
                
        # Restore backup if it exists
        if backup_path:
            shutil.copy2(backup_path, file_path)
            print_info(f"Restored from backup")
            
        return True
    except Exception as e:
        print_error(f"Error writing to file: {e}")
        
        # Restore backup if something went wrong
        if backup_path and backup_path.exists():
            try:
                shutil.copy2(backup_path, file_path)
                print_info(f"Restored from backup due to error")
            except Exception as restore_error:
                print_error(f"Failed to restore from backup: {restore_error}")
                
        return False

def test_json_operations(file_path):
    """Test JSON file operations"""
    try:
        # Read JSON
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print_success(f"JSON read successful: {file_path}")
            
            # Display the top-level keys
            print_info(f"Top-level keys: {', '.join(list(data.keys())[:10])}...")
            
            # Create a test key
            test_data = data.copy()
            test_data['_test_key'] = f"Test value {time.time()}"
            
            # Write to a temporary file
            temp_path = f"{file_path}.temp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2, ensure_ascii=False)
            print_success(f"JSON write successful: {temp_path}")
            
            # Read back and verify
            with open(temp_path, 'r', encoding='utf-8') as f:
                verify_data = json.load(f)
                if '_test_key' in verify_data:
                    print_success(f"JSON verification successful")
                else:
                    print_error(f"JSON verification failed: test key not found")
            
            # Clean up
            os.remove(temp_path)
            print_info(f"Removed temporary file: {temp_path}")
            
            return True
    except json.JSONDecodeError as e:
        print_error(f"JSON parsing error: {e}")
        return False
    except Exception as e:
        print_error(f"Error in JSON operations: {e}")
        return False

def test_binary_file_operations(file_path):
    """Test binary file operations"""
    if not Path(file_path).exists():
        print_warning(f"Binary file does not exist: {file_path}")
        return False
    
    try:
        # Calculate original hash
        original_hash = get_file_hash(file_path)
        
        # Read the file
        with open(file_path, 'rb') as f:
            data = f.read()
            print_success(f"Binary read successful: {file_path}")
            print_info(f"File size: {len(data):,} bytes")
        
        # Write to a temporary file
        temp_path = f"{file_path}.temp"
        with open(temp_path, 'wb') as f:
            f.write(data)
        print_success(f"Binary write successful: {temp_path}")
        
        # Verify hash
        temp_hash = get_file_hash(temp_path)
        if original_hash == temp_hash:
            print_success(f"Binary verification successful: hashes match")
        else:
            print_error(f"Binary verification failed: hash mismatch")
            print_info(f"Original hash: {original_hash}")
            print_info(f"Temp file hash: {temp_hash}")
        
        # Clean up
        os.remove(temp_path)
        print_info(f"Removed temporary file: {temp_path}")
        
        return True
    except Exception as e:
        print_error(f"Error in binary file operations: {e}")
        return False

def test_directory_operations():
    """Test directory operations"""
    # Create a temporary directory
    temp_dir = Path("temp_test_dir")
    
    try:
        # Create directory
        temp_dir.mkdir(exist_ok=True)
        print_success(f"Directory creation successful: {temp_dir}")
        
        # Create a subdirectory
        sub_dir = temp_dir / "subdir"
        sub_dir.mkdir(exist_ok=True)
        print_success(f"Subdirectory creation successful: {sub_dir}")
        
        # Create a file in the directory
        test_file = temp_dir / "test.txt"
        with open(test_file, 'w') as f:
            f.write("Test content")
        print_success(f"File creation in directory successful: {test_file}")
        
        # List directory contents
        files = list(temp_dir.glob("**/*"))
        print_info(f"Directory contains {len(files)} entries: {[f.name for f in files]}")
        
        # Remove the directory and contents
        shutil.rmtree(temp_dir)
        print_success(f"Directory removal successful: {temp_dir}")
        
        return True
    except Exception as e:
        print_error(f"Error in directory operations: {e}")
        
        # Clean up in case of error
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                print_info(f"Cleaned up directory after error: {temp_dir}")
            except:
                pass
                
        return False

def test_config_manager():
    """Test the config manager class"""
    try:
        # Import the config manager
        sys.path.append(".")
        from utils.config_manager import Config
        
        print_success(f"Successfully imported Config class")
        
        # Create a temporary config
        temp_config_path = "temp_config.json"
        config = Config(temp_config_path)
        
        # Set some values
        config.set("test.key1", "value1")
        config.set("test.key2", 123)
        config.set("test.nested.key", [1, 2, 3])
        
        # Save config
        config.save_config()
        print_success(f"Config saved successfully: {temp_config_path}")
        
        # Create a new instance and load config
        config2 = Config(temp_config_path)
        
        # Verify values
        if (config2.get("test.key1") == "value1" and 
            config2.get("test.key2") == 123 and 
            config2.get("test.nested.key") == [1, 2, 3]):
            print_success(f"Config values verified successfully")
        else:
            print_error(f"Config values verification failed")
        
        # Clean up
        os.remove(temp_config_path)
        print_info(f"Removed temporary config: {temp_config_path}")
        
        return True
    except ImportError:
        print_warning(f"Could not import Config class - skipping this test")
        return False
    except Exception as e:
        print_error(f"Error in config manager test: {e}")
        
        # Clean up in case of error
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            
        return False

def main():
    print_header("🔍 AMULET-AI COMPREHENSIVE FILE OPERATIONS TEST 🔍")
    
    # Keep track of test results
    results = {
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }
    
    # Test 1: Basic file operations
    print_header("📁 TEST 1: BASIC FILE OPERATIONS")
    
    important_files = [
        "config.json",
        "README.md",
        "requirements.txt",
        "docs/SYSTEM_GUIDE.md",
    ]
    
    for file_path in important_files:
        if test_file_exists(file_path):
            if test_file_read(file_path):
                results["passed"] += 1
            else:
                results["failed"] += 1
        else:
            results["skipped"] += 1
    
    # Test 2: JSON operations
    print_header("📊 TEST 2: JSON OPERATIONS")
    
    json_files = [
        "config.json",
        "ai_models/labels.json",
    ]
    
    for file_path in json_files:
        if test_file_exists(file_path):
            if test_json_operations(file_path):
                results["passed"] += 1
            else:
                results["failed"] += 1
        else:
            results["skipped"] += 1
    
    # Test 3: Writing operations
    print_header("✏️ TEST 3: WRITING OPERATIONS")
    
    test_content = "This is a test file.\nCreated by the Amulet-AI file test tool.\n"
    test_file = "test_write.txt"
    
    if test_file_write(test_file, test_content):
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 4: Binary file operations
    print_header("🖼️ TEST 4: BINARY FILE OPERATIONS")
    
    binary_files = [
        "ai_models/amulet_model.h5",
        "ai_models/amulet_model.tflite",
    ]
    
    for file_path in binary_files:
        if test_file_exists(file_path):
            if test_binary_file_operations(file_path):
                results["passed"] += 1
            else:
                results["failed"] += 1
        else:
            results["skipped"] += 1
    
    # Test 5: Directory operations
    print_header("📂 TEST 5: DIRECTORY OPERATIONS")
    
    if test_directory_operations():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 6: Config Manager
    print_header("⚙️ TEST 6: CONFIG MANAGER")
    
    if test_config_manager():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Print summary
    print_header("📋 TEST SUMMARY")
    print_info(f"Tests passed: {results['passed']}")
    print_info(f"Tests failed: {results['failed']}")
    print_info(f"Tests skipped: {results['skipped']}")
    
    if results["failed"] == 0:
        print_success("All tests completed successfully!")
    else:
        print_warning(f"{results['failed']} tests failed. Check the output for details.")

if __name__ == "__main__":
    main()
