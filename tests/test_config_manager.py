"""
Amulet-AI Config Manager Test
ทดสอบการทำงานของระบบจัดการการกำหนดค่า
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('config_test')

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

def test_config_import():
    """Test importing the config manager"""
    try:
        from utils.config_manager import Config, get_config, set_config
        print_success("Successfully imported Config class and utility functions")
        return Config, get_config, set_config
    except ImportError as e:
        print_error(f"Failed to import Config: {e}")
        print_info("Make sure you're running this script from the project root directory")
        return None, None, None
    except Exception as e:
        print_error(f"Error importing Config: {e}")
        return None, None, None

def test_config_create_load():
    """Test creating and loading a config file"""
    Config, _, _ = test_config_import()
    if not Config:
        return False
    
    # Create a temporary config file
    temp_config_path = "temp_config_test.json"
    
    try:
        # Create sample config data
        sample_config = {
            "test": {
                "string_value": "test string",
                "int_value": 42,
                "float_value": 3.14,
                "bool_value": True,
                "list_value": [1, 2, 3],
                "dict_value": {"key": "value"}
            }
        }
        
        # Write to file directly
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            json.dump(sample_config, f, indent=2)
            
        print_success(f"Created temporary config file: {temp_config_path}")
        
        # Create Config instance and load the file
        config = Config(temp_config_path)
        print_success("Loaded config file successfully")
        
        # Verify values
        if (config.get("test.string_value") == "test string" and
            config.get("test.int_value") == 42 and
            config.get("test.float_value") == 3.14 and
            config.get("test.bool_value") == True and
            config.get("test.list_value") == [1, 2, 3] and
            config.get("test.dict_value.key") == "value"):
            print_success("Config values loaded correctly")
            return True
        else:
            print_error("Config values don't match expected values")
            return False
            
    except Exception as e:
        print_error(f"Error in config create/load test: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print_info(f"Removed temporary config file: {temp_config_path}")

def test_config_set_get():
    """Test setting and getting config values"""
    Config, _, _ = test_config_import()
    if not Config:
        return False
    
    # Create a temporary config file
    temp_config_path = "temp_config_test.json"
    
    try:
        # Create an empty config
        config = Config(temp_config_path)
        
        # Set various types of values
        config.set("string_value", "test string")
        config.set("int_value", 42)
        config.set("float_value", 3.14)
        config.set("bool_value", True)
        config.set("list_value", [1, 2, 3])
        config.set("dict_value", {"key": "value"})
        config.set("nested.deeply.key", "nested value")
        
        # Save config
        config.save_config()
        print_success("Saved config with various value types")
        
        # Create a new instance to load from file
        config2 = Config(temp_config_path)
        
        # Verify values
        tests = [
            ("string_value", "test string", "string"),
            ("int_value", 42, "integer"),
            ("float_value", 3.14, "float"),
            ("bool_value", True, "boolean"),
            ("list_value", [1, 2, 3], "list"),
            ("dict_value.key", "value", "nested dictionary"),
            ("nested.deeply.key", "nested value", "deeply nested value")
        ]
        
        all_passed = True
        for key, expected, type_name in tests:
            actual = config2.get(key)
            if actual == expected:
                print_success(f"{type_name.capitalize()} value correct: {key} = {actual}")
            else:
                print_error(f"{type_name.capitalize()} value incorrect: {key} = {actual}, expected {expected}")
                all_passed = False
        
        return all_passed
            
    except Exception as e:
        print_error(f"Error in config set/get test: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print_info(f"Removed temporary config file: {temp_config_path}")

def test_config_update():
    """Test updating existing config values"""
    Config, _, _ = test_config_import()
    if not Config:
        return False
    
    # Create a temporary config file
    temp_config_path = "temp_config_test.json"
    
    try:
        # Create an initial config with values
        config = Config(temp_config_path)
        config.set("value1", "original")
        config.set("nested.value", 100)
        config.save_config()
        
        # Update the values
        config.set("value1", "updated")
        config.set("nested.value", 200)
        config.save_config()
        print_success("Updated config values")
        
        # Create a new instance to load from file
        config2 = Config(temp_config_path)
        
        # Verify updated values
        if (config2.get("value1") == "updated" and
            config2.get("nested.value") == 200):
            print_success("Config updates verified successfully")
            return True
        else:
            print_error("Config updates verification failed")
            return False
            
    except Exception as e:
        print_error(f"Error in config update test: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print_info(f"Removed temporary config file: {temp_config_path}")

def test_utility_functions():
    """Test the utility functions get_config and set_config"""
    _, get_config, set_config = test_config_import()
    if not get_config or not set_config:
        return False
    
    try:
        # Set a value using the utility function
        set_config("utility_test.key", "utility value")
        
        # Get the value
        value = get_config("utility_test.key")
        
        if value == "utility value":
            print_success("Utility functions working correctly")
            return True
        else:
            print_error(f"Utility functions not working correctly: got {value}")
            return False
            
    except Exception as e:
        print_error(f"Error in utility functions test: {e}")
        return False

def test_main_config():
    """Test the main config.json file"""
    try:
        # Import the config manager
        from utils.config_manager import Config
        
        # Load the main config
        main_config_path = "config.json"
        config = Config(main_config_path)
        
        # Try to access some keys
        keys_to_check = [
            "model_name",
            "batch_size", 
            "learning_rate",
            "image_processing.target_size",
            "categories"
        ]
        
        success = True
        for key in keys_to_check:
            value = config.get(key)
            if value is not None:
                print_success(f"Found key in main config: {key} = {value}")
            else:
                print_warning(f"Key not found in main config: {key}")
                success = False
        
        return success
    except Exception as e:
        print_error(f"Error testing main config: {e}")
        return False

def main():
    print_header("⚙️ AMULET-AI CONFIG MANAGER TEST ⚙️")
    
    # Test results tracking
    results = {
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }
    
    # Test 1: Config Import
    print_header("TEST 1: CONFIG IMPORT")
    Config, _, _ = test_config_import()
    if Config:
        results["passed"] += 1
    else:
        results["failed"] += 1
        print_error("Skipping remaining tests since import failed")
        return
    
    # Test 2: Config Create/Load
    print_header("TEST 2: CONFIG CREATE AND LOAD")
    if test_config_create_load():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 3: Config Set/Get
    print_header("TEST 3: CONFIG SET AND GET")
    if test_config_set_get():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 4: Config Update
    print_header("TEST 4: CONFIG UPDATE")
    if test_config_update():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 5: Utility Functions
    print_header("TEST 5: UTILITY FUNCTIONS")
    if test_utility_functions():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Test 6: Main Config
    print_header("TEST 6: MAIN CONFIG FILE")
    if test_main_config():
        results["passed"] += 1
    else:
        results["failed"] += 1
    
    # Print summary
    print_header("📋 TEST SUMMARY")
    print_info(f"Tests passed: {results['passed']}")
    print_info(f"Tests failed: {results['failed']}")
    print_info(f"Tests skipped: {results['skipped']}")
    
    if results["failed"] == 0:
        print_success("All config manager tests completed successfully!")
    else:
        print_warning(f"{results['failed']} tests failed. Check the output for details.")

if __name__ == "__main__":
    main()
