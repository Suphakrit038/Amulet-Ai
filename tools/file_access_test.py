"""
Amulet-AI File Access Test
ทดสอบการเข้าถึงและอ่านไฟล์สำคัญในระบบ Amulet-AI
"""

import sys
import json
import time
from pathlib import Path
import argparse

# Define color codes for terminal output
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

def read_file(file_path, encoding='utf-8', read_binary=False, max_lines=None):
    """Read a file and return its contents"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print_error(f"File does not exist: {file_path}")
        return None
    
    print_info(f"Reading file: {file_path}")
    
    try:
        if read_binary:
            with open(file_path, 'rb') as f:
                data = f.read(1024)  # Read first 1KB for binary files
                print_success(f"Successfully read binary file (first 1KB)")
                print_info(f"File size: {file_path.stat().st_size:,} bytes")
                return data
        else:
            with open(file_path, 'r', encoding=encoding) as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    print_success(f"Successfully read file (first {max_lines} lines)")
                    return ''.join(lines)
                else:
                    content = f.read()
                    print_success(f"Successfully read file")
                    return content
    except UnicodeDecodeError:
        print_warning(f"Encoding error with {encoding}. Trying with latin-1...")
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                if max_lines:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    print_success(f"Successfully read file with latin-1 encoding (first {max_lines} lines)")
                    return ''.join(lines)
                else:
                    content = f.read()
                    print_success(f"Successfully read file with latin-1 encoding")
                    return content
        except Exception as e:
            print_error(f"Error reading file with latin-1: {e}")
            return None
    except Exception as e:
        print_error(f"Error reading file: {e}")
        return None

def read_json_file(file_path, encoding='utf-8'):
    """Read a JSON file and return its contents"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print_error(f"File does not exist: {file_path}")
        return None
    
    print_info(f"Reading JSON file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
            print_success(f"Successfully parsed JSON file")
            return data
    except json.JSONDecodeError as e:
        print_error(f"JSON parsing error: {e}")
        # Try to read as plain text to see the issue
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read(1000)  # Read first 1000 chars
                print_info(f"File content (first 1000 chars):\n{content}")
        except:
            pass
        return None
    except UnicodeDecodeError:
        print_warning(f"Encoding error with {encoding}. Trying with latin-1...")
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                data = json.load(f)
                print_success(f"Successfully parsed JSON file with latin-1 encoding")
                return data
        except Exception as e:
            print_error(f"Error reading JSON file with latin-1: {e}")
            return None
    except Exception as e:
        print_error(f"Error reading JSON file: {e}")
        return None

def print_file_info(file_path):
    """Print detailed information about a file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print_error(f"File does not exist: {file_path}")
        return
    
    print_info(f"File information for: {file_path}")
    
    # Basic file stats
    stats = file_path.stat()
    print_info(f"Size: {stats.st_size:,} bytes")
    print_info(f"Last modified: {time.ctime(stats.st_mtime)}")
    print_info(f"Created: {time.ctime(stats.st_ctime)}")
    
    # Determine file type
    extension = file_path.suffix.lower()
    
    if extension in ['.py', '.txt', '.md', '.json', '.yml', '.yaml', '.html', '.css', '.js']:
        content = read_file(file_path, max_lines=10)
        if content:
            print_info("First 10 lines:")
            print("-" * 80)
            print(content)
            print("-" * 80)
    elif extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        print_info("Image file - binary content")
        try:
            from PIL import Image
            img = Image.open(file_path)
            print_info(f"Image format: {img.format}")
            print_info(f"Image size: {img.width}x{img.height}")
            print_info(f"Image mode: {img.mode}")
        except ImportError:
            print_warning("PIL not available for image information")
        except Exception as e:
            print_error(f"Error getting image info: {e}")
    elif extension in ['.json']:
        data = read_json_file(file_path)
        if data:
            print_info("JSON structure:")
            if isinstance(data, dict):
                print_info(f"Top-level keys: {', '.join(list(data.keys())[:10])}" + 
                           ("..." if len(data.keys()) > 10 else ""))
            elif isinstance(data, list):
                print_info(f"Array with {len(data)} items")
    else:
        print_info("Binary or unknown file format")
        read_file(file_path, read_binary=True)

def main():
    parser = argparse.ArgumentParser(description='Amulet-AI File Access Test')
    parser.add_argument('file_path', type=str, help='Path to the file to test')
    parser.add_argument('--json', action='store_true', help='Parse as JSON file')
    parser.add_argument('--binary', action='store_true', help='Read as binary file')
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding')
    parser.add_argument('--lines', type=int, help='Maximum number of lines to read')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"{Colors.BLUE}🔍 AMULET-AI FILE ACCESS TEST{Colors.ENDC}")
    print("="*80 + "\n")
    
    if args.json:
        data = read_json_file(args.file_path, encoding=args.encoding)
        if data:
            print_info("File content (JSON):")
            print(json.dumps(data, indent=2, ensure_ascii=False))
    elif args.binary:
        data = read_file(args.file_path, read_binary=True)
        if data:
            print_info("File content (first bytes, hex):")
            print(' '.join(f'{b:02x}' for b in data[:32]))
    else:
        print_file_info(args.file_path)
    
    print("\n" + "="*80)
    print(f"{Colors.GREEN}✅ File access test completed{Colors.ENDC}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
