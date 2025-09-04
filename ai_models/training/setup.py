"""
🚀 Installation script for AI model training dependencies
สคริปต์สำหรับติดตั้ง dependencies สำหรับการเทรนโมเดล
"""
import subprocess
import sys
import os
import argparse

def check_package_installed(package_name):
    """Check if a package is already installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_base_requirements():
    """Install base requirements for the project"""
    print("Installing base requirements...")
    packages = [
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "tqdm",
        "pillow",
        "requests"
    ]
    
    for package in packages:
        if not check_package_installed(package.split("==")[0].lower()):
            print(f"Installing {package}...")
            install_package(package)
        else:
            print(f"{package} already installed.")

def install_pytorch(cuda_version=None):
    """Install PyTorch with the appropriate CUDA version"""
    print("Installing PyTorch...")
    
    # Check if PyTorch is already installed
    if check_package_installed("torch"):
        print("PyTorch already installed.")
        return
    
    # Default to CPU version if no CUDA version specified
    if cuda_version is None or cuda_version == "cpu":
        install_package("torch==2.0.1")
        install_package("torchvision==0.15.2")
        install_package("torchaudio==2.0.2")
        print("Installed PyTorch (CPU version)")
        return
    
    # Install PyTorch with CUDA support
    if cuda_version == "11.7":
        install_package("torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html")
        install_package("torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html")
        install_package("torchaudio==2.0.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html")
    elif cuda_version == "11.8":
        install_package("torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html")
        install_package("torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html")
        install_package("torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html")
    elif cuda_version == "12.1":
        install_package("torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html")
        install_package("torchvision==0.16.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html")
        install_package("torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html")
    else:
        print(f"Unsupported CUDA version: {cuda_version}")
        print("Installing CPU version instead...")
        install_package("torch==2.0.1")
        install_package("torchvision==0.15.2")
        install_package("torchaudio==2.0.2")
    
    print(f"Installed PyTorch with CUDA {cuda_version}")

def install_extra_requirements():
    """Install extra requirements for advanced features"""
    print("Installing extra requirements...")
    packages = [
        "opencv-python",
        "albumentations",
        "tensorboard",
        "jupyter",
        "ipython"
    ]
    
    for package in packages:
        if not check_package_installed(package.split("==")[0].lower().replace("-", "_")):
            print(f"Installing {package}...")
            install_package(package)
        else:
            print(f"{package} already installed.")

def verify_installation():
    """Verify the installation"""
    print("\nVerifying installation...")
    
    try:
        import torch
        import torchvision
        import numpy
        import pandas
        import matplotlib
        import sklearn
        import PIL
        import tqdm
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
        
        print("\nAll required packages are installed!")
        return True
    except ImportError as e:
        print(f"Installation verification failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Install dependencies for AI model training')
    parser.add_argument('--cuda', type=str, choices=['cpu', '11.7', '11.8', '12.1'], default=None,
                        help='CUDA version to install PyTorch with (default: auto-detect)')
    parser.add_argument('--no-extra', action='store_true', help='Skip installing extra requirements')
    
    args = parser.parse_args()
    
    print("Installing dependencies for AI model training...")
    
    # Install base requirements
    install_base_requirements()
    
    # Install PyTorch
    install_pytorch(args.cuda)
    
    # Install extra requirements
    if not args.no_extra:
        install_extra_requirements()
    
    # Verify installation
    success = verify_installation()
    
    if success:
        print("\nSetup completed successfully!")
        print("You can now run the training script with: python run_training.py")
    else:
        print("\nSetup completed with some issues. Please check the errors above.")

if __name__ == "__main__":
    main()
