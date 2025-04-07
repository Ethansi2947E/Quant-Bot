#!/usr/bin/env python3
import os
import shutil
import sys
from pathlib import Path

def clean_pycache(start_dir='.'):
    """Remove __pycache__ directories and .pyc files recursively."""
    count_dirs = 0
    count_files = 0
    
    # Skip directories to avoid
    dirs_to_skip = ['.venv', 'venv', 'env', '.git']
    
    # Walk through directory structure
    for root, dirs, files in os.walk(start_dir):
        # Skip specified directories
        for skip_dir in dirs_to_skip:
            if skip_dir in dirs:
                dirs.remove(skip_dir)
                
        # Remove __pycache__ directories
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            print(f"Removing directory: {pycache_path}")
            shutil.rmtree(pycache_path)
            count_dirs += 1
            dirs.remove('__pycache__')  # Don't traverse into deleted directories
            
        # Remove .pyc files
        for file in files:
            if file.endswith('.pyc'):
                pyc_file = os.path.join(root, file)
                print(f"Removing file: {pyc_file}")
                os.remove(pyc_file)
                count_files += 1
    
    print(f"\nRemoved {count_dirs} __pycache__ directories and {count_files} .pyc files.")

if __name__ == "__main__":
    clean_pycache()
    print("\nTo prevent Python from creating __pycache__ directories, use one of these methods:")
    print("\n1. Environment variable (already added to your .env file):")
    print("   PYTHONDONTWRITEBYTECODE=1")
    print("\n2. Add this at the beginning of your main.py and other entry point files:")
    print("   import sys")
    print("   sys.dont_write_bytecode = True")
    print("\n3. Run Python with the -B flag:")
    print("   python -B your_script.py")
    print("\n4. Create a PYTHONDONTWRITEBYTECODE=1 file in your project root") 