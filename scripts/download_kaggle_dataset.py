"""
Dataset Downloader for Indian Classical Dance Mudra Dataset (49 Classes)
This script downloads and extracts the dataset from Kaggle directly into the required project structure.

Usage:
    1. Set up your Kaggle API key (kaggle.json in ~/.kaggle/)
    2. Run: python scripts/download_kaggle_dataset.py
"""

import os
import sys
import zipfile

def download_dataset():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_data_dir = os.path.join(project_root, "data")
    os.makedirs(target_data_dir, exist_ok=True)
    
    dataset_slug = "your_kaggle_username/indian-classical-dance-mudras-49"
    
    print("=" * 70)
    print(" Indian Classical Dance Mudra Dataset (49 Classes) Downloader")
    print("=" * 70)
    print(f"Target Directory: {target_data_dir}")
    print()
    
    try:
        import kagglehub
        print("[1/2] Downloading dataset using kagglehub...")
        path = kagglehub.dataset_download(dataset_slug)
        print(f"Downloaded to cache: {path}")
        print("[2/2] Ready! Link or copy dataset to data/v2_processed_split_dataset")
    except ImportError:
        print("[INFO] 'kagglehub' not installed. You can install it via: pip install kagglehub")
        print()
        print("Alternatively, download the dataset manually from:")
        print(f"  https://www.kaggle.com/datasets/{dataset_slug}")
        print()
        print("Then extract the zip file into:")
        print(f"  {os.path.join(target_data_dir, 'v2_processed_split_dataset')}")
        print("=" * 70)

if __name__ == "__main__":
    download_dataset()
