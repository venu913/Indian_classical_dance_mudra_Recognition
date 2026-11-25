import os
import shutil
import random
from pathlib import Path

DATA_DIR = Path(r"S:\urop_project\UROP_FINAL_2\data\v2_processed_13k_dataset")
OUTPUT_DIR = Path(r"S:\urop_project\UROP_FINAL_2\data\v2_processed_split_dataset")

# Standard 80/10/10 Split
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

def split_dataset():
    print("Splitting dataset into Train/Val/Test (80/10/10)...")
    
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split).mkdir(parents=True, exist_ok=True)
        
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR / d)]
    
    total_train, total_val, total_test = 0, 0, 0
    
    for cls in classes:
        for split in ['train', 'val', 'test']:
            (OUTPUT_DIR / split / cls).mkdir(exist_ok=True)
            
        images = list((DATA_DIR / cls).glob("*.jpg")) + list((DATA_DIR / cls).glob("*.png"))
        random.shuffle(images)
        
        train_count = int(len(images) * TRAIN_RATIO)
        val_count = int(len(images) * VAL_RATIO)
        
        train_imgs = images[:train_count]
        val_imgs = images[train_count:train_count+val_count]
        test_imgs = images[train_count+val_count:]
        
        for img in train_imgs:
            shutil.copy(img, OUTPUT_DIR / 'train' / cls / img.name)
            total_train += 1
            
        for img in val_imgs:
            shutil.copy(img, OUTPUT_DIR / 'val' / cls / img.name)
            total_val += 1
            
        for img in test_imgs:
            shutil.copy(img, OUTPUT_DIR / 'test' / cls / img.name)
            total_test += 1
            
    print(f"Dataset Split Complete!")
    print(f"Train: {total_train} images (80%)")
    print(f"Validation: {total_val} images (10%)")
    print(f"Test: {total_test} images (10%)")

if __name__ == "__main__":
    split_dataset()
