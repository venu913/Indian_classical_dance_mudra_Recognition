import os
import shutil
from pathlib import Path
import re

# Paths
KAGGLE_DIR = Path(r"S:\urop_project\UROP_FINAL_2\data\kaggle_dataset\images")
V1_MASTER_DIR = Path(r"S:\urop_project\UROP_FINAL\data\master_dataset")
V2_OUTPUT_DIR = Path(r"S:\urop_project\UROP_FINAL_2\data\v2_13k_dataset")

# The 50 Universal Kathak Mudras (Single & Double Hand)
KATHAK_CLASSES = {
    'alapadma', 'anjali', 'arala', 'ardhachandra', 'ardhapataka', 'bherunda', 'bhramara', 
    'chakra', 'chandrakala', 'chatura', 'garuda', 'hamsapaksha', 'hamsasya', 'kangula', 
    'kapittha', 'kapota', 'karkata', 'kartariswastika', 'katakamukha', 'katakavardhana', 
    'kartarimukha', 'khatva', 'kilaka', 'kurma', 'matsya', 'mayura', 'mrigasirsha', 
    'mukula', 'musti', 'nagabandha', 'padmakosa', 'pasha', 'pataka', 'pushpaputa', 
    'shakatam', 'samputa', 'sarpasirsha', 'shankha', 'shivalinga', 'shukatunda', 
    'shikhara', 'simhamukha', 'suchi', 'swastika', 'tamrachuda', 'trishula', 'varaha'
}

# Mapping South Indian Bharatanatyam Kaggle spellings to Universal Kathak spellings
SPELLING_MAP = {
    'alapadmam': 'alapadma',
    'aralam': 'arala',
    'ardhachandran': 'ardhachandra',
    'ardhapathaka': 'ardhapataka',
    'berunda': 'bherunda',
    'bramaram': 'bhramara',
    'chaturam': 'chatura',
    'hamsasyam': 'hamsasya',
    'kangulam': 'kangula',
    'kapith': 'kapittha',
    'kapotham': 'kapota',
    'karkatta': 'karkata',
    'katrimukha': 'kartarimukha',
    'mukulam': 'mukula',
    'mushti': 'musti',
    'pathaka': 'pataka',
    'sakata': 'shakatam',
    'shanka': 'shankha',
    'shukatundam': 'shukatunda',
    'sikharam': 'shikhara',
    'simhamukham': 'simhamukha',
    'swastikam': 'swastika',
    'tamarachudam': 'tamrachuda',
    'trishulam': 'trishula'
}

def build_v2_dataset():
    print(f"Building V2 13k Dataset at {V2_OUTPUT_DIR}...")
    V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create the 28 class folders
    for cls in KATHAK_CLASSES:
        (V2_OUTPUT_DIR / cls).mkdir(exist_ok=True)
        
    copied_kaggle = 0
    copied_master = 0
    skipped = 0
    
    # 1. Copy Kaggle Dataset
    if KAGGLE_DIR.exists():
        print("Scanning Kaggle Dataset...")
        all_images = list(KAGGLE_DIR.rglob("*.jpg")) + list(KAGGLE_DIR.rglob("*.png"))
        for img in all_images:
            raw_class_name = re.split(r'[\_\(]', img.name)[0].lower().strip()
            
            # Translate spelling if necessary
            if raw_class_name in SPELLING_MAP:
                raw_class_name = SPELLING_MAP[raw_class_name]
                
            if raw_class_name in KATHAK_CLASSES:
                target_file = V2_OUTPUT_DIR / raw_class_name / f"kaggle_{img.name}"
                if not target_file.exists():
                    shutil.copy(img, target_file)
                    copied_kaggle += 1
            else:
                skipped += 1
                
    # 2. Copy UROP_FINAL Master Dataset
    if V1_MASTER_DIR.exists():
        print("Scanning Master Dataset...")
        for split in ['train', 'val', 'test']:
            split_dir = V1_MASTER_DIR / split
            if split_dir.exists():
                for cls_dir in split_dir.iterdir():
                    if cls_dir.is_dir() and cls_dir.name.lower() in KATHAK_CLASSES:
                        for img in cls_dir.glob("*.*"):
                            target_file = V2_OUTPUT_DIR / cls_dir.name.lower() / f"master_{img.name}"
                            if not target_file.exists():
                                shutil.copy(img, target_file)
                                copied_master += 1

    print(f"Successfully aggregated {copied_kaggle} Kaggle images.")
    print(f"Successfully aggregated {copied_master} Master Dataset images.")
    print(f"Ignored {skipped} Kaggle images from unrelated classes.")

if __name__ == "__main__":
    build_v2_dataset()
