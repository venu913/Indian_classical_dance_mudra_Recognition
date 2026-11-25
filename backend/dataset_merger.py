import os
import shutil
import glob
from pathlib import Path

# Paths
KAGGLE_DIR = Path(r"S:\urop_project\UROP_FINAL_2\data\kaggle_dataset")
V1_MASTER_DIR = Path(r"S:\urop_project\UROP_FINAL\data\master_dataset_6ch\train")
V2_OUTPUT_DIR = Path(r"S:\urop_project\UROP_FINAL_2\data\v2_13k_dataset")

# The 26 Kathak Classes we care about
KATHAK_CLASSES = {
    'arala', 'ardhachandra', 'ardhapataka', 'bhramara', 'chandrakala', 
    'chatura', 'hamsapaksha', 'hamsasya', 'kapittha', 'kartarimukha', 
    'katakamukha', 'kangula', 'mayura', 'mrigasirsa', 'mukula', 'musti', 
    'padmakosa', 'pataka', 'sarpasirsha', 'shakatam', 'shikhara', 
    'simhamukha', 'suchi', 'tamrachuda', 'tripataka', 'trishula'
}

def clean_class_name(name):
    return name.lower().strip()

def build_v2_dataset():
    print(f"Building V2 13k Dataset at {V2_OUTPUT_DIR}...")
    V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create the 26 class folders
    for cls in KATHAK_CLASSES:
        (V2_OUTPUT_DIR / cls).mkdir(exist_ok=True)
        
    copied = 0
    skipped = 0
    
    # 1. Map and copy Kaggle Dataset (if it exists)
    if KAGGLE_DIR.exists():
        print("Scanning Kaggle Dataset...")
        # Kaggle might have subfolders like train/ test/ val/ or just class folders
        # We will dynamically search for class folders
        for root, dirs, files in os.walk(KAGGLE_DIR):
            for d in dirs:
                cls_name = clean_class_name(d)
                if cls_name in KATHAK_CLASSES:
                    cls_path = Path(root) / d
                    images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))
                    for img in images:
                        target_file = V2_OUTPUT_DIR / cls_name / f"kaggle_{img.name}"
                        if not target_file.exists():
                            shutil.copy(img, target_file)
                            copied += 1
                else:
                    skipped += len(list((Path(root) / d).glob("*.jpg")))
                    
    # 2. To build the *true* 6-channel dataset, we would need to run the Pipeline
    # over these raw images. But for now, we just aggregate the raw images.
    
    print(f"Successfully aggregated {copied} images for the 26 Kathak classes.")
    print(f"Ignored {skipped} images from unrelated classes.")

if __name__ == "__main__":
    build_v2_dataset()
