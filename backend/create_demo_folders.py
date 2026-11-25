import os
import shutil
import random
from pathlib import Path

# Paths
TRAIN_DIR = Path(r"S:\urop_project\UROP_FINAL_2\data\v2_split_dataset\train")
TEST_DIR = Path(r"S:\urop_project\UROP_FINAL_2\data\v2_split_dataset\test")
DEMO_DIR = Path(r"S:\urop_project\UROP_FINAL_2\presentation_demo_images")

def create_demo_folders():
    print(f"Creating Presentation Demo Folders at {DEMO_DIR}...")
    
    # 1. Create the 3 main folders
    seen_dir = DEMO_DIR / "1_already_trained_on_seen"
    unseen_dir = DEMO_DIR / "2_from_dataset_unseen"
    real_world_dir = DEMO_DIR / "3_not_from_dataset_real_world"
    
    for d in [seen_dir, unseen_dir, real_world_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Choose 5 random classes to demo
    classes = os.listdir(TRAIN_DIR)
    demo_classes = random.sample(classes, min(5, len(classes)))
    
    # 2. Copy 1 image per demo class from TRAIN (Seen)
    for cls in demo_classes:
        train_cls_dir = TRAIN_DIR / cls
        if train_cls_dir.exists():
            images = list(train_cls_dir.rglob("*.jpg")) + list(train_cls_dir.rglob("*.png"))
            if images:
                img_path = random.choice(images)
                shutil.copy(img_path, seen_dir / f"{cls}_seen.jpg")
                
    # 3. Copy 1 image per demo class from TEST (Unseen but from dataset)
    for cls in demo_classes:
        test_cls_dir = TEST_DIR / cls
        if test_cls_dir.exists():
            images = list(test_cls_dir.rglob("*.jpg")) + list(test_cls_dir.rglob("*.png"))
            if images:
                img_path = random.choice(images)
                shutil.copy(img_path, unseen_dir / f"{cls}_unseen.jpg")
                
    # 4. Add a README text file to the Real World folder
    readme_path = real_world_dir / "PUT_YOUR_OWN_PHOTOS_HERE.txt"
    with open(readme_path, "w") as f:
        f.write("Take photos using your phone or webcam and put them in this folder to prove to the professor that the AI works on completely new data!")
        
    print("Demo folders successfully generated!")

if __name__ == "__main__":
    create_demo_folders()
