import os
from PIL import Image
from pathlib import Path

def clean_dataset(data_dir):
    print(f"Scanning for corrupted images in {data_dir}...")
    corrupted_count = 0
    p = Path(data_dir)
    
    all_images = list(p.rglob("*.jpg")) + list(p.rglob("*.png"))
    total = len(all_images)
    
    for idx, img_path in enumerate(all_images):
        try:
            with Image.open(img_path) as img:
                img.verify() # verify that it is, in fact, an image
        except (IOError, SyntaxError, Exception) as e:
            print(f"Corrupted image found and deleted: {img_path}")
            os.remove(img_path)
            corrupted_count += 1
            
    print(f"Scanned {total} images.")
    print(f"Successfully deleted {corrupted_count} corrupted images.")

if __name__ == "__main__":
    clean_dataset(r"S:\urop_project\UROP_FINAL_2\data\v2_split_dataset")
