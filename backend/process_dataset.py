import os
import cv2
from pathlib import Path
from ultralytics import YOLO
from pipeline import enhance_image, contextual_crop
from tqdm import tqdm

RAW_DATASET_DIR = Path(r"S:\urop_project\UROP_FINAL_2\data\v2_13k_dataset")
PROCESSED_DATASET_DIR = Path(r"S:\urop_project\UROP_FINAL_2\data\v2_processed_13k_dataset")

def process_dataset():
    print(f"Loading YOLO11n for dataset processing...")
    yolo_model = YOLO("yolo11n.pt")
    
    if not RAW_DATASET_DIR.exists():
        print(f"Error: {RAW_DATASET_DIR} does not exist!")
        return

    PROCESSED_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    classes = [d.name for d in RAW_DATASET_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes to process.")
    
    total_images = sum(len(list((RAW_DATASET_DIR / c).glob("*.*"))) for c in classes)
    print(f"Total images to process: {total_images}")
    
    processed_count = 0
    failed_count = 0
    
    for cls in classes:
        cls_raw_dir = RAW_DATASET_DIR / cls
        cls_proc_dir = PROCESSED_DATASET_DIR / cls
        cls_proc_dir.mkdir(exist_ok=True)
        
        images = list(cls_raw_dir.glob("*.*"))
        
        for img_path in tqdm(images, desc=f"Processing {cls}"):
            try:
                # 1. Read image
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    failed_count += 1
                    continue
                    
                # 2. DIP Enhancement
                enhanced_bgr = enhance_image(img_bgr)
                
                # 3. YOLO Detection
                results = yolo_model(enhanced_bgr, verbose=False)
                
                if len(results[0].boxes) > 0:
                    # Get highest confidence box
                    box = results[0].boxes[0].xyxy[0].cpu().numpy()
                    bbox = [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                    
                    # 4. Contextual Crop (30% padding)
                    final_crop = contextual_crop(enhanced_bgr, bbox, padding_ratio=0.30)
                else:
                    # If YOLO fails (very rare), just use the enhanced full image
                    final_crop = enhanced_bgr
                    
                # 5. Save processed image
                out_path = cls_proc_dir / img_path.name
                cv2.imwrite(str(out_path), final_crop)
                processed_count += 1
                
            except Exception as e:
                #print(f"Failed on {img_path.name}: {e}")
                failed_count += 1

    print(f"\nProcessing Complete!")
    print(f"Successfully processed: {processed_count}")
    print(f"Failed/Skipped: {failed_count}")
    print(f"Processed dataset saved to: {PROCESSED_DATASET_DIR}")

if __name__ == "__main__":
    process_dataset()
