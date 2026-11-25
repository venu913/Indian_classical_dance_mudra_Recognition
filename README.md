# Preserving Classical Dance Heritage through AI: FastViT-T8 Mudra Recognition Platform

An end-to-end Computer Vision and Deep Learning system designed to recognize **49 Indian Classical Dance Mudras (Hastas)** from static images, integrating an adaptive digital image processing (DIP) pipeline, a **FastViT-T8** hybrid vision transformer, and an asynchronous full-stack web application.

---

## Key Highlights & Performance
- **Model Architecture**: FastViT-T8 with structural re-parameterization and custom 6-channel tensor input.
- **Accuracy**: **99.32% test accuracy** (3,949 / 3,976 correct predictions) across 49 gesture classes.
- **Macro Metrics**: Macro Precision: **0.9820** | Macro Recall: **0.9867** | Macro F1-score: **0.9835**.
- **Inference Latency**: **22.51 ms** on NVIDIA RTX 3050 Laptop GPU (**473.4 images/second** throughput).
- **Computer Vision Pipeline**: CIELAB CLAHE luminance equalization, Gaussian unsharp masking ($\sigma=2.0$), and YOLO-based hand detection with 30% adaptive contextual cropping.
- **Visualization**: Real-time **MediaPipe 21-keypoint skeletal landmark** extraction and overlay.
- **Scientific Audit**: Comprehensive data-integrity audit documenting 17.98% exact cross-split duplication to establish generalizability bounds.

---

## Project Structure
```
Kathak_Mudra_AI_Platform/
├── backend/
│   ├── app.py                      # FastAPI microservice
│   ├── pipeline.py                 # DIP enhancement + YOLO cropping + MediaPipe
│   ├── fastvit_v2.py               # FastViT-T8 model definition
│   ├── fastvit_v2_best.pth         # Trained weights checkpoint
│   ├── hand_yolo_best.pt           # YOLO hand detector weights
│   ├── train_v2_main.py            # Model training script
│   ├── evaluate_v2.py              # Test set evaluation script
│   ├── process_dataset.py          # Preprocessing utility
│   ├── split_dataset.py            # Dataset split generator
│   └── requirements.txt            # Python dependencies
├── frontend/
│   ├── src/                        # React UI source code
│   ├── public/                     # Static assets
│   ├── package.json                # Frontend dependencies
│   └── vite.config.js              # Vite bundler configuration
├── demo_images/                    # Sample test images for instant out-of-the-box evaluation
├── scripts/
│   └── download_kaggle_dataset.py  # 1-click Kaggle dataset downloader
├── run_project.bat                 # Windows 1-click launcher (starts backend + frontend)
├── start.bat                       # Lightweight background starter
└── requirements.txt                # Root Python environment requirements
```

---

## Quick Start (Running Locally)

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- CUDA-compatible GPU (recommended, but CPU supported)

### 2. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies
```bash
cd ../frontend
npm install
```

### 4. Run the Full Application (1-Click)
On Windows, simply double-click:
```bash
run_project.bat
```
Or start each service manually:
- **Backend**:
  ```bash
  cd backend
  python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Frontend**:
  ```bash
  cd frontend
  npm run dev
  ```
Navigate to **`http://localhost:5173`** in your browser.

---

## Dataset (Kaggle)
The full benchmark dataset contains **39,355 preprocessed images** arranged into `train` (31,466), `val` (3,913), and `test` (3,976) subsets across 49 mudra classes.

To download the full dataset:
1. Download from Kaggle: `https://www.kaggle.com/datasets/your_username/indian-classical-dance-mudras-49`
2. Extract the archive into the `data/` folder:
   ```
   data/v2_processed_split_dataset/
   ├── train/
   ├── val/
   └── test/
   ```

---

## Training & Evaluation
To retrain FastViT-T8 on the dataset:
```bash
python backend/train_v2_main.py
```
To evaluate the model on the test split:
```bash
python backend/evaluate_v2.py
```
