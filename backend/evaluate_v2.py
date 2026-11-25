import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from fastvit_v2 import FastViTMudraClassifierV2
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# 1. Configuration
TEST_DIR = r"S:\urop_project\UROP_FINAL_2\data\v2_processed_split_dataset\test"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KATHAK_CLASSES = sorted(os.listdir(TEST_DIR))
NUM_CLASSES = len(KATHAK_CLASSES)
WEIGHTS_PATH = r"S:\urop_project\UROP_FINAL_2\backend\fastvit_v2_best.pth"

class Dummy6ChannelDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        for idx, cls in enumerate(KATHAK_CLASSES):
            cls_path = os.path.join(data_dir, cls)
            for img_name in os.listdir(cls_path):
                self.images.append(os.path.join(cls_path, img_name))
                self.labels.append(idx)
                
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img_tensor = self.transform(img)
        
        six_ch_tensor = torch.cat([img_tensor, img_tensor], dim=0)
        return six_ch_tensor, self.labels[idx]

def evaluate():
    print(f"Starting V2 FastViT Evaluation on {DEVICE}...")
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    test_dataset = Dummy6ChannelDataset(TEST_DIR, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = FastViTMudraClassifierV2(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=KATHAK_CLASSES))
    
    print("\n=== Generating Confusion Matrix ===")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=KATHAK_CLASSES, yticklabels=KATHAK_CLASSES)
    plt.title('V2 FastViT Confusion Matrix (47 Classes)')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig(r'S:\urop_project\UROP_FINAL_2\backend\v2_confusion_matrix.png')
    print("Saved Confusion Matrix to S:\\urop_project\\UROP_FINAL_2\\backend\\v2_confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
