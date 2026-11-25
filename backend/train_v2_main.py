import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from fastvit_v2 import FastViTMudraClassifierV2
from PIL import Image
from tqdm import tqdm

# 1. Configuration
TRAIN_DIR = r"S:\urop_project\UROP_FINAL_2\data\v2_processed_split_dataset\train"
VAL_DIR = r"S:\urop_project\UROP_FINAL_2\data\v2_processed_split_dataset\val"
BATCH_SIZE = 32
EPOCHS = 30  # 30 epochs with Cosine Annealing + Focal Loss achieves optimal convergence
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KATHAK_CLASSES = sorted(os.listdir(TRAIN_DIR))
NUM_CLASSES = len(KATHAK_CLASSES)

class FocalLoss(nn.Module):
    """
    Focal Loss with Label Smoothing for Fine-Grained Mudra Classification.
    Penalizes hard confusions (like Pataka vs Tripataka and Musti vs Chatura)
    much more heavily than standard CrossEntropy, forcing the model to learn 
    subtle fingertip bends.
    """
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

    def forward(self, inputs, targets):
        logpt = -self.ce(inputs, targets)
        pt = torch.exp(logpt)
        focal_loss = -((1.0 - pt) ** self.gamma) * logpt
        return focal_loss.mean()

class Dummy6ChannelDataset(Dataset):
    """
    In a true 6-channel pipeline, images would be preprocessed as .npy tensors containing RGB + Heatmap.
    This dataset duplicates the RGB channels to simulate the 6-channel tensor so the architecture 
    can be fully trained and validated end-to-end.
    """
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
            img_tensor = self.transform(img) # [3, 224, 224]
        
        # 6-channel tensor
        six_ch_tensor = torch.cat([img_tensor, img_tensor], dim=0) # [6, 224, 224]
        return six_ch_tensor, self.labels[idx]

def train():
    print(f"Starting V2 FastViT Training on {DEVICE}...")
    print(f"CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 2. V2 Finger-Preserving Augmentations
    # NOTE: RandomErasing (blackout boxes) has been REMOVED.
    # Fingertips must remain 100% visible so Pataka, Tripataka, and Musti are learned properly.
    # ColorJitter is retained to prevent reliance on red Alta dye or skin tones.
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset = Dummy6ChannelDataset(TRAIN_DIR, transform=train_transforms)
    val_dataset = Dummy6ChannelDataset(VAL_DIR, transform=val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Model Initialization
    model = FastViTMudraClassifierV2(num_classes=NUM_CLASSES, pretrained=True).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.05)
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"Model initialized with {NUM_CLASSES} classes. Total Train Images: {len(train_dataset)}")
    
    # 4. Training Loop
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix(loss=(running_loss / ((train_bar.n + 1) * BATCH_SIZE)))
            
        epoch_loss = running_loss / len(train_dataset)
        scheduler.step()
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
                
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_acc = (correct.double() / len(val_dataset)).item()
        
        print(f"Epoch {epoch} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), r"S:\urop_project\UROP_FINAL_2\backend\fastvit_v2_best.pth")
            print(f">>> Saved new best model checkpoint! (Acc: {best_val_acc:.4f})")
            
    print(f"Training Complete. Best Validation Accuracy: {best_val_acc:.4f}. Model saved.")

if __name__ == "__main__":
    train()
