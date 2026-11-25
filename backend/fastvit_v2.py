import torch
import torch.nn as nn
import timm

class FastViTMudraClassifierV2(nn.Module):
    def __init__(self, num_classes=26, pretrained=True):
        super().__init__()
        # We use the robust Apple FastViT-T8 backbone
        self.backbone = timm.create_model('fastvit_t8.apple_in1k', pretrained=pretrained, in_chans=6)
        
        # Replace the classifier head for 26 classes
        in_features = self.backbone.head.fc.in_features
        self.backbone.head.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        """
        x: [B, 6, 224, 224] (3 RGB channels + 3 Skeleton Heatmap channels)
        """
        return self.backbone(x)

def load_v2_model(weights_path=None, device="cuda"):
    model = FastViTMudraClassifierV2(num_classes=26, pretrained=False)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model
