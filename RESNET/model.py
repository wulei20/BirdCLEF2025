import torch.nn as nn
import torchvision.models as models
import torch

class BirdNet(nn.Module):
    def __init__(self, num_labels):
        super().__init__()

        # Load ResNet18 backbone
        backbone = models.resnet18(weights=None)
        state_dict = torch.load("resnet18-f37072fd.pth")
        backbone.load_state_dict(state_dict)
        
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # [512, 1, 1] output from ResNet18
        self.img_head = nn.Sequential(nn.Flatten())                     # 512

        # Classifier layer
        self.classifier = nn.Linear(512, num_labels)

    def forward(self, img):
        x_img = self.backbone(img)          # [B,512,1,1]
        x_img = self.img_head(x_img)        # [B,512]

        logits = self.classifier(x_img)           # [B,num_labels]
        return torch.sigmoid(logits)          # [0,1] num_labels