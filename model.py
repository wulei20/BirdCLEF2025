import torch.nn as nn
import torchvision.models as models
import torch

#CNN model layers
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    #Define how input moves through layers
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    


class BirdNet(nn.Module):
    def __init__(self, num_labels):
        super().__init__()

        # 视觉分支：ResNet18 预训练，替换最后 fc
        backbone = models.resnet18(weights=None)
        state_dict = torch.load("resnet18-f37072fd.pth")
        backbone.load_state_dict(state_dict)
        
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # 输出 512×1×1
        self.img_head = nn.Sequential(nn.Flatten())                     # 512

        # 分类器
        self.classifier = nn.Linear(512, num_labels)

    def forward(self, img):
        x_img = self.backbone(img)          # [B,512,1,1]
        x_img = self.img_head(x_img)        # [B,512]

        logits = self.classifier(x_img)           # [B,num_labels]
        return torch.sigmoid(logits)          # 概率 [0‑1]