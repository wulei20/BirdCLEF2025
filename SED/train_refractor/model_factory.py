import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm
from typing import Dict, Optional, Type, Tuple
from .config import TrainingConfig

class ModelFactory:

    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def create_backbone(self) -> nn.Module:

        return timm.create_model(
            self.config.model_name,
            pretrained=True,
            in_chans=1,
            features_only=True,
            out_indices=[-1]
        )
    
    def create_attention_block(self) -> nn.Module:

        return AttentionBlock(
            in_features=self.config.feature_dim,
            out_features=self.config.num_classes
        )
    
    def create_normalization_layer(self) -> nn.Module:

        return nn.BatchNorm2d(self.config.n_mels)
    
    def create_classification_head(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.config.feature_dim, self.config.attention_dim),
            nn.ReLU(),
            nn.Linear(self.config.attention_dim, self.config.num_classes)
        )
    
    def create_loss_function(self, loss_type: str) -> nn.Module:
        if loss_type == "ce":
            return nn.CrossEntropyLoss(
                label_smoothing=self.config.label_smoothing,
                reduction='none'
            )
        else:  # BCE
            return nn.BCEWithLogitsLoss(reduction='none')
    
    def create_optimizer(self, parameters) -> torch.optim.Optimizer:

        return torch.optim.Adam(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def create_scheduler(self, optimizer) -> torch.optim.lr_scheduler._LRScheduler:

        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=self.config.get_stage_config("train_ce")["epochs"],
            eta_min=1e-6
        )


class AttentionBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=1, bias=True),
            nn.Tanh(),
            nn.Softmax(dim=-1)
        )
        self.classifier = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=1, bias=True),
            nn.Sigmoid())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention_weights = self.attention(x)
        class_activations = self.classifier(x)
        weighted_output = torch.sum(attention_weights * class_activations, dim=2)
        return weighted_output, attention_weights, class_activations
