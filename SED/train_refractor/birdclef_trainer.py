import torch
import pytorch_lightning as pl
import torch.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from .config import TrainingConfig
from .model_factory import ModelFactory
from .audio_processor import AudioProcessor
from .data_loader import create_data_loaders
import numpy as np
import pandas as pd
import sklearn.metrics
from typing import Dict, List, Tuple
import torch.functional as F

class BirdCLEFTrainer(pl.LightningModule):

    def __init__(self, config: TrainingConfig, stage: str):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.stage = stage
        self.stage_config = config.get_stage_config(stage)
        
        self.model_factory = ModelFactory(config)
        self.audio_processor = AudioProcessor(config)
        
        self.backbone = self.model_factory.create_backbone()
        self.normalization = self.model_factory.create_normalization_layer()
        self.attention = self.model_factory.create_attention_block()
        self.classifier = self.model_factory.create_classification_head()
        self.loss_fn = self.model_factory.create_loss_function(self.stage_config["loss"])
        
        self.train_metrics = {"loss": []}
        self.val_metrics = {"loss": [], "cmap": [], "ap": []}
        
        if stage == "train_ce":
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:

        spectrogram = self.audio_processor.process_audio(audio_input, self.training)
        spectrogram = self.audio_processor.create_spectrogram_converter()(spectrogram)
        spectrogram = self.audio_processor.create_db_converter()(spectrogram)
        spectrogram = self.audio_processor.normalize_spectrogram(spectrogram)

        if self.training:
            spectrogram = self.audio_processor.apply_time_frequency_masking(spectrogram)
        
        features = self._extract_features(spectrogram)
        
        clipwise_output, _, _ = self.attention(features)
        return clipwise_output

    def _extract_features(self, spectrogram: torch.Tensor) -> torch.Tensor:
        x = self.normalization(spectrogram)
        features = self.backbone(x)[0]

        x = torch.mean(features, dim=2)
        x_max = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x_avg = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        return x_max + x_avg

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        audio, targets, weights = batch
        logits = self(audio)
        loss = self._compute_loss(logits, targets, weights)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_metrics["loss"].append(loss.detach())
        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict:
        audio, targets, _ = batch
        logits = self(audio)
        loss = self._compute_loss(logits, targets)
        
        probs = torch.sigmoid(logits).detach()
        return {
            "val_loss": loss.detach(),
            "probs": probs,
            "targets": targets.detach()
        }

    def on_validation_epoch_end(self) -> Dict:
        """Calculate validation metrics at epoch end"""
        outputs = self.trainer.validation_loop.outputs
        if not outputs:
            return {}
            
        # Aggregate outputs
        all_probs = torch.cat([x["probs"] for x in outputs])
        all_targets = torch.cat([x["targets"] for x in outputs])
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        
        # Calculate metrics
        cmap_score = self._calculate_cmap(all_targets, all_probs)
        ap_score = sklearn.metrics.average_precision_score(
            all_targets.cpu().numpy(),
            all_probs.cpu().numpy(),
            average="macro"
        )
        
        # Log metrics
        self.log("val_loss", avg_loss, prog_bar=True)
        self.log("val_cmap", cmap_score, prog_bar=True)
        self.log("val_ap", ap_score, prog_bar=True)
        self.val_metrics["loss"].append(avg_loss.cpu().item())
        self.val_metrics["cmap"].append(cmap_score)
        self.val_metrics["ap"].append(ap_score)
        
        return {"val_loss": avg_loss, "val_cmap": cmap_score}

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                     weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = self.loss_fn(logits, targets)
        
        if weights is not None:
            loss = (loss * weights.unsqueeze(1)) / weights.sum()
            
        return loss.mean()

    def _calculate_cmap(self, targets: torch.Tensor, probs: torch.Tensor, 
                       padding_factor: int = 5) -> float:

        target_df = pd.DataFrame(
            targets.cpu().numpy(), 
            columns=self.config.bird_species
        )
        prob_df = pd.DataFrame(
            probs.cpu().numpy(),
            columns=self.config.bird_species
        )
        padding = pd.DataFrame(
            [[1.0] * len(target_df.columns)] * padding_factor,
            columns=target_df.columns
        )
        padded_targets = pd.concat([target_df, padding]).reset_index(drop=True)
        padded_probs = pd.concat([prob_df, padding]).reset_index(drop=True)
        
        return sklearn.metrics.average_precision_score(
            padded_targets.values,
            padded_probs.values,
            average="macro"
        )

    def configure_optimizers(self) -> Dict:
        optimizer = self.model_factory.create_optimizer(self.parameters())
        scheduler = self.model_factory.create_scheduler(optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }


def train_model(config: TrainingConfig, stage: str):

    train_loader, val_loader = create_data_loaders(
        config=config,
        stage=stage
    )
    
    model = BirdCLEFTrainer(config, stage)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.output_dir}/{stage}",
        filename=config.checkpoint_format,
        save_top_k=1,
        save_last=True,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor=config.monitor_metric,
        patience=10,
        mode=config.monitor_mode,
        verbose=True
    )
    
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        max_epochs=config.get_stage_config(stage)["epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        val_check_interval=0.5,
        deterministic=True,
        precision="16-mixed",
        enable_progress_bar=True,
        logger=True,
        log_every_n_steps=10
    )
    
    print(f"Starting {stage} training for {config.get_stage_config(stage)['epochs']} epochs...")
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )
    
    final_model_path = f"{config.output_dir}/{stage}/final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    print(f"Training completed. Model saved to {final_model_path}")
    
    return model
