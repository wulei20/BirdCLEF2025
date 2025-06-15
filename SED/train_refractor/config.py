from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class TrainingConfig:
    # Data parameters
    train_data_path: str = "data/train"
    batch_size: int = 32
    test_batch_size: int = 64
    num_workers: int = 4
    sample_rate: int = 32000
    duration: float = 5.0
    audio_length: int = sample_rate * duration
    
    # Feature extraction
    n_mels: int = 128
    f_min: int = 50
    f_max: int = 14000
    hop_length: int = 512
    n_fft: int = 2048
    spectrogram_normalization: str = "standard"  # "standard" or "custom"
    
    # Model architecture
    model_name: str = "sed_seresnext26t"
    feature_dim: int = 2048
    attention_dim: int = 512
    
    # Training parameters
    stages: Dict[str, Dict] = None  # Will be populated dynamically
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    label_smoothing: float = 0.1
    mix_beta: float = 0.5
    mixup_prob: float = 0.5
    mixup_double_prob: float = 0.3
    
    # Augmentation
    pitch_shift_range: tuple = (-4, 4)
    time_shift_range: tuple = (-0.5, 0.5)
    time_mask_param: int = 60
    freq_mask_param: int = 24
    gain_range: tuple = (-12, 12)
    gaussian_noise_range: tuple = (0.0001, 0.03)
    snr_range: tuple = (5, 15)
    
    # Experiment tracking
    output_dir: str = "experiments"
    checkpoint_format: str = "birdclef-{stage}-epoch{epoch:02d}-val_loss{val_loss:.2f}"
    monitor_metric: str = "val_cmap"
    monitor_mode: str = "max"
    
    def __post_init__(self):
        # Set stage-specific parameters
        self.stages = {
            "pretrain_bce": {
                "epochs": 40,
                "learning_rate": 1e-3,
                "loss": "bce"
            },
            "train_ce": {
                "epochs": 60,
                "learning_rate": 3e-4,
                "loss": "ce"
            }
        }
    
    def get_stage_config(self, stage: str) -> dict:
        return self.stages.get(stage, {})
