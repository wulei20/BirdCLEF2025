import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from ast import literal_eval
from typing import Dict, Tuple, Optional
from .config import TrainingConfig
from .audio_processor import AudioProcessor

class BirdSoundDataset(Dataset):

    def __init__(self, config: TrainingConfig, metadata: pd.DataFrame, 
                 labels: pd.DataFrame, is_training: bool = True):
        self.config = config
        self.metadata = metadata
        self.labels = labels
        self.is_training = is_training
        self.audio_processor = AudioProcessor(config)
        self.audio_length = config.audio_length
        
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.metadata.iloc[idx]
        label_vector = self.labels.iloc[idx].values.astype(np.float32)
        weight = self._calculate_sample_weight(row)
        
        audio = self._load_audio(row["file_path"])
        
        return {
            "audio": torch.tensor(audio).float(),
            "label": torch.tensor(label_vector).float(),
            "weight": torch.tensor(weight).float()
        }
    
    def _load_audio(self, file_path: str) -> np.ndarray:

        try:
            # Load audio
            audio, orig_sr = librosa.load(
                file_path, 
                sr=self.config.sample_rate,
                mono=True,
                duration=self.config.duration
            )
            
            # Handle short/long audio
            return self._process_audio_length(audio)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return np.zeros(self.audio_length)
    
    def _process_audio_length(self, audio: np.ndarray) -> np.ndarray:

        if len(audio) < self.audio_length:
            return np.pad(audio, (0, self.audio_length - len(audio)), mode='constant')
        elif len(audio) > self.audio_length:
            start = np.random.randint(0, len(audio) - self.audio_length)
            return audio[start:start + self.audio_length]
        return audio
    
    def _calculate_sample_weight(self, row: pd.Series) -> float:
        primary = row["primary_label"]
        secondary = row["secondary_labels"]
        
        if primary in self.config.bird_species:
            return 1.0 / self.species_counts.get(primary, 1)
        
        valid_secondary = [s for s in secondary if s in self.config.bird_species]
        if valid_secondary:
            return np.mean([1.0 / self.species_counts.get(s, 1) for s in valid_secondary])
        
        return 1.0  # Default weight


def create_data_loaders(config: TrainingConfig, stage: str) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    # Load and prepare metadata
    metadata_df, species_counts = _load_metadata(config)
    
    # Create label matrix
    label_matrix = np.zeros((len(metadata_df), len(config.bird_species)))
    label_df = pd.DataFrame(label_matrix, columns=config.bird_species)
    
    # Populate labels and counts
    _populate_labels(metadata_df, label_df, config, species_counts)
    
    # Split data
    train_df, val_df, train_labels, val_labels = _split_data(metadata_df, label_df, config, stage)
    
    # Create datasets
    train_dataset = BirdSoundDataset(config, train_df, train_labels, is_training=True)
    val_dataset = BirdSoundDataset(config, val_df, val_labels, is_training=False)
    
    # Create samplers
    sampler = _create_sampler(train_df, config, species_counts)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.test_batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        shuffle=False
    )
    
    return train_loader, val_loader


def _load_metadata(config: TrainingConfig) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Load metadata and calculate species counts"""
    metadata_df = pd.read_csv(os.path.join(config.train_data_path, "metadata.csv"))
    metadata_df["secondary_labels"] = metadata_df["secondary_labels"].apply(literal_eval)
    metadata_df["file_path"] = metadata_df["filename"].apply(
        lambda x: os.path.join(config.train_data_path, "audio", x)
    )
    
    # Initialize species counts
    species_counts = {species: 0.0 for species in config.bird_species}
    return metadata_df, species_counts


def _populate_labels(metadata_df: pd.DataFrame, label_df: pd.DataFrame, 
                    config: TrainingConfig, species_counts: Dict[str, float]):
    """Populate labels and update species counts"""
    for i, row in metadata_df.iterrows():
        primary = row["primary_label"]
        secondary = row["secondary_labels"]
        
        if primary in config.bird_species:
            label_df.loc[i, primary] = 1
            species_counts[primary] += 1
        
        for species in secondary:
            if species in config.bird_species:
                label_df.loc[i, species] = 0.5
                species_counts[species] += 0.5


def _split_data(metadata_df: pd.DataFrame, label_df: pd.DataFrame, 
               config: TrainingConfig, stage: str) -> Tuple:
    """Split data into training and validation sets"""
    # For simplicity, use last 10% as validation
    split_idx = int(len(metadata_df) * 0.9)
    
    train_df = metadata_df.iloc[:split_idx]
    val_df = metadata_df.iloc[split_idx:]
    
    train_labels = label_df.iloc[:split_idx]
    val_labels = label_df.iloc[split_idx:]
    
    return train_df, val_df, train_labels, val_labels


def _create_sampler(metadata_df: pd.DataFrame, config: TrainingConfig,
                   species_counts: Dict[str, float]) -> WeightedRandomSampler:
    """Create weighted sampler for imbalanced data"""
    weights = np.zeros(len(metadata_df))
    
    for i, row in metadata_df.iterrows():
        primary = row["primary_label"]
        secondary = row["secondary_labels"]
        
        if primary in config.bird_species:
            weights[i] = 1.0 / species_counts[primary]
        else:
            valid_secondary = [s for s in secondary if s in config.bird_species]
            if valid_secondary:
                weights[i] = np.mean([1.0 / species_counts[s] for s in valid_secondary])
            else:
                weights[i] = 1.0
    
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(weights),
        replacement=True
    )
