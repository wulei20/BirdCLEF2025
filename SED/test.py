import os
import gc
import warnings
import logging
import time
import math
import cv2
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from soundfile import SoundFile 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import timm
from tqdm.auto import tqdm
from glob import glob
import torchaudio
import random
import itertools
from typing import Union

import concurrent.futures

from helper import *

from model import *

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

# 创建配置对象
cfg = CFG()

print(f"Using device: {cfg.device}")
print(f"Loading taxonomy data...")
taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
species_ids = taxonomy_df['primary_label'].tolist()
num_classes = len(species_ids)
print(f"Number of classes: {num_classes}")

set_seed(cfg.seed)


def load_sample(path, cfg):
    audio, orig_sr = sf.read(path, dtype="float32")
    seconds = []
    audio_length = cfg.SR * cfg.target_duration
    step = audio_length
    for i in range(audio_length, len(audio) + step, step):
        start = max(0, i - audio_length)
        end = start + audio_length
        if end > len(audio):
            pass
        else:
            seconds.append(int(end/cfg.SR))

    audio = np.concatenate([audio,audio,audio])
    audios = []
    for i,second in enumerate(seconds):
        end_seconds = int(second)
        start_seconds = int(end_seconds - cfg.target_duration)
    
        end_index = int(cfg.SR * (end_seconds + (cfg.train_duration - cfg.target_duration) / 2) ) + len(audio) // 3
        start_index = int(cfg.SR * (start_seconds - (cfg.train_duration - cfg.target_duration) / 2) ) + len(audio) // 3
        end_pad = int(cfg.SR * (cfg.train_duration - cfg.target_duration) / 2) 
        start_pad = int(cfg.SR * (cfg.train_duration - cfg.target_duration) / 2) 
        y = audio[start_index:end_index].astype(np.float32)
        if i==0:
            y[:start_pad] = 0
        elif i==(len(seconds)-1):
            y[-end_pad:] = 0
        audios.append(y)

    return audios

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def find_model_files(cfg):
    """
    Find all .pth model files in the specified model directory
    """
    model_files = []
    
    model_dir = Path(cfg.model_path)
    
    for path in model_dir.glob('**/*.pth'):
        model_files.append(str(path))
    
    return model_files

def load_models(cfg, num_classes):
    """
    Load all found model files and prepare them for ensemble
    """
    models = []
    
    # model_files = find_model_files(cfg)
    model_files = cfg.model_files
    
    if not model_files:
        print(f"Warning: No model files found under {cfg.model_path}!")
        return models
    
    print(f"Found a total of {len(model_files)} model files.")
    
    for i, model_path in enumerate(model_files):
        try:
            print(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=torch.device(cfg.device), weights_only=False)
            cfg_temp = checkpoint['cfg']
            cfg_temp['device'] = cfg.device
            
            model = BirdCLEFModel(cfg_temp)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(cfg.device)
            model.eval()
            model.zero_grad()
            model.half().float()
            
            models.append(model)
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
    
    return models

def predict_on_spectrogram(audio_path, models, cfg, species_ids):
    """Process a single audio file and predict species presence for each 5-second segment"""
    audio_path = str(audio_path)
    predictions = []
    row_ids = []
    soundscape_id = Path(audio_path).stem

    print(f"Processing {soundscape_id}")
    audio_data = load_sample(audio_path, cfg)
    for segment_idx, audio_input in enumerate(audio_data):
        
        end_time_sec = (segment_idx + 1) * cfg.target_duration
        row_id = f"{soundscape_id}_{end_time_sec}"
        row_ids.append(row_id)
        
        mel_spec = torch.tensor(audio_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mel_spec = mel_spec.to(cfg.device)
        
        if len(models) == 1:
            with torch.no_grad():
                outputs = models[0].infer(mel_spec)
                final_preds = outputs.squeeze()
                # final_preds = torch.sigmoid(outputs).cpu().numpy().squeeze()

        else:
            segment_preds = []
            for model in models:
                with torch.no_grad():
                    outputs = model.infer(mel_spec)
                    probs = outputs.squeeze()
                    # probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
                    segment_preds.append(probs)

            
            final_preds = np.mean(segment_preds, axis=0)
                
        predictions.append(final_preds)

    predictions = np.stack(predictions,axis=0)
    
    return row_ids, predictions

def run_inference(cfg, models, species_ids):
    """Run inference on all test soundscapes"""
    test_files = list(Path(cfg.test_soundscapes).glob('*.ogg'))
    if len(test_files) == 0:
        test_files = sorted(glob(str(Path('/kaggle/input/birdclef-2025/train_soundscapes') / '*.ogg')))[:10]
    
    print(f"Found {len(test_files)} test soundscapes")

    all_row_ids = []
    all_predictions = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
        executor.map(
            predict_on_spectrogram,
            test_files,
            itertools.repeat(models),
            itertools.repeat(cfg),
            itertools.repeat(species_ids)
        )
    )

    for rids, preds in results:
        all_row_ids.extend(rids)
        all_predictions.extend(preds)
    
    return all_row_ids, all_predictions

def create_submission(row_ids, predictions, species_ids, cfg):
    """Create submission dataframe"""
    print("Creating submission dataframe...")

    submission_dict = {'row_id': row_ids}
    
    for i, species in enumerate(species_ids):
        submission_dict[species] = [pred[i] for pred in predictions]

    submission_df = pd.DataFrame(submission_dict)

    submission_df.set_index('row_id', inplace=True)

    sample_sub = pd.read_csv(cfg.submission_csv, index_col='row_id')

    missing_cols = set(sample_sub.columns) - set(submission_df.columns)
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} species columns in submission")
        for col in missing_cols:
            submission_df[col] = 0.0

    submission_df = submission_df[sample_sub.columns]

    submission_df = submission_df.reset_index()
    
    return submission_df


def smooth_submission(submission_path):
        """
        Post-process the submission CSV by smoothing predictions to enforce temporal consistency.
        
        For each soundscape (grouped by the file name part of 'row_id'), each row's predictions
        are averaged with those of its neighbors using defined weights.
        
        :param submission_path: Path to the submission CSV file.
        """
        print("Smoothing submission predictions...")
        sub = pd.read_csv(submission_path)
        cols = sub.columns[1:]
        # Extract group names by splitting row_id on the last underscore
        groups = sub['row_id'].str.rsplit('_', n=1).str[0].values
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            # Get indices for the current group
            idx = np.where(groups == group)[0]
            sub_group = sub.iloc[idx].copy()
            predictions = sub_group[cols].values
            new_predictions = predictions.copy()
            
            if predictions.shape[0] > 1:
                # Smooth the predictions using neighboring segments
                new_predictions[0] = (predictions[0] * 0.8) + (predictions[1] * 0.2)
                new_predictions[-1] = (predictions[-1] * 0.8) + (predictions[-2] * 0.2)
                for i in range(1, predictions.shape[0]-1):
                    new_predictions[i] = (predictions[i-1] * 0.2) + (predictions[i] * 0.6) + (predictions[i+1] * 0.2)
            # Replace the smoothed values in the submission dataframe
            sub.iloc[idx, 1:] = new_predictions
        
        sub.to_csv(submission_path, index=False)
        print(f"Smoothed submission saved to {submission_path}")

def main():
    start_time = time.time()
    print("Starting BirdCLEF-2025 inference...")

    models = load_models(cfg, num_classes)
    
    if not models:
        print("No models found! Please check model paths.")
        return
    
    print(f"Model usage: {'Single model' if len(models) == 1 else f'Ensemble of {len(models)} models'}")

    row_ids, predictions = run_inference(cfg, models, species_ids)

    submission_df = create_submission(row_ids, predictions, species_ids, cfg)

    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    smooth_submission(submission_path)
    
    end_time = time.time()
    print(f"Inference completed in {(end_time - start_time)/60:.2f} minutes")


if __name__ == "__main__":
    main()