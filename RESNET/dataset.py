import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from help_functions import *
import pandas as pd
from pathlib import Path

class BirdclefDataset(Dataset):
    def __init__(self, csv_path=TRAIN_CSV, ogg_file_root=TRAIN_AUDIO):
        self.df = pd.read_csv(csv_path)
        self.ogg_file_root = Path(ogg_file_root)

        # primary_label -> idx
        self.label2idx = {l: i for i, l in enumerate(sorted(self.df.primary_label.unique()))}
        self.num_labels = len(self.label2idx)

        self.samples = []
        for _, row in self.df.iterrows():
            ogg_path = self.ogg_file_root / Path(row.filename)
            if not ogg_path.exists():
                continue

            secondary_labels = []
            
            if pd.notna(row.secondary_labels):
                secondary_labels += eval(row.secondary_labels)  # 列表字符串 → list
            idxs = [self.label2idx[l] for l in secondary_labels if l in self.label2idx]

            target = torch.zeros(self.num_labels, dtype=torch.float32)
            target[idxs] = 0.9
            target[self.label2idx[row.primary_label]] = 1.0

            self.samples.append(
                dict(
                    ogg_path=ogg_path,
                    target=target,
                    primary_idx=self.label2idx[row.primary_label],
                    latitude=row.latitude if pd.notna(row.latitude) else 0.0,
                    longitude=row.longitude if pd.notna(row.longitude) else 0.0,
                )
            )

    def __len_of_label__(self):
        return self.num_labels

    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        filepath = s['ogg_path']

        #Loads audio and splits with helper function
        audio, _ = librosa.load(filepath, sr=SR)
        chunks = audio_split(audio, sr=SR)

        chunk = np.pad(audio, (0, SR*CHUNK_LEN - len(audio)), mode='constant') if len(chunks) == 0 else random.choice(chunks)
        #Convert chunks to actual spectrograms
        mel_res = mel_normalize(audio_to_mel_spectrogram(chunk))
        mel_res = torch.tensor(mel_res, dtype=torch.float32).unsqueeze(0)

        mel_res = ((mel_res - mel_res.min()) / (mel_res.max() - mel_res.min() + 1e-6)).unsqueeze(0)

        mel_res = torch.nn.functional.interpolate(mel_res, size=(224, 224), mode='bilinear', align_corners=False)

        mel_res = mel_res.repeat(1, 3, 1, 1).squeeze(0)  # [1,3,224,224] -> [3,224,224]

        # 归一化地理坐标到 [-1,1]
        lat = torch.tensor(s["latitude"] / 90.0, dtype=torch.float32)
        lon = torch.tensor(s["longitude"] / 180.0, dtype=torch.float32)

        return {
            "mel": mel_res,               # float tensor [1,128,313]
            "coords": torch.stack([lat, lon]),  # [2]
            "target": s["target"],      # multi‑hot [num_labels]
        }
