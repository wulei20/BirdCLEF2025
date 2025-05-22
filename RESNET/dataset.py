import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from help_functions import *
from PIL import Image
import pandas as pd
from pathlib import Path

class BirdclefDataset(Dataset):
    def __init__(self, csv_path=TRAIN_CSV, ogg_root=TRAIN_AUDIO):
        self.df = pd.read_csv(csv_path)
        self.ogg_root = Path(ogg_root)

        # primary_label -> idx
        self.label2idx = {l: i for i, l in enumerate(sorted(self.df.primary_label.unique()))}
        self.num_labels = len(self.label2idx)

        self.samples = []
        for _, row in self.df.iterrows():
            ogg_path = self.image_root / Path(row.filename)
            if not ogg_path.exists():
                continue

            # --------- 组装 multi‑label ---------
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

    # 供外部调用
    def __len_of_label__(self):
        return self.num_labels

    # -------- PyTorch API --------
    def __len__(self):  return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        filepath = s['ogg_path']

        #Loads audio and splits with helper function
        audio, _ = librosa.load(filepath, sr=SR)
        chunks = split_audio(audio, sr=SR)

        if len(chunks) == 0:
            #If the audio file is shorter than CHUNK_LEN, pad it with 0s
            chunk = np.pad(audio, (0, SR*CHUNK_LEN - len(audio)), mode='constant')
        else:
            chunk = random.choice(chunks)
        #Convert chunks to actual spectrograms
        mel = to_mel_spectrogram(chunk)
        mel = normalize_mel(mel)
        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)

        mel = torch.nn.functional.interpolate(mel.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)

        mel = mel.repeat(1, 3, 1, 1)  # [1,3,224,224]
        mel = mel.squeeze(0)         # [3,224,224]

        # 归一化地理坐标到 [-1,1]
        lat = torch.tensor(s["latitude"] / 90.0, dtype=torch.float32)
        lon = torch.tensor(s["longitude"] / 180.0, dtype=torch.float32)

        return {
            "mel": mel,               # float tensor [1,128,313]
            "coords": torch.stack([lat, lon]),  # [2]
            "target": s["target"],      # multi‑hot [num_labels]
        }




##Stores filepaths and labels inside BirdclefDataset object
#class BirdclefDataset(Dataset):
#    def __init__(self, filepaths, labels):
#        self.filepaths = filepaths
#        self.labels = labels
#        
##Returns the length of the amount of samples
#    def __len__(self):
#        return len(self.filepaths)
#
###Audio processing
#    def __getitem__(self, idx):
#        filepath = self.filepaths[idx]
#        label = self.labels[idx]
#
##Loads audio and splits with helper function
#        audio, _ = librosa.load(filepath, sr=SR)
#        chunks = split_audio(audio, sr=SR)
#
#        if len(chunks) == 0:
#            #If the audio file is shorter than CHUNK_LEN, pad it with 0s
#            chunk = np.pad(audio, (0, SR*CHUNK_LEN - len(audio)), mode='constant')
#        else:
#            chunk = random.choice(chunks)
#        #Convert chunks to actual spectrograms
#        mel = to_mel_spectrogram(chunk)
#        mel = normalize_mel(mel)
#        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
#
#        return mel, label
    


