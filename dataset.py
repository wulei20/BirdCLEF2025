import random
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from utils import split_audio, to_mel_spectrogram, normalize_mel, SR, CHUNK_LEN

#Stores filepaths and labels inside BirdclefDataset object
class BirdclefDataset(Dataset):
    def __init__(self, filepaths, labels):
        self.filepaths = filepaths
        self.labels = labels
        
#Returns the length of the amount of samples
    def __len__(self):
        return len(self.filepaths)

##Audio processing
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

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

        return mel, label