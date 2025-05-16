import librosa
import numpy as np
import torch
import os

#Parameters
AUDIO_DIR = '/kaggle/input/birdclef-2025/train_audio/' #locate the audio files
SR        = 32000 #Sampling rate 32K hz
CHUNK_LEN = 5  #5 seconds
N_MELS    = 128 #the amount of mel frequency bands used to convert audios to spectrogram
BATCH_SIZE = 32 #Choose batch size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Define when to use gpu or cpu
SAVE_PATH = '/kaggle/working/birdclef_cnn.pth' #Newly trained model
MODEL_PATH = '/kaggle/input/birdclef_stratifiedcnn/pytorch/stratified/2/birdclef_cnnstratified.pth' #Pretrained option


###Helper functions
#Split audio in 5 seconds function
def split_audio(audio, sr=SR, chunk_length=CHUNK_LEN):
    samples_per_chunk = chunk_length * sr
    chunks = []
    for i in range(0, len(audio), samples_per_chunk):
        chunk = audio[i:i + samples_per_chunk]
        if len(chunk) < samples_per_chunk:
            chunk = np.pad(chunk,
                           (0, samples_per_chunk - len(chunk)),
                           mode='constant')
        chunks.append(chunk)
    return chunks

#Converts audio chunks into mel spectrograms
def to_mel_spectrogram(audio_chunk, sr=SR, n_mels=N_MELS):
    mel = librosa.feature.melspectrogram(y=audio_chunk,
                                         sr=sr,
                                         n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db

#Normalizes all spectrograms created
def normalize_mel(mel_db):
    mel_db -= mel_db.min()
    max_val = mel_db.max()
    if max_val > 0:
        mel_db /= max_val
    else:
        mel_db[:] = 0.0
    return mel_db

def statistic_labels():
    #Create 2 empty lists
    ogg_paths = []
    labels = []

    #This part walks through all audio files in the directory and extracts the full paths and associated labels
    for root, _, files in os.walk(AUDIO_DIR):
        for fname in files:
            if fname.lower().endswith('.ogg'):
                ogg_paths.append(os.path.join(root, fname))
                label = os.path.basename(root)  #foldername = bird species
                labels.append(label)

    #Map the bird labels to integers
    unique_labels = sorted(list(set(labels)))
    return ogg_paths, labels, unique_labels