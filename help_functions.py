import librosa
import numpy as np
import torch
import os
import pandas as pd

#Parameters
AUDIO_DIR = '../data/train_audio/' #locate the audio files
SR        = 32000 #Sampling rate 32K hz
CHUNK_LEN = 5  #5 seconds
N_MELS    = 128 #the amount of mel frequency bands used to convert audios to spectrogram
BATCH_SIZE = 32 #Choose batch size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Define when to use gpu or cpu
SAVE_PATH = '../working/birdclef_cnn.pth' #Newly trained model
MODEL_PATH = '../working/birdclef_cnnstratified.pth' #Pretrained option
TEST_PATH = '../data/test_soundscapes/' #Test audio files
LABLE2IDX_FILE_PATH = '../working/label2idx.csv' #Label to index mapping
OUTPUT_CSV = '../working/submission.csv' #Output csv file
SAMPLE_CSV = '../data/sample_submission.csv' #Sample csv file for the test set
TRAIN_CSV = '../data/train.csv' #Train csv file
TRAIN_AUDIO = '../data/train_audio/' #Train audio files

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

def save_label2idx_to_csv(label2idx:dict):
    #Convert the dictionary to a DataFrame
    df = pd.DataFrame(list(label2idx.items()), columns=['label', 'index'])
    #Save the DataFrame to a CSV file
    df.to_csv(LABLE2IDX_FILE_PATH, index=False)
    print(f"✅ Label to index mapping saved to {LABLE2IDX_FILE_PATH}")

def get_label2idx(unique_labels):
    #Check if the CSV file exists
    if os.path.exists(LABLE2IDX_FILE_PATH):
        #Read the CSV file into a DataFrame
        df = pd.read_csv(LABLE2IDX_FILE_PATH)
        #Convert the DataFrame to a dictionary
        label2idx = dict(zip(df['label'], df['index']))
        return label2idx
    else:
        print(f"❌ Label to index mapping file not found at {LABLE2IDX_FILE_PATH}")
        label2idx = {label: idx for idx, label in enumerate(unique_labels)}
        #Save the mapping to a CSV file
        save_label2idx_to_csv(label2idx)
        return label2idx

def get_file_paths(folder):
    paths = []
    #This part walks through all audio files in the directory and extracts the full paths
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith('.ogg'):
                paths.append(os.path.join(root, fname))

    return paths

def compare_and_rearrange_final_csv(generated_list:list, generated_columns:list, sample_csv: pd.DataFrame):
    valid_keys = set(sample_csv.iloc[:, 0])
    unmatched = [row for row in generated_list if row[0] not in valid_keys]
    if unmatched:
        print(f"Unmatched keys: {unmatched}")
        return
    df_generated = pd.DataFrame(generated_list, columns=["row_id"] + generated_columns)
    if set(df_generated.columns) != set(sample_csv.columns):
        print("Column names do not match.")
        return
    df_aligned = df_generated.set_index('row_id').loc[sample_csv['row_id']].reset_index()
    df_aligned = df_aligned[sample_csv.columns]
    df_aligned.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Aligned CSV saved to {OUTPUT_CSV}")
