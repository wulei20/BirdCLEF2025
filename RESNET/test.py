import torch
import os
from tqdm import tqdm
from model import BirdNet
import matplotlib.pyplot as plt
import numpy as np
import librosa
from help_functions import *
import pandas as pd

def test_on_unmarked():
    df = pd.read_csv(TRAIN_CSV)
    label2idx = {l: i for i, l in enumerate(sorted(df.primary_label.unique()))}
    idx2label = {i: l for i, l in enumerate(sorted(df.primary_label.unique()))}
    num_labels = len(label2idx)
    print(f"Number of labels: {num_labels}")
    label_list = [idx2label[i] for i in range(len(label2idx))]


    # Initialize and load model, this code will favor using the newly trained model but if not detected it will use the pretrained option
    model = BirdNet(num_labels=206).to(device)

    if os.path.exists(SAVE_PATH):
        print(f"⚠️ Loading newly trained model from {SAVE_PATH}")
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    elif os.path.exists(MODEL_PATH):
        print(f"⚠️ Loading pretrained model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # else:
    #    raise FileNotFoundError("❌ No model file found at either SAVE_PATH or MODEL_PATH.")

    model.eval()

    test_paths = get_file_paths(TEST_PATH)  #Get the paths of the test audio files

    prob_sheet = []
    for path in tqdm(test_paths, desc="Predicting validation set", total=len(test_paths)):
        #Load audio
        audio, _ = librosa.load(path, sr=SR)
        audio_name = os.path.basename(path).replace('.ogg', '')
        chunks = audio_split(audio)

        for i, chunk in enumerate(chunks):
            if len(chunk) < CHUNK_LEN * SR:
                chunk = np.pad(chunk, (0, CHUNK_LEN * SR - len(chunk)))

            mel_val = audio_to_mel_spectrogram(chunk)
            mel_val = mel_normalize(mel_val)  #Normalization
            mel_val = torch.tensor(mel_val).unsqueeze(0).float().to(device)
            
            mel_val = (mel_val - mel_val.min()) / (mel_val.max() - mel_val.min() + 1e-6)
            mel_val = torch.nn.functional.interpolate(mel_val.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            mel_val = mel_val.repeat(1, 3, 1, 1)  # [1,3,224,224]


            with torch.no_grad():
                logits = model(mel_val)
                probs = logits.cpu().numpy().flatten()
                chunk_time = (i + 1) * CHUNK_LEN
                chunk_name = f"{audio_name}_{chunk_time}"
                prob_sheet.append([chunk_name] + probs.tolist())

    sample_df = pd.read_csv(SAMPLE_CSV)
    compare_and_rearrange_final_csv(prob_sheet, label_list, sample_df)


if __name__ == "__main__":
    test_on_unmarked()
    # test_on_marked()