import torch
import os
from tqdm import tqdm
from model import CNNModel, BirdNet
import matplotlib.pyplot as plt
import numpy as np
import librosa
from help_functions import *

def test_on_unmarked():
    _, _, unique_labels = statistic_labels() #Get the paths and labels of the audio files
    label2idx = get_label2idx(unique_labels) #Get the label to index mapping
    label_list = [label2idx[str(i)] for i in range(len(label2idx))]

    #Initialize and load model, this code will favor using the newly trained model but if not detected it will use the pretrained option
    model = BirdNet(num_classes=len(unique_labels)).to(device)

    if os.path.exists(SAVE_PATH):
        print(f"✅ Using newly trained model at {SAVE_PATH}")
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    elif os.path.exists(MODEL_PATH):
        print(f"⚠️ Newly trained model not found. Using pretrained model at {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    #else:
    #    raise FileNotFoundError("❌ No model file found at either SAVE_PATH or MODEL_PATH.")

    model.eval()

    test_paths = get_file_paths(TEST_PATH)  #Get the paths of the test audio files

    prob_sheet = []
    for path in tqdm(test_paths, desc="Predicting validation set", total=len(test_paths)):
        #Load audio
        audio, _ = librosa.load(path, sr=SR)
        audio_name = os.path.basename(path).replace('.ogg', '')
        chunks = split_audio(audio)

        for i, chunk in enumerate(chunks):
            if len(chunk) < CHUNK_LEN * SR:
                chunk = np.pad(chunk, (0, CHUNK_LEN * SR - len(chunk)))

            mel = to_mel_spectrogram(chunk)
            mel = normalize_mel(mel)  #Normalization
            mel = torch.tensor(mel).unsqueeze(0).float().to(device)
            
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
            mel = torch.nn.functional.interpolate(mel.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            mel = mel.repeat(1, 3, 1, 1)  # [1,3,224,224]


            with torch.no_grad():
                logits = model(mel)
                probs = logits.cpu().numpy().flatten()
                chunk_time = (i + 1) * CHUNK_LEN
                chunk_name = f"{audio_name}_{chunk_time}"
                prob_sheet.append([chunk_name] + probs.tolist())

    sample_df = pd.read_csv(SAMPLE_CSV)
    compare_and_rearrange_final_csv(prob_sheet, label_list, sample_df)


if __name__ == "__main__":
    test_on_unmarked()
    # test_on_marked()