import torch
import os
from tqdm import tqdm
from model import CNNModel
import matplotlib.pyplot as plt
import numpy as np
import librosa
from utils import SAVE_PATH, MODEL_PATH, SR, CHUNK_LEN, to_mel_spectrogram, normalize_mel, split_audio, statistic_labels, device

def test_model():
    _, _, unique_labels = statistic_labels() #Get the paths and labels of the audio files

    #Initialize and load model, this code will favor using the newly trained model but if not detected it will use the pretrained option
    model = CNNModel(num_classes=len(unique_labels)).to(device)

    if os.path.exists(SAVE_PATH):
        print(f"✅ Using newly trained model at {SAVE_PATH}")
        model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    elif os.path.exists(MODEL_PATH):
        print(f"⚠️ Newly trained model not found. Using pretrained model at {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        raise FileNotFoundError("❌ No model file found at either SAVE_PATH or MODEL_PATH.")

    from sklearn.metrics import accuracy_score, classification_report

    model.eval()
    val_preds = []
    val_true = []

    for path, label_idx in tqdm(zip(val_paths, val_labels), desc="Predicting 20% validation set", total=len(val_paths)):
        #Load audio
        audio, _ = librosa.load(path, sr=SR)
        chunks = split_audio(audio)

        chunk_probs = []

        for i, chunk in enumerate(chunks):
            if len(chunk) < CHUNK_LEN * SR:
                chunk = np.pad(chunk, (0, CHUNK_LEN * SR - len(chunk)))

            mel = to_mel_spectrogram(chunk)
            mel = normalize_mel(mel)  #Normalization
            mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).float().to(device)

            with torch.no_grad():
                logits = model(mel_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
                chunk_probs.append(probs)

        #Aggregate across chunks (e.g. mean of softmax probabilities)
        avg_probs = np.mean(chunk_probs, axis=0)
        pred_idx = int(np.argmax(avg_probs))

        val_preds.append(pred_idx)
        val_true.append(label_idx)

    #Evaluation metrics
    acc = accuracy_score(val_true, val_preds)
    print(f"\n✅ Validation accuracy: {acc:.4f}\n")

    labels_present = sorted(set(val_true) | set(val_preds))
    print("Classification Report:")
    print(classification_report(
        val_true,
        val_preds,
        labels=labels_present,
        target_names=[unique_labels[i] for i in labels_present],
        zero_division=0
    ))

    ##Additional output
    from collections import Counter

    train_dist = Counter(train_labels)
    val_dist = Counter(val_labels)

    print(f"Train classes: {len(train_dist)}")
    print(f"Val classes:   {len(val_dist)}")

    #Visualization
    plt.figure(figsize=(10, 4))
    plt.hist(train_dist.values(), bins=50, alpha=0.6, label='Train')
    plt.hist(val_dist.values(), bins=50, alpha=0.6, label='Validation')
    plt.legend()
    plt.title("")
    plt.xlabel("Number of samples per class")
    plt.ylabel("Number of classes")
    plt.show()
