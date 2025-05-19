import os
from help_functions import *
from dataset import BirdclefDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import CNNModel, BirdNet
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, random_split

#def prepare_dataset():
#    ogg_paths, labels, unique_labels = statistic_labels() #Get the paths and labels of the audio files
#
#    label2idx = get_label2idx(unique_labels) #Get the label to index mapping
#    # idx2label = {idx: label for label, idx in label2idx.items()}
#    labels_idx = [label2idx[label] for label in labels]
#
#    #Print label order (later used for the actual submission script)
#    print("✅ unique_labels (label → index mapping):")
#    for i, label in enumerate(unique_labels):
#        print(f"{i}: {label}")
#
#    full_dataset = BirdclefDataset(ogg_paths, labels_idx)
#
#    #80% training and 20% validation set splitting
#    train_paths, val_paths, train_labels, val_labels = train_test_split(
#        ogg_paths,
#        labels_idx,
#        test_size=0.2,
#        stratify=labels_idx,
#        random_state=42
#    )
#
#    #Create the datasets from created split
#    train_dataset = BirdclefDataset(train_paths, train_labels)
#    val_dataset   = BirdclefDataset(val_paths, val_labels)
#
#    ###This is the code for non-stratified splitting
#    #train_size = int(0.8 * len(full_dataset))
#    #val_size   = len(full_dataset) - train_size
#    #train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
#
#    #These data loaders below will be used later during training to feed batches into the model and to evaluate the performance during training
#    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
#    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
#    print(f"✅ {len(train_loader.dataset)} training samples")
#    print(f"✅ {len(val_loader.dataset)} validation samples")
#    return train_loader, val_loader, unique_labels, train_labels, val_labels

def prepare_dataset():
    ds = BirdclefDataset()
    train_ds, val_ds = random_split(ds, [0.9, 0.1])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    unique_len = ds.__len_of_label__()
    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset)
    return train_loader, val_loader, unique_len, train_len, val_len

def train_model(train_loader, val_loader, unique_len, train_len, val_len):
    #Training the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = BirdNet(num_labels=unique_len).to(device) #Initiliazes CNN model to output num_classes
    optimizer = optim.Adam(model.parameters(), lr=1e-3) #Adam optimizer is used for updating the model weights
    criterion = nn.CrossEntropyLoss() #Loss function used for classification (cross entropy)

    for epoch in range(10):  #Epoch number = 10, with 10 epochs this takes like 4 hours
        model.train() #Sets model in training mode
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            mel   = batch["mel"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            preds = model(mel)
            loss  = criterion(preds, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * mel.size(0)

        model.eval() #Sets the model in evaluation mode
        val_loss = 0.0

        with torch.no_grad(): #this makes sure the model validation is done without updating the weights
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                mel   = batch["mel"].to(device)
                target = batch["target"].to(device)
                preds = model(mel)
                loss  = criterion(preds, target)
                val_loss += loss.item() * mel.size(0)

        print(f"Epoch {epoch+1}: Train loss {train_loss/train_len:.4f} | Val loss {val_loss/val_len:.4f}") #Reports average training and validation loss for each epoch

    #Save the model
    torch.save(model.state_dict(), SAVE_PATH) #Saves the model to the path defined in utils.py
    print("✅ Model saved to", SAVE_PATH) #To make sure the model is saved

def valid_train_result(train_labels, val_labels, val_paths):
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

def show_single(label_id, file_name):
    ##Visualizing one spectrogram for poster
    #Construct the path to a known file
    # label_id = "1194042"
    # file_name = "CSA18783.ogg"
    sample_path = os.path.join(AUDIO_DIR, label_id, file_name)

    #Load and process
    y, _ = librosa.load(sample_path, sr=SR)
    chunks = split_audio(y)

    #Use the first chunk for visualization
    chunk = chunks[0]
    mel = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    #Plot without normalization (to preserve dB scale)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=SR, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    #plt.title(f"Mel spectrogram - label {label_id} ({file_name})")
    plt.title("")
    plt.tight_layout()
    plt.show()

train_loader, val_loader, unique_labels, train_labels, val_labels = prepare_dataset() #Prepares the dataset and splits it into training and validation sets
train_model(train_loader, val_loader, len(unique_labels), len(train_labels), len(val_labels)) #Trains the model
valid_train_result(train_labels, val_labels, val_loader.dataset.filepaths) #Validates the model with the 20% validation set