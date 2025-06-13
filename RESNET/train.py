import os
from help_functions import *
from dataset import BirdclefDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import BirdNet
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import librosa
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

def prepare_dataset():
    ds = BirdclefDataset()
    tds, vds = random_split(ds, [0.9, 0.1])
    train_loader = DataLoader(tds, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=True)
    val_loader   = DataLoader(vds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    unique_len = ds.__len_of_label__()
    train_len = len(train_loader.dataset)
    val_len = len(val_loader.dataset)
    return train_loader, val_loader, unique_len, train_len, val_len

def train_model(train_loader, val_loader, unique_len, train_len, val_len, train_from_checkpoint=False, checkpoint_path=None):
    # Set device for training
    LEARNING_RATE = 1e-4
    print(LEARNING_RATE)

    model = BirdNet(num_labels=unique_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    # Define the path to save the model
    if train_from_checkpoint and checkpoint_path:
        if os.path.exists(checkpoint_path):
            print(f"⚠️ Loading model from checkpoint at {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"❌ Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")

    for epoch in range(20):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            mel_graph   = batch["mel"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            preds = model(mel_graph)
            loss  = criterion(preds, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * mel_graph.size(0)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                mel_graph   = batch["mel"].to(device)
                target = batch["target"].to(device)
                preds = model(mel_graph)
                loss  = criterion(preds, target)
                val_loss += loss.item() * mel_graph.size(0)

        print(f"Epoch {epoch+1}: Train loss {train_loss/train_len:.10f} | Val loss {val_loss/val_len:.10f}")
        if epoch % 5 == 4:
            torch.save(model.state_dict(), SAVE_PATH + '_checkpoint_ep' + str(epoch+1) + '_lr' + str(LEARNING_RATE) + '.pth')
            print("Save model to path: " + SAVE_PATH + '_checkpoint_ep' + str(epoch+1) + '_lr' + str(LEARNING_RATE) + '.pth')

    # Save the final model
    torch.save(model.state_dict(), SAVE_PATH + '_lr' + str(LEARNING_RATE) + '.pth')
    print("✅ Model saved to", SAVE_PATH + '_lr' + str(LEARNING_RATE) + '.pth')

def show_single(label_id, file_name):
    audio_path = os.path.join(AUDIO_DIR, label_id, file_name)

    # Load the audio file and split it into chunks
    y, _ = librosa.load(audio_path, sr=SR)
    chunks = audio_split(y)

    # Use first chunk for visualization
    chunk = chunks[0]
    mel_graph = librosa.feature.melspectrogram(y=chunk, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel_graph, ref=np.max)

    # Display the mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_db, sr=SR, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format="%+2.0fdB")
    plt.tight_layout()
    plt.title(f"Mel Spectrogram of {file_name} ({label_id})")
    plt.show()

train_loader, val_loader, unique_len, train_len, val_len = prepare_dataset() # Prepares the dataset and splits it into training and validation sets
train_model(train_loader, val_loader, unique_len, train_len, val_len) # Trains the model