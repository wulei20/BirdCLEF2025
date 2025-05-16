import os
from utils import AUDIO_DIR, BATCH_SIZE, SAVE_PATH, statistic_labels
from dataset import BirdclefDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import CNNModel
from utils import device
import torch.nn as nn
import torch.optim as optim
import torch

def prepare_dataset():
    ogg_paths, labels, unique_labels = statistic_labels() #Get the paths and labels of the audio files

    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    # idx2label = {idx: label for label, idx in label2idx.items()}
    labels_idx = [label2idx[label] for label in labels]

    #Print label order (later used for the actual submission script)
    print("✅ unique_labels (label → index mapping):")
    for i, label in enumerate(unique_labels):
        print(f"{i}: {label}")

    full_dataset = BirdclefDataset(ogg_paths, labels_idx)

    #80% training and 20% validation set splitting
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        ogg_paths,
        labels_idx,
        test_size=0.2,
        stratify=labels_idx,
        random_state=42
    )

    #Create the datasets from created split
    train_dataset = BirdclefDataset(train_paths, train_labels)
    val_dataset   = BirdclefDataset(val_paths, val_labels)

    ###This is the code for non-stratified splitting
    #train_size = int(0.8 * len(full_dataset))
    #val_size   = len(full_dataset) - train_size
    #train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    #These data loaders below will be used later during training to feed batches into the model and to evaluate the performance during training
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"✅ {len(train_loader.dataset)} training samples")
    print(f"✅ {len(val_loader.dataset)} validation samples")
    return train_loader, val_loader, len(unique_labels), len(train_dataset), len(val_dataset)

def train_model(train_loader, val_loader, label_len, train_len, val_len):
    #Training the model
    model = CNNModel(num_classes=label_len).to(device) #Initiliazes CNN model to output num_classes
    optimizer = optim.Adam(model.parameters(), lr=1e-3) #Adam optimizer is used for updating the model weights
    criterion = nn.CrossEntropyLoss() #Loss function used for classification (cross entropy)

    for epoch in range(10):  #Epoch number = 10, with 10 epochs this takes like 4 hours
        model.train() #Sets model in training mode
        train_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)

        model.eval() #Sets the model in evaluation mode
        val_loss = 0.0

        with torch.no_grad(): #this makes sure the model validation is done without updating the weights
            for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)

        print(f"Epoch {epoch+1}: Train loss {train_loss/train_len:.4f} | Val loss {val_loss/val_len:.4f}") #Reports average training and validation loss for each epoch

    #Save the model
    torch.save(model.state_dict(), SAVE_PATH) #Saves the model to the path defined in utils.py
    print("✅ Model saved to", SAVE_PATH) #To make sure the model is saved
