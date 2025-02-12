import torch.optim as optim
from model import BandgapPredictionModel 
from pymatgen.core.composition import Composition as pmg_Composition
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import os
from preprocess_data import PreprocessData



# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



def main():
    preprocess = PreprocessData()
    preprocess.sample_size = 100000
    preprocess.batch_size = 32
    train_dataloader, val_dataloader, test_dataloader = PreprocessData().preprocess_data()
    for batch_idx, (data, target) in enumerate(train_dataloader):
        if batch_idx == 2:
            print(data, target)
            break
    print("Data Loaded")
 
    
    # --------------------------
    # 6) Initialize your model, loss, optimizer
    # --------------------------

    # 1) Define or import your model
    model = BandgapPredictionModel(num_elements=118, embedding_dim=128, 
                                num_heads=4, num_layers=3, num_queries=5)
    model.to(device)
    #model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001,)

    num_epochs = 120

    # 2) Prepare logs
    train_losses = []
    val_losses = []

    # 3) Training Loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_train_loss = 0.0
        for batch_element_ids, batch_bandgaps in train_dataloader:
            batch_element_ids = batch_element_ids.to(device)  # Move inputs to MPS
            batch_bandgaps = batch_bandgaps.to(device) 
            optimizer.zero_grad()
            predictions = model(batch_element_ids).squeeze(-1)
            loss = criterion(predictions, batch_bandgaps)
            #grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(batch_bandgaps)

        epoch_train_loss = running_train_loss / len(train_dataloader.dataset)

        # Validate
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_element_ids, val_bandgaps in val_dataloader:
                val_element_ids = val_element_ids.to(device)
                val_bandgaps = val_bandgaps.to(device)
                val_preds = model(val_element_ids).squeeze(-1)
                val_loss = criterion(val_preds, val_bandgaps)
                running_val_loss += val_loss.item() * len(val_bandgaps)

        epoch_val_loss = running_val_loss / len(val_dataloader.dataset)

        # Store
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

    # 4) Plot Train vs Val Loss
    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("predictions/train_val_loss.png")

    # 5) Final Test Evaluation
    test_predictions = []
    test_targets = []

    model.eval()
    with torch.no_grad():
        for test_element_ids, test_bandgaps in test_dataloader:
            test_element_ids = test_element_ids.to(device)
            test_bandgaps = test_bandgaps.to(device)
            preds = model(test_element_ids).squeeze(-1)
            test_predictions.extend(preds.tolist())
            test_targets.extend(test_bandgaps.tolist())

    # 6) Save Predictions
    df_preds = pd.DataFrame({
        "Actual_BG": test_targets,
        "Predicted_BG": test_predictions
    })
    df_preds.to_csv("predictions/test_predictions.csv", index=False)

    # 7) Plot Actual vs Predicted
    plt.figure(figsize=(6,6))
    plt.scatter(test_targets, test_predictions, alpha=0.5)
    # Diagonal line for reference
    min_val = min(min(test_targets), min(test_predictions))
    max_val = max(max(test_targets), max(test_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('Actual BG')
    plt.ylabel('Predicted BG')
    plt.title('Actual vs Predicted Bandgap')
    plt.legend()
    plt.savefig("predictions/actual_vs_predicted.png")

if __name__ == "__main__":
    main()