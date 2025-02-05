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



# Initialize model, loss, and optimizer


def collate_fn(batch):
    # batch is a list of (element_ids, bandgap) pairs
    element_ids_batch, bandgaps_batch = zip(*batch)
    element_ids_batch = [torch.tensor(e, dtype=torch.long) for e in element_ids_batch]
    bandgaps_batch = torch.tensor(bandgaps_batch, dtype=torch.float32)

    # Pad the element sequences
    padded_element_ids = pad_sequence(element_ids_batch, batch_first=True, padding_value=0)

    return padded_element_ids, bandgaps_batch

def get_atomic_number_from_ele(ele):
   return pmg_Composition(ele).elements[0].Z
def formula_to_id_seq(formula):
    ele_dict = pmg_Composition(formula).get_el_amt_dict()
    seq = []
    for ele in ele_dict.keys():
        seq+=[get_atomic_number_from_ele(ele)]*int(ele_dict[ele])
    return seq

def main():
    

    df = pd.read_csv("data.csv")
    # Clean up any weird formulas
    df = df[~df["formula"].isin(["nan","NaN","NAN"])].dropna(subset=["formula"])
    df['formula'] = df['formula'].apply(lambda x: str(x).replace("NaN", "Na1N"))

    bandgaps = df["BG"].values  # or df["BG"].tolist()
    formulas = df["formula"].tolist()

    # Build your dataset (list of (sequence, bandgap) pairs)
    dataset = []
    for formula, bg in zip(formulas, bandgaps):
        seq = formula_to_id_seq(formula)
        dataset.append((seq, bg))
    sample_size = 20000  # or however many you want
    subset = random.sample(dataset, sample_size)
    dataset = subset
    for i, (seq, bg) in enumerate(dataset):
        if len(seq) == 0:
            print(f"Empty sequence at index {i}, formula??, BG={bg}")
        if any(torch.isnan(torch.tensor(seq))):
            print(f"NaN in sequence at index {i}")
    # --------------------------
    # 4) Split into train/val/test
    # --------------------------
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)   # 80% train
    val_size = int(0.1 * dataset_size)     # 10% val
    test_size = dataset_size - train_size - val_size  # 10% test

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # for reproducibility
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # --------------------------
    # 5) Create DataLoaders
    # --------------------------
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn,)
    val_dataloader   = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,)
    test_dataloader  = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, )

    # --------------------------
    # 6) Initialize your model, loss, optimizer
    # --------------------------

    # 1) Define or import your model
    model = BandgapPredictionModel(num_elements=100, embedding_dim=64, 
                                num_heads=4, num_layers=3, num_queries=5)
    #model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 25

    # 2) Prepare logs
    train_losses = []
    val_losses = []

    # 3) Training Loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_train_loss = 0.0
        for batch_element_ids, batch_bandgaps in train_dataloader:
            #batch_element_ids = batch_element_ids.to(device)  # Move inputs to MPS
            #batch_bandgaps = batch_bandgaps.to(device) 
            optimizer.zero_grad()
            predictions = model(batch_element_ids).squeeze(-1)
            loss = criterion(predictions, batch_bandgaps)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * len(batch_bandgaps)

        epoch_train_loss = running_train_loss / len(train_dataloader.dataset)
        torch.mps.empty_cache()

        # Validate
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for val_element_ids, val_bandgaps in val_dataloader:
                #val_element_ids = val_element_ids.to(device)
                #val_bandgaps = val_bandgaps.to(device)
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
    plt.show()

    # 5) Final Test Evaluation
    test_predictions = []
    test_targets = []

    model.eval()
    with torch.no_grad():
        for test_element_ids, test_bandgaps in test_dataloader:
            #test_element_ids = test_element_ids.to(device)
            #test_bandgaps = test_bandgaps.to(device)
            preds = model(test_element_ids).squeeze(-1)
            test_predictions.extend(preds.tolist())
            test_targets.extend(test_bandgaps.tolist())

    # 6) Save Predictions
    df_preds = pd.DataFrame({
        "Actual_BG": test_targets,
        "Predicted_BG": test_predictions
    })
    df_preds.to_csv("test_predictions.csv", index=False)

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
    plt.show()
if __name__ == "__main__":
    main()