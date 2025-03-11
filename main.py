import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from set_based_model import SetBasedBandgapModel
from preprocess_set_data import SetBasedPreprocessing

# Create directory for saving predictions if it doesn't exist
os.makedirs("predictions", exist_ok=True)

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def train_set_based_model(pooling_type='attention'):
    # Initialize preprocessing
    preprocess = SetBasedPreprocessing()
    preprocess.sample_size = 10000  # Adjust as needed
    preprocess.batch_size = 32
    
    # Get data loaders
    train_dataloader, val_dataloader, test_dataloader, mean, std = preprocess.preprocess_data()
    print(f"Data loaded and preprocessed, using pooling type: {pooling_type}")
    
    # Check a sample batch
    for batch_idx, (element_ids, element_weights, bandgaps) in enumerate(train_dataloader):
        if batch_idx == 0:
            print(f"Sample batch element IDs shape: {element_ids.shape}")
            print(f"Sample batch element weights shape: {element_weights.shape}")
            print(f"Sample batch bandgaps shape: {bandgaps.shape}")
            break
    
    # Initialize model, loss function, and optimizer
    model = SetBasedBandgapModel(
        num_elements=118,
        embedding_dim=128,
        num_blocks=3,
        num_heads=4,
        hidden_dim=128,
        dropout=0.1,
        pooling_type=pooling_type
    )
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training settings
    num_epochs = 120
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15  # Early stopping patience
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_train_loss = 0.0
        
        for element_ids, element_weights, bandgaps in train_dataloader:
            element_ids = element_ids.to(device)
            element_weights = element_weights.to(device)
            bandgaps = bandgaps.to(device)
            
            optimizer.zero_grad()
            predictions = model(element_ids, element_weights)
            loss = criterion(predictions, bandgaps)
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * len(bandgaps)
        
        epoch_train_loss = running_train_loss / len(train_dataloader.dataset)
        
        # Validate
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for element_ids, element_weights, bandgaps in val_dataloader:
                element_ids = element_ids.to(device)
                element_weights = element_weights.to(device)
                bandgaps = bandgaps.to(device)
                
                val_preds = model(element_ids, element_weights)
                val_loss = criterion(val_preds, bandgaps)
                
                running_val_loss += val_loss.item() * len(bandgaps)
        
        epoch_val_loss = running_val_loss / len(val_dataloader.dataset)
        
        # Update learning rate scheduler
        scheduler.step(epoch_val_loss)
        
        # Store losses
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        # Check for best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f"predictions/best_set_model_{pooling_type}.pt")
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"predictions/set_model_{pooling_type}_train_val_loss.png")
    
    # Load best model for testing
    model.load_state_dict(torch.load(f"predictions/best_set_model_{pooling_type}.pt"))
    
    # Test evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for element_ids, element_weights, bandgaps in test_dataloader:
            element_ids = element_ids.to(device)
            element_weights = element_weights.to(device)
            bandgaps = bandgaps.to(device)
            
            preds = model(element_ids, element_weights)
            
            # Convert back to original scale
            preds_orig = preds.cpu().numpy() * std + mean
            targets_orig = bandgaps.cpu().numpy() * std + mean
            
            test_predictions.extend(preds_orig.tolist())
            test_targets.extend(targets_orig.tolist())
    
    # Calculate metrics
    mae = np.mean(np.abs(np.array(test_predictions) - np.array(test_targets)))
    rmse = np.sqrt(np.mean(np.square(np.array(test_predictions) - np.array(test_targets))))
    
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    
    # Save predictions
    df_preds = pd.DataFrame({
        "Actual_BG": test_targets,
        "Predicted_BG": test_predictions
    })
    df_preds.to_csv(f"predictions/set_model_{pooling_type}_test_predictions.csv", index=False)
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(test_targets, test_predictions, alpha=0.5)
    
    # Add diagonal line for reference
    min_val = min(min(test_targets), min(test_predictions))
    max_val = max(max(test_targets), max(test_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    plt.xlabel('Actual Bandgap (eV)')
    plt.ylabel('Predicted Bandgap (eV)')
    plt.title('Set-Based Model: Actual vs Predicted Bandgap')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"predictions/set_model_{pooling_type}_actual_vs_predicted.png")
    
    return model, mae, rmse

def run_experiments():
    """Run experiments with different pooling mechanisms."""
    results = {}
    
    # List of pooling types to experiment with
    pooling_types = [
        'weighted',       # Original weighted pooling
        'attention',      # Basic attention pooling
        'cross_attention', # Multiple query vectors
        'hierarchical',   # Hierarchical attention pooling
        'gated'           # Gated attention pooling
    ]
    
    for pooling_type in pooling_types:
        print(f"\n{'='*50}")
        print(f"EXPERIMENT: {pooling_type} pooling")
        print(f"{'='*50}\n")
        
        model, mae, rmse = train_set_based_model(pooling_type)
        results[pooling_type] = {'mae': mae, 'rmse': rmse}
    
    # Print summary of results
    print("\n\nEXPERIMENT RESULTS SUMMARY")
    print("="*50)
    print(f"{'Pooling Type':<20} {'MAE':<10} {'RMSE':<10}")
    print("-"*50)
    for pooling_type, metrics in results.items():
        print(f"{pooling_type:<20} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f}")

if __name__ == "__main__":
    # To run all experiments (takes a long time):
    # run_experiments()
    
    # For a single experiment with the best pooling mechanism:
    model, mae, rmse = train_set_based_model('gated')
    print(f"Final Test MAE: {mae:.4f}")
    print(f"Final Test RMSE: {rmse:.4f}")
    
    # Uncomment to run all experiments:
    # run_experiments()s