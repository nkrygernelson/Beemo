import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from set_based_model import SetBasedBandgapModel
from preprocess_set_data import MultiFidelityPreprocessing

# Map fidelity names to their integer representations
FIDELITY_MAP = {
    'GGA': 0,
    'GGAU': 1,
    'SCAN': 2,
    'GLLBSC': 3,
    'HSE': 4,
    'EXPT': 5
}

def prepare_datasets_for_multifidelity():
    """
    Prepare datasets for multi-fidelity training and individual fidelity testing.
    
    This function:
    1. Loads each fidelity dataset
    2. Creates train/test splits for each fidelity
    3. Combines the training portions for multi-fidelity training
    4. Keeps the test portions separate for individual evaluation
    
    Returns:
        tuple: (combined_train_df, test_datasets)
            - combined_train_df: DataFrame with combined training data from all fidelities
            - test_datasets: Dictionary with fidelity names as keys and test DataFrames as values
    """
    combined_train_df = pd.DataFrame()
    test_datasets = {}
    
    # Process each fidelity dataset
    for fidelity_name, fidelity_id in FIDELITY_MAP.items():
        print(f"Processing {fidelity_name} dataset...")
        
        # Load the dataset
        df = pd.read_csv(f'data/train/{fidelity_name}.csv')
        
        # Shuffle the dataset
        sample_frac = 1
        if fidelity_name == "GGA":
            sample_frac = 0.5
        if fidelity_name == "SCAN":
            sample_frac = 0.5
        
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        
        # Create train/test split (80/20)
        test_size = int(0.2 * len(df))
        test_df = df.iloc[:test_size].copy()
        train_df = df.iloc[test_size:].copy()
        
        # Add fidelity column
        train_df['fidelity'] = fidelity_id
        test_df['fidelity'] = fidelity_id
        
        # Store the test dataset
        test_datasets[fidelity_name] = test_df
        
        # Add training data to combined dataset
        combined_train_df = pd.concat([combined_train_df, train_df], ignore_index=True)
        
        print(f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    
    return combined_train_df, test_datasets

def create_test_dataloader(test_df, preprocess, mean, std):
    """
    Create a DataLoader for a test dataset.
    
    Args:
        test_df: DataFrame with test data
        preprocess: MultiFidelityPreprocessing instance
        mean: Mean value for normalization
        std: Standard deviation for normalization
        
    Returns:
        DataLoader for the test dataset
    """
    # Process the test data
    test_data = []
    for idx, row in test_df.iterrows():
        element_ids, element_weights = preprocess.formula_to_set_representation(row['formula'])
        test_data.append((element_ids, element_weights, int(row['fidelity']), row['BG']))
    
    # Apply normalization
    normalized_test_data = []
    for element_ids, element_weights, fid, bg in test_data:
        normalized_test_data.append((element_ids, element_weights, fid, (bg - mean) / std))
    
    # Create a DataLoader
    test_loader = torch.utils.data.DataLoader(
        normalized_test_data,
        batch_size=preprocess.batch_size,
        shuffle=False,
        collate_fn=preprocess.collate_fn
    )
    
    return test_loader

def train_multifidelity_model(combined_train_df, pooling_type='gated'):
    """
    Train a multi-fidelity model using the combined training dataset.
    
    Args:
        combined_train_df: DataFrame with combined training data
        pooling_type: Type of pooling to use in the model
        
    Returns:
        tuple: (model, mean, std, preprocess)
            - model: Trained model
            - mean: Mean value used for normalization
            - std: Standard deviation used for normalization
            - preprocess: MultiFidelityPreprocessing instance
    """
    # Save the combined training dataset
    os.makedirs("data", exist_ok=True)
    combined_train_df.to_csv("data/multifidelity_train.csv", index=False)
    
    # Initialize preprocessing
    preprocess = MultiFidelityPreprocessing()
    preprocess.sample_size = None  # Use all data
    preprocess.batch_size = 32
    
    # Calculate normalization statistics
    mean = combined_train_df['BG'].mean()
    std = combined_train_df['BG'].std()
    
    # Process the training data
    train_data = []
    for idx, row in combined_train_df.iterrows():
        element_ids, element_weights = preprocess.formula_to_set_representation(row['formula'])
        train_data.append((element_ids, element_weights, int(row['fidelity']), row['BG']))
    
    # Apply normalization
    normalized_train_data = []
    for element_ids, element_weights, fid, bg in train_data:
        normalized_train_data.append((element_ids, element_weights, fid, (bg - mean) / std))
    
    # Split into train/validation
    train_size = int(0.9 * len(normalized_train_data))
    val_size = len(normalized_train_data) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        normalized_train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create DataLoaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=preprocess.batch_size,
        shuffle=True,
        collate_fn=preprocess.collate_fn
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=preprocess.batch_size,
        shuffle=False,
        collate_fn=preprocess.collate_fn
    )
    
    print(f"Data processed: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Initialize model, loss function, and optimizer
    model = SetBasedBandgapModel(
        num_elements=118,
        embedding_dim=104,
        fidelity_dim=16,
        num_blocks=5,
        num_heads=10,
        hidden_dim=250,
        dropout=0.1,
        pooling_type=pooling_type
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
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
    
    # Create directory for saving predictions
    os.makedirs("predictions/multifidelity", exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        running_train_loss = 0.0
        
        for element_ids, element_weights, fidelity_ids, bandgaps in train_dataloader:
            element_ids = element_ids.to(device)
            element_weights = element_weights.to(device)
            fidelity_ids = fidelity_ids.to(device)
            bandgaps = bandgaps.to(device)
            
            optimizer.zero_grad()
            predictions = model(element_ids, fidelity_ids, element_weights)
            loss = criterion(predictions, bandgaps)
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * len(bandgaps)
        
        epoch_train_loss = running_train_loss / len(train_dataset)
        
        # Validate
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for element_ids, element_weights, fidelity_ids, bandgaps in val_dataloader:
                element_ids = element_ids.to(device)
                element_weights = element_weights.to(device)
                fidelity_ids = fidelity_ids.to(device)
                bandgaps = bandgaps.to(device)
                
                val_preds = model(element_ids, fidelity_ids, element_weights)
                val_loss = criterion(val_preds, bandgaps)
                
                running_val_loss += val_loss.item() * len(bandgaps)
        
        epoch_val_loss = running_val_loss / len(val_dataset)
        
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
            torch.save(model.state_dict(), f"predictions/multifidelity/best_model_{pooling_type}.pt")
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
    plt.title('Multi-Fidelity Model: Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"predictions/multifidelity/train_val_loss_{pooling_type}.png")
    
    # Load best model for testing
    model.load_state_dict(torch.load(f"predictions/multifidelity/best_model_{pooling_type}.pt"))
    
    return model, mean, std, preprocess

def evaluate_on_fidelity(model, test_loader, fidelity_name, device, mean, std):
    """
    Evaluate the model on a specific fidelity dataset.
    
    Args:
        model: Trained model
        test_loader: DataLoader for the specific fidelity test set
        fidelity_name: Name of the fidelity
        device: Device to run evaluation on
        mean: Mean value for denormalization
        std: Standard deviation for denormalization
        
    Returns:
        tuple: Metrics (MAE, RMSE, R²) and predictions/targets for plotting
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for element_ids, element_weights, fidelity_ids, bandgaps in test_loader:
            element_ids = element_ids.to(device)
            element_weights = element_weights.to(device)
            fidelity_ids = fidelity_ids.to(device)
            bandgaps = bandgaps.to(device)
            
            preds = model(element_ids, fidelity_ids, element_weights)
            
            # Denormalize
            preds_orig = preds.cpu().numpy() * std + mean
            targets_orig = bandgaps.cpu().numpy() * std + mean
            
            predictions.extend(preds_orig.tolist())
            targets.extend(targets_orig.tolist())
    
    # Calculate metrics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}, predictions, targets

def plot_fidelity_results(predictions, targets, fidelity_name, metrics):
    """
    Generate plots for a specific fidelity evaluation.
    
    Args:
        predictions: List of model predictions
        targets: List of actual values
        fidelity_name: Name of the fidelity
        metrics: Dictionary of performance metrics
    """
    # Create directory for fidelity plots
    os.makedirs("predictions/multifidelity/fidelity_plots", exist_ok=True)
    
    # Scatter plot of actual vs predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    
    # Add diagonal line for reference
    min_val = min(min(targets), min(predictions))
    max_val = max(max(targets), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    
    # Add metrics to plot
    plt.text(
        0.05, 0.95, 
        f"MAE: {metrics['mae']:.4f}\nRMSE: {metrics['rmse']:.4f}\nR²: {metrics['r2']:.4f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor='white', alpha=0.8)
    )
    
    plt.xlabel('Actual Bandgap (eV)')
    plt.ylabel('Predicted Bandgap (eV)')
    plt.title(f'Multi-Fidelity Model on {fidelity_name} Dataset')
    plt.grid(True)
    plt.savefig(f"predictions/multifidelity/fidelity_plots/{fidelity_name}_actual_vs_predicted.png")
    plt.close()
    
    # Save predictions to CSV
    df_preds = pd.DataFrame({
        "Actual_BG": targets,
        "Predicted_BG": predictions
    })
    df_preds.to_csv(f"predictions/multifidelity/fidelity_plots/{fidelity_name}_predictions.csv", index=False)
    
    # Return the plot file path for reference
    return f"predictions/multifidelity/fidelity_plots/{fidelity_name}_actual_vs_predicted.png"

def create_summary_plot(results):
    """
    Create a summary plot showing performance across all fidelities.
    
    Args:
        results: Dictionary with fidelity names as keys and metrics as values
    """
    # Extract metrics for plotting
    fidelities = list(results.keys())
    maes = [results[f]['mae'] for f in fidelities]
    rmses = [results[f]['rmse'] for f in fidelities]
    r2s = [results[f]['r2'] for f in fidelities]
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar chart for MAE and RMSE
    x = np.arange(len(fidelities))
    width = 0.35
    
    ax1.bar(x - width/2, maes, width, label='MAE')
    ax1.bar(x + width/2, rmses, width, label='RMSE')
    ax1.set_ylabel('Error (eV)')
    ax1.set_title('MAE and RMSE by Fidelity Level')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fidelities)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Bar chart for R²
    ax2.bar(x, r2s, width, label='R²', color='green')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score by Fidelity Level')
    ax2.set_xticks(x)
    ax2.set_xticklabels(fidelities)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("predictions/multifidelity/performance_summary.png")
    plt.close()
    
    # Save results to CSV
    df_results = pd.DataFrame({
        'Fidelity': fidelities,
        'MAE': maes,
        'RMSE': rmses,
        'R2': r2s
    })
    df_results.to_csv("predictions/multifidelity/performance_summary.csv", index=False)

def run_multifidelity_experiments():
    """
    Run the complete multi-fidelity training and evaluation workflow.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare datasets
    print("\n" + "="*50)
    print("Preparing Datasets")
    print("="*50 + "\n")
    
    combined_train_df, test_datasets = prepare_datasets_for_multifidelity()
    
    # Train multi-fidelity model
    print("\n" + "="*50)
    print("Training Multi-Fidelity Model")
    print("="*50 + "\n")
    
    model, mean, std, preprocess = train_multifidelity_model(combined_train_df, pooling_type='gated')
    
    # Evaluate on each fidelity
    results = {}
    plot_paths = {}
    
    print("\n" + "="*50)
    print("Evaluating on Individual Fidelity Datasets")
    print("="*50 + "\n")
    
    for fidelity_name, test_df in test_datasets.items():
        print(f"Evaluating on {fidelity_name} dataset...")
        
        # Create test DataLoader
        test_loader = create_test_dataloader(test_df, preprocess, mean, std)
        
        # Evaluate
        metrics, predictions, targets = evaluate_on_fidelity(
            model, test_loader, fidelity_name, device, mean, std
        )
        results[fidelity_name] = metrics
        
        # Generate plots
        plot_path = plot_fidelity_results(predictions, targets, fidelity_name, metrics)
        plot_paths[fidelity_name] = plot_path
        
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print()
    
    # Create a summary plot for all fidelities
    create_summary_plot(results)
    
    # Print summary of results
    print("\n" + "="*50)
    print("MULTI-FIDELITY EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    print(f"{'Fidelity':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-"*50)
    
    for fidelity_name, metrics in results.items():
        print(f"{fidelity_name:<10} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} {metrics['r2']:<10.4f}")
    
    return results, plot_paths

if __name__ == "__main__":
    # Run the multi-fidelity experiments
    results, plot_paths = run_multifidelity_experiments()