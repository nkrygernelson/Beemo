import os
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from set_based_model import SetBasedBandgapModel
from preprocess_set_data import MultiFidelityPreprocessing

# Define fidelity mapping (same as in the original code)
FIDELITY_MAP = {
    'GGA': 0,
    'GGAU': 1,
    'SCAN': 2,
    'GLLBSC': 3,
    'HSE': 4,
    'EXPT': 5
}

def load_normalization_params():
    """
    Load or compute normalization parameters for the model.
    
    Returns:
        mean: Mean value for normalization
        std: Standard deviation for normalization
    """
    try:
        # Try to load from a saved file
        norm_params = pd.read_csv("predictions/multifidelity/normalization_params.csv")
        mean = norm_params['mean'].values[0]
        std = norm_params['std'].values[0]
        print(f"Loaded normalization parameters: mean={mean:.4f}, std={std:.4f}")
    except:
        # If not found, try to compute from the training data
        try:
            train_df = pd.read_csv("data/multifidelity_train.csv")
            mean = train_df['BG'].mean()
            std = train_df['BG'].std()
            print(f"Computed normalization parameters from training data: mean={mean:.4f}, std={std:.4f}")
            
            # Save for future use
            norm_df = pd.DataFrame({'mean': [mean], 'std': [std]})
            os.makedirs("predictions/multifidelity", exist_ok=True)
            norm_df.to_csv("predictions/multifidelity/normalization_params.csv", index=False)
        except:
            # Use default values as last resort
            print("Using default normalization parameters")
            mean = 1.5  # Typical value for bandgaps in eV
            std = 1.0   # Approximate value
    
    return mean, std

def load_javier_home_dataset():
    """
    Load the javier_home dataset from the csv folder in data.
    
    Returns:
        df_hse: DataFrame with HSE data
        df_gga: DataFrame with GGA data
    """
    # Load the dataset
    try:
        df = pd.read_csv('data/csv/javier_home.csv')
    except:
        raise FileNotFoundError("Could not find the javier_home.csv file in data/csv/")
    
    # Print the columns to understand the data structure
    print(f"Columns in javier_home.csv: {df.columns.tolist()}")
    
    # Create separate DataFrames for HSE and GGA
    df_hse = pd.DataFrame()
    df_gga = pd.DataFrame()
    
    # Check for formula column
    formula_col = None
    for col in df.columns:
        if col.lower() in ['formula', 'compound', 'material']:
            formula_col = col
            break
    
    if formula_col is None:
        raise ValueError("Could not find a formula column in the dataset")
    
    # Check for HSE and GGA columns
    hse_col = None
    gga_col = None
    for col in df.columns:
        if 'hse' in col.lower():
            hse_col = col
        elif 'gga' in col.lower() and 'gga+u' not in col.lower():
            gga_col = col
    
    if hse_col is None:
        raise ValueError("Could not find an HSE column in the dataset")
    if gga_col is None:
        raise ValueError("Could not find a GGA column in the dataset")
    
    # Fill the DataFrames
    df_hse['formula'] = df[formula_col]
    df_hse['BG'] = df[hse_col]
    df_hse['fidelity'] = FIDELITY_MAP['HSE']
    
    df_gga['formula'] = df[formula_col]
    df_gga['BG'] = df[gga_col]
    df_gga['fidelity'] = FIDELITY_MAP['GGA']
    
    # Clean data (remove NaN values)
    df_hse = df_hse.dropna()
    df_gga = df_gga.dropna()
    
    print(f"Loaded {len(df_hse)} HSE samples and {len(df_gga)} GGA samples")
    
    return df_hse, df_gga

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

def load_trained_model(model_path, device):
    """
    Load the trained model.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    # Initialize model with the same parameters used during training
    model = SetBasedBandgapModel(
        num_elements=118,
        embedding_dim=104,
        fidelity_dim=16,
        num_blocks=5,
        num_heads=10,
        hidden_dim=250,
        dropout=0.1,
        pooling_type='gated'  # Using gated pooling as specified in multi_main.py
    )
    
    # Load the saved state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

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
    os.makedirs("predictions/multifidelity/javier_home", exist_ok=True)
    
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
    plt.savefig(f"predictions/multifidelity/javier_home/{fidelity_name}_actual_vs_predicted.png")
    plt.close()
    
    # Histogram of residuals
    residuals = np.array(targets) - np.array(predictions)
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Count')
    plt.title(f'Residual Distribution for {fidelity_name}')
    plt.grid(True)
    plt.savefig(f"predictions/multifidelity/javier_home/{fidelity_name}_residuals.png")
    plt.close()
    
    # Save predictions to CSV
    df_preds = pd.DataFrame({
        "Actual_BG": targets,
        "Predicted_BG": predictions,
        "Residual": residuals
    })
    df_preds.to_csv(f"predictions/multifidelity/javier_home/{fidelity_name}_predictions.csv", index=False)
    
    # Return the plot file path for reference
    return f"predictions/multifidelity/javier_home/{fidelity_name}_actual_vs_predicted.png"

def evaluate_on_javier_home():
    """
    Evaluate the model on the javier_home dataset.
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the javier_home dataset
    df_hse, df_gga = load_javier_home_dataset()
    
    # Initialize preprocessing
    preprocess = MultiFidelityPreprocessing()
    preprocess.batch_size = 32
    
    # Load normalization parameters
    mean, std = load_normalization_params()
    
    # Find the best model
    model_path = "predictions/multifidelity/best_model_gated.pt"
    if not os.path.exists(model_path):
        # Try different possible paths
        possible_paths = [
            "best_model_gated.pt",
            "models/best_model_gated.pt",
            "predictions/best_model_gated.pt"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Could not find the trained model at {model_path}")
    
    print(f"Loading model from {model_path}")
    
    # Load the trained model
    model = load_trained_model(model_path, device)
    
    # Create DataLoaders for HSE and GGA
    hse_loader = create_test_dataloader(df_hse, preprocess, mean, std)
    gga_loader = create_test_dataloader(df_gga, preprocess, mean, std)
    
    # Evaluate on HSE
    print("\nEvaluating on HSE data...")
    hse_metrics, hse_predictions, hse_targets = evaluate_on_fidelity(
        model, hse_loader, "HSE", device, mean, std
    )
    print(f"  MAE: {hse_metrics['mae']:.4f}")
    print(f"  RMSE: {hse_metrics['rmse']:.4f}")
    print(f"  R²: {hse_metrics['r2']:.4f}")
    
    # Evaluate on GGA
    print("\nEvaluating on GGA data...")
    gga_metrics, gga_predictions, gga_targets = evaluate_on_fidelity(
        model, gga_loader, "GGA", device, mean, std
    )
    print(f"  MAE: {gga_metrics['mae']:.4f}")
    print(f"  RMSE: {gga_metrics['rmse']:.4f}")
    print(f"  R²: {gga_metrics['r2']:.4f}")
    
    # Create plots for HSE and GGA
    plot_fidelity_results(hse_predictions, hse_targets, "javier_home_HSE", hse_metrics)
    plot_fidelity_results(gga_predictions, gga_targets, "javier_home_GGA", gga_metrics)
    
    # Create a summary DataFrame and save to CSV
    summary_df = pd.DataFrame({
        'Fidelity': ['HSE', 'GGA'],
        'MAE': [hse_metrics['mae'], gga_metrics['mae']],
        'RMSE': [hse_metrics['rmse'], gga_metrics['rmse']],
        'R2': [hse_metrics['r2'], gga_metrics['r2']]
    })
    
    print("\nSummary of Results:")
    print(summary_df)
    
    # Save summary to CSV
    summary_df.to_csv("predictions/multifidelity/javier_home_summary.csv", index=False)
    
    # Return the results
    return {
        'HSE': hse_metrics,
        'GGA': gga_metrics
    }

if __name__ == "__main__":
    # Run the evaluation on javier_home dataset
    evaluate_on_javier_home()