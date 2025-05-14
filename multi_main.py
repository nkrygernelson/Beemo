import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter # For TensorBoard
import os
import time # For unique run directory names
import sys # <<< ADDED: For checking if in Colab environment
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure these .py files are accessible (e.g., in the same directory or Python path)
# It's assumed they exist and are correct.
from set_based_model import SetBasedBandgapModel
from preprocess_set_data import MultiFidelityPreprocessing

# --- Google Drive Mounting & Base Path Definition ---
# IMPORTANT: Run these lines in a SEPARATE Colab cell BEFORE running the main script cell.
#
# from google.colab import drive
# import os # Already imported above
#
# GDRIVE_PROJECT_BASE_DIR = None # Initialize
# IS_GOOGLE_DRIVE_AVAILABLE = False # Initialize
#
# try:
#     print("Attempting to mount Google Drive...")
#     drive.mount('/content/drive', force_remount=True) # force_remount can be useful
#     # Define YOUR base path on Google Drive for ALL project outputs
#     # IMPORTANT: Create this folder in your Google Drive if it doesn't exist.
GDRIVE_PROJECT_BASE_DIR_EXAMPLE = '/content/drive/MyDrive/Beemo_MultiFidelity_Runs_V2' # <--- CUSTOMIZE THIS PATH
#     os.makedirs(GDRIVE_PROJECT_BASE_DIR_EXAMPLE, exist_ok=True)
#     GDRIVE_PROJECT_BASE_DIR = GDRIVE_PROJECT_BASE_DIR_EXAMPLE # Assign if successful
#     print(f"Google Drive successfully mounted. Output base directory set to: {GDRIVE_PROJECT_BASE_DIR}")
#     IS_GOOGLE_DRIVE_AVAILABLE = True
# except Exception as e_drive:
#     print(f"Google Drive mounting failed or not in a Colab environment: {e_drive}")
#     print("Outputs will be saved locally to './script_outputs/'")
#     # GDRIVE_PROJECT_BASE_DIR remains None
#     # IS_GOOGLE_DRIVE_AVAILABLE remains False
# --- End Google Drive Setup Cell ---


# Map fidelity names to their integer representations
FIDELITY_MAP = {
    'GGA': 0,
    'GGAU': 1,
    'SCAN': 2,
    'GLLBSC': 3,
    'HSE': 4,
    'EXPT': 5
}

def prepare_datasets_for_multifidelity(data_base_path='data/train'):
    """
    Prepare datasets for multi-fidelity training and individual fidelity testing.
    Handles missing files, empty dataframes, and ensures robust train/test splits.
    """
    combined_train_df = pd.DataFrame()
    test_datasets = {}
    any_data_processed = False

    for fidelity_name, fidelity_id in FIDELITY_MAP.items():
        print(f"Processing {fidelity_name} dataset...")
        file_path = os.path.join(data_base_path, f'{fidelity_name}.csv')

        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Skipping this fidelity.")
            continue
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                print(f"Warning: File {file_path} is empty. Skipping this fidelity.")
                continue
        except pd.errors.EmptyDataError:
            print(f"Warning: File {file_path} is empty or unreadable (EmptyDataError). Skipping.")
            continue
        except Exception as e_read:
            print(f"Warning: Could not read {file_path}: {e_read}. Skipping.")
            continue
        
        any_data_processed = True

        # User's original subsampling for speed during testing
        sample_frac = 1.0 
        if fidelity_name == "GGA": sample_frac = 0.5
        if fidelity_name == "SCAN": sample_frac = 0.5
        # Note: GGAU used 0.001 in user's original code, but 1.0 in the snippet they provided for fixing.
        # Sticking to the snippet's logic for GGA/SCAN, others 1.0.
        
        if 0 < sample_frac < 1.0 and not df.empty:
             df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        elif not df.empty: # Ensure shuffle even if frac is 1.0
             df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        if df.empty:
            print(f"Warning: DataFrame for {fidelity_name} is empty after sampling. Skipping.")
            continue

        # Robust train/test split logic
        if len(df) < 2:
            print(f"Warning: Less than 2 samples for {fidelity_name} ({len(df)}). Using all for training.")
            train_df = df.copy()
            test_df = pd.DataFrame(columns=df.columns) # Empty test_df
        else:
            test_size = int(0.2 * len(df))
            if test_size == 0: test_size = 1 # Ensure at least one test sample if len(df) >= 2
            
            # Ensure train_df is not empty if possible
            if len(df) - test_size <= 0: 
                train_df = df.iloc[test_size:].copy() # This might be empty if test_size == len(df)
                test_df = df.iloc[:test_size].copy()
                if train_df.empty and not test_df.empty: # If train became empty, move one from test
                    if len(test_df) > 1:
                        train_df = test_df.iloc[:1].copy()
                        test_df = test_df.iloc[1:].copy()
                    # If test_df also had only 1, train_df remains empty, test_df has 1.
            else:
                test_df = df.iloc[:test_size].copy()
                train_df = df.iloc[test_size:].copy()
        
        if train_df.empty and test_df.empty:
            print(f"Warning: Both train and test sets are empty for {fidelity_name} after splitting. Skipping.")
            continue
        
        # Use .loc to avoid SettingWithCopyWarning
        if not train_df.empty:
            train_df.loc[:, 'fidelity'] = fidelity_id
        if not test_df.empty:
            test_df.loc[:, 'fidelity'] = fidelity_id
        
        test_datasets[fidelity_name] = test_df
        if not train_df.empty:
             combined_train_df = pd.concat([combined_train_df, train_df], ignore_index=True)
        
        print(f"  {fidelity_name}: Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    
    if combined_train_df.empty:
        if not any_data_processed:
            raise FileNotFoundError(f"No data loaded. All source CSV files might be missing or empty in '{data_base_path}'. Please check your data paths and content.")
        else:
            raise ValueError("No training data was aggregated into combined_train_df. This could be due to all data going into test sets, or all fidelities being skipped after sampling/splitting. Check sampling fractions and individual dataset processing.")
            
    return combined_train_df, test_datasets

def create_test_dataloader(test_df, preprocess, mean, std, num_data_workers=0, current_batch_size=32):
    """ Create a DataLoader for a test dataset. """
    if test_df is None or test_df.empty:
        # print("Warning: test_df is empty or None in create_test_dataloader. Returning None.")
        return None

    test_data = []
    for idx, row in test_df.iterrows():
        formula = row.get('formula', None)
        bg_val = row.get('BG', None)
        fidelity_val = row.get('fidelity', None) # Ensure fidelity is present
        if formula is None or bg_val is None or fidelity_val is None:
            print(f"Warning: Missing 'formula', 'BG', or 'fidelity' in a row: {row.to_dict()}. Skipping row.")
            continue
        
        representation = preprocess.formula_to_set_representation(formula)
        if representation:
            element_ids, element_weights = representation
            test_data.append((element_ids, element_weights, int(fidelity_val), bg_val))
        else:
            print(f"Warning: Could not get representation for formula '{formula}'. Skipping row.")

    if not test_data: 
        # print(f"Warning: No valid test data after formula processing. Returning None.")
        return None
    
    current_std = std if (std != 0 and not np.isnan(std)) else 1.0
    normalized_test_data = [(eid, ew, fid, (bg - mean) / current_std) for eid, ew, fid, bg in test_data]
    
    return torch.utils.data.DataLoader(
        normalized_test_data, batch_size=current_batch_size, shuffle=False,
        collate_fn=preprocess.collate_fn, num_workers=num_data_workers,
        pin_memory=True if torch.cuda.is_available() else False # Enable pin_memory for CUDA
    )

def train_multifidelity_model(
    combined_train_df, 
    pooling_type='gated', 
    writer=None, 
    num_data_workers=0, 
    learning_rate=0.001, 
    num_model_epochs=120,
    model_config_params=None, # For model hyperparameters from config
    run_specific_output_path='.' # Base path for outputs of this specific run
    ):
    """ Train a multi-fidelity model, save checkpoints, and log to TensorBoard. """

    models_save_dir = os.path.join(run_specific_output_path, "models")
    plots_train_val_save_dir = os.path.join(run_specific_output_path, "plots_train_val")
    os.makedirs(models_save_dir, exist_ok=True)
    os.makedirs(plots_train_val_save_dir, exist_ok=True)
    
    preprocess = MultiFidelityPreprocessing()
    preprocess.sample_size = None 
    preprocess.batch_size = model_config_params.get("batch_size", 32) if model_config_params else 32
    
    if 'BG' not in combined_train_df.columns:
        raise KeyError("'BG' column missing from combined_train_df. Cannot calculate mean/std.")
    mean = combined_train_df['BG'].mean()
    std = combined_train_df['BG'].std()
    if std == 0 or np.isnan(std): std = 1.0

    train_data = []
    for idx, row in combined_train_df.iterrows():
        formula = row.get('formula', None); bg_val = row.get('BG', None); fidelity_val = row.get('fidelity', None)
        if formula is None or bg_val is None or fidelity_val is None: continue
        representation = preprocess.formula_to_set_representation(formula)
        if representation:
            element_ids, element_weights = representation
            train_data.append((element_ids, element_weights, int(fidelity_val), bg_val))

    if not train_data: print("Error: No training data after preprocessing."); return None, mean, std, preprocess, float('inf')
    normalized_train_data = [(eid, ew, fid, (bg - mean) / std) for eid, ew, fid, bg in train_data]
    
    if len(normalized_train_data) < 2: print("Error: <2 samples for train/val split."); return None, mean, std, preprocess, float('inf')
        
    train_size = int(0.9 * len(normalized_train_data))
    val_size = len(normalized_train_data) - train_size
    if train_size == 0 : train_size = 1; val_size = len(normalized_train_data) -1
    if val_size == 0 and len(normalized_train_data) > 1 : val_size = 1; train_size = len(normalized_train_data) -1

    train_dataset_obj, val_dataset_obj_maybe = torch.utils.data.random_split(
        normalized_train_data, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    val_dataset_obj = val_dataset_obj_maybe if val_size > 0 else None

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_obj, batch_size=preprocess.batch_size, shuffle=True,
        collate_fn=preprocess.collate_fn, num_workers=num_data_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_dataloader = None
    if val_dataset_obj:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset_obj, batch_size=preprocess.batch_size, shuffle=False,
            collate_fn=preprocess.collate_fn, num_workers=num_data_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
    
    print(f"DataLoaders: Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader) if val_dataloader else 'N/A'}")
    
    default_hparams = {"embedding_dim": 104, "fidelity_dim": 16, "num_blocks": 5, 
                       "num_heads": 10, "hidden_dim": 250, "dropout": 0.1, "weight_decay": 1e-5}
    current_hparams = default_hparams.copy()
    if model_config_params: current_hparams.update(model_config_params)

    if current_hparams["embedding_dim"] % current_hparams["num_heads"] != 0:
        original_heads = current_hparams["num_heads"]
        possible_heads = [h for h in [1, 2, 4, 5, 8, 10, 13, 20, 26, 52, 104] if current_hparams["embedding_dim"] % h == 0] 
        current_hparams["num_heads"] = possible_heads[-1] if possible_heads else 1
        print(f"Warning: embedding_dim {current_hparams['embedding_dim']} not divisible by num_heads {original_heads}. Adjusted num_heads to {current_hparams['num_heads']}.")

    model = SetBasedBandgapModel(
        num_elements=118, embedding_dim=current_hparams["embedding_dim"],
        fidelity_dim=current_hparams["fidelity_dim"], num_blocks=current_hparams["num_blocks"],
        num_heads=current_hparams["num_heads"], hidden_dim=current_hparams["hidden_dim"],
        dropout=current_hparams["dropout"], pooling_type=pooling_type
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    if str(device) != "cpu":
        try:
            print("Attempting to compile model with torch.compile()..."); model = torch.compile(model); print("Model compiled.")
        except Exception as e_compile: print(f"torch.compile() failed: {e_compile}. Proceeding without.")
    model.to(device)
    
    if writer and len(train_dataloader) > 0:
        try:
            sample_element_ids, sample_element_weights, sample_fidelity_ids, _ = next(iter(train_dataloader))
            # Ensure model passed to add_graph is the original if compiled
            model_to_log = model._orig_mod if hasattr(model, '_orig_mod') and model._orig_mod is not None else model
            writer.add_graph(model_to_log, (sample_element_ids.to(device), sample_fidelity_ids.to(device), sample_element_weights.to(device)))
            print("Model graph added to TensorBoard.")
        except Exception as e_graph: print(f"Could not add model graph to TensorBoard: {e_graph}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=current_hparams["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=current_hparams.get("scheduler_patience", 5))
    
    num_epochs = num_model_epochs
    train_losses_list, val_losses_list = [], []
    best_val_loss_for_saving = float('inf')
    patience_config, patience_counter = current_hparams.get("early_stopping_patience", 15), 0
    
    actual_epochs_trained = 0
    for epoch in range(num_epochs):
        actual_epochs_trained = epoch + 1
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        model.train()
        running_train_loss = 0.0
        for batch_idx, (element_ids, element_weights, fidelity_ids, bandgaps) in enumerate(train_dataloader):
            if batch_idx % (max(1, len(train_dataloader) // 5)) == 0: print(f"  Epoch {epoch+1}, Train batch {batch_idx+1}/{len(train_dataloader)}")
            element_ids,element_weights,fidelity_ids,bandgaps = element_ids.to(device),element_weights.to(device),fidelity_ids.to(device),bandgaps.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(element_ids, fidelity_ids, element_weights)
            loss = criterion(predictions, bandgaps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward(); optimizer.step()
            running_train_loss += loss.item() * len(bandgaps)
        
        epoch_train_loss = running_train_loss / len(train_dataset_obj) if len(train_dataset_obj) > 0 else 0.0
        train_losses_list.append(epoch_train_loss)
        if writer:
            writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch + 1)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)

        current_epoch_val_loss = float('inf')
        if val_dataloader:
            model.eval(); running_val_loss = 0.0
            with torch.no_grad():
                for element_ids_v,element_weights_v,fidelity_ids_v,bandgaps_v in val_dataloader:
                    element_ids_v,element_weights_v,fidelity_ids_v,bandgaps_v = element_ids_v.to(device),element_weights_v.to(device),fidelity_ids_v.to(device),bandgaps_v.to(device)
                    val_preds = model(element_ids_v, fidelity_ids_v, element_weights_v)
                    val_loss_item = criterion(val_preds, bandgaps_v)
                    running_val_loss += val_loss_item.item() * len(bandgaps_v)
            current_epoch_val_loss = running_val_loss / len(val_dataset_obj) if val_dataset_obj and len(val_dataset_obj) > 0 else float('inf')
            val_losses_list.append(current_epoch_val_loss)
            scheduler.step(current_epoch_val_loss)
            if writer: writer.add_scalar('Loss/val_epoch', current_epoch_val_loss, epoch + 1)
        else: val_losses_list.append(float('nan'))

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f} | Val Loss: {current_epoch_val_loss if val_dataloader else 'N/A':.4f}")
        
        if val_dataloader:
            if current_epoch_val_loss < best_val_loss_for_saving:
                best_val_loss_for_saving = current_epoch_val_loss; patience_counter = 0
                save_path = os.path.join(models_save_dir, f"best_model_{pooling_type}.pt")
                torch.save(model.state_dict(), save_path); print(f"Best model saved: {save_path} (Val Loss: {best_val_loss_for_saving:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience_config: print(f"Early stopping at epoch {epoch+1}."); break
        elif (epoch + 1) % 10 == 0 or epoch == num_epochs - 1: # Save periodically or at end if no validation
             save_path = os.path.join(models_save_dir, f"model_epoch_{epoch+1}_{pooling_type}.pt")
             torch.save(model.state_dict(), save_path); print(f"Model saved: {save_path} (epoch {epoch+1})")
    
    fig_tv = plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses_list) + 1), train_losses_list, label='Train Loss')
    if val_dataloader and any(not np.isnan(vl) for vl in val_losses_list):
        plt.plot(range(1, len(val_losses_list) + 1), val_losses_list, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.title(f'Training & Validation Loss ({pooling_type})')
    plt.legend(); plt.grid(True)
    plot_save_path = os.path.join(plots_train_val_save_dir, f"train_val_loss_{pooling_type}.png")
    plt.savefig(plot_save_path); print(f"Train/val loss plot saved to: {plot_save_path}")
    if writer: writer.add_figure(f'Plots_TrainVal/Loss_Curve_{pooling_type}', fig_tv, actual_epochs_trained)
    plt.close(fig_tv)

    best_model_chkpt_path = os.path.join(models_save_dir, f"best_model_{pooling_type}.pt")
    # Use actual_epochs_trained for the last saved model path
    last_saved_epoch_model_path = os.path.join(models_save_dir, f"model_epoch_{actual_epochs_trained}_{pooling_type}.pt")
    
    model_loaded_path = None
    if os.path.exists(best_model_chkpt_path) and val_dataloader :
        model.load_state_dict(torch.load(best_model_chkpt_path, map_location=device)); model_loaded_path = best_model_chkpt_path
    elif os.path.exists(last_saved_epoch_model_path): # Check if the last epoch model was saved (e.g. no validation)
         model.load_state_dict(torch.load(last_saved_epoch_model_path, map_location=device)); model_loaded_path = last_saved_epoch_model_path
    if model_loaded_path: print(f"Loaded model for evaluation from: {model_loaded_path}")
    else: print("Warning: Using model from final training state as no specific checkpoint was found/loaded.")
    
    return model, mean, std, preprocess, best_val_loss_for_saving, actual_epochs_trained


def evaluate_on_fidelity(model, test_loader, fidelity_name, device, mean, std):
    """ Evaluate model on a specific fidelity, return metrics and predictions. """
    if test_loader is None:
        print(f"No test data/loader for {fidelity_name}. Skipping its evaluation.")
        return {k: float('nan') for k in ['mae','rmse','r2']}, [], []
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for element_ids, element_weights, fidelity_ids, bandgaps in test_loader:
            element_ids,element_weights,fidelity_ids,bandgaps = element_ids.to(device),element_weights.to(device),fidelity_ids.to(device),bandgaps.to(device)
            preds = model(element_ids, fidelity_ids, element_weights)
            preds_orig = preds.cpu().numpy() * std + mean
            targets_orig = bandgaps.cpu().numpy() * std + mean
            preds_list.extend(preds_orig.flatten().tolist())
            targets_list.extend(targets_orig.flatten().tolist())
    if not targets_list or not preds_list: # Check if lists are empty after loop
        print(f"No predictions or targets generated for {fidelity_name} during evaluation.")
        return {k: float('nan') for k in ['mae','rmse','r2']}, [], []
    
    # Handle cases where all predictions or targets might be the same, leading to issues with r2_score
    if len(np.unique(targets_list)) < 2 or len(np.unique(preds_list)) < 2 and len(targets_list) > 1 :
        r2 = float('nan') # R2 is ill-defined or 0 if variance is zero
        print(f"Warning: R2 score is ill-defined for {fidelity_name} due to constant targets or predictions.")
    else:
        r2 = r2_score(targets_list, preds_list)

    return {'mae': mean_absolute_error(targets_list, preds_list), 
            'rmse': np.sqrt(mean_squared_error(targets_list, preds_list)), 
            'r2': r2}, preds_list, targets_list


def plot_fidelity_results(predictions, targets, fidelity_name, metrics, eval_plots_save_dir):
    """ Generate and save individual fidelity evaluation plots. Returns the figure object. """
    os.makedirs(eval_plots_save_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    if not predictions or not targets:
        plt.title(f'Evaluation: {fidelity_name} - No Data')
        plt.text(0.5,0.5,"No data to plot",ha='center', va='center')
    else:
        plt.scatter(targets, predictions, alpha=0.5, label="Predictions")
        min_val = min(min(targets,default=0), min(predictions,default=0)) 
        max_val = max(max(targets,default=1), max(predictions,default=1))
        if min_val == max_val : max_val +=1 # Avoid flat line if all values are same
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y=x)')
        plt.text(0.05,0.95,f"MAE: {metrics.get('mae', float('nan')):.4f}\nRMSE: {metrics.get('rmse', float('nan')):.4f}\nR²: {metrics.get('r2', float('nan')):.4f}",
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white',alpha=0.8), va='top')
    plt.xlabel('Actual Bandgap (eV)'); plt.ylabel('Predicted Bandgap (eV)')
    plt.title(f'Evaluation Results on {fidelity_name} Dataset')
    plt.legend(loc='lower right'); plt.grid(True)
    plot_file_path = os.path.join(eval_plots_save_dir, f"{fidelity_name}_actual_vs_predicted.png")
    plt.savefig(plot_file_path); print(f"Saved evaluation plot: {plot_file_path}")
    
    df_preds = pd.DataFrame({"Actual_BG": targets, "Predicted_BG": predictions})
    csv_file_path = os.path.join(eval_plots_save_dir, f"{fidelity_name}_predictions.csv")
    df_preds.to_csv(csv_file_path, index=False); print(f"Saved evaluation predictions CSV: {csv_file_path}")
    return fig

def create_summary_plot(results, summary_plots_save_dir):
    """ Generate and save a summary plot of metrics across all fidelities. Returns the figure object. """
    os.makedirs(summary_plots_save_dir, exist_ok=True)
    fidelities = list(results.keys())
    if not fidelities: 
        print("No results available to create a summary plot.")
        fig = plt.figure(figsize=(16,6)); plt.text(0.5,0.5,"No results for summary plot",ha='center'); return fig

    maes = [results[f].get('mae', float('nan')) for f in fidelities]
    rmses = [results[f].get('rmse', float('nan')) for f in fidelities]
    r2s = [results[f].get('r2', float('nan')) for f in fidelities]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(fidelities)); width = 0.35
    
    ax1.bar(x - width/2, maes, width, label='MAE', color='skyblue')
    ax1.bar(x + width/2, rmses, width, label='RMSE', color='salmon')
    ax1.set_ylabel('Error (eV)'); ax1.set_title('MAE and RMSE by Fidelity Level')
    ax1.set_xticks(x); ax1.set_xticklabels(fidelities, rotation=45, ha="right")
    ax1.legend(); ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    ax2.bar(x, r2s, width, label='R²', color='lightgreen')
    ax2.set_ylabel('R² Score'); ax2.set_title('R² Score by Fidelity Level')
    ax2.set_xticks(x); ax2.set_xticklabels(fidelities, rotation=45, ha="right")
    valid_r2s = [r for r in r2s if not np.isnan(r)] # Filter out NaNs for ylim calculation
    ax2.set_ylim(min(0, min(valid_r2s) if valid_r2s else 0) - 0.05, max(1, max(valid_r2s) if valid_r2s else 1) + 0.05)
    ax2.legend(); ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.suptitle("Overall Performance Summary", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    
    plot_file_path = os.path.join(summary_plots_save_dir, "performance_summary.png")
    plt.savefig(plot_file_path); print(f"Saved summary plot: {plot_file_path}")
    
    df_results = pd.DataFrame({'Fidelity': fidelities, 'MAE': maes, 'RMSE': rmses, 'R2': r2s})
    csv_file_path = os.path.join(summary_plots_save_dir, "performance_summary.csv")
    df_results.to_csv(csv_file_path, index=False); print(f"Saved summary metrics CSV: {csv_file_path}")
    return fig

def print_final_summary_results(results): # Helper function
    """ Prints the final summary table of results to console. """
    print("\n" + "="*70 + "\nMULTI-FIDELITY EXPERIMENT FINAL RESULTS SUMMARY\n" + "="*70)
    print(f"{'Fidelity':<10} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10}")
    print("-" * 55) # Adjusted width
    for fidelity_name, metrics in results.items():
        print(f"{fidelity_name:<10} | {metrics.get('mae', float('nan')):<10.4f} | {metrics.get('rmse', float('nan')):<10.4f} | {metrics.get('r2', float('nan')):<10.4f}")
    print("-" * 55)

def run_experiment_session(
    config, # Dictionary with all experiment parameters for this run
    gdrive_project_root_dir=None, # Base Google Drive directory for ALL project runs
    local_project_root_dir='./script_outputs_local' # Default local base directory for ALL runs
    ):
    """ 
    Orchestrates a full experiment run: data prep, training, evaluation, logging, and saving.
    Outputs are saved to a unique, timestamped run-specific directory either locally or on Google Drive.
    """
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    # Create a more descriptive run name using key config parameters
    run_name_parts = [
        config.get('pooling_type','defpool'),
        f"lr{config.get('lr',0.001)}",
        f"ep{config.get('epochs',0)}",
        f"bs{config.get('batch_size',0)}",
        current_time_str
    ]
    run_name = "_".join(run_name_parts)
    
    # Determine the absolute base directory for all project runs (GDrive or local)
    project_runs_base_dir = local_project_root_dir # Default to local
    is_colab_env = 'google.colab' in sys.modules # Check if in Colab
    
    if gdrive_project_root_dir and is_colab_env:
        # Only attempt to use GDrive if a path is given AND we are in Colab
        # The GDrive mounting should have happened BEFORE calling this function
        if os.path.exists(gdrive_project_root_dir) and os.path.isdir(gdrive_project_root_dir):
            project_runs_base_dir = gdrive_project_root_dir
            print(f"Outputs for all runs will be under Google Drive: {project_runs_base_dir}")
        else:
            print(f"Warning: Specified Google Drive root '{gdrive_project_root_dir}' does not exist or is not a directory. Defaulting to local path: {local_project_root_dir}")
            os.makedirs(project_runs_base_dir, exist_ok=True) # Ensure local default exists
    else:
        if gdrive_project_root_dir and not is_colab_env:
            print(f"Warning: Google Drive path specified but not in a Colab environment. Using local path: {local_project_root_dir}")
        else: # No GDrive path specified, or not in Colab
            print(f"Using local output directory for all runs: {local_project_root_dir}")
        os.makedirs(project_runs_base_dir, exist_ok=True) # Ensure local default exists

    # Create the unique directory for THIS specific run's outputs
    run_specific_output_path = os.path.join(project_runs_base_dir, run_name)
    os.makedirs(run_specific_output_path, exist_ok=True)
    print(f"All outputs for THIS run ({run_name}) will be saved under: {run_specific_output_path}")

    tensorboard_log_dir_for_run = os.path.join(run_specific_output_path, 'tensorboard_logs')
    writer = SummaryWriter(log_dir=tensorboard_log_dir_for_run)
    print(f"TensorBoard logs for this run will be in: {tensorboard_log_dir_for_run}")

    # Save the run configuration to a file for traceability
    config_save_path = os.path.join(run_specific_output_path, "run_config.txt")
    with open(config_save_path, 'w') as f_config:
        import json
        json.dump(config, f_config, indent=4)
    print(f"Run configuration saved to: {config_save_path}")


    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device selected for this run: {device}")

    results = {}
    actual_epochs_completed_in_train = 0

    try:
        print("\n" + "="*50 + "\nPreparing Datasets\n" + "="*50 + "\n")
        combined_train_df, test_datasets = prepare_datasets_for_multifidelity(
            data_base_path=config.get('data_input_dir', 'data/train') # Assumes 'data/train' is relative to script
        )
        if combined_train_df.empty: raise ValueError("No training data loaded from prepare_datasets_for_multifidelity.")

        print("\n" + "="*50 + "\nTraining Multi-Fidelity Model\n" + "="*50 + "\n")
        model, mean, std, preprocess, best_val_loss_train, actual_epochs_completed_in_train = train_multifidelity_model(
            combined_train_df, 
            pooling_type=config.get('pooling_type','gated'), 
            writer=writer,
            num_data_workers=config.get('num_workers',0),
            learning_rate=config.get('lr',0.001),
            num_model_epochs=config.get('epochs',120),
            model_config_params=config, # Pass the whole config for model hparams
            run_specific_output_path=run_specific_output_path # For saving models, train/val plots
        )
        if model is None: raise ValueError("Model training returned None (failed).")

        # Prepare hparams for TensorBoard logging
        hparams_for_tb = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))} # Filter for TB scalar types
        if best_val_loss_train is not None and not np.isnan(best_val_loss_train):
            hparams_for_tb['best_validation_loss_from_training'] = best_val_loss_train
        hparams_for_tb['actual_epochs_trained'] = actual_epochs_completed_in_train
        
        print("\n" + "="*50 + "\nEvaluating on Individual Fidelity Datasets\n" + "="*50 + "\n")
        # Define specific subdirectories for evaluation plots and CSVs
        eval_plots_save_dir = os.path.join(run_specific_output_path, "plots_evaluation_fidelity")
        os.makedirs(eval_plots_save_dir, exist_ok=True)

        for fidelity_name, test_df_fid in test_datasets.items():
            print(f"Evaluating on: {fidelity_name} dataset...")
            if test_df_fid.empty:
                print(f"  Test dataset for {fidelity_name} is empty. Skipping.")
                results[fidelity_name] = {k: float('nan') for k in ['mae','rmse','r2']}
                continue
            
            test_loader = create_test_dataloader(
                test_df_fid, preprocess, mean, std, 
                num_data_workers=config.get('num_workers',0),
                current_batch_size=config.get("batch_size", 32)
            )
            
            metrics, predictions, targets = evaluate_on_fidelity(model, test_loader, fidelity_name, device, mean, std)
            results[fidelity_name] = metrics
            
            if predictions or targets: # Ensure there's data to plot
                fig_fid = plot_fidelity_results(predictions, targets, fidelity_name, metrics, eval_plots_save_dir=eval_plots_save_dir)
                if writer and fig_fid: 
                    writer.add_figure(f'Eval_Plots_Per_Fidelity/{fidelity_name}_Actual_vs_Predicted', fig_fid, actual_epochs_completed_in_train)
                plt.close(fig_fid) # Close figure after logging
            print(f"  {fidelity_name}: MAE {metrics.get('mae',0):.4f}, RMSE {metrics.get('rmse',0):.4f}, R² {metrics.get('r2',0):.4f}")
        
        summary_plots_save_dir = os.path.join(run_specific_output_path, "plots_summary_evaluation")
        os.makedirs(summary_plots_save_dir, exist_ok=True)
        if results: # Ensure there are results before creating summary
            fig_sum = create_summary_plot(results, summary_plots_save_dir=summary_plots_save_dir)
            if writer and fig_sum: 
                writer.add_figure('Eval_Plots_Overall/Performance_Summary', fig_sum, actual_epochs_completed_in_train)
            plt.close(fig_sum) # Close figure

            # Log final metrics with hparams to TensorBoard
            final_metrics_to_log_tb = {}
            for f_name, mets in results.items():
                for m_name, m_val in mets.items():
                    if not np.isnan(m_val): final_metrics_to_log_tb[f'METRIC_{f_name}_{m_name}'] = m_val # Prefix to avoid collision
            if writer and final_metrics_to_log_tb:
                try: 
                    writer.add_hparams(hparams_for_tb, final_metrics_to_log_tb)
                    print("Hyperparameters and final metrics logged to TensorBoard.")
                except Exception as e_hparam: print(f"TensorBoard HParam logging error: {e_hparam}")
        
        print_final_summary_results(results)

    except FileNotFoundError as e_fnf: # Specific error for missing data
        print(f"FATAL ERROR: A required data file was not found: {e_fnf}")
        if writer: writer.add_text("RunFatalError", f"FileNotFound: {e_fnf}")
    except ValueError as e_val: # Specific error for data issues
        print(f"FATAL ERROR: A data validation issue occurred: {e_val}")
        if writer: writer.add_text("RunFatalError", f"ValueError: {e_val}")
    except Exception as e_main_run: # Catch-all for other errors
        print(f"FATAL ERROR during experiment run: {e_main_run}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        if writer: writer.add_text("RunFatalError", f"GeneralError: {str(e_main_run)}\n{traceback.format_exc()}")
    finally:
        if 'writer' in locals() and writer: # Ensure writer was initialized
            writer.close()
        print(f"Run '{run_name}' finished. All outputs located in: {run_specific_output_path if 'run_specific_output_path' in locals() else 'N/A'}")

    return results, run_specific_output_path if 'run_specific_output_path' in locals() else None


if __name__ == "__main__":
    # --- Google Drive Global Path (Should be set by the Colab setup cell) ---
    # This variable is expected to be defined if running in Colab with Drive mounted.
    # If not defined (e.g., running locally or Drive mount failed), it will be None.
    gdrive_active_base_path = None
    try:
        # This checks if GDRIVE_PROJECT_BASE_DIR was set by the Colab setup cell
        if 'GDRIVE_PROJECT_BASE_DIR' in locals() or 'GDRIVE_PROJECT_BASE_DIR' in globals():
            gdrive_active_base_path = GDRIVE_PROJECT_BASE_DIR if GDRIVE_PROJECT_BASE_DIR else None
        elif 'IS_GOOGLE_DRIVE_AVAILABLE' in locals() and IS_GOOGLE_DRIVE_AVAILABLE:
             # Fallback if only the flag was set but not the path directly (less ideal)
             # This part is a bit fragile, relying on the setup cell to define GDRIVE_PROJECT_BASE_DIR is better.
             gdrive_active_base_path = '/content/drive/MyDrive/Beemo_MultiFidelity_Runs_V2_Default' # A default if flag is true but path isn't
             print(f"Warning: IS_GOOGLE_DRIVE_AVAILABLE is True, but GDRIVE_PROJECT_BASE_DIR was not found. Using default: {gdrive_active_base_path}")
             os.makedirs(gdrive_active_base_path, exist_ok=True)

    except NameError:
        print("GDRIVE_PROJECT_BASE_DIR not found. Assuming local execution or Drive not mounted.")
        gdrive_active_base_path = None


    # ----- Configuration for the experiment run -----
    # You can define multiple configurations and loop through them if needed
    experiment_config = {
        "pooling_type": "gated",
        "lr": 0.001,
        "epochs": 10,  # Quick test run; set to 120-200 for full training
        "num_workers": 0, # Start with 0 for stability, increase based on your system (e.g., os.cpu_count()//2)
        "data_input_dir": 'data/train', # Relative path to where CSVs are stored (from project root)

        # Model Hyperparameters (can be filled from Optuna best_trial.params later)
        "embedding_dim": 104,
        "fidelity_dim": 16,
        "num_blocks": 5,
        "num_heads": 10, # Ensure embedding_dim is divisible by num_heads
        "hidden_dim": 250, # Corresponds to hidden_dim in SetBasedBandgapModel
        "dropout": 0.1,
        "batch_size": 32,
        "weight_decay": 1e-5,
        "scheduler_patience": 5, # For ReduceLROnPlateau
        "early_stopping_patience": 15
    }
    
    print(f"Starting experiment session with config: {experiment_config}")
    if gdrive_active_base_path:
        print(f"Targeting Google Drive base for outputs: {gdrive_active_base_path}")
    else:
        print(f"Targeting local directory for outputs: ./script_outputs_local")

    # The `run_experiment_session` function now handles GDrive path or local path for outputs
    results_summary, final_output_location = run_experiment_session(
        config=experiment_config,
        gdrive_project_root_dir=gdrive_active_base_path # Pass the GDrive base path (can be None)
        # local_project_root_dir can be customized here too if needed
    )
    
    print(f"\nExperiment session finished. All outputs for this run are located in: {final_output_location}")

    # ----- OPTIONAL: Hyperparameter Optimization with Optuna -----
    # To run HPO, you would uncomment and adapt the Optuna calling code here,
    # similar to previous examples, ensuring `objective_for_optuna` is defined
    # and uses the GDrive/local path logic for its trial outputs.
    # Example:
    # RUN_OPTUNA = False # Set to True to run HPO
    # if RUN_OPTUNA:
    #     import optuna
    #     # ... (Define objective_for_optuna similar to previous versions) ...
    #     # ... (Optuna study setup and study.optimize call) ...
    #     print("Optuna HPO finished. Best params:", study.best_trial.params)

