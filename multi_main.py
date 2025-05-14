import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter # For TensorBoard
import os
import time # For unique run directory names
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure these .py files are accessible (e.g., in the same directory or Python path)
# It's assumed they exist and are correct.
from set_based_model import SetBasedBandgapModel
from preprocess_set_data import MultiFidelityPreprocessing

# --- Google Drive Mounting & Base Path Definition ---
# IMPORTANT: Run these lines in a SEPARATE Colab cell BEFORE running the main script cell.
# from google.colab import drive
# import os
# try:
#     drive.mount('/content/drive', force_remount=True) # force_remount can be useful
#     # Define YOUR base path on Google Drive for ALL project outputs
GDRIVE_PROJECT_BASE_DIR = '/content/drive/MyDrive/Beemo_MultiFidelity_Runs' # <--- CUSTOMIZE THIS PATH
#     os.makedirs(GDRIVE_PROJECT_BASE_DIR, exist_ok=True)
#     print(f"Google Drive successfully mounted. Output base directory set to: {GDRIVE_PROJECT_BASE_DIR}")
#     # Set a flag or variable to indicate Drive is ready
#     IS_GOOGLE_DRIVE_AVAILABLE = True
# except Exception as e_drive:
#     print(f"Google Drive mounting failed or not in a Colab environment: {e_drive}")
#     print("Outputs will be saved locally to './script_outputs/'")
#     GDRIVE_PROJECT_BASE_DIR = None
#     IS_GOOGLE_DRIVE_AVAILABLE = False
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
    combined_train_df = pd.DataFrame()
    test_datasets = {}
    all_fidelities_processed_successfully = False

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
        
        all_fidelities_processed_successfully = True # At least one file was found and read

        sample_frac = 1.0 # Default
        if fidelity_name == "GGA":
            sample_frac = 0.5 # User's specified fraction
        if fidelity_name == "SCAN":
            sample_frac = 0.5 # User's specified fraction
        # GGAU will use sample_frac = 1.0 as per current logic

        if 0 < sample_frac <= 1.0 and not df.empty:
            df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
        elif not df.empty: # Ensure shuffle even if frac is 1.0
            df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        if df.empty:
            print(f"Warning: DataFrame for {fidelity_name} is empty after sampling. Skipping.")
            continue

        # Robust train/test split
        if len(df) < 2:
            print(f"Warning: Less than 2 samples for {fidelity_name} after sampling. Using all for training if available.")
            train_df = df.copy()
            test_df = pd.DataFrame(columns=df.columns) # Empty test_df
        else:
            test_size = int(0.2 * len(df))
            if test_size == 0: # Ensure at least one test sample if possible
                test_size = 1
            
            # Ensure train_df is not empty
            if len(df) - test_size <= 0: # If train would be empty or less
                if len(df) == 1: # Only one sample total
                    train_df = df.copy()
                    test_df = pd.DataFrame(columns=df.columns)
                else: # e.g. 2 samples, test_size = 1, train_size = 1
                    train_df = df.iloc[test_size:].copy() # Standard split
                    test_df = df.iloc[:test_size].copy()
            else:
                test_df = df.iloc[:test_size].copy()
                train_df = df.iloc[test_size:].copy()
        
        if train_df.empty and test_df.empty:
            continue

        train_df.loc[:, 'fidelity'] = fidelity_id
        test_df.loc[:, 'fidelity'] = fidelity_id
        
        test_datasets[fidelity_name] = test_df
        if not train_df.empty:
             combined_train_df = pd.concat([combined_train_df, train_df], ignore_index=True)
        
        print(f"  {fidelity_name}: Train: {len(train_df)} samples, Test: {len(test_df)} samples")
    
    if combined_train_df.empty:
        if not all_fidelities_processed_successfully:
            raise FileNotFoundError(f"No data loaded. Source CSV files might be missing or empty in '{data_base_path}'.")
        else:
            raise ValueError("No training data was aggregated. Check sampling or data content.")
            
    return combined_train_df, test_datasets

def create_test_dataloader(test_df, preprocess, mean, std, num_data_workers=0, current_batch_size=32):
    if test_df.empty:
        return None
    test_data = []
    for idx, row in test_df.iterrows():
        # Robustly get 'formula' and 'BG', handle missing columns if necessary
        formula = row.get('formula', None)
        bg_val = row.get('BG', None)
        if formula is None or bg_val is None:
            print(f"Warning: Missing 'formula' or 'BG' in a row: {row.to_dict()}. Skipping row.")
            continue
        
        representation = preprocess.formula_to_set_representation(formula)
        if representation:
            element_ids, element_weights = representation
            test_data.append((element_ids, element_weights, int(row['fidelity']), bg_val))
        else:
            print(f"Warning: Could not get representation for formula '{formula}'. Skipping row.")

    if not test_data: return None
    
    current_std = std if std != 0 else 1.0
    normalized_test_data = [(eid, ew, fid, (bg - mean) / current_std) for eid, ew, fid, bg in test_data]
    
    return torch.utils.data.DataLoader(
        normalized_test_data, batch_size=current_batch_size, shuffle=False,
        collate_fn=preprocess.collate_fn, num_workers=num_data_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

def train_multifidelity_model(
    combined_train_df, 
    pooling_type='gated', 
    writer=None, 
    num_data_workers=0, 
    learning_rate=0.001, 
    num_model_epochs=120,
    model_config_params=None, # For model hyperparameters
    run_specific_output_path='.' # Base path for outputs of this specific run
    ):

    models_save_dir = os.path.join(run_specific_output_path, "models")
    plots_save_dir = os.path.join(run_specific_output_path, "plots_train_val")
    os.makedirs(models_save_dir, exist_ok=True)
    os.makedirs(plots_save_dir, exist_ok=True)

    # Save the combined training dataset used for this run (for inspection)
    # data_intermediate_dir = os.path.join(run_specific_output_path, "data_intermediate")
    # os.makedirs(data_intermediate_dir, exist_ok=True)
    # combined_train_df.to_csv(os.path.join(data_intermediate_dir, "multifidelity_train_for_run.csv"), index=False)
    
    preprocess = MultiFidelityPreprocessing()
    preprocess.sample_size = None 
    preprocess.batch_size = model_config_params.get("batch_size", 32) if model_config_params else 32
    
    if 'BG' not in combined_train_df.columns:
        raise KeyError("'BG' column missing from combined_train_df. Cannot calculate mean/std.")
    mean = combined_train_df['BG'].mean()
    std = combined_train_df['BG'].std()
    if std == 0 or np.isnan(std): std = 1.0 # Avoid division by zero or NaN std

    train_data = []
    for idx, row in combined_train_df.iterrows():
        formula = row.get('formula', None)
        bg_val = row.get('BG', None)
        if formula is None or bg_val is None: continue
        representation = preprocess.formula_to_set_representation(formula)
        if representation:
            element_ids, element_weights = representation
            train_data.append((element_ids, element_weights, int(row['fidelity']), bg_val))

    if not train_data: print("Error: No training data after preprocessing."); return None, mean, std, preprocess, float('inf')
    normalized_train_data = [(eid, ew, fid, (bg - mean) / std) for eid, ew, fid, bg in train_data]
    
    if len(normalized_train_data) < 2: print("Error: <2 samples for train/val split."); return None, mean, std, preprocess, float('inf')
        
    train_size = int(0.9 * len(normalized_train_data))
    val_size = len(normalized_train_data) - train_size
    if train_size == 0 and len(normalized_train_data) > 0 : train_size = 1; val_size = len(normalized_train_data) -1
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
    
    default_model_hparams = {"embedding_dim": 104, "fidelity_dim": 16, "num_blocks": 5, 
                           "num_heads": 10, "hidden_dim": 250, "dropout": 0.1, "weight_decay": 1e-5}
    current_hparams = default_model_hparams.copy()
    if model_config_params: current_hparams.update(model_config_params)

    # Ensure num_heads is compatible
    if current_hparams["embedding_dim"] % current_hparams["num_heads"] != 0:
        print(f"Warning: embedding_dim {current_hparams['embedding_dim']} not divisible by num_heads {current_hparams['num_heads']}. Adjusting num_heads.")
        # Find largest valid num_heads or default
        possible_heads = [h for h in [1, 2, 4, 8, 10, 13, 16, 26, 52, 104] if current_hparams["embedding_dim"] % h == 0] # Example factors
        current_hparams["num_heads"] = possible_heads[-1] if possible_heads else 1 # Fallback
        print(f"Adjusted num_heads to {current_hparams['num_heads']}")

    model = SetBasedBandgapModel(
        num_elements=118, embedding_dim=current_hparams["embedding_dim"],
        fidelity_dim=current_hparams["fidelity_dim"], num_blocks=current_hparams["num_blocks"],
        num_heads=current_hparams["num_heads"], hidden_dim=current_hparams["hidden_dim"],
        dropout=current_hparams["dropout"], pooling_type=pooling_type
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    if str(device) != "cpu": # torch.compile might be slow or have issues on CPU for some models
        try:
            print("Attempting to compile model with torch.compile()..."); model = torch.compile(model); print("Model compiled.")
        except Exception as e: print(f"torch.compile() failed: {e}. Proceeding without.")
    model.to(device)
    
    if writer: # Log model graph if SummaryWriter is provided
        try:
            sample_element_ids, sample_element_weights, sample_fidelity_ids, _ = next(iter(train_dataloader))
            writer.add_graph(model, (sample_element_ids.to(device), sample_fidelity_ids.to(device), sample_element_weights.to(device)))
            print("Model graph added to TensorBoard.")
        except Exception as e_graph: print(f"Could not add model graph to TensorBoard: {e_graph}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=current_hparams["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) # verbose removed
    
    num_epochs = num_model_epochs
    train_losses, val_losses_list = [], [] # Renamed val_losses to val_losses_list
    best_val_loss_for_saving = float('inf')
    patience_config, patience_counter = 15, 0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        model.train()
        running_train_loss = 0.0
        for batch_idx, (element_ids, element_weights, fidelity_ids, bandgaps) in enumerate(train_dataloader):
            if batch_idx % (max(1, len(train_dataloader) // 5)) == 0: print(f"  Epoch {epoch+1}, Train batch {batch_idx+1}/{len(train_dataloader)}")
            element_ids,element_weights,fidelity_ids,bandgaps = element_ids.to(device),element_weights.to(device),fidelity_ids.to(device),bandgaps.to(device)
            optimizer.zero_grad(set_to_none=True) # More efficient
            predictions = model(element_ids, fidelity_ids, element_weights)
            loss = criterion(predictions, bandgaps)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            loss.backward(); optimizer.step()
            running_train_loss += loss.item() * len(bandgaps)
        
        epoch_train_loss = running_train_loss / len(train_dataset_obj) if len(train_dataset_obj) > 0 else 0.0
        train_losses.append(epoch_train_loss)
        if writer:
            writer.add_scalar('Loss/train_epoch', epoch_train_loss, epoch + 1)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch + 1)

        current_epoch_val_loss = float('inf') # Default if no validation
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
        elif (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
             save_path = os.path.join(models_save_dir, f"model_epoch_{epoch+1}_{pooling_type}.pt")
             torch.save(model.state_dict(), save_path); print(f"Model saved: {save_path} (epoch {epoch+1})")
    
    # Plotting
    fig_tv = plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    if val_dataloader and any(not np.isnan(vl) for vl in val_losses_list):
        plt.plot(range(1, len(val_losses_list) + 1), val_losses_list, label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.title(f'Loss ({pooling_type})')
    plt.legend(); plt.grid(True)
    plot_save_path = os.path.join(plots_save_dir, f"train_val_loss_{pooling_type}.png")
    plt.savefig(plot_save_path); print(f"Train/val loss plot: {plot_save_path}")
    if writer: writer.add_figure(f'Plots_TrainVal/Loss_Curve_{pooling_type}', fig_tv, epoch + 1)
    plt.close(fig_tv)

    # Load best or last saved model
    best_model_chkpt_path = os.path.join(models_save_dir, f"best_model_{pooling_type}.pt")
    last_saved_epoch_model_path = os.path.join(models_save_dir, f"model_epoch_{epoch+1}_{pooling_type}.pt") # epoch+1 is correct from loop end
    
    model_loaded_path = None
    if os.path.exists(best_model_chkpt_path) and val_dataloader:
        model.load_state_dict(torch.load(best_model_chkpt_path, map_location=device)); model_loaded_path = best_model_chkpt_path
    elif os.path.exists(last_saved_epoch_model_path):
         model.load_state_dict(torch.load(last_saved_epoch_model_path, map_location=device)); model_loaded_path = last_saved_epoch_model_path
    if model_loaded_path: print(f"Loaded model for evaluation from: {model_loaded_path}")
    else: print("Warning: Using model from final training state as no specific checkpoint was found/loaded.")
    
    return model, mean, std, preprocess, best_val_loss_for_saving


def evaluate_on_fidelity(model, test_loader, fidelity_name, device, mean, std):
    if test_loader is None:
        print(f"No test data/loader for {fidelity_name}. Skipping eval.")
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
    if not targets_list: return {k: float('nan') for k in ['mae','rmse','r2']}, [], []
    return {'mae': mean_absolute_error(targets_list, preds_list), 
            'rmse': np.sqrt(mean_squared_error(targets_list, preds_list)), 
            'r2': r2_score(targets_list, preds_list)}, preds_list, targets_list


def plot_fidelity_results(predictions, targets, fidelity_name, metrics, eval_plots_save_dir): # Added save_dir
    os.makedirs(eval_plots_save_dir, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    if not predictions or not targets:
        plt.title(f'{fidelity_name} - No Data'); plt.text(0.5,0.5,"No data",ha='center')
    else:
        plt.scatter(targets, predictions, alpha=0.5)
        min_val = min(min(targets,default=0), min(predictions,default=0)); max_val = max(max(targets,default=1), max(predictions,default=1))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        plt.text(0.05,0.95,f"MAE: {metrics.get('mae',0):.4f}\nRMSE: {metrics.get('rmse',0):.4f}\nR²: {metrics.get('r2',0):.4f}",
                 transform=plt.gca().transAxes, bbox=dict(facecolor='w',alpha=0.8), va='top')
    plt.xlabel('Actual BG (eV)'); plt.ylabel('Predicted BG (eV)'); plt.title(f'Eval: {fidelity_name}')
    plt.legend(loc='lower right'); plt.grid(True)
    plot_file = os.path.join(eval_plots_save_dir, f"{fidelity_name}_actual_vs_predicted.png")
    plt.savefig(plot_file); print(f"Saved: {plot_file}")
    df_preds = pd.DataFrame({"Actual_BG": targets, "Predicted_BG": predictions})
    csv_file = os.path.join(eval_plots_save_dir, f"{fidelity_name}_predictions.csv")
    df_preds.to_csv(csv_file, index=False); print(f"Saved: {csv_file}")
    return fig

def create_summary_plot(results, summary_plots_save_dir): # Added save_dir
    os.makedirs(summary_plots_save_dir, exist_ok=True)
    fidelities = list(results.keys())
    if not fidelities: fig = plt.figure(); plt.text(0.5,0.5,"No results",ha='center'); return fig
    maes = [results[f].get('mae', float('nan')) for f in fidelities]
    rmses = [results[f].get('rmse', float('nan')) for f in fidelities]
    r2s = [results[f].get('r2', float('nan')) for f in fidelities]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(len(fidelities)); width = 0.35
    ax1.bar(x-width/2, maes, width, label='MAE'); ax1.bar(x+width/2, rmses, width, label='RMSE')
    ax1.set_ylabel('Error (eV)'); ax1.set_title('MAE & RMSE by Fidelity'); ax1.set_xticks(x)
    ax1.set_xticklabels(fidelities, rotation=45, ha="right"); ax1.legend(); ax1.grid(axis='y',alpha=0.7)
    ax2.bar(x, r2s, width, label='R²', color='g'); ax2.set_ylabel('R² Score'); ax2.set_title('R² by Fidelity')
    ax2.set_xticks(x); ax2.set_xticklabels(fidelities, rotation=45, ha="right")
    valid_r2s = [r for r in r2s if not np.isnan(r)]
    ax2.set_ylim(min(0, min(valid_r2s) if valid_r2s else 0) - 0.05, max(1, max(valid_r2s) if valid_r2s else 1) + 0.05)
    ax2.legend(); ax2.grid(axis='y', alpha=0.7); plt.tight_layout()
    plot_file = os.path.join(summary_plots_save_dir, "performance_summary.png")
    plt.savefig(plot_file); print(f"Saved: {plot_file}")
    df_results = pd.DataFrame({'Fidelity': fidelities, 'MAE': maes, 'RMSE': rmses, 'R2': r2s})
    csv_file = os.path.join(summary_plots_save_dir, "performance_summary.csv")
    df_results.to_csv(csv_file, index=False); print(f"Saved: {csv_file}")
    return fig

def run_experiment_session(
    config, # Dictionary with all experiment parameters
    gdrive_root_dir=None, # Base Google Drive directory for all project runs
    local_root_dir='./script_outputs' # Default local directory
    ):
    """ Orchestrates a full experiment run: data, train, eval, log, save. """
    current_time_str = time.strftime("%Y%m%d-%H%M%S")
    run_name_suffix = f"{config.get('pooling_type','defaultpool')}_{config.get('lr',0.001)}_{current_time_str}"
    
    # Determine base output path (GDrive or local)
    active_root_dir = local_root_dir
    if gdrive_root_dir and ('google.colab' in str(globals().get('__name__')) or 'google.colab' in sys.modules): # Check if in Colab
        # Attempt to use GDrive path only if it seems like GDrive is mounted and accessible
        # A more robust check would be `os.path.exists('/content/drive/MyDrive')` after mount attempt
        try:
            if not os.path.exists(gdrive_root_dir): # User needs to ensure gdrive_root_dir itself exists
                 print(f"Warning: Specified Google Drive root '{gdrive_root_dir}' does not exist. Falling back to local.")
            else:
                 active_root_dir = gdrive_root_dir
                 print(f"Using Google Drive for outputs: {active_root_dir}")
        except Exception as e_gdrive_check: # Catch any error if path is invalid for os.path.exists
            print(f"Error checking Google Drive path '{gdrive_root_dir}': {e_gdrive_check}. Falling back to local.")
            active_root_dir = local_root_dir # Fallback
    else:
        print(f"Not using Google Drive (or not in Colab). Using local output directory: {active_root_dir}")

    run_specific_output_path = os.path.join(active_root_dir, f"run_{run_name_suffix}")
    os.makedirs(run_specific_output_path, exist_ok=True)
    print(f"All outputs for this run will be under: {run_specific_output_path}")

    tensorboard_log_dir = os.path.join(run_specific_output_path, 'tensorboard_logs')
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    print(f"TensorBoard logs for this run: {tensorboard_log_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device for this run: {device}")

    try:
        print("\n" + "="*50 + "\nPreparing Datasets\n" + "="*50 + "\n")
        combined_train_df, test_datasets = prepare_datasets_for_multifidelity(
            data_base_path=config.get('data_input_dir', 'data/train')
        )
        if combined_train_df.empty: raise ValueError("No training data.")

        print("\n" + "="*50 + "\nTraining Multi-Fidelity Model\n" + "="*50 + "\n")
        model, mean, std, preprocess, best_val_loss_from_train = train_multifidelity_model(
            combined_train_df, 
            pooling_type=config.get('pooling_type','gated'), 
            writer=writer,
            num_data_workers=config.get('num_workers',0),
            learning_rate=config.get('lr',0.001),
            num_model_epochs=config.get('epochs',120),
            model_config_params=config, # Pass the whole config for model hparams
            run_specific_output_path=run_specific_output_path
        )
        if model is None: raise ValueError("Model training failed.")

        results = {}
        hparams_for_tb_log = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool, list))}
        # Add best val loss to hparams if available
        if best_val_loss_from_train is not None and not np.isnan(best_val_loss_from_train):
            hparams_for_tb_log['best_validation_loss'] = best_val_loss_from_train
        
        print("\n" + "="*50 + "\nEvaluating on Individual Fidelity Datasets\n" + "="*50 + "\n")
        eval_plots_dir = os.path.join(run_specific_output_path, "plots_eval_fidelity")

        for fidelity_name, test_df_fid in test_datasets.items():
            print(f"Evaluating: {fidelity_name}...")
            if test_df_fid.empty:
                results[fidelity_name] = {k: float('nan') for k in ['mae','rmse','r2']}; continue
            
            test_loader = create_test_dataloader(
                test_df_fid, preprocess, mean, std, 
                num_data_workers=config.get('num_workers',0),
                current_batch_size=config.get("batch_size", 32) # Use consistent batch size
            )
            
            metrics, predictions, targets = evaluate_on_fidelity(model, test_loader, fidelity_name, device, mean, std)
            results[fidelity_name] = metrics
            
            if predictions or targets: # Ensure there's data to plot
                fig_fid = plot_fidelity_results(predictions, targets, fidelity_name, metrics, eval_plots_save_dir=eval_plots_dir)
                if writer: writer.add_figure(f'Eval_Plots/{fidelity_name}_Actual_vs_Predicted', fig_fid, config.get('epochs',120))
                plt.close(fig_fid)
            print(f"  {fidelity_name}: MAE {metrics.get('mae',0):.4f}, RMSE {metrics.get('rmse',0):.4f}, R² {metrics.get('r2',0):.4f}")
        
        summary_plots_dir = os.path.join(run_specific_output_path, "plots_summary")
        if results:
            fig_sum = create_summary_plot(results, summary_plots_save_dir=summary_plots_dir)
            if writer: writer.add_figure('Eval_Plots/Performance_Summary_Overall', fig_sum, config.get('epochs',120))
            plt.close(fig_sum)

            final_metrics_to_log_tb = {}
            for f_name, mets in results.items():
                for m_name, m_val in mets.items():
                    if not np.isnan(m_val): final_metrics_to_log_tb[f'{f_name}_{m_name}'] = m_val
            if writer and final_metrics_to_log_tb:
                try: writer.add_hparams(hparams_for_tb_log, final_metrics_to_log_tb)
                except Exception as e: print(f"TB HParam logging error: {e}")
        
        print_summary_results(results)

    except Exception as e_main_run:
        print(f"ERROR during experiment run: {e_main_run}")
        import traceback
        traceback.print_exc()
        if writer: writer.add_text("RunError", str(e_main_run))
    finally:
        if writer: writer.close()
        print(f"Run finished. Outputs are in: {run_specific_output_path}")

    return results, run_specific_output_path


def print_summary_results(results): # Helper to print final results
    print("\n" + "="*50 + "\nMULTI-FIDELITY EXPERIMENT RESULTS SUMMARY\n" + "="*50)
    print(f"{'Fidelity':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
    print("-"*50)
    for fidelity_name, metrics in results.items():
        print(f"{fidelity_name:<10} {metrics.get('mae', float('nan')):<10.4f} {metrics.get('rmse', float('nan')):<10.4f} {metrics.get('r2', float('nan')):<10.4f}")


if __name__ == "__main__":
    # --- Google Drive Global Path (Set after mounting in Colab) ---
    # This variable will be used by run_experiment_session if not None
    # To use local storage, set GDRIVE_PROJECT_BASE_DIR = None
    # GDRIVE_PROJECT_BASE_DIR = '/content/drive/MyDrive/Beemo_Project_Runs' # Example
    # For this script to run standalone, we'll assume it's defined or None
    # In Colab, you would set this after drive.mount()
    
    # Attempt to get the GDrive path if defined by the mounting cell, else None
    # This is a bit of a trick to make it work if the mounting cell was run
    try:
        # This line assumes GDRIVE_PROJECT_BASE_DIR was defined in a previous cell in Colab
        # If running as a .py file standalone, this will likely cause NameError unless defined globally
        # For a .py file, you'd set GDRIVE_PROJECT_BASE_DIR = "/path/to/gdrive/if/mounted/else/None"
        gdrive_path_for_run = GDRIVE_PROJECT_BASE_DIR 
    except NameError:
        print("GDRIVE_PROJECT_BASE_DIR not defined (e.g., Drive not mounted or script run outside Colab with that setup). Using local outputs.")
        gdrive_path_for_run = None


    # ----- Configuration for the experiment run -----
    # You can define multiple configurations and loop through them if needed
    experiment_settings = {
        "pooling_type": "gated",
        "lr": 0.0005, # Example: different LR
        "epochs": 15,  # Quick test run; set to 120-200 for full training
        "num_workers": 0, # Start with 0 for stability, increase based on your system (e.g., os.cpu_count()//2)
        "data_input_dir": 'data/train', # Relative path to where CSVs are stored

        # Model Hyperparameters (can be filled from Optuna best_trial.params)
        "embedding_dim": 104,
        "fidelity_dim": 16,
        "num_blocks": 5,
        "num_heads": 10, # Ensure embedding_dim is divisible by num_heads
        "hidden_dim": 250, # Corresponds to hidden_dim in SetBasedBandgapModel
        "dropout": 0.1,
        "batch_size": 32,
        "weight_decay": 1e-5
    }
    
    print(f"Starting experiment session with config: {experiment_settings}")
    # The `run_experiment_session` function now handles GDrive path or local path for outputs
    results_summary, output_location = run_experiment_session(
        config=experiment_settings,
        gdrive_root_dir=gdrive_path_for_run # Pass the GDrive base path (can be None)
    )
    
    print(f"\nExperiment session finished. All outputs located in: {output_location}")

    # If you want to run Optuna, you'd structure its call here,
    # using the objective_for_optuna function (defined separately as in previous examples)
    # and passing combined_train_df to it.
    # For example:
    # if "RUN_OPTUNA_HPO" in os.environ and os.environ["RUN_OPTUNA_HPO"] == "1":
    #     import optuna
    #     print("\n" + "="*70 + "\nSTARTING OPTUNA HPO\n" + "="*70)
    #     # ... (Optuna setup and study.optimize call as shown previously) ...