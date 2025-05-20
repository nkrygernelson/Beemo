import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Assuming these are in your Colab environment or correctly pathed
from set_based_model import SetBasedBandgapModel
from preprocess_set_data import MultiFidelityPreprocessing


class MultiTrainer:
    def __init__(self, subsample_dict = None, model_params = None, training_params = None,fidelity_map = None, property_name = None, fidelities_dir = None):
        if model_params is None:
            self.model_params = {
                "num_elements": 118,
                "num_fidelities":5,
                "embedding_dim": 100,
                "fidelity_dim": 20,
                "num_blocks": 5,
                "num_heads": 10,
                "hidden_dim": 250,
                "dropout": 0.1,
                "pooling_type": "gated"}
        else: 
            self.model_params = model_params
        if training_params is None:
            self.training_params = {
                "epochs": 10, "batch_size": 32, "learning_rate": 0.001, "weight_decay": 1e-5, "multi_train_split":0.8,}
        else:
            self.training_params = training_params
        if subsample_dict is None:
            self.subsample_dict = {"pbe": 1, "scan": 1,
                                "gllb-sc": 1, "hse": 1, "expt": 1}
        else:
            self.subsample_dict = subsample_dict
        if fidelity_map is None:
            self.fidelity_map = {
                'pbe': 0,
                'scan': 1,
                'gllb-sc': 2,
                'hse': 3,
                'expt': 4
            }
        else:
            self.fidelity_map = fidelity_map
        if property_name is None:
            self.property_name = "BG"
        else:
            self.property_name = property_name
        self.device = torch.device("cuda" if torch.cuda.is_available(
        ) else "cpu" if torch.backends.mps.is_available() else "cpu")
        # --- Google Drive Configuration ---
        if fidelities_dir is None:
            self.fidelities_dir = "train_homemade"
        else:
            self.fidelities_dir = fidelities_dir
        self.GOOGLE_DRIVE = False  # SET THIS TO True TO ENABLE GOOGLE DRIVE SAVING
        self.MAC = True

        self.DRIVE_MOUNT_POINT = '/content/drive'
        # *** CUSTOMIZE THIS TO YOUR PREFERRED GDRIVE FOLDER ***
        self.YOUR_PROJECT_GDRIVE_FOLDER = 'AI_Bandgap_Project/MultiFidelityOutput'
        self.DRIVE_BASE_SAVE_PATH = os.path.join(self.DRIVE_MOUNT_POINT, 'MyDrive', self.YOUR_PROJECT_GDRIVE_FOLDER)

        # This will be "" if not using Google Drive, or the DRIVE_BASE_SAVE_PATH if using it.
        if self.GOOGLE_DRIVE:
            self.save_prefix = self.DRIVE_BASE_SAVE_PATH
        else:
            self.save_prefix = ""


    def prepare_datasets_for_multifidelity(self, subsample_dict = None):
        """
        Prepare datasets for multi-fidelity training and individual fidelity testing.
        """
        if subsample_dict is None:
            subsample_dict = self.subsample_dict

        combined_train_df = pd.DataFrame()
        test_datasets = {}
        train_stats = {}
        train_split = self.training_params["multi_train_split"]

        # Process each fidelity dataset
        for fidelity_name, fidelity_id in self.fidelity_map.items():
            print(f"Processing {fidelity_name} dataset...")

            # Construct path for loading data
            # Assumes input data is in 'data/train/' relative to save_prefix if GOOGLE_DRIVE is True
            # Or locally if GOOGLE_DRIVE is False
            data_file_path = os.path.join(
                self.save_prefix, 'data', self.fidelities_dir, f'{fidelity_name}.csv')

            # Fallback to local path if file not found at prefixed path (e.g., if inputs are always local)
            # only try local if prefixed path failed AND prefix exists
            if not os.path.exists(data_file_path) and self.save_prefix:
                print(
                    f"File not found at {data_file_path}, trying local path 'data/train/{fidelity_name}.csv'")
                local_path_check = f'data/train/{fidelity_name}.csv'
                if os.path.exists(local_path_check):
                    data_file_path = local_path_check
                else:
                    print(
                        f"ERROR: Data file for {fidelity_name} not found at {data_file_path} or {local_path_check}")
                    print(
                        "Please ensure your data .csv files (e.g., GGA.csv) are in the correct location:")
                    print(
                        f"  - If GOOGLE_DRIVE=True: In '{os.path.join(self.DRIVE_BASE_SAVE_PATH, 'data', 'train')}' on your Google Drive.")
                    print(
                        f"  - If GOOGLE_DRIVE=False: In the local './data/train/' directory.")
                    print(f"Skipping {fidelity_name} dataset.")
                    continue
            # Not using drive and file not found locally
            elif not os.path.exists(data_file_path) and not self.save_prefix:
                print(
                    f"ERROR: Data file for {fidelity_name} not found at {data_file_path}")
                print(
                    "Please ensure your data .csv files are in the local './data/train/' directory.")
                print(f"Skipping {fidelity_name} dataset.")
                continue

            print(f"  Loading from: {data_file_path}")
            df = pd.read_csv(data_file_path)

            # Shuffle the dataset
            df = df.drop_duplicates()
            df.dropna()
            if self.subsample_dict:
                sample_frac = self.subsample_dict.get(fidelity_name, 1)

            df = df.sample(frac=sample_frac,
                        random_state=42).reset_index(drop=True)

            # Create train/test split (80/20)
            test_size = int((1-train_split) * len(df))
            test_df = df.iloc[:test_size].copy()
            train_df = df.iloc[test_size:].copy()
            print(train_df.columns)
            mean = train_df[self.property_name].mean()
            std = train_df[self.property_name].std()
            train_stats[fidelity_name] = {
                "mean": mean,
                "std": std,
                "num_samples": len(train_df)
            }
            # Add fidelity column
            train_df['fidelity'] = fidelity_id
            test_df['fidelity'] = fidelity_id

            # Store the test dataset
            test_datasets[fidelity_name] = test_df

            # Add training data to combined dataset
            combined_train_df = pd.concat(
                [combined_train_df, train_df], ignore_index=True)

            print(
                f"  Train: {len(train_df)} samples, Test: {len(test_df)} samples")

        if combined_train_df.empty and self.fidelity_map:
            print("WARNING: No data was loaded into combined_train_df. Check data paths and file availability.")
       
        return combined_train_df, test_datasets, train_stats


    def create_test_dataloader(self, test_df, preprocess, mean, std,):
        """
        Create a DataLoader for a test dataset.
        """
        # Process the test data
        test_data = []
        for idx, row in test_df.iterrows():
            element_ids, element_weights = preprocess.formula_to_set_representation(
                row['formula'])
            test_data.append((element_ids, element_weights,
                            int(row['fidelity']), row[self.property_name]))

        # Apply normalization
        normalized_test_data = []
        for element_ids, element_weights, fid, bg in test_data:
            normalized_test_data.append(
                (element_ids, element_weights, fid, (bg - mean) / std))

        # Create a DataLoader
        test_loader = torch.utils.data.DataLoader(
            normalized_test_data,
            batch_size=self.training_params["batch_size"],
            shuffle=False,
            collate_fn=preprocess.collate_fn
        )
        return test_loader
    


    def train_multifidelity_model(self, combined_train_df, model_params=None, pooling_type="gated"):
        """
        Train a multi-fidelity model using the  combined training dataset.
        """
        if model_params is None:
            model_params = self.model_params
        train_split = self.training_params["multi_train_split"]
        #global save_prefix  # Ensure we're using the globally set save_prefix

        # Define paths for saving, prefixed if GOOGLE_DRIVE is True
        data_dir_path = os.path.join(self.save_prefix, "data")
        predictions_dir_path = os.path.join(
            self.save_prefix, "predictions", "multifidelity")
        combined_train_df = combined_train_df.drop_duplicates()
        # Save the combined training dataset
        os.makedirs(data_dir_path, exist_ok=True)
        combined_train_csv_path = os.path.join(
            data_dir_path, "multifidelity_train.csv")
        combined_train_df.to_csv(combined_train_csv_path, index=False)
        print(f"Combined training data saved to: {combined_train_csv_path}")

        # Initialize preprocessing
        preprocess = MultiFidelityPreprocessing()
        preprocess.batch_size = self.training_params["batch_size"]


        # Calculate normalization statistics
        mean = combined_train_df[self.property_name].mean()
        std = combined_train_df[self.property_name].std()

        # Process the training data
        train_data = []
        for idx, row in combined_train_df.iterrows():
            element_ids, element_weights = preprocess.formula_to_set_representation(
                row['formula'])
            train_data.append((element_ids, element_weights,
                            int(row['fidelity']), row[self.property_name]))

        # Apply normalization
        normalized_train_data = []
        for element_ids, element_weights, fid, bg in train_data:
            normalized_train_data.append(
                (element_ids, element_weights, fid, (bg - mean) / std))

        # Split into train/validation
        train_size = int(train_split*len(normalized_train_data))
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

        print(
            f"Data processed: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

        # Initialize model, loss function, and optimizer
        if self.model_params:
            model = SetBasedBandgapModel(
                num_elements=self.model_params["num_elements"],
                embedding_dim=self.model_params["embedding_dim"],
                num_fidelities=self.model_params["num_fidelities"],
                fidelity_dim=self.model_params["fidelity_dim"],
                num_blocks=self.model_params["num_blocks"],
                num_heads=self.model_params["num_heads"],
                hidden_dim=self.model_params["hidden_dim"],
                dropout=self.model_params["dropout"],
                pooling_type=self.model_params["pooling_type"]
            )
    
        device = torch.device("cuda" if torch.cuda.is_available(
        ) else "cpu" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        self.device = device
        model.to(device)
        #model = torch.compile(model, mode ="default")
        learning_rate = self.training_params.get("learning_rate", 0.001)
        weight_decay = self.training_params.get("weight_decay", 1e-5)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5,
        )

        # Training settings
        num_epochs = self.training_params["epochs"]
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_val = 15  # Renamed from patience to avoid conflict with module
        patience_counter = 0

        # Create directory for saving predictions
        os.makedirs(predictions_dir_path, exist_ok=True)
        best_model_path = os.path.join(
            predictions_dir_path, f"best_model_{pooling_type}.pt")

        # Training loop
        for epoch in range(num_epochs):
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

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * len(bandgaps)

            epoch_train_loss = running_train_loss / len(train_dataset)

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
            scheduler.step(epoch_val_loss)
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(
                    f"New best model saved to {best_model_path} with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience_val:
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
        plot_loss_path = os.path.join(
            predictions_dir_path, f"train_val_loss_{pooling_type}.png")
        plt.savefig(plot_loss_path)
        print(f"Training/validation loss plot saved to: {plot_loss_path}")
        plt.close()

        # Load best model for testing
        model.load_state_dict(torch.load(best_model_path))
        print(f"Loaded best model from: {best_model_path}")

        return model,preprocess


    def evaluate_on_fidelity(self, model, test_loader,  mean, std, device=None):
        '''
        Evaluates the given model on a test dataset with a specific fidelity, returning regression metrics.

        Args:
            model (torch.nn.Module): The model to evaluate.
            test_loader (torch.utils.data.DataLoader): DataLoader providing test data batches.
            mean (float or np.ndarray): Mean value(s) used for denormalizing predictions and targets.
            std (float or np.ndarray): Standard deviation(s) used for denormalizing predictions and targets.
            device (torch.device or str, optional): Device to run evaluation on. Defaults to self.device.

        Returns:
            Tuple[dict, list, list]:
                - metrics (dict): Dictionary containing 'mae', 'rmse', and 'r2' regression metrics.
                - predictions (list): List of denormalized model predictions.
                - targets (list): List of denormalized ground truth values.
        '''
        model.eval()
        predictions = []
        targets = []
        if device is None:
            device = self.device
        with torch.no_grad():
            for element_ids, element_weights, fidelity_ids, bandgaps in test_loader:
                element_ids = element_ids.to(device)
                element_weights = element_weights.to(device)
                fidelity_ids = fidelity_ids.to(device)
                bandgaps = bandgaps.to(device)

                preds = model(element_ids, fidelity_ids, element_weights)

                preds_orig = preds.cpu().numpy() * std + mean
                targets_orig = bandgaps.cpu().numpy() * std + mean

                predictions.extend(preds_orig.tolist())
                targets.extend(targets_orig.tolist())

        mae = mean_absolute_error(targets, predictions)
        rmse = np.sqrt(mean_squared_error(targets, predictions))
        r2 = r2_score(targets, predictions)

        return {'mae': mae, 'rmse': rmse, 'r2': r2}, predictions, targets


    def plot_fidelity_results(self, predictions, targets, fidelity_name, metrics):
        """
        Generate plots for a specific fidelity evaluation.
        """

        fidelity_plots_dir = os.path.join(
            self.save_prefix, "predictions", "multifidelity", "fidelity_plots")
        os.makedirs(fidelity_plots_dir, exist_ok=True)

        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.5)

        # Added default for empty lists
        min_val = min(min(targets, default=0), min(predictions, default=0))
        # Added default for empty lists
        max_val = max(max(targets, default=1), max(predictions, default=1))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

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

        plot_file_path = os.path.join(
            fidelity_plots_dir, f"{fidelity_name}_actual_vs_predicted.png")
        plt.savefig(plot_file_path)
        plt.close()
        print(f"Fidelity plot for {fidelity_name} saved to: {plot_file_path}")

        df_preds = pd.DataFrame({
            "Actual_BG": targets,
            "Predicted_BG": predictions
        })
        preds_csv_path = os.path.join(
            fidelity_plots_dir, f"{fidelity_name}_predictions.csv")
        df_preds.to_csv(preds_csv_path, index=False)
        print(f"Predictions CSV for {fidelity_name} saved to: {preds_csv_path}")

        return plot_file_path


    def create_summary_plot(self, results):
        """
        Create a summary plot showing performance across all fidelities.
        """
        save_prefix = self.save_prefix  # Ensure we're using the globally set save_prefix
        summary_dir = os.path.join(save_prefix, "predictions", "multifidelity")
        # Already created in train_multifidelity_model, but good to be sure
        os.makedirs(summary_dir, exist_ok=True)

        fidelities = list(results.keys())
        if not fidelities:
            print("No results to create a summary plot.")
            return

        maes = [results[f]['mae'] for f in fidelities]
        rmses = [results[f]['rmse'] for f in fidelities]
        r2s = [results[f]['r2'] for f in fidelities]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
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

        ax2.bar(x, r2s, width, label='R²', color='green')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score by Fidelity Level')
        ax2.set_xticks(x)
        ax2.set_xticklabels(fidelities)
        ax2.set_ylim(min(0, min(r2s if r2s else [0])), 1)  # Adjust y-lim for R2
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        summary_plot_path = os.path.join(summary_dir, "performance_summary.png")
        plt.savefig(summary_plot_path)
        plt.close()
        print(f"Performance summary plot saved to: {summary_plot_path}")

        df_results = pd.DataFrame({
            'Fidelity': fidelities,
            'MAE': maes,
            'RMSE': rmses,
            'R2': r2s
        })
        summary_csv_path = os.path.join(summary_dir, "performance_summary.csv")
        df_results.to_csv(summary_csv_path, index=False)
        print(f"Performance summary CSV saved to: {summary_csv_path}")


    def run_multifidelity_experiments(self):
        """
        Run the complete multi-fidelity training and evaluation workflow.
        """
        save_prefix = self.save_prefix  # Crucial for functions below to know the correct base path

        # --- Google Drive Mounting and Path Setup ---
        if self.GOOGLE_DRIVE:
            try:
                from google.colab import drive
                drive.mount(self.DRIVE_MOUNT_POINT, force_remount=True)
                save_prefix = self.DRIVE_BASE_SAVE_PATH  # Set the global prefix
                print(
                    f"Successfully mounted Google Drive. Saving outputs to: {save_prefix}")
                # Create base output directories on Drive
                os.makedirs(os.path.join(save_prefix, "data", "train"),
                            exist_ok=True)  # For input data structure
                os.makedirs(os.path.join(save_prefix, "predictions",
                            "multifidelity", "fidelity_plots"), exist_ok=True)
            except ImportError:
                print("Not running in Google Colab or 'google.colab' not found. 'GOOGLE_DRIVE' is True but cannot mount.")
                print("Saving outputs locally instead.")
                save_prefix = ""  # Fallback to local saving
                # Ensure local directories exist
                os.makedirs(os.path.join(
                    save_prefix, "data", "train"), exist_ok=True)
                os.makedirs(os.path.join(save_prefix, "predictions",
                            "multifidelity", "fidelity_plots"), exist_ok=True)
            except Exception as e:
                print(f"An error occurred during Google Drive setup: {e}")
                print("Saving outputs locally instead.")
                save_prefix = ""  # Fallback to local saving
                os.makedirs(os.path.join(
                    save_prefix, "data", "train"), exist_ok=True)
                os.makedirs(os.path.join(save_prefix, "predictions",
                            "multifidelity", "fidelity_plots"), exist_ok=True)
        else:
            print("GOOGLE_DRIVE flag is False. Saving outputs locally.")
            save_prefix = ""  # Ensure it's empty for local saves
            # Ensure local directories exist for local execution
            os.makedirs(os.path.join(save_prefix, "data", "train"), exist_ok=True)
            os.makedirs(os.path.join(save_prefix, "predictions",
                        "multifidelity", "fidelity_plots"), exist_ok=True)
        # --- End of Google Drive Setup ---

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # This print is now inside train_multifidelity_model
        # print(f"Using device: {device}") # Already printed in train_multifidelity_model

        print("\n" + "="*50)
        print("Preparing Datasets")
        print("="*50 + "\n")
        combined_train_df, test_datasets, train_stats= self.prepare_datasets_for_multifidelity(
            subsample_dict=self.subsample_dict)

        if combined_train_df.empty:
            print("Halting execution: combined_train_df is empty after prepare_datasets_for_multifidelity.")
            print("Please check your data CSV files and their paths.")
            return {}, {}

        print("\n" + "="*50)
        print("Training Multi-Fidelity Model")
        print("="*50 + "\n")
        # Pass model_params from the __main__ block
        model, preprocess = self.train_multifidelity_model(
            combined_train_df, model_params=self.model_params, pooling_type=self.model_params.get('pooling_type', 'gated'))

        results = {}
        plot_paths = {}

        print("\n" + "="*50)
        print("Evaluating on Individual Fidelity Datasets")
        print("="*50 + "\n")

        if not test_datasets:
            print("No test datasets to evaluate. Skipping evaluation.")
        else:
            for fidelity_name, test_df in test_datasets.items():
                print(f"Evaluating on {fidelity_name} dataset...")
                mean, std = train_stats["fidelity_name"]["mean"], train_stats["fidelity_name"]["std"]
                test_loader = self.create_test_dataloader(
                    test_df, preprocess, mean, std)
                metrics, predictions, targets = self.evaluate_on_fidelity(
                    model, test_loader, device, mean, std
                )
                results[fidelity_name] = metrics
                plot_path = self.plot_fidelity_results(
                    predictions, targets, fidelity_name, metrics)
                plot_paths[fidelity_name] = plot_path
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  R²: {metrics['r2']:.4f}")
                print()

        if results:
            self.create_summary_plot(results)
        else:
            print("No results from evaluation to summarize.")

        print("\n" + "="*50)
        print("MULTI-FIDELITY EXPERIMENT RESULTS SUMMARY")
        print("="*50)
        if results:
            print(f"{'Fidelity':<10} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
            print("-"*50)
            for fidelity_name, metrics in results.items():
                print(
                    f"{fidelity_name:<10} {metrics['mae']:<10.4f} {metrics['rmse']:<10.4f} {metrics['r2']:<10.4f}")
        else:
            print("No results to display.")

        return results, plot_paths

    '''
    if __name__ == "__main__":
        # Make model_params global so run_multifidelity_experiments can pass it to train_multifidelity_model
        model_params_global = {
            "num_elements": 118,
            "embedding_dim": 14,
            "fidelity_dim": 6,
            "num_blocks": 5,
            "num_heads": 10,
            "hidden_dim": 250,
            "dropout": 0.1,
            "pooling_type": "gated"
        }
        model_params_global = {
            "epochs": 10,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "num_elements": 118,
            "embedding_dim": 100,
            "fidelity_dim": 20,
            "num_blocks": 5,
            "num_heads": 10,
            "hidden_dim": 250,
            "dropout": 0.1,
            "pooling_type": "gated"}
        subsample_dict = {"pbe": 0.5, "scan": 0.1, "gllb-sc": 0.1}

        results, plot_paths = run_multifidelity_experiments(
            subsample_dict=subsample_dict)

        print("\n--- Experiment Finished ---")
        if self.GOOGLE_DRIVE and self.save_prefix:  # Check if save_prefix was successfully set
            print(
                f"All outputs should be saved in your Google Drive under: {self.save_prefix}")
        elif self.GOOGLE_DRIVE and not self.save_prefix:
            print(
                "GOOGLE_DRIVE was True, but an issue occurred. Outputs saved locally (if any).")
        else:
            print(
                f"All outputs saved locally (current directory structure). Base prefix was: '{self.save_prefix}'")
        '''