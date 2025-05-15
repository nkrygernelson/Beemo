import torch
import pandas as pd
import numpy as np
import random
from pymatgen.core.composition import Composition as pmg_Composition
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

class SetBasedPreprocessing:
    def __init__(self):
        self.sample_size = 30000
        self.split = [0.8, 0.1, 0.1]
        self.property_key = "BG"
        self.batch_size = 128
        self.max_elements = 10  # Max elements to consider per formula
    
    def get_atomic_number_from_ele(self, ele):
        return pmg_Composition(ele).elements[0].Z
    
    def formula_to_set_representation(self, formula):
        """
        Convert a chemical formula to a set representation with elements and fractional weights.
        
        Args:
            formula: Chemical formula string (e.g., "Fe2O3")
            
        Returns:
            element_ids: List of atomic numbers
            element_weights: List of fractional weights
        """
        try:
            comp = pmg_Composition(formula)
            elements = []
            weights = []
            
            # Get total number of atoms
            total_atoms = comp.num_atoms
            
            # Extract elements and their fractional amounts
            for element, amount in comp.items():
                elements.append(element.Z)  # Atomic number
                weights.append(amount / total_atoms)  # Fractional amount
            
            # Sort by atomic number (optional, makes visualization easier)
            sorted_pairs = sorted(zip(elements, weights), key=lambda x: x[0])
            elements, weights = zip(*sorted_pairs) if sorted_pairs else ([], [])
            
            # Pad to max_elements
            padding_needed = self.max_elements - len(elements)
            if padding_needed > 0:
                elements = list(elements) + [0] * padding_needed  # 0 for padding
                weights = list(weights) + [0.0] * padding_needed
            elif padding_needed < 0:
                # If more elements than max_elements, truncate (rare case)
                elements = list(elements)[:self.max_elements]
                weights = list(weights)[:self.max_elements]
                # Renormalize weights
                weight_sum = sum(weights)
                weights = [w / weight_sum for w in weights]
                
            return elements, weights
        except Exception as e:
            print(f"Error processing formula {formula}: {e}")
            # Return default for invalid formulas (Hydrogen)
            return [1] + [0] * (self.max_elements - 1), [1.0] + [0.0] * (self.max_elements - 1)
    
    def collate_fn(self, batch):
        """
        Collate function for set-based representation.
        
        Args:
            batch: List of (element_ids, element_weights, bandgap) tuples
            
        Returns:
            element_ids_batch: [batch_size, max_elements] tensor
            element_weights_batch: [batch_size, max_elements] tensor
            bandgaps_batch: [batch_size] tensor
        """
        element_ids_batch, element_weights_batch, bandgaps_batch = zip(*batch)
        
        # Convert to tensors
        element_ids_batch = torch.tensor(element_ids_batch, dtype=torch.long)
        element_weights_batch = torch.tensor(element_weights_batch, dtype=torch.float32)
        bandgaps_batch = torch.tensor(bandgaps_batch, dtype=torch.float32)
        
        return element_ids_batch, element_weights_batch, bandgaps_batch
    
    def normalize_target(self, dataset):
        bandgaps = [bg for _, _, bg in dataset]
        mean = np.mean(bandgaps)
        std = np.std(bandgaps)
        
        normalized_dataset = []
        for element_ids, element_weights, bg in dataset:
            normalized_dataset.append((element_ids, element_weights, (bg - mean) / std))
        
        return normalized_dataset, mean, std
    
    def nan_hook(self, dataset):
        for i, (element_ids, element_weights, bg) in enumerate(dataset):
            if np.isnan(bg):
                print(f"Found nan at index {i}")
                print(f"Element IDs: {element_ids}")
                print(f"Element weights: {element_weights}")
                print(f"Bandgap: {bg}")
                break
    
    def preprocess_data(self):
        # Load data
        df = pd.read_csv("data/data.csv")
        
        # Clean up formulas
        df = df[~df["formula"].isin(["nan","NaN","NAN"])].dropna(subset=["formula"])
        df['formula'] = df['formula'].apply(lambda x: str(x).replace("NaN", "Na1N"))
        
        bandgaps = df[self.property_key].values  
        formulas = df["formula"].tolist()
        
        # Build dataset with set representation
        dataset = []
        for formula, bg in zip(formulas, bandgaps):
            element_ids, element_weights = self.formula_to_set_representation(formula)
            dataset.append((element_ids, element_weights, bg))
        
        # Sample if needed
        if self.sample_size is not None and self.sample_size < len(dataset):
            dataset = random.sample(dataset, self.sample_size)
        
        # Check for NaNs
        self.nan_hook(dataset)
        
        # Normalize targets
        dataset, mean, std = self.normalize_target(dataset)
        
        # Split into train/val/test
        dataset_size = len(dataset)
        train_size = int(self.split[0] * dataset_size)
        val_size = int(self.split[1] * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
        
        # Return dataloaders and normalization parameters
        return train_dataloader, val_dataloader, test_dataloader, mean, std


class MultiFidelityPreprocessing:
    def __init__(self):
        self.sample_size = 30000
        self.split = [0.8, 0.1, 0.1]
        self.property_key = "BG"
        self.batch_size = 128
        self.max_elements = 10  # Max elements to consider per formula
    
    def get_atomic_number_from_ele(self, ele):
        return pmg_Composition(ele).elements[0].Z
    
    def formula_to_set_representation(self, formula):
        """
        Convert a chemical formula to a set representation with elements and fractional weights.
        
        Args:
            formula: Chemical formula string (e.g., "Fe2O3")
            
        Returns:
            element_ids: List of atomic numbers
            element_weights: List of fractional weights
        """
        try:
            comp = pmg_Composition(formula)
            elements = []
            weights = []
            
            # Get total number of atoms
            total_atoms = comp.num_atoms
            
            # Extract elements and their fractional amounts
            for element, amount in comp.items():
                elements.append(element.Z)  # Atomic number
                weights.append(amount / total_atoms)  # Fractional amount
            
            # Sort by atomic number (optional, makes visualization easier)
            sorted_pairs = sorted(zip(elements, weights), key=lambda x: x[0])
            elements, weights = zip(*sorted_pairs) if sorted_pairs else ([], [])
            
            # Pad to max_elements
            padding_needed = self.max_elements - len(elements)
            if padding_needed > 0:
                elements = list(elements) + [0] * padding_needed  # 0 for padding
                weights = list(weights) + [0.0] * padding_needed
            elif padding_needed < 0:
                # If more elements than max_elements, truncate (rare case)
                elements = list(elements)[:self.max_elements]
                weights = list(weights)[:self.max_elements]
                # Renormalize weights
                weight_sum = sum(weights)
                weights = [w / weight_sum for w in weights]
                
            return elements, weights
        except Exception as e:
            print(f"Error processing formula {formula}: {e}")
            # Return default for invalid formulas (Hydrogen)
            return [1] + [0] * (self.max_elements - 1), [1.0] + [0.0] * (self.max_elements - 1)
    
    def collate_fn(self, batch):
        """
        Collate function for multi-fidelity set-based representation.
        
        Args:
            batch: List of (element_ids, element_weights, fidelity, bandgap) tuples
            
        Returns:
            element_ids_batch: [batch_size, max_elements] tensor
            element_weights_batch: [batch_size, max_elements] tensor
            fidelity_ids_batch: [batch_size] tensor
            bandgaps_batch: [batch_size] tensor
        """
        element_ids_batch, element_weights_batch, fidelity_ids_batch, bandgaps_batch = zip(*batch)
        
        # Convert to tensors
        element_ids_batch = torch.tensor(element_ids_batch, dtype=torch.long)
        element_weights_batch = torch.tensor(element_weights_batch, dtype=torch.float32)
        fidelity_ids_batch = torch.tensor(fidelity_ids_batch, dtype=torch.long)
        bandgaps_batch = torch.tensor(bandgaps_batch, dtype=torch.float32)
        
        return element_ids_batch, element_weights_batch, fidelity_ids_batch, bandgaps_batch
    
    def normalize_target(self, dataset):
        bandgaps = [bg for _, _, _, bg in dataset]  # Updated for 4-tuple dataset
        mean = np.mean(bandgaps)
        std = np.std(bandgaps)
        
        normalized_dataset = []
        for element_ids, element_weights, fid, bg in dataset:  # Updated for 4-tuple dataset
            normalized_dataset.append((element_ids, element_weights, fid, (bg - mean) / std))
        
        return normalized_dataset, mean, std
    
    def nan_hook(self, dataset):
        for i, (element_ids, element_weights, fid, bg) in enumerate(dataset):  # Updated for 4-tuple dataset
            if np.isnan(bg):
                print(f"Found nan at index {i}")
                print(f"Element IDs: {element_ids}")
                print(f"Element weights: {element_weights}")
                print(f"Fidelity: {fid}")
                print(f"Bandgap: {bg}")
                break
    
    def preprocess_data(self):
        # Load data
        df = pd.read_csv("data/all_data.csv")
        
        # Clean up formulas
        df = df[~df["formula"].isin(["nan","NaN","NAN"])].dropna(subset=["formula"])
        df = df.dropna()
        df['formula'] = df['formula'].apply(lambda x: str(x).replace("NaN", "Na1N"))
        
        bandgaps = df[self.property_key].values  
        formulas = df["formula"].tolist()
        fidelities = df["fidelity"].values
        
        # Build dataset with set representation
        dataset = []
        for formula, bg, fid in zip(formulas, bandgaps, fidelities):
            element_ids, element_weights = self.formula_to_set_representation(formula)
            dataset.append((element_ids, element_weights, int(fid), bg))
            
        # Sample if needed
        if self.sample_size is not None and self.sample_size < len(dataset):
            dataset = random.sample(dataset, self.sample_size)
        
        # Check for NaNs
        self.nan_hook(dataset)
        
        # Normalize targets
        dataset, mean, std = self.normalize_target(dataset)
        
        # Split into train/val/test
        dataset_size = len(dataset)
        train_size = int(self.split[0] * dataset_size)
        val_size = int(self.split[1] * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create DataLoaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=self.collate_fn
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=self.collate_fn
        )
        
        # Return dataloaders and normalization parameters
        return train_dataloader, val_dataloader, test_dataloader, mean, std
