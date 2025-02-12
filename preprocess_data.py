
import torch
import pandas as pd
import numpy as np
import random
from pymatgen.core.composition import Composition as pmg_Composition
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
class PreprocessData:
    def __init__(self):
        self.sample_size = 30000
        self.split = [0.8, 0.1, 0.1]
        self.property_key = "BG"
        self.batch_size = 128
    def collate_fn(self, batch):
        # batch is a list of (element_ids, bandgap) pairs
        element_ids_batch, bandgaps_batch = zip(*batch)
        element_ids_batch = [torch.tensor(e, dtype=torch.long) for e in element_ids_batch]
        bandgaps_batch = torch.tensor(bandgaps_batch, dtype=torch.float32)

        # Pad the element sequences
        padded_element_ids = pad_sequence(element_ids_batch, batch_first=True, padding_value=0)

        return padded_element_ids, bandgaps_batch


    def get_atomic_number_from_ele(self, ele):
        return pmg_Composition(ele).elements[0].Z
    def formula_to_id_seq(self, formula):
        ele_dict = pmg_Composition(formula).get_el_amt_dict()
        seq = []
        for ele in ele_dict.keys():
            seq+=[self.get_atomic_number_from_ele(ele)]*int(ele_dict[ele])
        return seq
    
    def normalize_target(self, dataset):
        bandgaps = [bg for seq, bg in dataset]
        mean = np.mean(bandgaps)
        std = np.std(bandgaps)
        for i in range(len(dataset)):
            seq, bg = dataset[i]
            dataset[i] = (seq, (bg - mean) / std)
        return dataset
    def nan_hook(self, dataset):
        for i in range(len(dataset)):
            seq, bg = dataset[i]
            if np.isnan(bg):
                print(f"Found nan at index {i}")
                print(seq)
                print(bg)
                break
    def preprocess_data(self):
        df = pd.read_csv("data/data.csv")
        # Clean up any weird formulas
        df = df[~df["formula"].isin(["nan","NaN","NAN"])].dropna(subset=["formula"])
        df['formula'] = df['formula'].apply(lambda x: str(x).replace("NaN", "Na1N"))
        bandgaps = df[self.property_key].values  
        formulas = df["formula"].tolist()
        # Build your dataset (list of (sequence, bandgap) pairs)
        dataset = []
        for formula, bg in zip(formulas, bandgaps):
            seq = self.formula_to_id_seq(formula)
            dataset.append((seq, bg))
        if self.sample_size is not None:
            sample_size = self.sample_size  # or however many you want
            subset = random.sample(dataset, sample_size)
            dataset = subset
        self.nan_hook(dataset)

        dataset = self.normalize_target(dataset)
      
        # --------------------------
        # 4) Split into train/val/test
        # --------------------------
        dataset_size = len(dataset)
        train_size = int(self.split[0]* dataset_size)   # 80% train
        val_size = int(self.split[1] * dataset_size)     # 10% val
        test_size = dataset_size - train_size - val_size  # 10% test

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # for reproducibility
        )

       

        # --------------------------
        # 5) Create DataLoaders
        # --------------------------
        train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=self.collate_fn,)
        val_dataloader   = DataLoader(val_dataset, batch_size=128, shuffle=False, collate_fn=self.collate_fn,)
        test_dataloader  = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=self.collate_fn, )
        return train_dataloader, val_dataloader, test_dataloader
