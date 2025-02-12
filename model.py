import torch
import torch.nn as nn
import pandas as pd
from pymatgen.core.composition import Composition as pmg_Composition
from torch.utils.data import TensorDataset, DataLoader

class ElementEmbedding(nn.Module):
    def __init__(self, num_elements, embedding_dim):
        super(ElementEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_elements, embedding_dim)

    
    def forward(self, element_ids):
        return self.embedding(element_ids)
    
class SelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_layers, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
    
    def forward(self, embeddings, src_key_padding_mask=None):
        return self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)

class AttentionPooling(nn.Module):
    def __init__(self, embedding_dim):
        super(AttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(embedding_dim, 1)

    def forward(self, attended_elements):
        attention_scores = self.attention_weights(attended_elements)  # [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)  # Softmax across seq_len
        global_representation = (attention_weights * attended_elements).sum(dim=1)  # Weighted sum
        return global_representation


class PredictionMLP(nn.Module):
    def __init__(self, input_dim):
        super(PredictionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single output for bandgap
        )
    
    def forward(self, global_representation):
        return self.mlp(global_representation)
    
class BandgapPredictionModel(nn.Module):
    def __init__(self, num_elements, embedding_dim, num_heads, num_layers):
        super(BandgapPredictionModel, self).__init__()
        self.element_embedding = ElementEmbedding(num_elements, embedding_dim)
        self.attention_block = SelfAttentionBlock(embedding_dim, num_heads, num_layers)
        self.prediction = PredictionMLP(embedding_dim)  # No aggregation
    
    def forward(self, element_ids):
        embeddings = self.element_embedding(element_ids)  # Step 1
        mask = (element_ids == 0)
        attended_elements = self.attention_block(embeddings, src_key_padding_mask=mask)  # Step 2
        
        # Convert sequence of embeddings to a fixed-size vector
        mean_representation = attended_elements.mean(dim=1)  # Step 3
        
        bandgap = self.prediction(mean_representation).squeeze(-1)  # Step 4
        return bandgap
