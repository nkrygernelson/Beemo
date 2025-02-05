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
    
class MotifDiscovery(nn.Module):
    def __init__(self, embedding_dim, num_queries):
        super(MotifDiscovery, self).__init__()
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, embedding_dim))
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first=True)
    
    def forward(self, element_embeddings):
        # Self-attention between queries and elements
        batch_size = element_embeddings.size(0)
        query_embeddings = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Add batch dimension
        attended_motifs, _ = self.attention(query_embeddings, element_embeddings, element_embeddings)
        return attended_motifs 
class HierarchicalAggregation(nn.Module):
    def __init__(self, embedding_dim):
        super(HierarchicalAggregation, self).__init__()
        self.global_context = nn.Parameter(torch.randn(1, embedding_dim))
        self.attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=4, batch_first = True)
    
    def forward(self, motif_embeddings):
        batch_size = motif_embeddings.size(0)

        # Insert batch dimension, make it shape [batch_size, 1, embed_dim]
        context = self.global_context.unsqueeze(0).expand(batch_size, 1, -1)
        # seq_len = 1 (one token), but for each example in the batch

        aggregated, _ = self.attention(context, motif_embeddings, motif_embeddings)
        # aggregated -> [batch_size, 1, embedding_dim]

        # Usually, we remove the extra "seq_len=1" dimension:
        return aggregated.squeeze(1)  # [batch_size, embedding_dim]

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
    def __init__(self, num_elements, embedding_dim, num_heads, num_layers, num_queries):
        super(BandgapPredictionModel, self).__init__()
        self.element_embedding = ElementEmbedding(num_elements, embedding_dim)
        self.attention_block = SelfAttentionBlock(embedding_dim, num_heads, num_layers)
        self.motif_discovery = MotifDiscovery(embedding_dim, num_queries)
        self.aggregation = HierarchicalAggregation(embedding_dim)
        self.prediction = PredictionMLP(embedding_dim)
    
    def forward(self, element_ids):
        embeddings = self.element_embedding(element_ids)  # Step 1
        mask = (element_ids == 0)
        attended_elements = self.attention_block(embeddings, src_key_padding_mask=mask)  # Step 2
        motifs = self.motif_discovery(attended_elements)  # Step 3
        global_representation = self.aggregation(motifs)  # Step 4
        bandgap = self.prediction(global_representation).squeeze(-1) # Step 5
        return bandgap
    
