import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_pooling import (
    AttentionPooling, 
    CrossAttentionPooling, 
    HierarchicalAttentionPooling, 
    GatedAttentionPooling
)

class ElementEmbedding(nn.Module):
    """Embeds elements based on their atomic numbers."""
    def __init__(self, num_elements=118, embedding_dim=128):
        super(ElementEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_elements + 1, embedding_dim)  # +1 for padding
    
    def forward(self, element_ids):
        """
        Args:
            element_ids: tensor of element atomic numbers [batch_size, max_elements]
        Returns:
            Embedded element representations [batch_size, max_elements, embedding_dim]
        """
        return self.embedding(element_ids)


class FidelityEmbedding(nn.Module):
    """Embeds the fidelity of the data."""
    def __init__(self, num_fidelities = 6, embedding_dim=16):
        super(FidelityEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_fidelities, embedding_dim)
    
    def forward(self, fidelity):
        """
        Args:
            fidelity: tensor of data fidelity [batch_size]
        Returns:
            Embedded fidelity representations [batch_size, embedding_dim]
        """
        return self.embedding(fidelity)

class SetAttentionBlock(nn.Module):
    """Permutation-invariant self-attention block for processing sets."""
    def __init__(self, embedding_dim, num_heads=4, dropout=0.1):
        super(SetAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),  # Using GELU instead of ReLU for better performance
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input embeddings [batch_size, set_size, embedding_dim]
            mask: Optional attention mask [batch_size, set_size]
        Returns:
            Processed embeddings [batch_size, set_size, embedding_dim]
        """
        # Self-attention with pre-norm (better training stability)
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, key_padding_mask=mask, need_weights=False)
        x = x + residual
        
        # Feed-forward network with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x

class WeightedPooling(nn.Module):
    """Pools element representations using their fractional weights."""
    def __init__(self, embedding_dim):
        super(WeightedPooling, self).__init__()
        # Optional: learnable parameters to adjust the importance of each dimension
        self.importance = nn.Parameter(torch.ones(embedding_dim))
        
    def forward(self, element_embeddings, element_weights, mask=None):
        """
        Args:
            element_embeddings: [batch_size, max_elements, embedding_dim]
            element_weights: [batch_size, max_elements] - fractional weights of each element
            mask: [batch_size, max_elements] - True for padding elements
        Returns:
            Weighted pooled representation [batch_size, embedding_dim]
        """
        # Apply mask if provided
        if mask is not None:
            # Create a float mask (1.0 for real elements, 0.0 for padding)
            float_mask = (~mask).float()
            # Ensure weights are zero for padding elements
            element_weights = element_weights * float_mask
        
        # Apply learned importance to embeddings
        weighted_embeddings = element_embeddings * self.importance
        
        # Perform weighted sum (batch matrix multiplication)
        # [batch_size, 1, max_elements] * [batch_size, max_elements, embedding_dim]
        weighted_sum = torch.bmm(element_weights.unsqueeze(1), weighted_embeddings).squeeze(1)
        
        # Normalize by the sum of weights
        weight_sums = element_weights.sum(dim=1, keepdim=True)
        # Avoid division by zero
        weight_sums = torch.clamp(weight_sums, min=1e-10)
        normalized_sum = weighted_sum / weight_sums
        
        return normalized_sum

class DeepSet(nn.Module):
    """Deep Sets architecture for permutation-invariant processing of element sets."""
    def __init__(self, embedding_dim, num_blocks=3, num_heads=4, dropout=0.1, pooling_type='attention'):
        super(DeepSet, self).__init__()
        self.blocks = nn.ModuleList([
            SetAttentionBlock(embedding_dim, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        
        # Choose pooling mechanism
        self.pooling_type = pooling_type
        if pooling_type == 'weighted':
            self.pooling = WeightedPooling(embedding_dim)
        elif pooling_type == 'attention':
            self.pooling = AttentionPooling(embedding_dim, num_heads=num_heads)
        elif pooling_type == 'cross_attention':
            self.pooling = CrossAttentionPooling(embedding_dim, num_queries=4, num_heads=num_heads)
        elif pooling_type == 'hierarchical':
            self.pooling = HierarchicalAttentionPooling(embedding_dim, num_motifs=4, num_heads=num_heads)
        elif pooling_type == 'gated':
            self.pooling = GatedAttentionPooling(embedding_dim)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
    def forward(self, element_embeddings, element_weights, mask=None):
        """
        Args:
            element_embeddings: [batch_size, max_elements, embedding_dim]
            element_weights: [batch_size, max_elements] - fractional weights
            mask: [batch_size, max_elements] - True for padding elements
        Returns:
            Global composition representation [batch_size, embedding_dim]
        """
        x = element_embeddings
        
        # Apply set attention blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Pool with selected mechanism to get the global representation
        if self.pooling_type == 'attention':
            global_repr, _ = self.pooling(x, element_weights, mask)
        else:
            global_repr = self.pooling(x, element_weights, mask)
        
        return global_repr

class PredictionHead(nn.Module):
    """MLP head for predicting bandgap from the global representation."""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1):
        super(PredictionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: Global composition representation [batch_size, input_dim]
        Returns:
            Predicted bandgap [batch_size]
        """
        return self.mlp(x).squeeze(-1)

class SetBasedBandgapModel(nn.Module):
    """Complete model for bandgap prediction using set-based representation."""
    def __init__(self, num_elements=118, embedding_dim=128, fidelity_dim=16,
                 num_blocks=3, num_heads=4, hidden_dim=128, dropout=0.1,
                 pooling_type='attention'):
        super(SetBasedBandgapModel, self).__init__()
        self.element_embedding = ElementEmbedding(num_elements, embedding_dim)
        self.fidelity_embedding = FidelityEmbedding(fidelity_dim, fidelity_dim)
        self.deep_set = DeepSet(embedding_dim+fidelity_dim, num_blocks, num_heads, dropout, pooling_type)
        self.prediction_head = PredictionHead(embedding_dim+fidelity_dim, hidden_dim, dropout)
        
    def forward(self, element_ids, fidelities, element_weights=None):
        # Create padding mask (True for padding elements)
        mask = (element_ids == 0)
        
        # Embed elements
        element_embeddings = self.element_embedding(element_ids)  # [batch_size, max_elements, embedding_dim]
        fidelity_embedding = self.fidelity_embedding(fidelities)  # [batch_size, fidelity_dim]
        
        # Expand fidelity embedding to add a dimension that matches max_elements
        expanded_fidelity = fidelity_embedding.unsqueeze(1).expand(-1, element_embeddings.size(1), -1)
        
        # Concatenate along the last dimension
        embeddings = torch.cat([element_embeddings, expanded_fidelity], dim=2)
        
        # If no weights provided, use uniform weights for non-padding elements
        if element_weights is None:
            element_weights = (~mask).float()
            # Normalize weights per example
            weight_sums = element_weights.sum(dim=1, keepdim=True)
            element_weights = element_weights / torch.clamp(weight_sums, min=1e-10)
        
        # Process with Deep Sets
        global_repr = self.deep_set(embeddings, element_weights, mask)
        
        # Predict bandgap
        bandgap = self.prediction_head(global_repr)
        
        return bandgap