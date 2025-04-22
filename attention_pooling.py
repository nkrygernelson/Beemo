import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    """Pools element representations using learned attention weights."""
    def __init__(self, embedding_dim, num_heads=4):
        super(AttentionPooling, self).__init__()
        
        # Global context vector (learnable)
        self.context_vector = nn.Parameter(torch.randn(embedding_dim))
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Output projection
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, element_embeddings, element_weights=None, mask=None):
        """
        Args:
            element_embeddings: [batch_size, max_elements, embedding_dim]
            element_weights: [batch_size, max_elements] - optional stoichiometric weights
            mask: [batch_size, max_elements] - True for padding elements
        Returns:
            Attention-pooled representation [batch_size, embedding_dim]
        """
        batch_size = element_embeddings.shape[0]
        
        # Create a query from the context vector (expanded for batch)
        # Shape: [batch_size, 1, embedding_dim]
        query = self.context_vector.view(1, 1, -1).expand(batch_size, 1, -1)
        
        # Apply multi-head attention
        # Query: the context vector 
        # Key/Value: the element embeddings
        global_repr, attention_weights = self.attention(
            query, 
            element_embeddings, 
            element_embeddings,
            key_padding_mask=mask,
            need_weights=True,
            average_attn_weights=False  # Get all attention heads
        )
        
        # If stoichiometric weights provided, incorporate them
        if element_weights is not None:
            # Create weight mask (weights=0 for padding elements)
            if mask is not None:
                weight_mask = (~mask).float()
                element_weights = element_weights * weight_mask
            
            # Normalize the attention weights by the stoichiometric weights
            # Use only the first attention head for simplicity
            first_head_attn = attention_weights[:, 0, 0, :]  # [batch_size, max_elements]
            
            # Adjust attention with element weights (chemical prior)
            combined_weights = first_head_attn * element_weights
            combined_weights = combined_weights / combined_weights.sum(dim=-1, keepdim=True).clamp(min=1e-10)
            
            # Apply combined weights (batch matrix multiplication)
            # [batch_size, 1, max_elements] * [batch_size, max_elements, embedding_dim]
            weighted_sum = torch.bmm(combined_weights.unsqueeze(1), element_embeddings)
            global_repr = weighted_sum  # [batch_size, 1, embedding_dim]
        
        # Final projection and normalization
        global_repr = self.norm(global_repr.squeeze(1))
        global_repr = self.proj(global_repr)  # [batch_size, embedding_dim]
        
        return global_repr, attention_weights


class CrossAttentionPooling(nn.Module):
    """Pools element representations using multiple learnable query vectors."""
    def __init__(self, embedding_dim, num_queries=4, num_heads=4, dropout=0.1):
        super(CrossAttentionPooling, self).__init__()
        
        # Learnable query vectors (like "property detectors")
        self.query_vectors = nn.Parameter(torch.randn(num_queries, embedding_dim))
        
        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Combine multiple query outputs
        self.combine = nn.Sequential(
            nn.Linear(num_queries * embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU()
        )
        
    def forward(self, element_embeddings, element_weights=None, mask=None):
        """
        Args:
            element_embeddings: [batch_size, max_elements, embedding_dim]
            element_weights: [batch_size, max_elements] - optional chemical weights
            mask: [batch_size, max_elements] - True for padding elements
        Returns:
            Pooled representation [batch_size, embedding_dim]
        """
        batch_size = element_embeddings.shape[0]
        num_queries = self.query_vectors.shape[0]
        
        # Expand queries for batch dimension [num_queries, embedding_dim] -> [batch_size, num_queries, embedding_dim]
        queries = self.query_vectors.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Each query attends to the elements (cross-attention)
        attended_queries, _ = self.cross_attention(
            queries,  # Queries: the learnable property detectors
            element_embeddings,  # Keys: the element embeddings
            element_embeddings,  # Values: the element embeddings
            key_padding_mask=mask,
            need_weights=False
        )
        # attended_queries: [batch_size, num_queries, embedding_dim]
        
        # Optional: incorporate stoichiometric weights
        if element_weights is not None:
            # Use element_weights to modulate the cross-attention (implementation omitted for brevity)
            pass
            
        # Flatten and combine the attended features from all queries
        flat_queries = attended_queries.reshape(batch_size, -1)  # [batch_size, num_queries*embedding_dim]
        pooled = self.combine(flat_queries)  # [batch_size, embedding_dim]
        
        return pooled


class HierarchicalAttentionPooling(nn.Module):
    """
    Hierarchical pooling that first identifies local motifs (element groups),
    then pools these motifs to a global representation.
    """
    def __init__(self, embedding_dim, num_motifs=4, num_heads=4, dropout=0.1):
        super(HierarchicalAttentionPooling, self).__init__()
        
        # First level: identify motifs
        self.motif_queries = nn.Parameter(torch.randn(num_motifs, embedding_dim))
        self.motif_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Second level: pool motifs to global representation
        self.global_query = nn.Parameter(torch.randn(1, embedding_dim))
        self.global_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=1,  # Simpler attention for final pooling
            dropout=dropout,
            batch_first=True
        )
        
        # Normalization and projection
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, element_embeddings, element_weights=None, mask=None):
        """
        Args:
            element_embeddings: [batch_size, max_elements, embedding_dim]
            element_weights: [batch_size, max_elements] - optional chemical weights
            mask: [batch_size, max_elements] - True for padding elements
        Returns:
            Hierarchically pooled representation [batch_size, embedding_dim]
        """
        batch_size = element_embeddings.shape[0]
        
        # Level 1: Extract motifs
        motif_queries = self.motif_queries.unsqueeze(0).expand(batch_size, -1, -1)
        motifs, _ = self.motif_attention(
            motif_queries,
            element_embeddings,
            element_embeddings,
            key_padding_mask=mask,
            need_weights=False
        )
        motifs = self.norm1(motifs)
        
        # Level 2: Global pooling over motifs
        global_query = self.global_query.unsqueeze(0).expand(batch_size, -1, -1)
        global_repr, _ = self.global_attention(
            global_query,
            motifs,
            motifs,
            need_weights=False
        )
        global_repr = self.norm2(global_repr.squeeze(1))
        
        return global_repr


class GatedAttentionPooling(nn.Module):
    """
    Gated attention pooling that learns to selectively combine 
    element representations with adaptive importance weights.
    """
    def __init__(self, embedding_dim, hidden_dim=None):
        super(GatedAttentionPooling, self).__init__()
        if hidden_dim is None:
            hidden_dim = embedding_dim
            
        # Attention mechanism
        self.gate_nn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Optional projection
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, element_embeddings, element_weights=None, mask=None):
        """
        Args:
            element_embeddings: [batch_size, max_elements, embedding_dim]
            element_weights: [batch_size, max_elements] - optional chemical weights
            mask: [batch_size, max_elements] - True for padding elements
        Returns:
            Pooled representation [batch_size, embedding_dim]
        """
        # Compute attention gates for each element
        gates = self.gate_nn(element_embeddings)  # [batch_size, max_elements, 1]
        
        # Apply mask if provided
        if mask is not None:
            # Set attention to -inf for masked elements
            gates = gates.masked_fill(mask.unsqueeze(-1), -1e9)
        
        # Compute attention weights with softmax
        attention_weights = F.softmax(gates, dim=1)  # [batch_size, max_elements, 1]
        
        # Optional: incorporate stoichiometric element weights
        if element_weights is not None:
            # Create a proper shape for element_weights
            element_weights = element_weights.unsqueeze(-1)  # [batch_size, max_elements, 1]
            
            # Combine learned attention with chemical prior (element weights)
            combined_weights = attention_weights * element_weights
            combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True).clamp(min=1e-10)
            attention_weights = combined_weights
        
        # Apply attention weights to element embeddings
        weighted_embeddings = element_embeddings * attention_weights  # [batch_size, max_elements, embedding_dim]
        
        # Sum to get global representation
        global_repr = weighted_embeddings.sum(dim=1)  # [batch_size, embedding_dim]
        
        # Optional projection
        global_repr = self.proj(global_repr)
        
        return global_repr