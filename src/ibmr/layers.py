"""
Reusable neural network layers for the CLS memory system.

Includes:
    - GCNLayer: Edge-feature-aware graph convolutional layer
    - EdgeAwareGCN: Stacked GCN with residual connections and layer norm
    - PrototypeAttention: Multihead attention over learned prototype slots
    - MLP: Configurable multi-layer perceptron
    - ContrastiveLoss: NT-Xent loss for pattern embedding training

All layers are torch.compile-compatible and follow bf16/fp16 safety practices.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GCNLayer(nn.Module):
    """Edge-feature-aware Graph Convolutional Layer.

    Extends standard GCN convolution to incorporate edge features in
    message passing. This mitigates the heterophily problem in agricultural
    disease KGs (healthy vs. infected fields are neighbors but must remain
    distinct in representation space).

    Shape conventions:
        x:    (N, in_dim)           — Node features
        edge_index: (2, E)         — Edge adjacency (source, target)
        edge_attr:  (E, edge_dim)   — Edge features
        return:     (N, out_dim)    — Updated node features
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        edge_dim: int = 16,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_residual = use_residual and in_dim == out_dim
        self.use_layer_norm = use_layer_norm

        # Node feature transformation                                      # (in_dim,) → (out_dim,)
        self.node_proj = nn.Linear(in_dim, out_dim, bias=False)

        # Edge feature projection into message space
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, out_dim, bias=False),                      # (edge_dim,) → (out_dim,)
            nn.GELU(),
        )

        # Combine node + edge features for message
        self.msg_proj = nn.Linear(out_dim * 2, out_dim, bias=False)       # (2*out_dim,) → (out_dim,)

        # Optional layer norm and dropout
        if use_layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        # Learned gating for heterophily: how much to trust neighbor messages
        self.gate = nn.Sequential(
            nn.Linear(out_dim, 1),                                          # (out_dim,) → (1,)
            nn.Sigmoid(),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize weights with truncated normal (He-like) initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) — Node feature matrix.
            edge_index: (2, E) — COO edge indices [source, target].
            edge_attr: (E, edge_dim) — Edge feature matrix, optional.

        Returns:
            (N, out_dim) — Updated node features.
        """
        N = x.shape[0]                                                      # number of nodes

        # Transform source node features                                   # (N, out_dim)
        h = self.node_proj(x)

        # Message passing: aggregate neighbor info
        if edge_attr is not None and edge_index.shape[1] > 0:
            src, tgt = edge_index[0], edge_index[1]                        # (E,), (E,)

            # Source node features (sender)                                # (E, out_dim)
            h_src = h[src]

            # Edge features                                                # (E, out_dim)
            e = self.edge_proj(edge_attr)

            # Combine source + edge features into message                  # (E, out_dim)
            msg_input = torch.cat([h_src, e], dim=-1)                      # (E, 2*out_dim)
            msg = self.msg_proj(msg_input)                                 # (E, out_dim)

            # Heterophily gate: learn per-message how much to trust        # (E, 1)
            gate = self.gate(msg)

            # Gated scatter-add aggregation
            aggr = torch.zeros(N, self.out_dim, device=x.device, dtype=x.dtype)
            aggr = aggr.scatter_add(0, tgt.unsqueeze(-1).expand(-1, self.out_dim), gate * msg)

            # Degree normalization: average instead of sum
            deg = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
            deg = deg.scatter_add(0, tgt.unsqueeze(-1), torch.ones_like(gate))
            deg = deg.clamp(min=1)                                         # avoid div by zero
            aggr = aggr / deg                                              # (N, out_dim)
        else:
            aggr = torch.zeros_like(h)                                     # no edges → no aggregation

        # Combine self + neighbor with gating
        self_gate = self.gate(h)                                           # (N, 1)
        out = self_gate * h + (1 - self_gate) * aggr                       # (N, out_dim)

        # Residual connection
        if self.use_residual:
            out = out + x[:, :self.out_dim] if out.shape == x.shape else out + x

        # Layer norm + dropout
        if self.use_layer_norm:
            # Cast to float32 before LayerNorm for numerical safety.
            # Ensure norm params are also float32 to avoid mixed-dtype error
            # when model is in bf16/fp16 mode.
            out = F.layer_norm(
                out.float(),
                self.norm.normalized_shape,
                self.norm.weight.float() if self.norm.weight is not None else None,
                self.norm.bias.float() if self.norm.bias is not None else None,
                self.norm.eps,
            ).to(out.dtype)                                                  # (N, out_dim)

        out = self.dropout(out)                                            # (N, out_dim)
        return out


class EdgeAwareGCN(nn.Module):
    """Stacked GCN with residual connections and layer normalization.

    Stacks multiple GCNLayer instances with residual connections between
    them and optional global normalization.

    Shape conventions:
        x:           (N, in_dim)
        edge_index:  (2, E)
        edge_attr:   (E, edge_dim)  (optional)
        return:      (N, out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 3,
        edge_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers

        # Input projection
        self.input_proj = nn.Linear(in_dim, hidden_dim)                   # (in_dim,) → (hidden_dim,)

        # Stacked GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layers):
            layer_in = hidden_dim
            layer_out = hidden_dim if i < n_layers - 1 else out_dim
            self.gcn_layers.append(
                GCNLayer(
                    in_dim=layer_in,
                    out_dim=layer_out,
                    edge_dim=edge_dim,
                    dropout=dropout,
                    use_layer_norm=True,
                    use_residual=(i > 0),  # residual from previous hidden
                )
            )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) — Node features.
            edge_index: (2, E) — Edge adjacencies.
            edge_attr: (E, edge_dim) — Edge features, optional.

        Returns:
            (N, out_dim) — Final node embeddings after all GCN layers.
        """
        h = self.input_proj(x)                                             # (N, hidden_dim)
        h = self.dropout(h)

        for gcn in self.gcn_layers:
            h = gcn(h, edge_index, edge_attr)                              # (N, hidden_dim) or (N, out_dim)

        return h


class PrototypeAttention(nn.Module):
    """Multihead attention over learned prototype pattern slots.

    Attends over a set of learnable prototype vectors given a subgraph
    encoding query. This implements the "pattern matching" mechanism
    where episodic subgraphs are compared against learned semantic patterns.

    Shape conventions:
        query:      (B, 1, P)        — Subgraph encoding projected to pattern space
        prototypes: (S, P)           — S learned prototype slots, each of dim P
        return:     (B, P), (B, 1, S) — Attended pattern, attention weights
    """

    def __init__(self, pattern_dim: int, n_prototypes: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pattern_dim = pattern_dim
        self.n_prototypes = n_prototypes
        self.n_heads = n_heads

        # Learnable prototype vectors (semantic memory slots)              # (S, P)
        self.prototype_vectors = nn.Parameter(torch.randn(n_prototypes, pattern_dim))

        # Multihead cross-attention: query attends over prototypes as keys/values
        self.attention = nn.MultiheadAttention(
            embed_dim=pattern_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(pattern_dim, pattern_dim),                           # (P,) → (P,)
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize prototype vectors on the unit hypersphere."""
        with torch.no_grad():
            # Normalize prototypes to lie on unit sphere
            norm = self.prototype_vectors.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            self.prototype_vectors.data = self.prototype_vectors.data / norm

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, 1, P) — Projected subgraph encoding.

        Returns:
            pattern: (B, P) — Attended pattern embedding.
            weights: (B, 1, S) — Attention weights over prototypes (before softmax).
        """
        B = query.shape[0]
        orig_dtype = query.dtype

        # Expand prototypes for batch dimension                            # (B, S, P)
        prototypes = self.prototype_vectors.unsqueeze(0).expand(B, -1, -1)

        # Cross-attend: query attends over prototype slots.
        # Cast to float32 before softmax for numerical safety (bf16/fp16
        # softmax can be unstable). When the parent model was converted to
        # bf16/fp16 via .bfloat16() / .half(), the attention module's weights
        # are also lower-precision — temporarily promote to float32 for this
        # call so the explicit .float() casts on inputs are effective.
        query_f32 = query.float()
        prototypes_f32 = prototypes.float()

        needs_float32 = self.attention.in_proj_weight.dtype != torch.float32
        if needs_float32:
            self.attention = self.attention.float()

        pattern, weights = self.attention(query_f32, prototypes_f32, prototypes_f32)

        if needs_float32:
            # Restore attention to original dtype to match parent model
            self.attention = self.attention.to(orig_dtype)

        # pattern: (B, 1, P), weights: (B, 1, S)
        pattern = pattern.to(orig_dtype)                                   # (B, 1, P)
        weights = weights.to(orig_dtype)                                   # (B, 1, S)

        pattern = self.output_proj(pattern).squeeze(1)                     # (B, P)

        return pattern, weights


class MLP(nn.Module):
    """Configurable multi-layer perceptron with GELU activations.

    Shape conventions:
        x: (..., in_dim) → (..., out_dim)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_dim) → (..., out_dim)"""
        return self.net(x)


class SubgraphPooling(nn.Module):
    """Pool subgraph node embeddings to a single graph-level vector.

    Supports mean and max pooling, respecting padding masks for batched inputs.

    Shape conventions:
        x:    (B, N, D)  — Node features
        mask: (B, N)     — Boolean mask (True = valid node)
        return: (B, D)    — Pooled graph-level embedding
    """

    def __init__(self, pooling: str = "mean"):
        super().__init__()
        assert pooling in ("mean", "max"), f"Unsupported pooling: {pooling}"
        self.pooling = pooling

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) — Batched node features.
            mask: (B, N) — Boolean mask, True for valid nodes.

        Returns:
            (B, D) — Pooled graph-level embedding.
        """
        # Apply mask: zero out padding nodes
        x = x * mask.unsqueeze(-1)                                        # (B, N, D)

        if self.pooling == "mean":
            # Mean over valid nodes
            node_counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)     # (B, 1)
            pooled = x.sum(dim=1) / node_counts                           # (B, D)
        elif self.pooling == "max":
            # Set masked positions to a large negative value for max pooling
            x = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled = x.max(dim=1).values                                  # (B, D)

        return pooled


class ContrastiveLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    Pulls positive pairs together and pushes negative pairs apart in
    the pattern embedding space.

    Shape conventions:
        embeddings: (B, D) — Pattern embeddings
        labels:     (B,)   — Class labels; same label = positive pair

    Reference: Chen et al., "A Simple Framework for Contrastive Learning
    of Visual Representations", ICML 2020.
    """

    def __init__(self, temperature: float = 0.1, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D) — L2-normalized pattern embeddings.
            labels: (B,) — Class labels. If None, uses each sample as its own class.

        Returns:
            Scalar loss tensor.
        """
        B = embeddings.shape[0]

        # L2-normalize embeddings
        embeddings = F.normalize(embeddings.float(), dim=-1)              # (B, D)

        # Pairwise cosine similarity matrix
        sim = embeddings @ embeddings.T                                   # (B, B) symmetric

        # Scale by temperature
        sim = sim / self.temperature                                      # (B, B)

        # Create positive mask: same class = positive
        if labels is not None:
            pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)         # (B, B) boolean
        else:
            # Each sample is its own class (instance discrimination)
            pos_mask = torch.eye(B, device=embeddings.device, dtype=torch.bool)

        # Remove self-pairs from positive mask
        self_mask = torch.eye(B, device=embeddings.device, dtype=torch.bool)
        pos_mask = pos_mask & ~self_mask                                  # (B, B)

        # Numerical stability: subtract max per row
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()          # (B, B)

        # Compute loss: -log(sum(pos) / sum(all))
        exp_sim = sim.exp()                                               # (B, B)
        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)                 # (B,)
        all_sum = exp_sim.sum(dim=1) - exp_sim.diag()                     # (B,)  exclude self

        # Avoid log(0)
        pos_sum = pos_sum.clamp(min=1e-8)
        all_sum = all_sum.clamp(min=1e-8)

        loss = -(pos_sum / all_sum).log().mean()                          # scalar

        # Add margin-based hinge loss for additional separation
        if self.margin > 0 and labels is not None:
            neg_mask = ~pos_mask & ~self_mask                             # (B, B)
            neg_sim = sim[neg_mask].clamp(min=0)                         # (num_neg,)
            if neg_sim.numel() > 0:                                      # guard against empty (all-same-class)
                margin_loss = (self.margin - neg_sim).clamp(min=0).mean()
                loss = loss + 0.1 * margin_loss

        return loss.to(embeddings.dtype)


# Alias for backward compatibility
def contrastive_loss(
    embeddings: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    margin: float = 1.0,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Functional interface for ContrastiveLoss."""
    criterion = ContrastiveLoss(temperature=temperature, margin=margin)
    return criterion(embeddings, labels)
