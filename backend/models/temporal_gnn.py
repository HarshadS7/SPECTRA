"""
Temporal GNN — Multi-horizon forecasting model
================================================
Architecture: Dual GCNConv (spatial) → Bidirectional LSTM (temporal)
              → LayerNorm → Dropout → K per-horizon linear heads
Checkpoint  : temporal_gnn_v1.pth

Input : x  [N, T, F]       N nodes, T timesteps, F features
        edge_index [2, E]   graph connectivity

Output: forecasts [K, N, 1] for K horizons
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class TemporalGNN(nn.Module):
    """
    GCN encoder + bidirectional LSTM + multi-horizon forecast heads.

    Parameters
    ----------
    node_features : int
        Number of input features per node per timestep (default 2).
    hidden_dim : int
        Dimensionality of GCN output and LSTM hidden state.
    num_horizons : int
        How many future steps to predict (e.g. 1, 3, 5 days).
    dropout : float
        Dropout applied after the LSTM.
    """

    def __init__(
        self,
        node_features: int = 4,
        hidden_dim: int = 64,
        num_horizons: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons

        # Spatial encoder — two GCN layers
        self.gcn1 = GCNConv(node_features, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # Temporal encoder
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)  # unidirectional

        # Per-horizon linear heads
        self.horizon_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(num_horizons)]
        )

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, N, T, F] (batched) or [N, T, F] (single)
        edge_index : Tensor [2, E] or list of tensors for batched

        Returns
        -------
        forecasts : Tensor [B, K, N] (batched) or [K, N] (single)
        """
        if x.dim() == 3:
            return self._forward_single(x, edge_index).squeeze(-1)
        return self._forward_batch(x, edge_index)

    def _forward_single(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        num_nodes, seq_len, num_feats = x.size()

        x_flat = x.reshape(-1, num_feats)
        ei_expanded = self._expand_edge_index(edge_index, num_nodes, seq_len)

        h = self.relu(self.gcn1(x_flat, ei_expanded))
        h = self.relu(self.gcn2(h, ei_expanded))
        h = h.view(num_nodes, seq_len, self.hidden_dim)

        lstm_out, _ = self.lstm(h)
        lstm_out = self.layer_norm(lstm_out)
        context = self.dropout(lstm_out[:, -1, :])

        forecasts = torch.stack([head(context) for head in self.horizon_heads], dim=0)
        return forecasts

    def _forward_batch(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, seq_len, num_feats = x.size()

        x_flat = x.reshape(-1, num_feats)

        if isinstance(edge_index, list):
            ei_expanded = torch.cat(
                [
                    self._expand_edge_index(ei, num_nodes, seq_len)
                    + b * num_nodes * seq_len
                    for b, ei in enumerate(edge_index)
                ],
                dim=1,
            )
        else:
            ei_expanded = self._expand_edge_index(edge_index, num_nodes, seq_len)
            ei_expanded = torch.cat(
                [ei_expanded + b * num_nodes * seq_len for b in range(batch_size)],
                dim=1,
            )

        h = self.relu(self.gcn1(x_flat, ei_expanded))
        h = self.relu(self.gcn2(h, ei_expanded))
        h = h.view(batch_size, num_nodes, seq_len, self.hidden_dim)
        h = h.view(batch_size * num_nodes, seq_len, self.hidden_dim)

        lstm_out, _ = self.lstm(h)
        lstm_out = self.layer_norm(lstm_out)
        context = self.dropout(lstm_out[:, -1, :])
        context = context.view(batch_size, num_nodes, -1)

        forecasts = torch.stack([head(context) for head in self.horizon_heads], dim=1)
        return forecasts.squeeze(-1)

    @staticmethod
    def _expand_edge_index(
        edge_index: torch.Tensor, num_nodes: int, seq_len: int
    ) -> torch.Tensor:
        offsets = torch.arange(seq_len, device=edge_index.device) * num_nodes
        ei_list = [edge_index + o for o in offsets]
        return torch.cat(ei_list, dim=1)

    def forecast_single(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        return self._forward_single(x, edge_index).squeeze(-1)
