"""
Aegis-CNI: Graph Attention Autoencoder (GAE) for Anomaly Detection

Stage A of the Aegis-CNI dual-stage pipeline. The GAE is trained exclusively
on benign provenance graphs. During inference, novel attack subgraphs produce
high reconstruction error — serving as a signature-less anomaly detector.

Architecture:
    Encoder: 2-layer GATConv → latent embeddings
    Decoder: Inner product + linear reconstruction
    Loss: MSE reconstruction loss (L_recon)

Inspired by CONTINUUM framework (ST-GNN autoencoder for APT detection).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

# Try importing PyG; fall back to pure-PyTorch if not available
try:
    from torch_geometric.nn import GATConv, GCNConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GraphAttentionEncoder(nn.Module):
    """
    2-layer Graph Attention Network encoder.
    Compresses node features into a low-dimensional latent space.
    """

    def __init__(self, in_channels: int = 16, hidden_channels: int = 32,
                 out_channels: int = 16, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if HAS_PYG:
            self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                                 dropout=dropout, concat=True)
            self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                                 dropout=dropout, concat=False)
        else:
            # Fallback: simple MLP-based encoder
            self.fc1 = nn.Linear(in_channels, hidden_channels)
            self.fc2 = nn.Linear(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x, edge_index):
        if HAS_PYG:
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.conv2(x, edge_index)
        else:
            x = F.elu(self.fc1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.fc2(x)
        return x


class GraphAttentionDecoder(nn.Module):
    """
    Reconstructs original node features from latent embeddings.
    Uses a learned linear projection + neighbor aggregation.
    """

    def __init__(self, latent_dim: int = 16, out_channels: int = 16):
        super().__init__()
        self.reconstruct = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ELU(),
            nn.Linear(latent_dim * 2, out_channels),
        )

    def forward(self, z):
        return self.reconstruct(z)


class GraphAttentionAutoencoder(nn.Module):
    """
    Complete GAE model for anomaly detection.

    Training: minimize reconstruction loss on benign graphs.
    Inference: high reconstruction error → anomalous nodes/subgraphs.
    """

    def __init__(self, in_channels: int = 16, hidden_channels: int = 32,
                 latent_dim: int = 16, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = GraphAttentionEncoder(
            in_channels, hidden_channels, latent_dim, heads, dropout
        )
        self.decoder = GraphAttentionDecoder(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z)
        return x_hat, z

    def compute_reconstruction_loss(self, x, x_hat):
        """Per-node reconstruction error."""
        return torch.mean((x - x_hat) ** 2, dim=1)

    def compute_anomaly_scores(self, x, edge_index):
        """
        Returns per-node anomaly scores and graph-level anomaly score.
        """
        self.eval()
        with torch.no_grad():
            x_hat, z = self.forward(x, edge_index)
            node_scores = self.compute_reconstruction_loss(x, x_hat)
            graph_score = torch.mean(node_scores).item()
        return node_scores.numpy(), graph_score, z.numpy()


class SimpleGAEDetector:
    """
    High-level wrapper that handles graph → tensor conversion,
    model inference, and anomaly score computation.

    Works with or without PyTorch Geometric installed.
    """

    def __init__(self, feature_dim: int = 16, hidden_dim: int = 32,
                 latent_dim: int = 16):
        self.model = GraphAttentionAutoencoder(
            in_channels=feature_dim,
            hidden_channels=hidden_dim,
            latent_dim=latent_dim,
        )
        self.feature_dim = feature_dim
        self.is_trained = False

        # Baseline stats for normalizing anomaly scores
        self.baseline_mean = 0.0
        self.baseline_std = 1.0

    def nodes_edges_to_tensors(
        self, nodes: list, edges: list
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Convert ProvenanceNode/ProvenanceEdge lists to PyTorch tensors.
        Returns: (node_features, edge_index, id_to_idx_map)
        """
        # Build node ID → index mapping
        id_to_idx = {n.id if hasattr(n, 'id') else n['id']: i for i, n in enumerate(nodes)}

        # Build feature matrix
        features = []
        for n in nodes:
            if hasattr(n, 'features'):
                f = n.features
            else:
                f = np.zeros(self.feature_dim, dtype=np.float32)
            if len(f) < self.feature_dim:
                f = np.pad(f, (0, self.feature_dim - len(f)))
            features.append(f[:self.feature_dim])

        x = torch.tensor(np.array(features, dtype=np.float32))

        # Build edge index
        src_ids, tgt_ids = [], []
        for e in edges:
            src = e.source if hasattr(e, 'source') else e['source']
            tgt = e.target if hasattr(e, 'target') else e['target']
            if src in id_to_idx and tgt in id_to_idx:
                src_ids.append(id_to_idx[src])
                tgt_ids.append(id_to_idx[tgt])

        if len(src_ids) == 0:
            # Add self-loops if no edges
            src_ids = list(range(len(nodes)))
            tgt_ids = list(range(len(nodes)))

        edge_index = torch.tensor([src_ids, tgt_ids], dtype=torch.long)

        return x, edge_index, id_to_idx

    def train_on_normal(self, normal_graphs: list, epochs: int = 50,
                        lr: float = 0.001):
        """
        Train the GAE on a list of normal (benign) graphs.
        Each graph is a tuple of (nodes, edges).
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        all_losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for nodes, edges in normal_graphs:
                x, edge_index, _ = self.nodes_edges_to_tensors(nodes, edges)

                optimizer.zero_grad()
                x_hat, z = self.model(x, edge_index)
                loss = F.mse_loss(x_hat, x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(1, len(normal_graphs))
            all_losses.append(avg_loss)

        # Set baseline from final epoch
        self.baseline_mean = np.mean(all_losses[-5:])
        self.baseline_std = max(np.std(all_losses[-5:]), 0.001)
        self.is_trained = True

        return all_losses

    def detect_anomalies(self, nodes: list, edges: list) -> Tuple[dict, float, np.ndarray]:
        """
        Run anomaly detection on a provenance graph.

        Returns:
            node_scores: dict mapping node_id → anomaly_score (0-1 normalized)
            graph_score: overall graph anomaly score (0-1)
            embeddings: latent node embeddings from the encoder
        """
        x, edge_index, id_to_idx = self.nodes_edges_to_tensors(nodes, edges)
        idx_to_id = {v: k for k, v in id_to_idx.items()}

        raw_scores, raw_graph_score, embeddings = self.model.compute_anomaly_scores(
            x, edge_index
        )

        # Normalize to 0-1 range using baseline stats
        if self.is_trained:
            normalized_scores = np.clip(
                (raw_scores - self.baseline_mean) / (self.baseline_std * 5), 0, 1
            )
            graph_score = float(np.clip(
                (raw_graph_score - self.baseline_mean) / (self.baseline_std * 3), 0, 1
            ))
        else:
            # Without training, use raw sigmoid
            normalized_scores = 1 / (1 + np.exp(-raw_scores * 2))
            graph_score = float(1 / (1 + np.exp(-raw_graph_score * 2)))

        node_scores = {}
        for idx, score in enumerate(normalized_scores):
            if idx in idx_to_id:
                node_scores[idx_to_id[idx]] = float(score)

        return node_scores, graph_score, embeddings

    def get_graph_embedding(self, nodes: list, edges: list) -> np.ndarray:
        """
        Get a fixed-size graph-level embedding by mean-pooling node embeddings.
        Used as input to the LSTM stage predictor.
        """
        x, edge_index, _ = self.nodes_edges_to_tensors(nodes, edges)
        self.model.eval()
        with torch.no_grad():
            _, z = self.model(x, edge_index)
            graph_emb = torch.mean(z, dim=0).numpy()
        return graph_emb

    def save_model(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'baseline_mean': self.baseline_mean,
            'baseline_std': self.baseline_std,
            'is_trained': self.is_trained,
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state'])
        self.baseline_mean = checkpoint['baseline_mean']
        self.baseline_std = checkpoint['baseline_std']
        self.is_trained = checkpoint['is_trained']
