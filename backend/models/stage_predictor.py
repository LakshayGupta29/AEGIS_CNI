"""
Aegis-CNI: LSTM Kill Chain Stage Predictor

Stage B of the Aegis-CNI dual-stage pipeline. Takes sequential graph-level
embeddings from the GAE encoder and predicts the adversary's current position
in the MITRE ATT&CK kill chain.

Architecture:
    Input: sequence of graph embeddings [g_1, g_2, ..., g_t]
    Model: 2-layer LSTM → fully connected → softmax over 8 stages
    Loss: Cross-entropy + temporal contrastive loss (reduces prediction flip rate)

Inspired by StageFinder (GLOBECOM 2026).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


# MITRE ATT&CK ICS stage names
STAGE_NAMES = [
    "Normal Operations",
    "Initial Access",
    "Execution",
    "Persistence",
    "Evasion",
    "Discovery",
    "Lateral Movement",
    "Impact",
]

STAGE_COLORS = [
    "#22c55e", "#eab308", "#f97316", "#f97316",
    "#ef4444", "#ef4444", "#dc2626", "#991b1b",
]


class LSTMStagePredictor(nn.Module):
    """
    2-layer LSTM that maps sequences of graph embeddings to MITRE ATT&CK
    stage probability distributions.

    (h_t, c_t) = f_LSTM(g_t, h_{t-1}, c_{t-1})
    p_t = softmax(W_stage * h_t + b_stage)
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64,
                 num_stages: int = 8, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_stages = num_stages

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_stages),
        )

    def forward(self, x: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Args:
            x: (batch, seq_len, input_dim) or (seq_len, input_dim) graph embeddings
            hidden: optional (h_0, c_0) tuple

        Returns:
            logits: (batch, seq_len, num_stages) raw stage predictions
            hidden: updated (h_n, c_n) state
            embeddings: (batch, seq_len, hidden_dim) LSTM hidden states
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension

        lstm_out, hidden = self.lstm(x, hidden)
        logits = self.classifier(lstm_out)

        return logits, hidden, lstm_out

    def predict_stage(self, x: torch.Tensor,
                      hidden: Optional[Tuple] = None):
        """
        Get the predicted stage probabilities for the latest time step.

        Returns:
            probs: (num_stages,) probability distribution
            predicted_stage: int — most likely stage
            confidence: float — probability of the predicted stage
            hidden: updated LSTM state
        """
        self.eval()
        with torch.no_grad():
            logits, hidden, _ = self.forward(x, hidden)
            # Take the last time step
            last_logits = logits[0, -1, :]  # (num_stages,)
            probs = F.softmax(last_logits, dim=0)
            predicted_stage = torch.argmax(probs).item()
            confidence = probs[predicted_stage].item()

        return probs.numpy(), predicted_stage, confidence, hidden


class TemporalContrastiveLoss(nn.Module):
    """
    Temporal contrastive loss component from StageFinder.

    Forces representational consistency across adjacent time steps,
    reducing the Temporal Flip Rate (TFR) — the tendency of the model
    to rapidly oscillate between predicted stages.

    L_ctr = -1/(T-1) * sum_t log(exp(sim(h_t, g_{t+1})/tau) /
                                  sum_{g- in N} exp(sim(h_t, g-)/tau))
    """

    def __init__(self, temperature: float = 0.07, num_negatives: int = 8):
        super().__init__()
        self.temperature = temperature
        self.num_negatives = num_negatives

    def forward(self, hidden_states: torch.Tensor,
                graph_embeddings: torch.Tensor):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_dim) LSTM outputs
            graph_embeddings: (batch, seq_len, input_dim) input graph embeds

        Returns:
            contrastive_loss: scalar
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        if seq_len < 2:
            return torch.tensor(0.0)

        total_loss = 0.0
        count = 0

        for t in range(seq_len - 1):
            h_t = hidden_states[:, t, :]          # (batch, hidden_dim)
            g_next = graph_embeddings[:, t + 1, :]  # (batch, input_dim)

            # Project g_next to hidden_dim if needed
            if g_next.shape[-1] != hidden_dim:
                g_next = F.adaptive_avg_pool1d(
                    g_next.unsqueeze(1), hidden_dim
                ).squeeze(1)

            # Positive similarity
            pos_sim = F.cosine_similarity(h_t, g_next, dim=-1) / self.temperature

            # Negative samples: random other time steps
            neg_sims = []
            for _ in range(self.num_negatives):
                neg_t = torch.randint(0, seq_len, (1,)).item()
                if neg_t == t + 1:
                    neg_t = (neg_t + 1) % seq_len
                g_neg = graph_embeddings[:, neg_t, :]
                if g_neg.shape[-1] != hidden_dim:
                    g_neg = F.adaptive_avg_pool1d(
                        g_neg.unsqueeze(1), hidden_dim
                    ).squeeze(1)
                neg_sim = F.cosine_similarity(h_t, g_neg, dim=-1) / self.temperature
                neg_sims.append(neg_sim)

            # InfoNCE loss
            neg_sims = torch.stack(neg_sims, dim=-1)  # (batch, num_neg)
            all_sims = torch.cat([pos_sim.unsqueeze(-1), neg_sims], dim=-1)
            labels = torch.zeros(batch_size, dtype=torch.long, device=h_t.device)
            loss = F.cross_entropy(all_sims, labels)

            total_loss += loss
            count += 1

        return total_loss / max(count, 1)


class StagePredictor:
    """
    High-level wrapper for LSTM-based MITRE ATT&CK stage prediction.

    Maintains internal LSTM state across time steps for continuous
    inference during the simulation.
    """

    def __init__(self, embedding_dim: int = 16, hidden_dim: int = 64,
                 num_stages: int = 8):
        self.model = LSTMStagePredictor(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_stages=num_stages,
        )
        self.hidden_state = None
        self.embedding_history: List[np.ndarray] = []
        self.prediction_history: List[int] = []
        self.is_trained = False

    def reset_state(self):
        """Reset LSTM hidden state for a new simulation."""
        self.hidden_state = None
        self.embedding_history = []
        self.prediction_history = []

    def predict(self, graph_embedding: np.ndarray) -> dict:
        """
        Given a graph-level embedding from the GAE, predict the current
        MITRE ATT&CK stage.

        Returns dict with:
            - stages: list of {name, probability, color, active} for each stage
            - currentStage: int — index of most likely stage
            - confidence: float — probability of predicted stage
        """
        self.embedding_history.append(graph_embedding)

        # Use last N embeddings for temporal context
        window = min(len(self.embedding_history), 10)
        sequence = np.array(self.embedding_history[-window:], dtype=np.float32)
        x = torch.tensor(sequence).unsqueeze(0)  # (1, seq_len, dim)

        probs, predicted_stage, confidence, self.hidden_state = \
            self.model.predict_stage(x, self.hidden_state)

        self.prediction_history.append(predicted_stage)

        # Build stage info list
        stages = []
        for i in range(len(STAGE_NAMES)):
            stages.append({
                "id": i,
                "name": STAGE_NAMES[i],
                "probability": float(probs[i]),
                "color": STAGE_COLORS[i],
                "active": i <= predicted_stage,
            })

        return {
            "stages": stages,
            "currentStage": predicted_stage,
            "confidence": confidence,
        }

    def train_on_sequences(self, sequences: List[List[np.ndarray]],
                           labels: List[List[int]], epochs: int = 30,
                           lr: float = 0.001):
        """
        Train the LSTM on labeled sequences.

        Args:
            sequences: list of embedding sequences (each is a list of np arrays)
            labels: list of stage label sequences (each is a list of ints)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        ce_loss_fn = nn.CrossEntropyLoss()
        contrastive_loss_fn = TemporalContrastiveLoss()

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for seq, lbl in zip(sequences, labels):
                x = torch.tensor(np.array(seq, dtype=np.float32)).unsqueeze(0)
                y = torch.tensor(lbl, dtype=torch.long).unsqueeze(0)

                optimizer.zero_grad()
                logits, _, lstm_hidden_states = self.model(x)

                # Cross-entropy loss
                ce_loss = ce_loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

                # Temporal contrastive loss
                ctr_loss = contrastive_loss_fn(lstm_hidden_states, x)

                # Combined loss (lambda = 0.3 for contrastive)
                loss = ce_loss + 0.3 * ctr_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        self.is_trained = True

    def save_model(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'is_trained': self.is_trained,
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state'])
        self.is_trained = checkpoint['is_trained']
