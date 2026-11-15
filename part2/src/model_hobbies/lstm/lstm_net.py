from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from numpy.lib.stride_tricks import sliding_window_view

def set_torch_seed(seed: int = 42) -> None:
    """Fixa sementes para reprodutibilidade exata exigida pelo check_task2.py."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TorchLSTMNet(nn.Module):
    """Rede LSTM padrão com projeção linear final."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        eff_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=eff_dropout,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # pega estado do último passo temporal da janela
        return self.head(last_step).squeeze(-1)



def make_sliding_windows(X_2d, y_1d=None, seq_len=7):
    """
    Converts (N, F) → (N, seq_len, F) causal windows with left-padding.
    Same contract as before; ~100× faster (no Python loop).
    """
    n_samples, n_features = X_2d.shape
    if seq_len <= 1:
        X_3d = X_2d[:, np.newaxis, :].astype(np.float32, copy=False)
        return X_3d, (y_1d.astype(np.float32) if y_1d is not None else None)

    pad = np.repeat(X_2d[:1], seq_len - 1, axis=0).astype(np.float32, copy=False)
    X_padded = np.concatenate([pad, X_2d.astype(np.float32, copy=False)], axis=0)

    # (N, F, seq_len) view → (N, seq_len, F); copy so the result is contiguous
    X_3d = sliding_window_view(X_padded, seq_len, axis=0).transpose(0, 2, 1)
    X_3d = np.ascontiguousarray(X_3d)

    y_out = y_1d.astype(np.float32) if y_1d is not None else None
    return X_3d, y_out


def train_torch_lstm(
    X_3d: np.ndarray,
    y_1d: np.ndarray,
    hidden_size: int = 32,
    num_layers: int = 1,
    dropout: float = 0.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 40,
    batch_size: int = 64,
    seed: int = 42,
    device: str | None = None,
) -> TorchLSTMNet:
    """Treina uma instância de TorchLSTMNet e a devolve em modo eval()."""
    set_torch_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = TorchLSTMNet(
        input_size=X_3d.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(dev)

    dataset = TensorDataset(torch.from_numpy(X_3d), torch.from_numpy(y_1d))
    # shuffle=False preserva a ordem temporal nos batches
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    return model.cpu()