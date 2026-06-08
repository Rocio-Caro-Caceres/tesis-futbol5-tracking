"""
train_lstm.py
=============
Entrenamiento de una red LSTM bidireccional con attention pooling para
clasificar ventanas temporales en {Pass, Duel, Foul}.

Arquitectura:
    Linear projection (F -> H)
        -> BiLSTM x L capas (H hidden, dropout)
        -> Attention pooling (masked por `valid_frames`)
        -> Dense (H*2 -> H) + ReLU + Dropout
        -> Dense (H -> n_classes)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import TARGET_LABELS, TrainConfig
from .io_utils import info, stage, warn, write_json


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------
class WindowDataset:
    """Envuelve (X_seq, sample_mask, frame_mask, y)."""

    def __init__(self, X: np.ndarray, sample_mask: np.ndarray, frame_mask: np.ndarray | None, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.sample_mask = sample_mask.astype(np.bool_)
        self.frame_mask = frame_mask.astype(np.bool_) if frame_mask is not None else np.ones((len(X), X.shape[1]), dtype=bool)
        self.y = y.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.frame_mask[idx], self.y[idx]


def _build_loaders(
    X: np.ndarray, meta: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
    batch_size: int, frame_mask: np.ndarray | None = None,
) -> tuple[Any, Any, int]:
    import torch  # type: ignore
    from torch.utils.data import DataLoader

    y = meta["label_idx"].to_numpy()
    keep = y >= 0
    if not keep.all():
        warn(f"Filtrando {(~keep).sum()} filas con label_idx<0")
        X = X[keep]
        y = y[keep]
        meta = meta.iloc[np.where(keep)[0]].reset_index(drop=True)
        if frame_mask is not None:
            frame_mask = frame_mask[keep]
        train_idx = np.array([i for i, m in enumerate(keep) if m and i in set(train_idx)])
        val_idx   = np.array([i for i, m in enumerate(keep) if m and i in set(val_idx)])

    valid = (meta["valid_frames"].to_numpy() > 0)
    if not valid.all():
        info(f"{(~valid).sum()} eventos sin frames validos: se incluyen igual (mask los ignora)")

    fm_train = frame_mask[train_idx] if frame_mask is not None else None
    fm_val   = frame_mask[val_idx]   if frame_mask is not None else None

    train_ds = WindowDataset(X[train_idx], valid[train_idx], fm_train, y[train_idx])
    val_ds   = WindowDataset(X[val_idx],   valid[val_idx],   fm_val,   y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, int(X.shape[2])


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def _build_model(n_features: int, n_classes: int, cfg: TrainConfig, device: str):
    import torch
    import torch.nn as nn

    class EventLSTM(nn.Module):
        def __init__(self, n_features, n_classes, hidden, layers, dropout):
            super().__init__()
            self.proj = nn.Linear(n_features, hidden)
            self.lstm = nn.LSTM(
                input_size=hidden,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
                bidirectional=True,
            )
            self.attn = nn.Linear(2 * hidden, 1)
            self.head = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, frame_mask=None):
            # x: (B, T, F)
            h = self.proj(x)              # (B, T, H)
            h = self.dropout(h)
            h, _ = self.lstm(h)           # (B, T, 2H)
            a = self.attn(h).squeeze(-1)  # (B, T)
            # Mask: frame_mask es (B, T) bool. True = frame con datos.
            if frame_mask is None:
                frame_present = (x.abs().sum(dim=-1) > 0).float()
            else:
                frame_present = frame_mask.float()
            a = a.masked_fill(frame_present == 0, -1e9)
            w = torch.softmax(a, dim=-1)
            pooled = torch.einsum("bt,btd->bd", w, h)
            return self.head(pooled)

    model = EventLSTM(
        n_features=n_features,
        n_classes=n_classes,
        hidden=cfg.lstm_hidden,
        layers=cfg.lstm_layers,
        dropout=cfg.dropout,
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def _evaluate(model, loader, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    model.eval()
    losses, all_y, all_pred = [], [], []
    with torch.no_grad():
        for x, frame_mask, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x, frame_mask=frame_mask.to(device))
            loss = F.cross_entropy(logits, y, reduction="sum")
            losses.append(loss.item())
            all_y.append(y.cpu().numpy())
            all_pred.append(logits.argmax(dim=-1).cpu().numpy())
    if not all_y:
        return 0.0, 0.0, np.zeros(0), np.zeros(0)
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)
    avg_loss = float(np.sum(losses) / len(y_true))
    from sklearn.metrics import accuracy_score, f1_score
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return avg_loss, f1, y_true, y_pred


def train_lstm(
    X_seq: np.ndarray,
    meta_df: pd.DataFrame,
    cfg: TrainConfig,
    frame_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Entrena el LSTM y devuelve metricas + path del modelo."""
    import torch
    import torch.nn.functional as F

    if X_seq.shape[0] != len(meta_df):
        raise ValueError("X_seq y meta_df desalineados")
    device = _resolve_device(cfg.device)
    info(f"LSTM device: {device}")

    # Reusar el split por match
    from .train_xgb import split_by_match, compute_metrics
    train_idx, val_idx = split_by_match(meta_df, cfg.val_fraction, cfg.split_seed)

    train_loader, val_loader, n_features = _build_loaders(
        X_seq, meta_df, train_idx, val_idx, cfg.batch_size, frame_mask=frame_mask
    )
    model = _build_model(n_features, len(TARGET_LABELS), cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "lstm_model.pt"

    # Pesos por clase
    if cfg.class_weights is not None:
        cw = torch.tensor(cfg.class_weights, dtype=torch.float32, device=device)
    else:
        cw = None

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        ep_loss, n_batches = 0.0, 0
        for x, frame_mask, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            logits = model(x, frame_mask=frame_mask.to(device))
            loss = F.cross_entropy(logits, y, weight=cw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += float(loss.item())
            n_batches += 1
        train_loss = ep_loss / max(1, n_batches)
        val_loss, val_f1, _, _ = _evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        history.append({
            "epoch":     epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "val_f1":     val_f1,
        })
        info(f"  epoch {epoch:02d}/{cfg.epochs}  train_loss={train_loss:.4f}  "
             f"val_loss={val_loss:.4f}  val_f1={val_f1:.3f}")
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience:
                info(f"Early stopping en epoch {epoch}")
                break

    train_time = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({
        "state_dict":   model.state_dict(),
        "n_features":   n_features,
        "n_classes":    len(TARGET_LABELS),
        "config":       {k: getattr(cfg, k) for k in
                         ("lstm_hidden", "lstm_layers", "dropout", "batch_size", "lr")},
        "label_order":  list(TARGET_LABELS),
    }, model_path)
    info(f"LSTM entrenado en {train_time:.1f}s -> {model_path}")

    # Prediccion final
    t0 = time.perf_counter()
    val_loss, val_f1, y_val, y_pred = _evaluate(model, val_loader, device)
    n_val = max(1, len(y_val))
    infer_ms = (time.perf_counter() - t0) * 1000 / n_val

    metrics = compute_metrics(y_val, y_pred)
    metrics["train_time_s"] = train_time
    metrics["inference_ms_per_sample"] = float(infer_ms)
    metrics["model_path"]   = str(model_path)
    metrics["history"]      = history
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Entrena LSTM para Etapa 4")
    p.add_argument("--features-dir", default="ETAPA4_data/features")
    p.add_argument("--output-dir",   default="ETAPA4_data/models")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--hidden",       type=int,   default=128)
    p.add_argument("--layers",       type=int,   default=2)
    p.add_argument("--dropout",      type=float, default=0.3)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default="auto")
    args = p.parse_args(argv)

    cfg = TrainConfig(
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lstm_hidden=args.hidden,
        lstm_layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        device=args.device,
        output_dir=Path(args.output_dir),
    )

    fd = Path(args.features_dir)
    X_seq = np.load(fd / "X_seq.npy")
    meta  = pd.read_parquet(fd / "meta.parquet")
    mask_path = fd / "mask_seq.npy"
    frame_mask = np.load(mask_path) if mask_path.exists() else None
    info(f"Cargados X_seq={X_seq.shape}, meta={len(meta)} filas"
         + (f", mask_seq={frame_mask.shape}" if frame_mask is not None else ", sin mask_seq"))

    with stage("train-lstm"):
        metrics = train_lstm(X_seq, meta, cfg, frame_mask=frame_mask)

    write_json(Path(args.output_dir) / "lstm_metrics.json", metrics)
    info(f"LSTM accuracy={metrics['accuracy']:.3f}  f1_macro={metrics['f1_macro']:.3f}")
    return 0
