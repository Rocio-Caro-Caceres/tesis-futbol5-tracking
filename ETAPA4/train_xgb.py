"""
train_xgb.py
============
Entrenamiento del clasificador XGBoost para Etapa 4.

Por que XGBoost:
    * Rapido de entrenar (CPU friendly).
    * Feature importances nativas (util para la tesis).
    * Funciona bien con vectores fijos de features aplanados de una ventana.

Por que NO captura la dinamica temporal por si mismo:
    * Por eso el feature vector ya viene "aplanado con resumen" desde
      features.build_features (mean/std/min/max/last por columna + flat).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import TARGET_LABELS, TrainConfig
from .io_utils import info, stage, warn, write_json


# ---------------------------------------------------------------------------
# Split por match (nunca por frame: filtra)
# ---------------------------------------------------------------------------
def split_by_match(
    meta_df: pd.DataFrame,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Devuelve (train_idx, val_idx) barajando MATCHES (no filas). Asi no hay
    leakage entre particiones: un match esta 100% en train o 100% en val.
    """
    matches = np.array(sorted(meta_df["match_id"].unique().tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(matches)
    n_val = max(1, int(round(len(matches) * val_fraction)))
    val_matches = set(matches[:n_val].tolist())
    val_mask = meta_df["match_id"].isin(val_matches).to_numpy()
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    info(f"split: {len(train_idx)} train / {len(val_idx)} val "
         f"({len(matches) - n_val} matches train, {n_val} val)")
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Accuracy, F1 macro, precision/recall por clase, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support,
        confusion_matrix,
    )
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prf = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(TARGET_LABELS))), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(TARGET_LABELS))))
    return {
        "accuracy":   acc,
        "f1_macro":   f1,
        "per_class": {
            TARGET_LABELS[i]: {
                "precision": float(prf[0][i]),
                "recall":    float(prf[1][i]),
                "f1":        float(prf[2][i]),
                "support":   int(prf[3][i]),
            }
            for i in range(len(TARGET_LABELS))
        },
        "confusion_matrix": cm.tolist(),
        "label_order":     list(TARGET_LABELS),
    }


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
def train_xgb(
    X_flat: np.ndarray,
    meta_df: pd.DataFrame,
    cfg: TrainConfig,
) -> dict[str, Any]:
    """Entrena un XGBClassifier multiclase y devuelve metricas + path del modelo."""
    import xgboost as xgb  # type: ignore

    if X_flat.shape[0] != len(meta_df):
        raise ValueError(
            f"X_flat ({X_flat.shape[0]}) y meta_df ({len(meta_df)}) desalineados"
        )
    y = meta_df["label_idx"].to_numpy()
    if (y < 0).any():
        # Filtrar filas con label desconocida
        keep = y >= 0
        warn(f"Filtrando {(~keep).sum()} filas con label_idx<0")
        X_flat = X_flat[keep]
        y = y[keep]
        meta_df = meta_df.iloc[np.where(keep)[0]].reset_index(drop=True)

    train_idx, val_idx = split_by_match(meta_df, cfg.val_fraction, cfg.split_seed)
    X_train, X_val = X_flat[train_idx], X_flat[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Pesos por clase
    sample_weight = None
    if cfg.class_weights is not None:
        sample_weight = np.array([cfg.class_weights[c] for c in y_train], dtype=np.float32)
    else:
        # Balanceo automatico via sample_count / class_count
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight("balanced", y_train)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "xgb_model.json"

    params = {
        "objective":        "multi:softprob",
        "num_class":        len(TARGET_LABELS),
        "max_depth":        6,
        "eta":              0.1,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "tree_method":      "hist",
        "eval_metric":      "mlogloss",
        "verbosity":        1,
        "seed":             cfg.split_seed,
    }

    info(f"Entrenando XGBoost con {X_train.shape[0]} filas, {X_train.shape[1]} features")
    t0 = time.perf_counter()
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=20,
        verbose_eval=50,
    )
    booster.save_model(model_path)
    train_time = time.perf_counter() - t0
    info(f"XGBoost entrenado en {train_time:.1f}s -> {model_path}")

    # Prediccion
    t0 = time.perf_counter()
    y_pred = booster.predict(dval).argmax(axis=1)
    infer_ms = (time.perf_counter() - t0) * 1000 / max(1, len(y_val))

    metrics = compute_metrics(y_val, y_pred)
    metrics["train_time_s"]   = train_time
    metrics["inference_ms_per_sample"] = float(infer_ms)
    metrics["model_path"]     = str(model_path)
    metrics["best_iteration"] = getattr(booster, "best_iteration", None)

    # Feature importances
    importance = booster.get_score(importance_type="gain")
    top = sorted(importance.items(), key=lambda kv: -kv[1])[:30]
    metrics["top_30_feature_importance"] = [
        {"feature_idx": int(k.replace("f", "")), "gain": float(v)}
        for k, v in top
    ]
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Entrena XGBoost para Etapa 4")
    p.add_argument("--features-dir", default="ETAPA4_data/features",
                   help="Carpeta con X_seq.npy / X_flat.npy / meta.parquet")
    p.add_argument("--output-dir", default="ETAPA4_data/models",
                   help="Donde guardar el modelo y metricas")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    cfg = TrainConfig(
        val_fraction=args.val_fraction,
        split_seed=args.seed,
        output_dir=Path(args.output_dir),
    )

    fd = Path(args.features_dir)
    X_flat = np.load(fd / "X_flat.npy")
    meta = pd.read_parquet(fd / "meta.parquet")
    info(f"Cargados X_flat={X_flat.shape}, meta={len(meta)} filas")

    with stage("train-xgb"):
        metrics = train_xgb(X_flat, meta, cfg)

    write_json(Path(args.output_dir) / "xgb_metrics.json", metrics)
    info(f"XGB accuracy={metrics['accuracy']:.3f}  f1_macro={metrics['f1_macro']:.3f}")
    return 0
