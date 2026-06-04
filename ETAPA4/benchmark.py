"""
benchmark.py
============
Entrena ambos modelos (XGBoost y LSTM) con la misma particion y compara
sus metricas lado a lado. Pensado para la defensa de tesis: muestra que
vale la pena el costo del LSTM (si vale).

Métricas reportadas:
    * accuracy, F1 macro, precision/recall por clase
    * confusion matrix
    * tiempo de entrenamiento
    * latencia de inferencia (ms / muestra)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import TrainConfig
from .io_utils import info, stage, write_json


def run_benchmark(
    X_seq: np.ndarray,
    X_flat: np.ndarray,
    meta: pd.DataFrame,
    out_dir: Path,
    cfg: TrainConfig,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {"xgb": None, "lstm": None}

    # XGBoost
    from .train_xgb import train_xgb
    with stage("benchmark-xgb"):
        t0 = time.perf_counter()
        out["xgb"] = train_xgb(X_flat, meta, cfg)
        out["xgb"]["wallclock_s"] = time.perf_counter() - t0
    write_json(out_dir / "xgb_metrics.json", out["xgb"])

    # LSTM
    from .train_lstm import train_lstm
    with stage("benchmark-lstm"):
        t0 = time.perf_counter()
        out["lstm"] = train_lstm(X_seq, meta, cfg)
        out["lstm"]["wallclock_s"] = time.perf_counter() - t0
    write_json(out_dir / "lstm_metrics.json", out["lstm"])

    # Tabla resumen
    summary = {
        "xgb": {
            "accuracy":               out["xgb"]["accuracy"],
            "f1_macro":               out["xgb"]["f1_macro"],
            "wallclock_s":            out["xgb"]["wallclock_s"],
            "inference_ms_per_sample": out["xgb"]["inference_ms_per_sample"],
        },
        "lstm": {
            "accuracy":               out["lstm"]["accuracy"],
            "f1_macro":               out["lstm"]["f1_macro"],
            "wallclock_s":            out["lstm"]["wallclock_s"],
            "inference_ms_per_sample": out["lstm"]["inference_ms_per_sample"],
        },
    }
    write_json(out_dir / "benchmark_summary.json", summary)
    info("\n========== BENCHMARK ==========")
    info(f"XGBoost: acc={summary['xgb']['accuracy']:.3f}  f1={summary['xgb']['f1_macro']:.3f}  "
         f"train={summary['xgb']['wallclock_s']:.1f}s  "
         f"infer={summary['xgb']['inference_ms_per_sample']:.3f}ms/sample")
    info(f"LSTM:    acc={summary['lstm']['accuracy']:.3f}  f1={summary['lstm']['f1_macro']:.3f}  "
         f"train={summary['lstm']['wallclock_s']:.1f}s  "
         f"infer={summary['lstm']['inference_ms_per_sample']:.3f}ms/sample")
    info("================================")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Benchmark XGBoost vs LSTM para Etapa 4")
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
        output_dir=Path(args.output_dir),
    )

    fd = Path(args.features_dir)
    X_seq  = np.load(fd / "X_seq.npy")
    X_flat = np.load(fd / "X_flat.npy")
    meta   = pd.read_parquet(fd / "meta.parquet")
    info(f"Cargados X_seq={X_seq.shape}  X_flat={X_flat.shape}  meta={len(meta)}")

    run_benchmark(X_seq, X_flat, meta, Path(args.output_dir), cfg)
    return 0
