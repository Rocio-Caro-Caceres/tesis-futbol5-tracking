"""
ETAPA4/__main__.py
==================
CLI unificado del paquete Etapa 4. Subcomandos:

    python -m ETAPA4 download   [args]   -> baja Labels + tracking
    python -m ETAPA4 normalize  [args]   -> SoccerNet -> Parquet (y opcional DB)
    python -m ETAPA4 features   [args]   -> ventanas temporales -> X_seq/X_flat
    python -m ETAPA4 train-xgb  [args]   -> entrena XGBoost
    python -m ETAPA4 train-lstm [args]   -> entrena LSTM
    python -m ETAPA4 benchmark  [args]   -> entrena ambos y compara
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ETAPA4",
        description="Etapa 4: Automatizacion de eventos complejos (Pases/Duelos/Faltas) via ML",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- download
    dl = sub.add_parser("download", help="Descargar Labels-v2 + tracking de SoccerNet (sin video)")
    dl.add_argument("--output-dir", default="data/soccernet")
    dl.add_argument("--splits", nargs="+", default=["train"],
                    choices=["train", "valid", "test", "challenge"])
    dl.add_argument("--soccernet-password", default="s0cc3rn3t",
                    help="Password academica por default. Overrideable via SOCCERNET_PASSWORD.")
    dl.add_argument("--tracking-format", default="mot", choices=["mot", "jsonl"],
                    help="Formato de tracking: 'mot' (MOT20 CSV, default) o 'jsonl' (legacy).")

    # --- normalize
    nm = sub.add_parser("normalize", help="SoccerNet -> Parquet atomico (tracking_events shape)")
    nm.add_argument("--raw-dir", default=None,
                    help="Carpeta con SoccerNet ya bajado. Si se omite, usa --output-dir de download.")
    nm.add_argument("--output-dir", default="ETAPA4_data/normalized")
    nm.add_argument("--ingest", action="store_true",
                    help="Ademas de normalizar, ingestar en la DB (tracking_events + event_training_labels)")
    nm.add_argument("--match-id", default=None,
                    help="Filtrar ingesta a un solo match_id")
    nm.add_argument("--tracking-format", default="mot", choices=["mot", "jsonl"],
                    help="Formato de tracking: 'mot' (MOT20 CSV) o 'jsonl' (legacy).")

    # --- features
    ft = sub.add_parser("features", help="Ventanas temporales -> X_seq.npy / X_flat.npy")
    ft.add_argument("--normalized-dir", default="ETAPA4_data/normalized")
    ft.add_argument("--output-dir", default="ETAPA4_data/features")
    ft.add_argument("--window", type=int, default=25,
                    help="W: frames a cada lado del evento (total = 2W+1)")
    ft.add_argument("--top-k", type=int, default=4)

    # --- train-xgb
    tx = sub.add_parser("train-xgb", help="Entrena XGBoost")
    tx.add_argument("--features-dir", default="ETAPA4_data/features")
    tx.add_argument("--output-dir",   default="ETAPA4_data/models")
    tx.add_argument("--val-fraction", type=float, default=0.2)
    tx.add_argument("--seed",         type=int,   default=42)

    # --- train-lstm
    tl = sub.add_parser("train-lstm", help="Entrena LSTM")
    tl.add_argument("--features-dir", default="ETAPA4_data/features")
    tl.add_argument("--output-dir",   default="ETAPA4_data/models")
    tl.add_argument("--epochs",       type=int,   default=30)
    tl.add_argument("--batch-size",   type=int,   default=64)
    tl.add_argument("--lr",           type=float, default=1e-3)
    tl.add_argument("--hidden",       type=int,   default=128)
    tl.add_argument("--layers",       type=int,   default=2)
    tl.add_argument("--dropout",      type=float, default=0.3)
    tl.add_argument("--val-fraction", type=float, default=0.2)
    tl.add_argument("--seed",         type=int,   default=42)
    tl.add_argument("--device",       default="auto")

    # --- benchmark
    bm = sub.add_parser("benchmark", help="Entrena XGB y LSTM y compara")
    bm.add_argument("--features-dir", default="ETAPA4_data/features")
    bm.add_argument("--output-dir",   default="ETAPA4_data/models")
    bm.add_argument("--epochs",       type=int,   default=30)
    bm.add_argument("--batch-size",   type=int,   default=64)
    bm.add_argument("--lr",           type=float, default=1e-3)
    bm.add_argument("--hidden",       type=int,   default=128)
    bm.add_argument("--layers",       type=int,   default=2)
    bm.add_argument("--dropout",      type=float, default=0.3)
    bm.add_argument("--val-fraction", type=float, default=0.2)
    bm.add_argument("--seed",         type=int,   default=42)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    cmd = args.cmd

    if cmd == "download":
        from .config import DownloadConfig
        from .soccernet_io import download
        cfg = DownloadConfig(
            output_dir=Path(args.output_dir),
            split=tuple(args.splits),
            password=args.soccernet_password,
            tracking_format=args.tracking_format,
        )
        download(cfg)
        return 0

    if cmd == "normalize":
        from .normalize import main as normalize_main
        ns = [
            "--output-dir", str(args.output_dir),
            "--tracking-format", str(args.tracking_format),
        ]
        if args.raw_dir:
            ns += ["--raw-dir", str(args.raw_dir)]
        if args.ingest:
            ns += ["--ingest"]
        if args.match_id:
            ns += ["--match-id", str(args.match_id)]
        return normalize_main(ns)

    if cmd == "features":
        from .features import main as features_main
        return features_main([
            "--normalized-dir", str(args.normalized_dir),
            "--output-dir",     str(args.output_dir),
            "--window",         str(args.window),
            "--top-k",          str(args.top_k),
        ])

    if cmd == "train-xgb":
        from .train_xgb import main as train_xgb_main
        return train_xgb_main([
            "--features-dir", str(args.features_dir),
            "--output-dir",   str(args.output_dir),
            "--val-fraction", str(args.val_fraction),
            "--seed",         str(args.seed),
        ])

    if cmd == "train-lstm":
        from .train_lstm import main as train_lstm_main
        return train_lstm_main([
            "--features-dir", str(args.features_dir),
            "--output-dir",   str(args.output_dir),
            "--epochs",       str(args.epochs),
            "--batch-size",   str(args.batch_size),
            "--lr",           str(args.lr),
            "--hidden",       str(args.hidden),
            "--layers",       str(args.layers),
            "--dropout",      str(args.dropout),
            "--val-fraction", str(args.val_fraction),
            "--seed",         str(args.seed),
            "--device",       str(args.device),
        ])

    if cmd == "benchmark":
        from .benchmark import main as benchmark_main
        return benchmark_main([
            "--features-dir", str(args.features_dir),
            "--output-dir",   str(args.output_dir),
            "--epochs",       str(args.epochs),
            "--batch-size",   str(args.batch_size),
            "--lr",           str(args.lr),
            "--hidden",       str(args.hidden),
            "--layers",       str(args.layers),
            "--dropout",      str(args.dropout),
            "--val-fraction", str(args.val_fraction),
            "--seed",         str(args.seed),
        ])

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
