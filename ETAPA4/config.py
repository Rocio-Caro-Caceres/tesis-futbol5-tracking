"""
config.py
=========
Configuracion central de Etapa 4. Dataclasses inmutables (slots=True) para que
los argumentos viajen limpios entre submodulos.

Atributos canonicos:
    PipelineConfig.field_dims  -- dimensiones del campo (FIFA 11-a-side default)
    FeaturesConfig.window_frames  -- tamano de la ventana temporal (W en cada
                                    lado del frame del evento, total 2W+1)
    TrainConfig.split_seed  -- semilla del split train/val POR MATCH
                              (nunca por frame: eso filtra)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Constantes del proyecto
# ---------------------------------------------------------------------------
TARGET_LABELS: tuple[str, ...] = ("Pass", "Duel", "Foul")
# Mapeo del label crudo SoccerNet (que a veces viene con espacios o minusculas)
LABEL_ALIASES: dict[str, str] = {
    "Pass": "Pass",
    "Pass ": "Pass",
    "pass": "Pass",
    "Duel": "Duel",
    "duel": "Duel",
    "Foul": "Foul",
    "foul": "Foul",
}


# ---------------------------------------------------------------------------
# Campo
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FieldDims:
    length_m: float = 105.0
    width_m: float = 68.0


# ---------------------------------------------------------------------------
# Descarga SoccerNet
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class DownloadConfig:
    output_dir: Path = Path("data/soccernet")
    split: tuple[str, ...] = ("train",)
    files: tuple[str, ...] = (
        "1_Labels-v2.json",
        "2_Labels-v2.json",
        "Labels-v2.json",
        "1_tracking.jsonl.gz",
        "2_tracking.jsonl.gz",
    )
    # La password "s0cc3rn3t" es la estandar para uso academico no comercial.
    # Si tenes credenciales propias, overridea con --soccernet-password o env.
    password: str = "s0cc3rn3t"
    # Si ya tenes los archivos en disco, podes saltarte la descarga.
    raw_dir: Path | None = None
    # Formato de tracking: "mot" (MOT20 CSV, default) o "jsonl" (JSONL.GZ legacy).
    tracking_format: str = "mot"


# ---------------------------------------------------------------------------
# Normalizacion
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class NormalizeConfig:
    field_dims: FieldDims = field(default_factory=FieldDims)
    # Por defecto SoccerNet publica tracking a 25 fps. Si la carpeta de
    # tracking no tiene la convencion esperada, lo inferimos por la cantidad
    # de frames del primer half.
    fps: float = 25.0
    # Si la coordenada 'position' del tracking viene normalizada [0,1]
    # (es la convencion actual del challenge), True. Si viene en metros, False.
    tracking_normalized: bool = True
    # Equivalencia 'role' -> object_type. La pelota de SoccerNet tiene role=ball.
    ball_role_marker: str = "ball"
    # Persistencia del output normalizado.
    output_dir: Path = Path("ETAPA4_data/normalized")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class FeaturesConfig:
    # Tamano de ventana: W frames hacia cada lado del frame del evento.
    # total = 2*W + 1. 25 frames = 1 segundo a 25 fps.
    window_frames: int = 25
    # Frecuencia de submuestreo dentro de la ventana (1 = todos los frames).
    stride: int = 1
    # Cantidad de oponentes / companeros mas cercanos que se incluyen como
    # features auxiliares (ademas del actor y target).
    top_k_neighbors: int = 4
    # Suavizado de trayectoria (ventana del moving average en frames).
    smooth_window: int = 3
    # Persistencia del output (Parquet columnar para entrenamiento rapido).
    output_dir: Path = Path("ETAPA4_data/features")
    # Target labels que se incluyen en el dataset de entrenamiento.
    target_labels: tuple[str, ...] = TARGET_LABELS
    # Cota de FPS (para etiquetar timestamps en features).
    fps: float = 25.0


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class TrainConfig:
    # Split por match (no por frame) para evitar leakage.
    val_fraction: float = 0.2
    split_seed: int = 42
    # Tamano del batch para LSTM.
    batch_size: int = 64
    # Cantidad de epocas (LSTM). XGBoost usa early_stopping interno.
    epochs: int = 30
    # Paciencia para early stopping en LSTM (sobre val_loss).
    patience: int = 5
    # Tamaño del hidden del LSTM.
    lstm_hidden: int = 128
    # Cantidad de capas LSTM.
    lstm_layers: int = 2
    # Dropout entre capas LSTM y en la cabeza densa.
    dropout: float = 0.3
    # Learning rate de Adam.
    lr: float = 1e-3
    # Weight decay.
    weight_decay: float = 1e-5
    # Pesos por clase (None = balanceados). Util cuando hay imbalance.
    class_weights: tuple[float, float, float] | None = None
    # Persistencia del modelo y metricas.
    output_dir: Path = Path("ETAPA4_data/models")
    # Device para LSTM.
    device: Literal["auto", "cpu", "cuda"] = "auto"


# ---------------------------------------------------------------------------
# Pipeline (agregador)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class PipelineConfig:
    download: DownloadConfig = field(default_factory=DownloadConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
