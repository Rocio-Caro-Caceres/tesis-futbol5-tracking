"""
ETAPA4
======

Etapa 4: Automatizacion de eventos complejos (Pases, Duelos, Faltas) via ML.

Pipeline:
    1. download   -- baja Labels-v2.json + tracking de SoccerNet (sin video)
    2. normalize  -- convierte el formato SoccerNet a filas atomicas tipo
                     tracking_events + filas de event_training_labels
    3. features   -- construye ventanas temporales (posiciones, velocidades,
                     distancias relativas) y las guarda como Parquet
    4. train-xgb  -- entrena un XGBoost
    5. train-lstm -- entrena una red LSTM bidireccional con atencion
    6. benchmark  -- compara ambos modelos lado a lado

Convencion: paquete en MAYUSCULAS (igual que METRICAS) para no shadowear el
modulo de nombre generico en imports.
"""

__version__ = "0.1.0"
