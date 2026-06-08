# Pipeline Diagram — Tesis Fútbol 5 Tracking

> Ver también [`pipeline-diagram.excalidraw`](./pipeline-diagram.excalidraw) — abrir en [excalidraw.com](https://excalidraw.com) (File → Load → pegar el JSON).

## Diagrama de Flujo

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                        ETAPA 1 — Video Tracking                               │
│                  tracking.py · YOLOv8m + ByteTrack + Re-ID                    │
│                                                                               │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│   │  Video   │───▶│ YOLOv8m  │───▶│ ByteTrack│───▶│  Re-ID   │               │
│   │  .mp4    │    │ detección│    │   × 2    │    │ oclusión │               │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘               │
│                                                        │                     │
│                                              ┌─────────┴─────────┐            │
│                                              ▼                   ▼            │
│                                       ┌──────────┐       ┌──────────┐        │
│                                       │ CSV      │       │ JSON     │        │
│                                       │ legacy   │       │ chunks   │        │
│                                       └──────────┘       └──────────┘        │
│                                              │                   │            │
└──────────────────────────────────────────────┼───────────────────┼────────────┘
                                               │                   │
               ┌───────────────────────────────┘                   │
               ▼                                                   ▼
┌──────────────────────────────────────────────┐  ┌───────────────────────────┐
│         ETAPA 3 — Kinematic Metrics          │  │   ETAPA 2 — DB Ingestion  │
│        python -m METRICAS · NumPy/PyTorch    │  │  scripts/ingest.py        │
│                                              │  │                           │
│  ┌──────────┐   ┌──────────┐  ┌──────────┐  │  │  ┌──────────┐             │
│  │ CSV      │──▶│ Smoothing│─▶│Kinematics│  │  │  │ JSON     │──▶ ingest   │
│  │ tracking │   │ CMA w=5  │  │ v, a     │  │  │  │ chunks   │    idemp.   │
│  └──────────┘   └──────────┘  └──────────┘  │  │  └──────────┘     │       │
│                        │                    │  │                   ▼       │
│                        ▼                    │  │  ┌──────────────────┐     │
│  ┌──────────┐   ┌──────────┐               │  │  │   PostgreSQL +   │     │
│  │ Distance │──▶│ Heatmap  │               │  │  │ TimescaleDB +    │     │
│  │ cumdist  │   │ .npz/csv │               │  │  │ PostGIS          │     │
│  └──────────┘   └──────────┘               │  │  │                  │     │
│        │                                    │  │  │ tracking_events  │     │
│        ▼ (opcional)                         │  │  │ match_summary    │     │
│  ┌──────────┐   ┌──────────┐               │  │  │ game_events      │     │
│  │Possession│   │ Events   │               │  │  │ loaded_chunks    │     │
│  │   FSM    │   │ gol/save │               │  │  │ event_training_  │     │
│  └──────────┘   └──────────┘               │  │  │   labels         │     │
│        │           │                        │  │  │ ml_performance_  │     │
│        ▼           ▼                        │  │  │   logs           │     │
│  ┌──────────────────────────┐               │  │  └──────────────────┘     │
│  │  Salidas:                │               │  └───────────────────────────┘
│  │  players_metrics.csv     │
│  │  summary_per_track.csv   │
│  │  summary_match.json      │
│  │  heatmaps/<id>.npz       │
│  │  possession_*.csv        │
│  │  events.csv/goals.csv    │
│  └──────────────────────────┘
│
│                        ┌─────────────────────────────────────────────────────────┐
│                        │       ETAPA 4 — SoccerNet ML (independiente)            │
│                        │      python -m ETAPA4 · XGBoost + LSTM                 │
│                        │                                                         │
│                        │  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│                        │  │ Download  │──▶│ Normalize│──▶│ Features │            │
│                        │  │ SoccerNet │   │ → Parquet│   │ → .npy   │            │
│                        │  └──────────┘   └──────────┘   └──────────┘            │
│                        │                                     │                  │
│                        │                                     ▼                  │
│                        │                            ┌──────────────────┐        │
│                        │                            │  train-xgb       │        │
│                        │                            │  train-lstm      │        │
│                        │                            └────────┬─────────┘        │
│                        │                                     ▼                  │
│                        │                            ┌──────────────────┐        │
│                        │                            │   benchmark      │        │
│                        │                            │   → metrics.json │        │
│                        │                            └──────────────────┘        │
│                        │                                                         │
│                        │  Targets: Pass / Duel / Foul                            │
│                        │  Split: by match (no frame leakage)                    │
│                        │  Away jersey: +100 offset                              │
│                        └─────────────────────────────────────────────────────────┘
```

## Leyenda

| Color | Etapa | Entrypoint | Dependencias |
|-------|-------|-----------|-------------|
| 🔵 Azul | 1 — Tracking | `tracking.py` | GPU, Windows (DirectShow) |
| 🟢 Verde | 2 — DB | `python -m scripts.ingest` | PostgreSQL (Docker) |
| 🟠 Naranja | 3 — Métricas | `python -m METRICAS` | CPU-only (probado ✅) |
| 🟣 Púrpura | 4 — ML | `python -m ETAPA4` | SoccerNet API (independiente) |

## Notas

- **tracking.py** requiere Windows + GPU + video hardcodeado; no tiene CLI flags.
- **Homografía es identidad** — `x_m`/`y_m` son coordenadas de píxel normalizadas, no metros reales.
- **Team detection** no implementada — todos los jugadores se guardan como `'unknown'`.
- **README desactualizado** — la referencia autoritativa es `AGENTS.md`.
- **METRICAS probado** con datos sintéticos ✅ — pipeline completo funciona correctamente.
