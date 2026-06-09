# Checkpoint: Etapa 4 - 09-Jun-2026

## Estado actual

### Dependencias instaladas (pip --break-system-packages)
- SoccerNet (v0.1.62, sin dep resolver)
- tqdm, requests
- huggingface_hub (v1.16.4)
- boto3, botocore, s3transfer, jmespath
- google-measurement-protocol, prices
- click (v8.4.1), typer (v0.25.1), httpx, hf-xet, httpcore, h11, anyio

### Faltantes (no críticos para download, sí para normalize/features/train)
- matplotlib (para SoccerNet, no necesario para Etapa 4 propia)
- pycocoevalcap (104 MB, timeout)
- scikit-video

### Datos descargados

#### `data/soccernet/` (via `python -m ETAPA4 download --splits train`)
Solo **Labels-v2.json** (18 partidos de `england_epl/2014-2015` y `2015-2016`).
Los `*_tracking.jsonl.gz` devuelven **404** — no existen en el servidor ownCloud.

#### `data/soccernet_tracking/tracking/train.zip` (via `downloadDataTask(task="tracking")`)
157 MB, pero **formato MOT** (CSV + imágenes), NO el jsonl.gz que espera `normalize.py`.

### Problema estructural

El pipeline `ETAPA4/normalize.py:_track_to_atomic_rows()` espera archivos
`*_tracking.jsonl.gz` por partido con formato:
```json
{"role": "player", "jersey": 10, "position": [0.5, 0.3], "frame": 123, "team": "home"}
```
Ese formato ya no está disponible en el SoccerNet API actual. El tracking disponible
via `downloadDataTask(task="tracking")` viene en formato MOT20 (CSV con 10 columnas:
frame, track_id, bbox_x, bbox_y, width, height, conf, -1, -1, -1).

### Para retomar

1. **Elegir estrategia:**
   - (a) Adaptar `normalize.py` y `soccernet_io.py` para parsear tracking formato MOT
   - (b) Buscar fuente alternativa de jsonl.gz (HuggingFace, o generar synthetic)
   - (c) Probar desde PowerShell/Windows (mejor conectividad)

2. **Comandos a ejecutar después de resolver el tracking:**
   ```bash
   python3 -m ETAPA4 normalize
   python3 -m ETAPA4 features
   python3 -m ETAPA4 train-xgb
   python3 -m ETAPA4 train-lstm
   python3 -m ETAPA4 benchmark
   ```

3. **Si se elige (a), editar:**
   - `ETAPA4/soccernet_io.py` — agregar lector de tracking MOT
   - `ETAPA4/normalize.py` — adaptar `_track_to_atomic_rows` para el formato MOT
   - `ETAPA4/download` en `soccernet_io.py` — usar `downloadDataTask` en vez de `downloadGames`

### Archivos clave
- `ETAPA4/__init__.py` — docstring del pipeline
- `ETAPA4/soccernet_io.py` — download + iteración de partidos
- `ETAPA4/normalize.py` — conversión a tracking_events + training labels
- `ETAPA4/features.py` — ventanas temporales para ML
- `ETAPA4/train_xgb.py` / `train_lstm.py` / `benchmark.py`
- `ETAPA4/config.py` — defaults y dataclasses
- `requirements-etapa4.txt` — deps (SoccerNet, pandas, numpy, pyarrow, sklearn, xgboost, torch, tqdm)
