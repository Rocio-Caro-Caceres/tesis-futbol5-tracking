# Checkpoint: Etapa 4 - 09-Jun-2026

## Estado actual — Pipeline completo ejecutado

### Pipeline ejecutado de punta a punta

| Paso | Estado | Output |
|------|--------|--------|
| `download` | OK | 39 Labels-v2.json + 57 clips MOT (9GB zip) |
| `normalize` | OK | 57 clips → 733,002 tracking rows, 285 synthetic events |
| `features` | OK | X_seq=(285,51,49), X_flat=(285,2744), mask_seq=(285,51) |
| `train-xgb` | OK | acc=0.436, F1=0.282, 8.9s train, 0.37ms/sample |
| `train-lstm` | OK | acc=0.709, F1=0.277, 9.2s train, 0.59ms/sample |
| `benchmark` | OK | `ETAPA4_data/models/benchmark_summary.json` |

### Nota sobre metrics
Los metrics son baseline-level porque los event labels son **synthetic** (random).
El pipeline funciona end-to-end con datos reales de tracking MOT.

### Datos en disco

```
data/soccernet/
├── england_epl/           # 39 Labels-v2.json (3 seasons)
└── train/                 # 57 clips MOT extraidos (SNMOT-060 a SNMOT-170)
    ├── SNMOT-060/
    │   ├── gt/gt.txt      # MOT20 CSV (10 cols)
    │   ├── seqinfo.ini    # 1920x1080, 25fps, 750 frames
    │   └── img1/          # JPEG frames
    └── ...

ETAPA4_data/
├── normalized/            # 57 dirs, cada uno con tracking.parquet + events.parquet
├── features/              # X_seq.npy, X_flat.npy, mask_seq.npy, meta.parquet
└── models/                # xgb_model.json, lstm_model.pt, benchmark_summary.json
```

### Archivos modificados

| Archivo | Cambios |
|---------|---------|
| `ETAPA4/soccernet_io.py` | +MOT reader, GameDirMot, iter_games_mot, download con downloadDataTask |
| `ETAPA4/normalize.py` | +MOT normalizer, fix Labels-v2 format (dict/annotations), fix position field |
| `ETAPA4/config.py` | +tracking_format field |
| `ETAPA4/__main__.py` | +--tracking-format CLI flag |

### Limitaciones conocidas
- Labels-v2.json (500 broadcast games) NO matchean con tracking clips (12 games) — sin gameID mapping
- Synthetic events creados para habilitar el pipeline completo
- MOT: sin team info, sin jersey numbers, clips de 30s

### Proximo paso
Con event labels reales (si se obtiene el gameID mapping o Labels-v2 del tracking challenge), re-entrenar con labels verdaderos.
