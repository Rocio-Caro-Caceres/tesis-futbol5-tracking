# Sistema de Tracking para Análisis de Fútbol 5

Detección y seguimiento de jugadores y pelota en video mediante visión computacional offline.

Desarrollado como Proyecto Final de carrera — por Sola Bru Marcelo, Bichir Cisneros Francisco y Caro Caceres Rocio.

---

## ¿Qué hace este sistema?

- Detecta jugadores y la pelota en video usando YOLOv8m
- Asigna un ID único y persistente a cada jugador (máximo 13)
- Re-identifica jugadores cuando el tracker los pierde
- Maneja oclusiones (cuando dos jugadores se cruzan)
- Exporta las coordenadas de cada jugador por frame a un CSV
- Exporta las posiciones de la pelota a un CSV separado
- Genera un video de salida con los IDs y bounding boxes dibujados

---

## Requisitos de hardware

- GPU NVIDIA con mínimo 6GB VRAM (desarrollado con RTX 4050 Laptop)
- Windows 10 u 11
- Python 3.12

> ⚠️ Sin GPU NVIDIA el sistema va a correr muy lento. El código detecta automáticamente si hay GPU disponible.

---

## Stack tecnológico

| Componente | Herramienta | Versión |
|---|---|---|
| Lenguaje | Python | 3.12 |
| Detección | YOLOv8m (ultralytics) | 8.4.37 |
| Tracker jugadores | ByteTrack (boxmot) | 17.0.0 |
| Tracker pelota | ByteTrack (boxmot) | 17.0.0 |
| Re-ID | No se usa (jugadores con ropa similar, cámara lejana) | — |
| Video / frames | OpenCV | 4.13.0 |
| Datos | Pandas | 2.3.3 |
| Deep Learning | PyTorch + CUDA 12.1 | 2.5.1 |

---

## Instalación paso a paso

### 1 — Instalar Python 3.12

Entrá a https://python.org/downloads y bajá Python 3.12.

> ⚠️ Durante la instalación **tildá la opción "Add Python to PATH"** antes de hacer click en Install. Si no lo hacés, nada va a funcionar.

Verificá que quedó bien abriendo PowerShell:

```powershell
python --version
# Debe decir: Python 3.12.x
```

### 2 — Instalar Git

Entrá a https://git-scm.com/download/win e instalalo con todas las opciones por defecto.

Verificá:

```powershell
git --version
# Debe decir: git version 2.x.x
```

### 3 — Clonar el repositorio

```powershell
git clone https://github.com/Rocio-Caro-Caceres/tesis-futbol5-tracking.git
cd tesis-futbol5-tracking
```

### 4 — Crear el entorno virtual

```powershell
python -m venv venv
.\venv\Scripts\activate
```

Cuando el entorno está activo vas a ver `(venv)` al principio de cada línea en la terminal.

### 5 — Instalar PyTorch con soporte CUDA

Este paso es separado porque necesita una URL especial:

```powershell
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

Verificar que la GPU fue reconocida:

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
# Debe decir: CUDA: True
# y el nombre de tu GPU
```

> ⚠️ Si dice `CUDA: False` no sigas. Significa que los drivers de NVIDIA no están instalados o no son compatibles con CUDA 12.1. Bajá los drivers desde https://nvidia.com/drivers

### 6 — Instalar el resto de dependencias

```powershell
pip install -r requirements.txt
```

Verificar que todo quedó bien:

```powershell
python -c "from ultralytics import YOLO; from boxmot import ByteTrack; import cv2; import pandas; print('Todo OK')"
# Debe decir: Todo OK
```

### 7 — Agregar el video

Los videos no están en el repositorio por su tamaño. Copiá tu archivo `.mp4` dentro de la carpeta `videos/`.

Después abrí `tracking.py` y modificá esta línea con la ruta a tu video:

```python
video_path = r'C:\ruta\a\tu\video\partido.mp4'
```

---

## Cómo ejecutar

```powershell
# 1. Activar el entorno virtual (siempre primero)
.\venv\Scripts\activate

# 2. Correr el sistema
python tracking.py
```

El sistema va a imprimir en consola información de cada 150 frames. Presioná `Q` en la ventana del video para detener.

---

## Parámetros configurables

Al principio de `tracking.py` podés ajustar estos valores sin necesidad de entender el resto del código:

| Parámetro | Valor por defecto | Qué hace |
|---|---|---|
| `MINUTO_INICIO` | 20 | Desde qué minuto del video empezar |
| `MAX_JUGADORES` | 13 | Máximo de personas a trackear |
| `MARGEN_ARRIBA` | 50 | Píxeles desde arriba que se ignoran |
| `DIST_MAX_REID` | 200 | Distancia máxima en px para re-identificar |
| `IOU_OCLUSION` | 0.4 | Superposición mínima para detectar oclusión |

---

## Archivos de salida

Después de ejecutar se generan en la carpeta `videos/`:

| Archivo | Contenido |
|---|---|
| `resultado_tesis.avi` | Video con bounding boxes e IDs dibujados |
| `datos_finales_tesis.csv` | Coordenadas de jugadores por frame |
| `pelota_tesis.csv` | Posiciones de la pelota por frame |

### Estructura del CSV de jugadores

```
frame, id, x, y, x1, y1, x2, y2, en_oclusion
```

- `frame` — número de frame
- `id` — ID del jugador (1 al 13)
- `x, y` — centro del jugador en píxeles
- `x1, y1, x2, y2` — esquinas del bounding box
- `en_oclusion` — True si estaba tapado por otro jugador

### Estructura del CSV de pelota

```
frame, x, y
```

---

## Estructura del repositorio

```
tesis-futbol5-tracking/
│
├── tracking.py          ← código principal
├── requirements.txt     ← dependencias
├── .gitignore           ← excluye videos, modelos y venv
├── README.md            ← esta guía
│
├── models/              ← carpeta vacía
│   └── .gitkeep         (yolov8m.pt se descarga automático)
│
└── videos/              ← carpeta vacía
    └── .gitkeep         (agregá tu video acá)
```

---

## Primera ejecución — qué esperar

La primera vez que corrás el sistema va a descargar automáticamente el modelo `yolov8m.pt` (~50MB). Esto solo pasa una vez, después queda guardado localmente.

---

## Problemas frecuentes

**El sistema no encuentra el video**
Verificá que la ruta en `video_path` esté bien escrita y que el archivo exista en la carpeta `videos/`.

**CUDA: False**
Los drivers de NVIDIA no están instalados o no son CUDA 12.1. Bajá los drivers desde https://nvidia.com/drivers

**Los IDs arrancan desde un número alto**
Significa que el kernel no se reinició entre ejecuciones. Cerrá la terminal, volvé a abrirla, activá el venv y corré de nuevo.

**La pelota no se detecta**
YOLOv8m tiene dificultades con pelotas pequeñas en cámara lejana. Se detecta en aproximadamente el 40-60% de los frames. Para mejorar esto se necesita fine-tuning del modelo con imágenes de tu cancha — trabajo futuro.

**Muchos IDs para pocos jugadores**
Ajustá `DIST_MAX_REID` a un valor más alto (ej: 300) para que el sistema sea más tolerante al re-identificar.

---

## Contexto del proyecto y trabajo futuro

Este sistema fue desarrollado con una sola cámara mal posicionada como prueba de concepto. Para la versión final de la tesis se planea:

- **Dos cámaras** bien posicionadas a media altura cubriendo cada una medio campo
- **Re-identificación por apariencia** (OSNet) que con las cámaras bien posicionadas va a funcionar correctamente
- **Fine-tuning** del modelo de detección con imágenes de la cancha específica para mejorar la detección de la pelota
- **Homografía** para transformar coordenadas de píxeles a metros reales en el campo
- **Heatmaps** de posicionamiento táctico por jugador
- **Métricas físicas** como distancia recorrida y velocidad máxima

---

## Etapa 1 — Persistencia en PostgreSQL (TimescaleDB + PostGIS)

Esta etapa reemplaza la salida CSV por una base relacional con series
temporales. El diseño atómico es **una fila por objeto por frame**.

### Arquitectura

```
  tracking.py ──► chunks/<match_id>/chunk_*.json ──► ingest.py ──► PostgreSQL
                                                            │
                                                            ├─ tracking_events  (hypertable)
                                                            ├─ game_events      (hypertable)
                                                            └─ match_summary    (catalogo)
```

### Levantar la base de datos

Requisitos: Docker Desktop con `docker compose`.

```powershell
cd db
copy .env.example .env       # editar credenciales si queres
docker compose up -d
docker compose logs -f db    # esperar al "database system is ready"
```

La primera vez se ejecuta `db/sql/01_schema.sql` ... `04_grants.sql` en orden,
creando extensiones, tablas, hypertables e índices.

### Instalar dependencias de la etapa 1

```powershell
pip install -r requirements-db.txt
```

### Generar chunks JSON (en lugar de CSV)

`tracking.py` ahora escribe un JSON cada 150 frames bajo `chunks/<match_id>/`:

```
chunks/
└── partido_f5/
    ├── match.json
    ├── chunk_00000.json
    ├── chunk_00001.json
    └── ...
```

Cada chunk tiene la forma:

```json
{
  "match_id": "partido_f5",
  "schema_version": 1,
  "metadata": { "fps": 24, "width": 1920, "height": 1080,
                "field_length_m": 105, "field_width_m": 68,
                "homography": null, "start_frame": 0, "end_frame": 149 },
  "tracking": [ {"frame": 0, "track_id": 1, "object_type": "player",
                 "team": "unknown", "x": 640, "y": 360,
                 "x_norm": 0.5, "y_norm": 0.5, "x_m": 52.5, "y_m": 34.0,
                 "bbox_x1": 600, "bbox_y1": 320, "bbox_x2": 680, "bbox_y2": 400,
                 "confidence": 0.87, "in_occlusion": false} ],
  "ball":    [ {"frame": 0, "x": 320, "y": 240, "x_norm": 0.25, "y_norm": 0.22,
                 "x_m": 26.25, "y_m": 15.0, "confidence": 0.5} ]
}
```

### Cargar los chunks en la base (Bulk Insert con COPY)

```powershell
# Un solo partido
python -m scripts.ingest --match-id partido_f5

# Todos los partidos bajo chunks/
python -m scripts.ingest --all

# Smoke test de lo que quedo en la DB
python -m scripts.verify_db --match-id partido_f5
```

La ingesta es **idempotente**: si se vuelve a correr, los chunks ya cargados
(mismo `sha256`) se saltean. La geometría PostGIS se materializa en el
`INSERT` final con `ST_SetSRID(ST_MakePoint(x_m, y_m), 3857)`.

### Variables de entorno para la ingesta

```
PGHOST=localhost
PGPORT=5432
PGDATABASE=futbol5
PGUSER=futbol5
PGPASSWORD=futbol5
```

### Índices creados (Etapa 1)

| Tabla              | Índice                                                | Tipo   |
|--------------------|-------------------------------------------------------|--------|
| `tracking_events`  | `(match_id, track_id, timestamp_ms DESC)`             | B-tree |
| `tracking_events`  | `(match_id, frame)`                                   | B-tree |
| `tracking_events`  | `geom`                                                | GIST   |
| `tracking_events`  | `timestamp_ms`                                        | BRIN   |
| `game_events`      | `(match_id, timestamp_ms DESC)`                       | B-tree |
| `game_events`      | `(match_id, event_type, timestamp_ms DESC)`           | B-tree |
| `game_events`      | `start_geom`                                          | GIST   |
| `game_events`      | `metadata`                                            | GIN    |

### Estructura del repositorio (actualizada)

```
tesis-futbol5-tracking/
├── tracking.py                ← emite chunks JSON (modificado)
├── tracking/                  ← paquete nuevo
│   ├── chunk_writer.py        ← ventana de 150 frames -> chunk_NNNNN.json
│   └── homography.py          ← pixel -> norm/m (identidad por ahora)
├── db/                        ← todo lo de la base
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── .env.example
│   ├── initdb-extensions.sh
│   └── sql/
│       ├── 01_schema.sql      ← tablas + PKs + comentarios
│       ├── 02_hypertables.sql ← create_hypertable + compresion
│       ├── 03_indexes.sql     ← indices compuestos + GIST/BRIN/GIN
│       └── 04_grants.sql      ← rol de aplicacion
├── scripts/
│   ├── ingest.py              ← COPY + ON CONFLICT, idempotente
│   └── verify_db.py           ← smoke test
├── chunks/                    ← salida de tracking.py (gitignored)
├── requirements.txt
├── requirements-db.txt
└── README.md
```

