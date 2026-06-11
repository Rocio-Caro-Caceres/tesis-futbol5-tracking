# Issues — Hito: Convertir datos de visión por computadora a estadísticas

**MVP**: Mapa de calor, posesión, distancia recorrida, eventos de gol.

---

## ✅ Completadas

### 1. Extraer datos crudos de visión por computadora y guardar raw
- **Estado**: Completada
- **Archivo**: `tracking.py` (línea 257-402), `tracking/chunk_writer.py`
- **Descripción**: El pipeline de tracking recibe video, detecta jugadores y pelota con YOLOv8 + ByteTrack, y guarda los datos por frame en chunks JSON atómicos (150 frames por archivo) en `chunks/<match_id>/`. Cada chunk contiene posición pixel, normalizada y en metros (identidad), bounding box, confianza y estado de oclusión.

### 2. Transformar datos para legibilidad y tipo de dato
- **Estado**: Completada
- **Archivo**: `METRICAS/metrica_loader.py`
- **Descripción**: El loader normaliza los datos de entrada: acepta CSV con columnas `x_m`/`y_m` o `x_norm`/`y_norm`, reescala coordenadas normalizadas a metros usando las dimensiones del campo, y formatea el DataFrame con tipos consistentes (`track_id` int, `frame` int, `team` str, etc.).

### 3. Implementar algoritmo de mapa de calor
- **Estado**: Completada
- **Archivo**: `METRICAS/heatmap.py` (164 líneas)
- **Descripción**: Genera heatmaps 2D de densidad de presencia por `track_id`. Usa `numpy.histogram2d` + filtro gaussiano de `scipy`. Salida en `.npz` (binario) y `.csv` (legible). Convención `(ny, nx)` — filas = Y, columnas = X.

### 4. Implementar algoritmo de posesión
- **Estado**: Completada
- **Archivo**: `METRICAS/possession.py` (533 líneas)
- **Descripción**: Máquina de estados (FSM) que asigna la posesión de la pelota frame a frame. Considera proximidad jugador-pelota, cambio de velocidad de la pelota (touch detectado), radios de robo y control, e histéresis para evitar flickering. Genera: posesión por equipo (%), posesión por jugador (segundos, toques), spells (secuencias de posesión continua).

### 5. Implementar algoritmo de distancia recorrida
- **Estado**: Completada
- **Archivo**: `METRICAS/distance.py` (141 líneas)
- **Descripción**: Calcula distancia acumulada por `track_id` usando dist euclídea entre frames consecutivos. Incluye umbral `min_step=0.03m` para filtrar ruido de jugadores estáticos. Devuelve distancia total por jugador y distancia por frame (`cumdist_m`).

### 6. Implementar detección de eventos de gol
- **Estado**: Completada
- **Archivo**: `METRICAS/events.py` (516 líneas)
- **Descripción**: Detección geométrica de goles por intersección pelota-polígono del arco. También detecta atajadas (velocidad del tiro + proximidad del arquero + desaceleración). Requiere configurar `--keeper-ids` o `goalkeeper_team` para que la detección funcione.

### 7. Orquestador del pipeline de métricas (CLI)
- **Estado**: Completada
- **Archivo**: `METRICAS/pipeline.py` (531 líneas), `METRICAS/__main__.py`
- **Descripción**: CLI ejecutable como `python -m METRICAS --input <csv> --output-dir out/`. Encadena las 5 etapas (suavizado → cinemática → distancia → heatmap → eventos/posesión). Genera `players_metrics.csv`, `summary_per_track.csv`, `summary_match.json`, heatmaps, posesión y eventos.

### 8. Suavizado de posiciones
- **Estado**: Completada
- **Archivo**: `METRICAS/smoothing.py` (173 líneas)
- **Descripción**: Media móvil central con ventana configurable (default=5). NaN-safe en bordes. Agrega columnas `x_m_smooth` y `y_m_smooth` al DataFrame.

### 9. Cinemática: velocidad y aceleración
- **Estado**: Completada
- **Archivo**: `METRICAS/kinematics.py` (242 líneas)
- **Descripción**: Calcula vx, vy, velocidad (magnitud), ax, ay, aceleración (magnitud) por frame usando diferencias centrales. Incluye cap de velocidad máxima (default 12 m/s) para filtrar outliers de tracking.

---

## 🔲 Pendientes

### 10. Crear conversor chunks JSON → CSV para METRICAS
- **Estado**: Pendiente
- **Bloquea MVP**: Sí — sin esto no hay input para METRICAS
- **Archivo a crear**: `scripts/chunks_to_csv.py` (propuesto)
- **Descripción**: Script que lea todos los `chunk_*.json` de `chunks/<match_id>/`, fusione `tracking[]` y `ball[]` en un DataFrame plano, y escriba un CSV con columnas: `frame, track_id, object_type, team, x_m, y_m, x_norm, y_norm`. Este CSV es el input directo que METRICAS necesita.
- **Pasos**:
  1. Parsear cada chunk JSON (usar `tracking/chunk_writer.py` como referencia del schema)
  2. Concatenar tracking + ball de todos los chunks
  3. Ordenar por `frame` ascendente
  4. Escribir CSV con columnas compatibles con `METRICAS/pipeline.py:_load()`
- **Criterio de aceptación**: `python -m METRICAS --input output.csv --output-dir out/ --possession --events --keeper-ids <ids> --field-length 42 --field-width 25` ejecuta sin errores y genera las 4 métricas del MVP.

### 11. Calibrar homografía real (pixel → metros del campo)
- **Estado**: Pendiente
- **Bloquea MVP**: Parcial — las métricas funcionan pero en unidades normalizadas, no en metros reales
- **Archivo a modificar**: `tracking/homography.py`
- **Descripción**: Actualmente usa transformación identidad (`x_m = x_norm * field_length`). Se necesita calibrar con puntos de control del video (esquinas del campo, líneas conocidas) para obtener una matriz de homografía real que convierta píxeles a metros reales del campo de fútbol 5.
- **Pasos**:
  1. Identificar puntos de control en el video (esquinas del campo, marca del centro, área)
  2. Definir sus coordenadas reales (campo 42×25m)
  3. Calcular matriz de homografía con `cv2.findHomography()`
  4. Integrar en `PixelToField` para que aplique la transformación real
- **Criterio de aceptación**: Un punto en una esquina del campo en el video se transforma a la coordenada real esperada (±0.5m).

### 12. Implementar detección de equipos (home/away)
- **Estado**: Pendiente
- **Bloquea MVP**: Parcial — posesión por equipo no funciona sin asignación de equipos
- **Archivo a modificar**: `tracking.py` (línea 118: `TEAM_BY_ID = {}`)
- **Descripción**: Asignar cada `track_id` a un equipo ('home' o 'away'). Opciones: detección por color de jersey, asignación manual por rango de IDs, o clustering de posiciones. La posesión FSM necesita el campo `team` para calcular el porcentaje por equipo.
- **Pasos**:
  1. Definir estrategia (color-based, manual, o clustering)
  2. Implementar asignación en `tracking.py` antes de escribir chunks
  3. Verificar que `possession.py` recibe `team` correctamente
- **Criterio de aceptación**: `summary_match.json` muestra porcentajes de posesión por equipo (no solo 'unknown').

### 13. Configurar dimensiones de campo futsal (42×25m) como default
- **Estado**: Pendiente
- **Bloquea MVP**: Parcial — el default 105×68 produce métricas incorrectas
- **Archivo a modificar**: `METRICAS/metrica_loader.py` (línea 56) y `tracking/chunk_writer.py` (línea 135)
- **Descripción**: Cambiar los defaults de `FieldDims` de `(105, 68)` a `(42, 25)` para que coincida con el campo de fútbol 5 del proyecto. Esto afecta el reescalamiento de coordenadas normalizadas a metros y los cálculos de heatmap y eventos.
- **Criterio de aceptación**: Ejecutar `python -m METRICAS --input <csv> --output-dir out/` sin flags de campo produce resultados correctos para futsal 42×25m.

### 14. Configurar IDs de arqueros para detección de goles
- **Estado**: Pendiente
- **Bloquea MVP**: Parcial — goles no se detectan sin configurar arqueros
- **Archivo a modificar**: `tracking.py` (agregar constante `GOALKEEPER_IDS`)
- **Descripción**: La detección de eventos en `events.py` requiere saber quiénes son los arqueros para detectar atajadas y confirmar goles. Se debe definir una lista de `track_id` de arqueros (o un equipo de arquero) que se pase al pipeline.
- **Pasos**:
  1. Identificar los `track_id` de los arqueros en el video
  2. Agregar constante `GOALKEEPER_IDS` en `tracking.py`
  3. Pasar los IDs al pipeline de METRICAS via CLI o configuración
- **Criterio de aceptación**: `events.csv` contiene al menos un evento de gol cuando la pelota entra al arco.

### 15. Generar JSON de estadísticas formateado para la web
- **Estado**: Pendiente
- **Bloquea MVP**: Parcial — `summary_match.json` existe pero puede no tener el formato que la web necesita
- **Archivo a modificar**: `METRICAS/pipeline.py` (línea 321-361)
- **Descripción**: El `summary_match.json` actual tiene un resumen parcial. Se necesita un JSON completo y estructurado que la web pueda consumir directamente, incluyendo: datos del heatmap (bins + valores), posesión por equipo y por jugador, distancia por jugador, y lista de eventos de gol con timestamp y metadata.
- **Criterio de aceptación**: Un frontend puede renderizar las 4 métricas (heatmap, posesión, distancia, goles) usando solo el JSON generado.
