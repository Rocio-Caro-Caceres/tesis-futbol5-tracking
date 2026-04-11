import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from boxmot import ByteTrack
from boxmot.trackers.bytetrack.basetrack import BaseTrack
import gc

# =============================================================
# 0. LIMPIEZA
# =============================================================
try:
    del tracker, tracker_pelota, model, vid, out
    gc.collect()
    torch.cuda.empty_cache()
    print("Variables anteriores liberadas.")
except:
    print("Sin sesión anterior.")

BaseTrack._count = 0
print("Contador de IDs reseteado.")

# =============================================================
# 1. MODELOS
# =============================================================
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Dispositivo: {'GPU (CUDA)' if device == '0' else 'CPU'}")

model = YOLO('yolov8m.pt')

# Tracker de jugadores
tracker = ByteTrack(
    track_thresh=0.15,
    match_thresh=0.9,
    track_buffer=150,
    frame_rate=24,
)

# Tracker separado solo para la pelota
tracker_pelota = ByteTrack(
    track_thresh=0.1,    # muy bajo porque la pelota es difícil de detectar
    match_thresh=0.8,
    track_buffer=30,     # la pelota no desaparece tanto tiempo
    frame_rate=24,
)
print("Trackers iniciados OK")

# =============================================================
# 2. VIDEO
# =============================================================
video_path = r'C:\Users\rocio\Desktop\Quinto Año\Proyecto Final\Tesis\tesis-futbol5-tracking\videos\partido_f5.mp4'
vid = cv2.VideoCapture(video_path)
if not vid.isOpened():
    raise FileNotFoundError(f"No se pudo abrir: {video_path}")

fps_lectura  = vid.get(cv2.CAP_PROP_FPS)
total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
width        = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
height       = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = int(round(fps_lectura))

print(f"Video: {width}x{height} @ {fps_lectura:.3f} FPS | {total_frames/fps_lectura/60:.1f} min")

MINUTO_INICIO = 20
frame_inicio  = int(MINUTO_INICIO * 60 * fps_lectura)
vid.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)
print(f"Saltando al minuto {MINUTO_INICIO} (frame {frame_inicio})")

# =============================================================
# 3. SALIDA
# =============================================================
output_dir = Path('videos')
output_dir.mkdir(exist_ok=True)
out = cv2.VideoWriter(
    str(output_dir / 'resultado_tesis.avi'),
    cv2.VideoWriter_fourcc(*'XVID'),
    fps, (width, height)
)
if not out.isOpened():
    raise RuntimeError("No se pudo crear el video de salida.")
print(f"Guardando en: {(output_dir / 'resultado_tesis.avi').resolve()}")

# =============================================================
# 4. PARÁMETROS
# =============================================================
MAX_JUGADORES       = 13
MARGEN_ARRIBA       = 50
MARGEN_LADOS        = 20
MARGEN_ABAJO        = 20
RATIO_MIN           = 0.8
RATIO_MAX           = 7.0
FRAMES_PARA_VALIDAR = 48
MOVIMIENTO_MINIMO   = 18
FRAMES_MIN_AUSENTE  = 8
DIST_MAX_REID       = 200

# Oclusión: si dos cajas se superponen más de este porcentaje
# no reasignamos IDs hasta que se separen
IOU_OCLUSION = 0.4

# =============================================================
# 5. ESTADO DEL POOL
# =============================================================
id_map             = {}
reverse_map        = {}
pool_disponible    = list(range(1, MAX_JUGADORES + 1))
ultima_posicion    = {}
ultimo_frame_visto = {}
historial_posiciones = defaultdict(list)
ids_estaticos        = set()

# Para la pelota
pos_pelota_historial = []  # lista de (frame, cx, cy) para el CSV

# =============================================================
# 6. FUNCIONES
# =============================================================
def color_por_id(nuestro_id):
    np.random.seed(int(nuestro_id) * 31)
    return tuple(int(x) for x in np.random.randint(60, 255, 3))

def deteccion_valida_jugador(x1, y1, x2, y2):
    if y1 < MARGEN_ARRIBA:                        return False
    if x1 < MARGEN_LADOS and x2 < MARGEN_LADOS:  return False
    if x1 > width - MARGEN_LADOS:                 return False
    if y1 > height - MARGEN_ABAJO:                return False
    ratio = max(y2 - y1, 1) / max(x2 - x1, 1)
    if ratio < RATIO_MIN or ratio > RATIO_MAX:    return False
    return True

def calcular_iou(box1, box2):
    """
    Calcula la superposición entre dos cajas.
    Usado para detectar oclusiones entre jugadores.
    box = (x1, y1, x2, y2)
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0

    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def hay_oclusion(x1, y1, x2, y2, tracks_activos):
    """
    Devuelve True si esta caja se superpone significativamente
    con algún track ya procesado en este frame.
    En ese caso no re-asignamos ID — esperamos a que se separen.
    """
    for (tx1, ty1, tx2, ty2) in tracks_activos:
        iou = calcular_iou((x1, y1, x2, y2), (tx1, ty1, tx2, ty2))
        if iou > IOU_OCLUSION:
            return True
    return False

def distancia(pos1, pos2):
    return np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)

def resolver_id(bytetrack_id, cx, cy, frame_actual, ids_activos_este_frame):
    # Caso 1: ya conocido
    if bytetrack_id in id_map:
        nuestro_id = id_map[bytetrack_id]
        if nuestro_id in ids_activos_este_frame:
            return None  # duplicado en el frame
        ultima_posicion[nuestro_id]    = (cx, cy)
        ultimo_frame_visto[nuestro_id] = frame_actual
        return nuestro_id

    # Caso 2: pool con lugar
    if pool_disponible:
        nuestro_id = pool_disponible.pop(0)
        id_map[bytetrack_id]           = nuestro_id
        reverse_map[nuestro_id]        = bytetrack_id
        ultima_posicion[nuestro_id]    = (cx, cy)
        ultimo_frame_visto[nuestro_id] = frame_actual
        print(f"  [NUEVO] ByteID:{bytetrack_id} → ID:{nuestro_id} "
              f"(pool restante:{len(pool_disponible)})")
        return nuestro_id

    # Caso 3: pool lleno → re-identificar
    candidatos = {}
    for nid in ultima_posicion:
        if nid in ids_activos_este_frame:
            continue
        frames_ausente = frame_actual - ultimo_frame_visto.get(nid, 0)
        if frames_ausente < FRAMES_MIN_AUSENTE:
            continue
        dist = distancia(ultima_posicion[nid], (cx, cy))
        if dist <= DIST_MAX_REID:
            candidatos[nid] = dist

    if candidatos:
        mejor_id   = min(candidatos, key=lambda nid: candidatos[nid])
        dist_final = candidatos[mejor_id]

        viejo_bt = reverse_map.get(mejor_id)
        if viejo_bt and viejo_bt in id_map:
            del id_map[viejo_bt]

        id_map[bytetrack_id]         = mejor_id
        reverse_map[mejor_id]        = bytetrack_id
        ultima_posicion[mejor_id]    = (cx, cy)
        ultimo_frame_visto[mejor_id] = frame_actual
        print(f"  [REID] ByteID:{bytetrack_id} → ID:{mejor_id} "
              f"(dist={dist_final:.0f}px)")
        return mejor_id

    return None

# =============================================================
# 7. LOOP PRINCIPAL
# =============================================================
datos_tracking    = []
frame_count       = frame_inicio
total_dets_raw    = 0
total_dets_filt   = 0
frames_sin_tracks = 0

print(f"\n--- TRACKING MIN {MINUTO_INICIO} | POOL {MAX_JUGADORES} IDs ---")
print("Verde=YOLO | Color+ID=Jugador | Amarillo=Pelota | Q para detener\n")

try:
    while True:
        ret, frame = vid.read()
        if not ret:
            print("Fin del video.")
            break

        # ── DETECCIÓN — personas Y pelota en una sola pasada ──
        results = model.predict(
            frame, imgsz=1280, conf=0.15,  # bajo para capturar la pelota
            device=device,
            classes=[0, 32],   # 0=persona, 32=sports ball
            verbose=False
        )

        # Separar detecciones por clase
        dets_jugadores = []
        dets_pelota    = []

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for det in results[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = det
                cls = int(cls)
                if cls == 0:  # persona
                    if deteccion_valida_jugador(x1, y1, x2, y2):
                        dets_jugadores.append(det)
                elif cls == 32:  # pelota
                    dets_pelota.append(det)

        n_dets_raw = len(results[0].boxes)
        total_dets_raw += n_dets_raw

        dets_jug_np = (np.array(dets_jugadores, dtype=np.float32)
                       if dets_jugadores else np.empty((0, 6), dtype=np.float32))
        dets_pel_np = (np.array(dets_pelota, dtype=np.float32)
                       if dets_pelota else np.empty((0, 6), dtype=np.float32))

        n_dets = len(dets_jug_np)
        total_dets_filt += n_dets

        # Dibujar cajas YOLO de jugadores en verde
        for det in dets_jug_np:
            cv2.rectangle(frame,
                (int(det[0]), int(det[1])), (int(det[2]), int(det[3])),
                (0, 220, 0), 1)

        # ── TRACKING JUGADORES ─────────────────────────────────
        tracks_jug = tracker.update(dets_jug_np, frame)

        frames_procesados      = frame_count - frame_inicio
        ids_activos_este_frame = set()
        cajas_activas          = []   # para detección de oclusión

        if tracks_jug is not None and len(tracks_jug) > 0:
            for t in tracks_jug:
                x1, y1, x2, y2, bytetrack_id = (int(t[0]), int(t[1]),
                                                  int(t[2]), int(t[3]), int(t[4]))
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # ── FILTRO DE OCLUSIÓN ──────────────────────────
                # Si este jugador se superpone con otro ya procesado
                # en este frame, mantenemos el ID que ya tenía pero
                # no forzamos una nueva asignación
                en_oclusion = hay_oclusion(x1, y1, x2, y2, cajas_activas)

                if en_oclusion and bytetrack_id in id_map:
                    # Está en oclusión pero ya conocemos su ID → dibujar con indicador
                    nuestro_id = id_map[bytetrack_id]
                    if nuestro_id not in ids_activos_este_frame:
                        ids_activos_este_frame.add(nuestro_id)
                        cajas_activas.append((x1, y1, x2, y2))
                        ultima_posicion[nuestro_id]    = (cx, cy)
                        ultimo_frame_visto[nuestro_id] = frame_count
                        # Dibujar con borde punteado (color más tenue) para indicar oclusión
                        color = color_por_id(nuestro_id)
                        color_tenue = tuple(max(0, c - 80) for c in color)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color_tenue, 2)
                        label = f'ID:{nuestro_id}~'  # ~ indica oclusión
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
                        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color_tenue, -1)
                        cv2.putText(frame, label, (x1+2, y1-2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
                    continue

                elif en_oclusion and bytetrack_id not in id_map:
                    # En oclusión y no conocemos quién es → no asignar ID nuevo
                    # ByteTrack lo va a re-asociar cuando salga de la oclusión
                    cajas_activas.append((x1, y1, x2, y2))
                    continue

                # Sin oclusión → resolver ID normalmente
                nuestro_id = resolver_id(
                    bytetrack_id, cx, cy,
                    frame_count, ids_activos_este_frame
                )
                if nuestro_id is None:
                    continue

                ids_activos_este_frame.add(nuestro_id)
                cajas_activas.append((x1, y1, x2, y2))

                # Filtro de objetos estáticos
                historial_posiciones[nuestro_id].append((cx, cy))
                if nuestro_id in ids_estaticos:
                    continue

                if len(historial_posiciones[nuestro_id]) >= FRAMES_PARA_VALIDAR:
                    pos = historial_posiciones[nuestro_id][-FRAMES_PARA_VALIDAR:]
                    xs  = [p[0] for p in pos]
                    ys  = [p[1] for p in pos]
                    if max(max(xs)-min(xs), max(ys)-min(ys)) < MOVIMIENTO_MINIMO:
                        ids_estaticos.add(nuestro_id)
                        pool_disponible.append(nuestro_id)
                        pool_disponible.sort()
                        viejo_bt = reverse_map.pop(nuestro_id, None)
                        if viejo_bt and viejo_bt in id_map:
                            del id_map[viejo_bt]
                        print(f"  [ESTÁTICO] ID:{nuestro_id} descartado, slot devuelto")
                        continue
                    historial_posiciones[nuestro_id] = historial_posiciones[nuestro_id][-12:]

                # Dibujar jugador
                color = color_por_id(nuestro_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                label = f'ID:{nuestro_id}'
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(frame, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
                cv2.putText(frame, label, (x1+2, y1-3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)

                datos_tracking.append({
                    'frame': frame_count, 'id': nuestro_id,
                    'x': cx, 'y': cy,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'en_oclusion': False
                })
        else:
            frames_sin_tracks += 1

        # ── TRACKING PELOTA ────────────────────────────────────
        tracks_pel = tracker_pelota.update(dets_pel_np, frame)

        if tracks_pel is not None and len(tracks_pel) > 0:
            # Tomamos solo el track con mayor confianza (hay una sola pelota)
            mejor_track = max(tracks_pel, key=lambda t: t[5])  # t[5] = conf
            bx1, by1, bx2, by2 = (int(mejor_track[0]), int(mejor_track[1]),
                                   int(mejor_track[2]), int(mejor_track[3]))
            bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2

            # Dibujar pelota en amarillo
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 255), 3)
            cv2.circle(frame, (bcx, bcy), 6, (0, 255, 255), -1)
            cv2.putText(frame, 'PELOTA', (bx1, by1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

            pos_pelota_historial.append({
                'frame': frame_count,
                'x': bcx, 'y': bcy
            })

        # ── HUD ────────────────────────────────────────────────
        ids_asignados = MAX_JUGADORES - len(pool_disponible) - len(ids_estaticos)
        minuto_actual = frame_count / fps_lectura / 60
        pelota_detectada = tracks_pel is not None and len(tracks_pel) > 0
        cv2.putText(frame,
            f'Min:{minuto_actual:.1f} | '
            f'Dets:{n_dets}/{n_dets_raw} | '
            f'IDs:{ids_asignados}/{MAX_JUGADORES} | '
            f'Activos:{len(ids_activos_este_frame)} | '
            f'Pelota:{"SI" if pelota_detectada else "NO"}',
            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)

        if frames_procesados % 150 == 0 and frames_procesados > 0:
            print(f"  Min {minuto_actual:.1f} | "
                  f"IDs:{ids_asignados}/{MAX_JUGADORES} | "
                  f"Activos:{len(ids_activos_este_frame)} | "
                  f"Estáticos:{len(ids_estaticos)} | "
                  f"Pelota:{'SI' if pelota_detectada else 'NO'}")

        out.write(frame)
        frame_count += 1

        cv2.imshow('TESIS - TRACKING', cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Detenido por usuario.")
            break

# =============================================================
# 8. CIERRE Y EXPORTACIÓN
# =============================================================
finally:
    vid.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n========== RESUMEN FINAL ==========")
    print(f"Frames procesados:         {frame_count - frame_inicio}")
    print(f"Detecciones totales:       {total_dets_raw}")
    print(f"Frames sin tracks:         {frames_sin_tracks}")
    print(f"IDs estáticos descartados: {len(ids_estaticos)} → {sorted(ids_estaticos)}")

    if datos_tracking:
        df = pd.DataFrame(datos_tracking)
        df = df[~df['id'].isin(ids_estaticos)]
        csv_path = output_dir / 'datos_finales_tesis.csv'
        df.to_csv(str(csv_path), index=False)
        print(f"\nIDs únicos jugadores: {df['id'].nunique()}")
        print(f"Registros jugadores:  {len(df)}")
        print(f"CSV en: {csv_path.resolve()}")
        print("\nFrames visibles por ID:")
        print(df.groupby('id').size().sort_values(ascending=False).to_string())

    if pos_pelota_historial:
        df_pelota = pd.DataFrame(pos_pelota_historial)
        csv_pelota = output_dir / 'pelota_tesis.csv'
        df_pelota.to_csv(str(csv_pelota), index=False)
        print(f"\nFrames con pelota detectada: {len(df_pelota)}")
        print(f"CSV pelota en: {csv_pelota.resolve()}")
    else:
        print("\n⚠ Pelota no detectada — YOLOv8m con cámara lejana tiene dificultades")
        print("  Considerá fine-tunear el modelo con imágenes de tu cancha")

    print("====================================")