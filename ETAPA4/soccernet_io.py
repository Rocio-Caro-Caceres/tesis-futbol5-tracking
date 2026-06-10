"""
soccernet_io.py
===============
Capa de extraccion de datos de SoccerNet.

Responsabilidad UNICA: bajar y localizar los archivos crudos (Labels-v2.json
y tracking de cada partido/mitad), sin descargar nunca el video.

Soporta dos formatos de tracking:
  * "jsonl": JSONL.GZ legacy (tracking.jsonl.gz por mitad) — ya no disponible
    en SoccerNet actual.
  * "mot": MOT20 CSV (gt.txt por clip de 30s, via downloadDataTask).

Estrategia:
    * `download()` envuelve `SoccerNetDownloader` con defaults academicos.
    * `iter_games()` / `iter_games_mot()` recorren la carpeta descargada.
    * Si el usuario ya tiene los archivos (--raw-dir), NO descargamos nada.
"""
from __future__ import annotations

import configparser
import csv
import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import DownloadConfig
from .io_utils import info, warn


# ---------------------------------------------------------------------------
# Modelo de un partido (JSONL legacy)
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class GameDir:
    """
    Ubicacion y archivos relevantes de un partido SoccerNet.

    Attributes:
        match_id: id estable tipo 'england_epl_2014-2015_chelsea_burnley'
                  (slug del nombre de la carpeta original).
        path: carpeta raiz del partido en disco.
        labels_path: ruta al Labels-v2.json combinado (o None si falta).
        half_labels: dict {1: path, 2: path} a 1_Labels-v2.json/2_Labels-v2.json
        tracking_paths: dict {1: path, 2: path} a {1,2}_tracking.jsonl.gz
    """
    match_id: str
    path: Path
    labels_path: Path | None
    half_labels: dict[int, Path]
    tracking_paths: dict[int, Path]

    def has_minimum(self) -> bool:
        """Tracking de al menos una mitad + al menos un Labels-v2."""
        return bool(self.tracking_paths) and (
            self.labels_path is not None or bool(self.half_labels)
        )


# ---------------------------------------------------------------------------
# Modelo de un clip MOT
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class GameDirMot:
    """
    Ubicacion y archivos relevantes de un clip de tracking MOT20.

    Attributes:
        match_id: slug unico del clip (ej 'snmot12_train_clip001').
        path: carpeta del clip en disco.
        mot_gt_path: ruta al gt/gt.txt (ground truth MOT20).
        seqinfo_path: ruta al seqinfo.ini (dimensiones, fps).
        labels_path: ruta al Labels-v2.json del juego padre (o None).
        img_w: ancho de imagen del clip (de seqinfo.ini).
        img_h: alto de imagen del clip (de seqinfo.ini).
        fps: frames por segundo del clip (de seqinfo.ini).
        seq_length: cantidad total de frames del clip.
    """
    match_id: str
    path: Path
    mot_gt_path: Path
    seqinfo_path: Path
    labels_path: Path | None = None
    img_w: int = 1920
    img_h: int = 1080
    fps: float = 25.0
    seq_length: int = 0

    def has_minimum(self) -> bool:
        """Al menos tiene un gt.txt."""
        return self.mot_gt_path.is_file()


# ---------------------------------------------------------------------------
# Slug de match_id
# ---------------------------------------------------------------------------
def slugify_match_id(folder_name: str) -> str:
    """
    '2014-2015\\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley'
        -> '2014-2015_2015-02-21_chelsea_1_1_burnley'
    """
    s = re.sub(r"\b\d{1,2}-\d{2}\b\s*", "", folder_name)
    s = re.sub(r"\s+\d+\s*-\s*\d+\s+", " ", s)
    s = s.replace(" - ", "_").replace("-", "-")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


def _slugify_clip(root_name: str, clip_name: str) -> str:
    """Genera un match_id unico para un clip MOT."""
    root_slug = slugify_match_id(root_name) if root_name else "unknown"
    clip_slug = slugify_match_id(clip_name) if clip_name else "clip0"
    return f"mot_{root_slug}_{clip_slug}"


# ---------------------------------------------------------------------------
# Walker — JSONL legacy
# ---------------------------------------------------------------------------
TRACKING_PATTERNS = (
    re.compile(r"^([12])_tracking\.jsonl?(\.gz)?$"),
    re.compile(r"^tracking_([12])\.jsonl?(\.gz)?$"),
)
LABELS_COMBINED = re.compile(r"^Labels-v2\.json$")
LABELS_HALF = re.compile(r"^([12])_Labels-v2\.json$")


def _classify(path: Path) -> tuple[str, int | None] | None:
    """Devuelve ('tracking', half) | ('labels', None) | ('half_labels', half) | None."""
    name = path.name
    m = TRACKING_PATTERNS[0].match(name)
    if m:
        return ("tracking", int(m.group(1)))
    m = TRACKING_PATTERNS[1].match(name)
    if m:
        return ("tracking", int(m.group(1)))
    if LABELS_COMBINED.match(name):
        return ("labels", None)
    m = LABELS_HALF.match(name)
    if m:
        return ("half_labels", int(m.group(1)))
    return None


def _walk(root: Path) -> Iterator[GameDir]:
    """
    Yield GameDir por cada carpeta que contenga al menos un tracking y labels.
    """
    seen: set[Path] = set()
    for track_path in root.rglob("*_tracking.jsonl*"):
        if not (track_path.suffix in (".json", ".jsonl")
                or track_path.name.endswith(".json.gz")
                or track_path.name.endswith(".jsonl.gz")):
            continue
        game_dir = track_path.parent
        if game_dir in seen:
            continue
        seen.add(game_dir)

        labels_path: Path | None = None
        half_labels: dict[int, Path] = {}
        tracking_paths: dict[int, Path] = {}
        for p in game_dir.iterdir():
            if not p.is_file():
                continue
            cls = _classify(p)
            if cls is None:
                continue
            kind, half = cls
            if kind == "tracking":
                tracking_paths[half] = p
            elif kind == "labels":
                labels_path = p
            elif kind == "half_labels" and half is not None:
                half_labels[half] = p

        if not tracking_paths:
            continue
        if labels_path is None and not half_labels:
            continue

        match_id = slugify_match_id(game_dir.name)
        yield GameDir(
            match_id=match_id,
            path=game_dir,
            labels_path=labels_path,
            half_labels=half_labels,
            tracking_paths=tracking_paths,
        )


def iter_games(cfg: DownloadConfig) -> Iterator[GameDir]:
    """Punto de entrada JSONL: itera GameDir segun la configuracion."""
    if cfg.raw_dir is not None:
        root = Path(cfg.raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"--raw-dir no existe: {root}")
        info(f"Walking local SoccerNet tree at {root}")
        yield from _walk(root)
        return

    root = Path(cfg.output_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"No existe {root}. Correr 'python -m ETAPA4 download' primero, "
            f"o pasar --raw-dir apuntando a una carpeta con SoccerNet ya bajado."
        )
    info(f"Walking downloaded SoccerNet tree at {root}")
    yield from _walk(root)


# ---------------------------------------------------------------------------
# Walker — MOT20 tracking
# ---------------------------------------------------------------------------
def _parse_seqinfo(seqinfo_path: Path) -> dict:
    """
    Lee un seqinfo.ini estilo MOTChallenge y devuelve dict con:
    imWidth, imHeight, frameRate, seqLength, name.
    """
    cp = configparser.ConfigParser()
    cp.read(str(seqinfo_path))
    result: dict = {}
    if cp.has_section("Sequence"):
        for key in ("imWidth", "imHeight", "frameRate", "seqLength", "name"):
            if cp.has_option("Sequence", key):
                result[key] = cp.get("Sequence", key)
    # Convertir tipos numericos
    for int_key in ("imWidth", "imHeight", "seqLength"):
        if int_key in result:
            try:
                result[int_key] = int(result[int_key])
            except (ValueError, TypeError):
                pass
    if "frameRate" in result:
        try:
            result["frameRate"] = float(result["frameRate"])
        except (ValueError, TypeError):
            pass
    return result


def _find_labels_for_clip(clip_path: Path, labels_index: dict[str, Path]) -> Path | None:
    """
    Intenta encontrar un Labels-v2.json que corresponda al clip MOT.
    Busca por nombre de juego en el path del clip.
    """
    # El path del clip tipicamente contiene el nombre del juego:
    # .../tracking/train/<game_name>/<clip_name>/gt/gt.txt
    # Intentamos matchear con los labels disponibles.
    parts = [p.lower() for p in clip_path.parts]
    for key, label_path in labels_index.items():
        key_parts = key.lower().split("_")
        # Si al menos 3 tokens del key aparecen en el path del clip
        matches = sum(1 for kp in key_parts if any(kp in pp for pp in parts))
        if matches >= 3:
            return label_path
    return None


def _walk_mot(root: Path, labels_index: dict[str, Path] | None = None) -> Iterator[GameDirMot]:
    """
    Recorre la estructura MOT20 y yield GameDirMot por cada clip.

    Estructura esperada:
        <root>/<split>/<game_or_clip>/gt/gt.txt
        <root>/<split>/<game_or_clip>/seqinfo.ini

    Tambien soporta:
        <root>/<split>/<game>/<clip>/gt/gt.txt
    """
    # Buscar todos los gt.txt (ground truth MOT20)
    for gt_path in sorted(root.rglob("gt.txt")):
        clip_dir = gt_path.parent.parent  # subir de gt/ al clip
        if not clip_dir.is_dir():
            continue

        seqinfo_path = clip_dir / "seqinfo.ini"
        info_map: dict = {}
        if seqinfo_path.is_file():
            info_map = _parse_seqinfo(seqinfo_path)

        clip_name = info_map.get("name", clip_dir.name)
        # Buscar el nombre del juego padre (2 niveles arriba del clip o 1)
        parent_game = clip_dir.parent.name if clip_dir.parent != root else "unknown"
        # Si hay un nivel extra (split), subir mas
        if parent_game.lower() in ("train", "test", "challenge", "valid"):
            parent_game = clip_dir.parent.parent.name if clip_dir.parent.parent != root else "unknown"

        match_id = _slugify_clip(parent_game, clip_name)

        labels_path = None
        if labels_index:
            labels_path = _find_labels_for_clip(gt_path, labels_index)

        yield GameDirMot(
            match_id=match_id,
            path=clip_dir,
            mot_gt_path=gt_path,
            seqinfo_path=seqinfo_path,
            labels_path=labels_path,
            img_w=info_map.get("imWidth", 1920),
            img_h=info_map.get("imHeight", 1080),
            fps=info_map.get("frameRate", 25.0),
            seq_length=info_map.get("seqLength", 0),
        )


def _build_labels_index(root: Path) -> dict[str, Path]:
    """
    Indexa todos los Labels-v2.json encontrados bajo root.
    Devuelve {slug_match_id: path}.
    """
    index: dict[str, Path] = {}
    for label_path in root.rglob("Labels-v2.json"):
        game_dir = label_path.parent
        match_id = slugify_match_id(game_dir.name)
        index[match_id] = label_path
    return index


def iter_games_mot(cfg: DownloadConfig) -> Iterator[GameDirMot]:
    """Punto de entrada MOT: itera GameDirMot segun la configuracion."""
    if cfg.raw_dir is not None:
        root = Path(cfg.raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"--raw-dir no existe: {root}")
        info(f"Walking local MOT tracking tree at {root}")
    else:
        root = Path(cfg.output_dir)
        if not root.exists():
            raise FileNotFoundError(
                f"No existe {root}. Correr 'python -m ETAPA4 download' primero, "
                f"o pasar --raw-dir apuntando a una carpeta con SoccerNet ya bajado."
            )
        info(f"Walking downloaded MOT tracking tree at {root}")

    # Indexar labels si existen (para intentar matchear con clips)
    labels_index = _build_labels_index(root)
    if labels_index:
        info(f"Labels-v2.json encontrados: {len(labels_index)}")

    yield from _walk_mot(root, labels_index)


# ---------------------------------------------------------------------------
# Lectores de archivos
# ---------------------------------------------------------------------------
def read_tracking_jsonl(path: Path) -> Iterator[dict]:
    """
    Lee un tracking.jsonl[.gz] y emite dicts por linea.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                warn(f"Linea invalida en {path.name}: {e}")
                continue


def read_tracking_mot(path: Path) -> Iterator[dict]:
    """
    Lee un gt.txt / det.txt en formato MOT20 (10 columnas CSV) y emite
    dicts por linea con campos tipados.

    Formato MOT20:
        frame_id, track_id, bbox_x, bbox_y, width, height, conf, -1, -1, -1
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            try:
                frame_id = int(row[0])
                track_id = int(row[1])
                bbox_x = float(row[2])
                bbox_y = float(row[3])
                width = float(row[4])
                height = float(row[5])
                conf = float(row[6])
            except (ValueError, IndexError):
                continue
            yield {
                "frame": frame_id,
                "track_id": track_id,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "width": width,
                "height": height,
                "confidence": conf,
            }


def read_labels(path: Path) -> list[dict]:
    """
    Lee un archivo Labels-v2.json (lista de eventos).
    Soporta dos formatos:
      1. Array directo: [{"label": ...}, ...]
      2. Dict con key "annotations": {"annotations": [{"label": ...}, ...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Formato real: dict con key "annotations"
        annotations = data.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError(f"{path}: 'annotations' no es una lista, recibi {type(annotations)}")
        return annotations
    if isinstance(data, list):
        return data
    raise ValueError(f"{path}: esperaba lista o dict, recibi {type(data)}")


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def download(cfg: DownloadConfig) -> None:
    """
    Wrapper delgadito sobre SoccerNetDownloader. Hace 3 cosas:
      1. Filtra la lista de archivos para NUNCA bajar .mp4 (video).
      2. Fija la password academica por default.
      3. Muestra cuanto se descargo al final.

    Si tracking_format es "mot", tambien descarga tracking via downloadDataTask.
    """
    forbidden = {
        "1_baidu_soccer_embeddings.npy",
        "2_baidu_soccer_embeddings.npy",
    }
    files = tuple(f for f in cfg.files if f not in forbidden)
    if any(f.endswith(".mp4") for f in files):
        raise ValueError(
            f"DownloadConfig.files incluye video (.mp4). Eso esta prohibido en Etapa 4."
        )

    try:
        from SoccerNet.Downloader import SoccerNetDownloader  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Falta el paquete 'SoccerNet'. Instalalo con "
            "'pip install -r requirements-etapa4.txt'."
        ) from e

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    d = SoccerNetDownloader(LocalDirectory=str(out))
    d.password = cfg.password

    # 1) Descargar Labels-v2.json (siempre)
    label_files = [f for f in files if "Labels" in f or "labels" in f]
    if label_files:
        info(f"Descargando Labels split={list(cfg.split)} files={label_files} -> {out}")
        d.downloadGames(files=label_files, split=list(cfg.split))

    # 2) Descargar tracking segun formato
    if cfg.tracking_format == "mot":
        info(f"Descargando tracking MOT (downloadDataTask) split={list(cfg.split)} -> {out}")
        try:
            d.downloadDataTask(task="tracking", split=list(cfg.split))
            info("Tracking MOT descargado correctamente.")
        except Exception as e:
            warn(f"Error descargando tracking MOT: {e}")
            info("Intentando continuar con labels solamente...")
    else:
        # JSONL legacy
        jsonl_files = [f for f in files if "tracking" in f]
        if jsonl_files:
            info(f"Descargando tracking JSONL split={list(cfg.split)} files={jsonl_files} -> {out}")
            d.downloadGames(files=jsonl_files, split=list(cfg.split))

    # Resumen
    n_labels = sum(1 for _ in out.rglob("Labels-v2.json"))
    n_mot = sum(1 for _ in out.rglob("gt.txt"))
    n_jsonl = sum(1 for _ in out.rglob("*_tracking.jsonl*"))
    info(f"Descarga completa. {n_labels} Labels-v2.json, {n_mot} clips MOT, {n_jsonl} JSONL tracking.")


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
def count_games(cfg: DownloadConfig) -> int:
    if cfg.tracking_format == "mot":
        return sum(1 for _ in iter_games_mot(cfg))
    return sum(1 for _ in iter_games(cfg))
