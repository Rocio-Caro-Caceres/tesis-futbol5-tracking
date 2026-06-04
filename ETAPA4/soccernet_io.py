"""
soccernet_io.py
===============
Capa de extraccion de datos de SoccerNet.

Responsabilidad UNICA: bajar y localizar los archivos crudos (Labels-v2.json
y tracking.jsonl.gz de cada mitad), sin descargar nunca el video.

Estrategia:
    * `download()` envuelve `SoccerNetDownloader` con defaults academicos.
    * `iter_games()` recorre recursivamente la carpeta descargada y emite
      tuplas GameDir(match_id, path, [labels_path], {half: tracking_path}).
    * Si el usuario ya tiene los archivos (--raw-dir), NO descargamos nada.
"""
from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import DownloadConfig
from .io_utils import info, warn


# ---------------------------------------------------------------------------
# Modelo de un partido
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
# Slug de match_id
# ---------------------------------------------------------------------------
def slugify_match_id(folder_name: str) -> str:
    """
    '2014-2015\\2015-02-21 - 18-00 Chelsea 1 - 1 Burnley'
        -> '2014-2015_2015-02-21_chelsea_1_1_burnley'
    """
    # Quitar la parte de la hora tipo "18-00"
    s = re.sub(r"\b\d{1,2}-\d{2}\b\s*", "", folder_name)
    # Quitar marcadores tipo " 1 - 1 " -> "1_1"
    s = re.sub(r"\s+\d+\s*-\s*\d+\s+", " ", s)
    s = s.replace(" - ", "_").replace("-", "-")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s


# ---------------------------------------------------------------------------
# Walker
# ---------------------------------------------------------------------------
# Regex para detectar las dos variantes de tracking que se vieron en el dataset.
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

    SoccerNet deposita los juegos en arboles tipo:
        <root>/<league>/<season>/<game-folder>/{1,2}_*.json
    Por eso usamos `root.rglob('*_tracking.jsonl*')` y derivamos el game dir.
    """
    seen: set[Path] = set()
    for track_path in root.rglob("*_tracking.jsonl*"):
        # Aceptar .json, .jsonl, .json.gz, .jsonl.gz
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
    """Punto de entrada: itera GameDir segun la configuracion."""
    if cfg.raw_dir is not None:
        root = Path(cfg.raw_dir)
        if not root.exists():
            raise FileNotFoundError(f"--raw-dir no existe: {root}")
        info(f"Walking local SoccerNet tree at {root}")
        yield from _walk(root)
        return

    # Si no hay raw_dir, asumimos que ya se descargo en cfg.output_dir.
    root = Path(cfg.output_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"No existe {root}. Correr 'python -m ETAPA4 download' primero, "
            f"o pasar --raw-dir apuntando a una carpeta con SoccerNet ya bajado."
        )
    info(f"Walking downloaded SoccerNet tree at {root}")
    yield from _walk(root)


# ---------------------------------------------------------------------------
# Lectores de archivos
# ---------------------------------------------------------------------------
def read_tracking_jsonl(path: Path) -> Iterator[dict]:
    """
    Lee un tracking.jsonl[.gz] y emite dicts por linea.
    Acepta tanto jsonl (una deteccion por linea) como jsonl.gz.
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


def read_labels(path: Path) -> list[dict]:
    """Lee un archivo Labels-v2.json (lista de eventos)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: esperaba una lista de eventos, recibi {type(data)}")
    return data


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------
def download(cfg: DownloadConfig) -> None:
    """
    Wrapper delgadito sobre SoccerNetDownloader. Hace 3 cosas:
      1. Filtra la lista de archivos para NUNCA bajar .mp4 (video).
      2. Fija la password academica por default.
      3. Muestra cuanto se descargo al final.
    """
    forbidden = {
        "1_baidu_soccer_embeddings.npy",  # embeddings -> ignorarlos
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

    info(
        f"Descargando SoccerNet split={list(cfg.split)} files={list(files)} -> {out}"
    )
    d = SoccerNetDownloader(LocalDirectory=str(out))
    d.password = cfg.password
    d.downloadGames(files=list(files), split=list(cfg.split))

    n_games = sum(1 for _ in out.rglob("*_tracking.jsonl*"))
    info(f"Descarga completa. {n_games} partidos detectados con tracking.")


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
def count_games(cfg: DownloadConfig) -> int:
    return sum(1 for _ in iter_games(cfg))
