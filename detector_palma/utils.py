"""Utilidades transversales: consola, importaciones perezosas y seleccion de dispositivo."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


def configurar_consola() -> None:
    """Fuerza UTF-8 en stdout/stderr para que los acentos no rompan la salida en Windows."""
    for flujo in (sys.stdout, sys.stderr):
        try:
            flujo.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


class DependenciaFaltante(RuntimeError):
    """Se lanza cuando falta un paquete y queremos dar instrucciones de instalacion."""


def importar(modulo: str, ayuda: str):
    """Importa `modulo` o lanza un error explicando como instalarlo."""
    try:
        return importlib.import_module(modulo)
    except ImportError as exc:
        raise DependenciaFaltante(
            f"No se ha podido importar '{modulo}'.\n"
            f"  Motivo: {exc}\n"
            f"  Solucion: {ayuda}"
        ) from exc


def importar_opcional(modulo: str):
    """Devuelve el modulo o None si no esta disponible."""
    try:
        return importlib.import_module(modulo)
    except ImportError:
        return None


def resolver_dispositivo(preferencia: str = "auto") -> str:
    """Traduce 'auto' al mejor dispositivo disponible y valida el resto."""
    torch = importar_opcional("torch")
    if torch is None:
        return "cpu"
    if preferencia != "auto":
        if preferencia.startswith("cuda") and not torch.cuda.is_available():
            print("[aviso] Se ha pedido CUDA pero no hay GPU NVIDIA visible. Se usara CPU.")
            return "cpu"
        return preferencia
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configurar_hilos(hilos: int | None) -> None:
    """Limita los hilos de PyTorch/OpenMP, util para medir consumo de forma reproducible en CPU."""
    if not hilos or hilos <= 0:
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(hilos))
    torch = importar_opcional("torch")
    if torch is not None:
        torch.set_num_threads(hilos)
    cv2 = importar_opcional("cv2")
    if cv2 is not None:
        cv2.setNumThreads(hilos)


def directorio_salida(base: str | os.PathLike | None, backend: str, video: str | os.PathLike) -> Path:
    """Construye (y crea) la carpeta de resultados para un backend y un video dados."""
    if base:
        destino = Path(base)
    else:
        destino = Path("salidas") / backend / Path(video).stem
    destino.mkdir(parents=True, exist_ok=True)
    # Ruta absoluta: backends como Cutie/Track-Anything hacen os.chdir al repo.
    return destino.resolve()


def anadir_ruta_repo(ruta: str | os.PathLike | None) -> None:
    """Agrega el directorio de un repositorio clonado al sys.path para poder importarlo."""
    if not ruta:
        return
    ruta = str(Path(ruta).expanduser().resolve())
    if ruta not in sys.path:
        sys.path.insert(0, ruta)
