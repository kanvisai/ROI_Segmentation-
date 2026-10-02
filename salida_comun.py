"""Carpetas de salida y tabla comparativa, iguales para cada tecnología."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

CAMPOS = [
    "clip",
    "backend",
    "ok",
    "frames",
    "fps",
    "ms_frame",
    "ms_segmentacion",
    "ram_pico_mb",
    "vram_pico_mb",
    "cpu_pct",
    "objetos",
    "desapariciones",
    "frames_con_objeto",
    "codigo_salida",
    "error",
    "carpeta",
]


def carpeta_de(raiz: Path, tecnologia: str, video: Path) -> Path:
    """salidas/<tecnologia>/<clip>/. Si la raíz ya se llama como la tecnología, no se duplica."""
    base = raiz if raiz.name == tecnologia else raiz / tecnologia
    destino = base / video.stem
    destino.mkdir(parents=True, exist_ok=True)
    return destino


def raiz_salidas(raiz: Path, tecnologia: str) -> Path:
    return raiz.parent if raiz.name == tecnologia else raiz


class Medidor:
    """Tiempo, RAM, VRAM y CPU de una pasada sobre un vídeo."""

    def __init__(self, backend: str):
        self.backend = backend
        self._t0 = time.perf_counter()
        self._etapas: dict[str, list[float]] = defaultdict(list)
        self._ram_pico = 0
        self._proceso = None
        try:
            import psutil

            self._proceso = psutil.Process()
            self._proceso.cpu_percent(None)
            self._ram_pico = self._proceso.memory_info().rss
        except Exception:
            self._proceso = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    @contextmanager
    def etapa(self, nombre: str):
        inicio = time.perf_counter()
        try:
            yield
        finally:
            self._etapas[nombre].append(time.perf_counter() - inicio)
            self._mirar_ram()

    def _mirar_ram(self) -> None:
        if self._proceso is None:
            return
        try:
            self._ram_pico = max(self._ram_pico, self._proceso.memory_info().rss)
        except Exception:
            pass

    def cerrar(self, frames: int) -> dict:
        duracion = max(1e-9, time.perf_counter() - self._t0)
        self._mirar_ram()
        cpu = 0.0
        if self._proceso is not None:
            try:
                cpu = float(self._proceso.cpu_percent(None))
            except Exception:
                cpu = 0.0
        vram = 0.0
        try:
            import torch

            if torch.cuda.is_available():
                vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        except Exception:
            vram = 0.0
        etapas = {}
        for nombre, valores in self._etapas.items():
            if not valores:
                continue
            etapas[nombre] = {
                "calls": len(valores),
                "mean_ms": round(1000 * sum(valores) / len(valores), 2),
                "total_s": round(sum(valores), 3),
            }
        return {
            "backend": self.backend,
            "processed_frames": frames,
            "total_duration_s": round(duracion, 3),
            "mean_fps": round(frames / duracion, 2) if frames else 0.0,
            "ms_per_frame": round(1000 * duracion / frames, 2) if frames else None,
            "peak_ram_mb": round(self._ram_pico / (1024 * 1024), 1),
            "peak_vram_mb": round(vram, 1),
            "mean_cpu_pct": round(cpu, 1),
            "stages": etapas,
        }


def publicar_comparativa(
    raiz: Path,
    tecnologia: str,
    clip: str,
    backend: str,
    carpeta: Path,
    metricas: dict | None,
    objetos: int,
    desapariciones: int,
    frames_con_objeto: int,
    ok: bool = True,
    codigo: int = 0,
    error: str = "",
) -> Path:
    """Actualiza la fila de este clip y esta tecnología en salidas/comparativa/."""
    metricas = metricas or {}
    seg = (metricas.get("stages") or metricas.get("etapas") or {}).get("segmentacion") or {}
    fila = {
        "clip": clip,
        "backend": backend,
        "ok": ok,
        "frames": metricas.get("processed_frames", metricas.get("frames_procesados", "")),
        "fps": metricas.get("mean_fps", metricas.get("fps_medio", "")),
        "ms_frame": metricas.get("ms_per_frame", metricas.get("ms_por_frame", "")),
        "ms_segmentacion": seg.get("mean_ms", seg.get("media_ms", "")),
        "ram_pico_mb": metricas.get("peak_ram_mb", metricas.get("ram_proceso_pico_mb", "")),
        "vram_pico_mb": metricas.get("peak_vram_mb", metricas.get("vram_pico_mb", "")),
        "cpu_pct": metricas.get("mean_cpu_pct", metricas.get("cpu_medio_pct", "")),
        "objetos": objetos,
        "desapariciones": desapariciones,
        "frames_con_objeto": frames_con_objeto,
        "codigo_salida": codigo,
        "error": error,
        "carpeta": str(carpeta),
    }
    destino = raiz_salidas(raiz, tecnologia) / "comparativa"
    destino.mkdir(parents=True, exist_ok=True)
    csv_path = destino / "comparativa.csv"
    filas = leer_comparativa(csv_path)
    previas = [
        item
        for item in filas
        if (item.get("clip"), item.get("backend")) == (clip, backend)
    ]
    filas = [item for item in filas if item not in previas]
    # Columnas añadidas a mano al CSV (revisión, notas): la fila nueva las hereda.
    for previa in previas:
        for campo, valor in previa.items():
            if campo not in CAMPOS:
                fila.setdefault(campo, valor)
    filas.append(fila)
    return volcar_comparativa(destino, filas)


def leer_comparativa(csv_path: Path) -> list[dict]:
    """Lee el CSV. Un editor de hojas de cálculo puede haberlo guardado en latin-1."""
    if not csv_path.is_file():
        return []
    for codificacion in ("utf-8-sig", "latin-1"):
        try:
            with csv_path.open(encoding=codificacion, newline="") as fichero:
                return list(csv.DictReader(fichero))
        except UnicodeDecodeError:
            continue
    return []


def volcar_comparativa(destino: Path, filas: list[dict]) -> Path:
    """Escribe comparativa.csv, .json y .md con las filas dadas, ordenadas por clip."""
    destino.mkdir(parents=True, exist_ok=True)
    columnas = list(CAMPOS)
    for item in filas:
        for campo in item:
            if campo not in columnas:
                columnas.append(campo)
    filas = [{campo: item.get(campo, "") for campo in columnas} for item in filas]
    filas.sort(key=lambda item: (item.get("clip") or "", item.get("backend") or ""))
    csv_path = destino / "comparativa.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fichero:
        escritor = csv.DictWriter(fichero, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)
    (destino / "comparativa.json").write_text(
        json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lineas = [
        "# Comparison",
        "",
        "One row per clip and technology.",
        "",
        "| " + " | ".join(columnas) + " |",
        "|" + "|".join("---" for _ in columnas) + "|",
    ]
    for item in filas:
        celdas = []
        for campo in columnas:
            valor = item.get(campo, "")
            if campo == "ok":
                valor = "si" if str(valor) in ("True", "true", "si") else "NO"
            celdas.append(str(valor) if valor not in (None, "") else "-")
        lineas.append("| " + " | ".join(celdas) + " |")
    (destino / "comparativa.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return csv_path
