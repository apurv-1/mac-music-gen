"""mac-music-gen: local text-to-music on the Mac's GPU, tuned for background music."""

from .api import Bed, Diagnosis, doctor, fetch_weights, generate, master_file, package_version
from .errors import (
    EngineError,
    EngineNotInstalled,
    GpuUnavailable,
    InvalidRequest,
    MacMusicGenError,
    WeightsMissing,
)
from .master import Mastering, Report

__version__ = package_version()

__all__ = [
    "Bed",
    "Diagnosis",
    "EngineError",
    "EngineNotInstalled",
    "GpuUnavailable",
    "InvalidRequest",
    "MacMusicGenError",
    "Mastering",
    "Report",
    "WeightsMissing",
    "doctor",
    "fetch_weights",
    "generate",
    "master_file",
]
