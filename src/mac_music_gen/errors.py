"""Every error mac-music-gen raises on purpose. Each carries the CLI exit code for its kind."""

from __future__ import annotations


class MacMusicGenError(Exception):
    exit_code = 1


class InvalidRequest(MacMusicGenError, ValueError):
    """The arguments cannot produce a bed. Raised before any GPU work starts."""

    exit_code = 2


class WeightsMissing(MacMusicGenError):
    exit_code = 3


class GpuUnavailable(MacMusicGenError):
    exit_code = 4


class EngineNotInstalled(MacMusicGenError):
    exit_code = 5


class EngineError(MacMusicGenError):
    exit_code = 5


NO_GPU_HELP = (
    "the Mac's GPU is not visible to this process (PyTorch MPS or MLX Metal is unavailable). "
    "Agent sandboxes block Metal, so run the command in your own terminal. "
    "Pass --allow-cpu only if you accept a run that takes many minutes."
)
