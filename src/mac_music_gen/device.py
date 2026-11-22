"""Pick the GPU, and refuse to quietly run on the CPU.

A 30-second bed takes seconds on Apple's GPU and many minutes on the CPU. A slow CPU run
that looks like it worked is the failure this module exists to prevent: if no GPU is
visible, generation stops with an explanation, unless the caller passes --allow-cpu, in
which case a warning is printed and recorded in the sidecar.
"""

from __future__ import annotations

import sys


class GpuUnavailable(RuntimeError):
    pass


NO_GPU_HELP = (
    "no GPU is visible to PyTorch (neither Apple MPS nor CUDA). "
    "If you are inside an agent sandbox, Metal is blocked there: run the command in your own terminal. "
    "Pass --allow-cpu only if you really want a run that takes many minutes."
)


def detect() -> str:
    """Return 'mps', 'cuda' or 'cpu' for what PyTorch can actually use right now."""
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def choose(allow_cpu: bool = False) -> str:
    found = detect()
    if found != "cpu":
        return found
    if not allow_cpu:
        raise GpuUnavailable(NO_GPU_HELP)
    print("WARNING: running on the CPU because --allow-cpu was passed. Expect many minutes.", file=sys.stderr)
    return "cpu"


def mlx_available() -> bool:
    """True when Apple's MLX can see a GPU. ACE-Step runs its DiT and VAE on MLX when it can."""
    try:
        import mlx.core as mx
    except Exception:
        return False
    try:
        return bool(mx.metal.is_available())
    except Exception:
        return False
