"""Run ACE-Step 1.5 text-to-music on the GPU and hand back raw audio.

Only the diffusion model (the "turbo" DiT), the VAE and the text encoder are loaded. The
5 Hz language model is skipped: it writes lyrics and plans song structure, which a bed
under narration does not need, and skipping it saves memory and time. On Apple Silicon
ACE-Step runs the DiT and the VAE on MLX and the text encoder on PyTorch MPS.

Everything is kept resident on the GPU. Offloading to the CPU between phases helps cards
with little VRAM, but on unified memory it only adds copies.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import weights
from .prompt import Conditioning

TURBO_STEPS = 8
TURBO_SHIFT = 3.0  # ACE-Step's documented value for the turbo checkpoint
MIN_SECONDS, MAX_SECONDS = 10, 600  # ACE-Step's supported duration range


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Raw:
    audio: np.ndarray  # (channels, samples), float32
    sample_rate: int
    seed: int
    timings: dict
    runtime: dict


def _model_device(handler) -> str:
    model = getattr(handler, "model", None)
    try:
        return next(model.parameters()).device.type
    except Exception:
        return "unknown"


def generate(
    cond: Conditioning, seconds: float, seed: int, device: str, checkpoints: Path, steps: int = TURBO_STEPS
) -> Raw:
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise EngineError(f"ACE-Step generates {MIN_SECONDS} to {MAX_SECONDS} seconds at a time, asked for {seconds:g}")

    os.environ.update(weights.offline_env(checkpoints))
    t0 = time.perf_counter()
    try:
        import torch
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
    except ImportError as exc:
        raise EngineError(f"ACE-Step is not installed ({exc}). Install the GPU extra: `uv sync --extra gpu`.") from exc

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(checkpoints.parent),
        config_path=weights.DIT_MODEL,
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
        use_mlx_dit=True,
    )
    if not ok:
        raise EngineError(f"ACE-Step failed to load: {status}")
    runtime = {
        "requested_device": device,
        "acestep_device": str(getattr(handler, "device", "unknown")),
        "model_device": _model_device(handler),
        "mlx_dit": bool(getattr(handler, "use_mlx_dit", False)),
        "mlx_vae": bool(getattr(handler, "use_mlx_vae", False)),
        "torch": torch.__version__,
    }
    # The device check that matters: ACE-Step resolves devices itself, so confirm it kept
    # the GPU we asked for instead of trusting the argument.
    if device != "cpu" and runtime["acestep_device"] != device:
        raise EngineError(f"asked ACE-Step for {device} but it loaded on {runtime['acestep_device']}")
    t_load = time.perf_counter()

    params = GenerationParams(
        task_type="text2music",
        caption=cond.caption,
        lyrics=cond.lyrics,
        instrumental=not cond.vocals,
        vocal_language="en" if cond.vocals else "unknown",
        bpm=cond.bpm,
        keyscale=cond.keyscale,
        duration=float(seconds),
        inference_steps=steps,
        shift=TURBO_SHIFT,
        seed=seed,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        # mac-music-gen masters to LUFS itself; ACE-Step's peak normalisation would only be undone.
        enable_normalization=False,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
    with tempfile.TemporaryDirectory(prefix="mac-music-gen-") as tmp:
        result = generate_music(handler, None, params, config, save_dir=tmp)
    t_gen = time.perf_counter()

    if not getattr(result, "success", False):
        raise EngineError(getattr(result, "error", None) or getattr(result, "status_message", "generation failed"))
    audios = getattr(result, "audios", None) or []
    if not audios or audios[0].get("tensor") is None:
        raise EngineError("ACE-Step reported success but returned no audio")
    tensor = audios[0]["tensor"]
    audio = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor, np.float32)
    return Raw(
        audio=audio,
        sample_rate=int(audios[0].get("sample_rate", 48000)),
        seed=seed,
        timings={"load_s": round(t_load - t0, 2), "generate_s": round(t_gen - t_load, 2)},
        runtime=runtime,
    )
