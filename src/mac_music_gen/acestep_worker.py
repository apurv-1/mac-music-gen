"""Child process that runs ACE-Step 1.5 on the GPU. The only module that imports torch, mlx or acestep.

The parent process starts it as `python -m mac_music_gen.acestep_worker <job_dir>`, with the
offline environment already set, after writing `job_dir/request.json`:

    {"checkpoints": str, "allow_cpu": bool,
     "jobs": [{"caption", "lyrics", "vocals", "bpm", "keyscale", "seconds", "seed", "steps"}]}

The worker loads the model once, runs every job, writes `raw-<i>.wav` as float32 with
shape (samples, channels), and finally writes `result.json`:

    {"ok": bool, "error_kind": str | None, "error": str | None, "runtime": {...}, "load_s": float,
     "jobs": [{"raw_path", "sample_rate", "seed", "generate_s"}]}

Running in a child process keeps three promises the parent cannot keep in-process:
offline mode holds even if the host already imported huggingface_hub, all Metal memory is
released when the job ends, and a crash in the model cannot take the host down with it.

`python -m mac_music_gen.acestep_worker --probe` prints what the GPU stack can see as JSON.

The device guard lives here, because only this process knows where the model really
loaded. ACE-Step falls back to the CPU when a device is missing, and from MLX to PyTorch
when MLX fails, both with only a log line. MLX and PyTorch draw noise differently, so a
silent switch would also change what a seed produces. Either fallback is an error here
unless the caller passed allow_cpu.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

TURBO_SHIFT = 3.0  # ACE-Step's documented value for the turbo checkpoint
DIT_MODEL = "acestep-v15-turbo"


class Failure(Exception):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def probe() -> dict:
    """Report what the GPU stack can see, without loading the model."""
    out: dict = {"engine_installed": False, "mps": False, "mlx_metal": False, "macos": platform.mac_ver()[0]}
    try:
        import torch

        out["torch"] = torch.__version__
        out["mps"] = bool(torch.backends.mps.is_available())
    except Exception as exc:
        out["error"] = f"PyTorch is not importable: {exc}"
        return out
    try:
        import mlx.core as mx

        out["mlx"] = getattr(mx, "__version__", "unknown")
        out["mlx_metal"] = bool(mx.metal.is_available())
    except Exception as exc:
        out["error"] = f"MLX is not importable: {exc}"
        return out
    try:
        import acestep.inference  # noqa: F401

        out["engine_installed"] = True
    except Exception as exc:
        out["error"] = f"ACE-Step is not importable: {exc}"
    return out


def check_gpu_before_load(found: dict, allow_cpu: bool) -> str:
    """Return the device to load on, or raise when the GPU is missing and CPU was not allowed."""
    if not found.get("engine_installed"):
        raise Failure("engine_not_installed", found.get("error") or "ACE-Step is not installed")
    if found.get("mps") and found.get("mlx_metal"):
        return "mps"
    if allow_cpu:
        print("WARNING: no GPU visible; running on the CPU because allow_cpu was set.", file=sys.stderr)
        return "cpu"
    from mac_music_gen.errors import NO_GPU_HELP

    raise Failure("gpu_unavailable", NO_GPU_HELP)


def _param_device(module) -> str:
    try:
        return next(module.parameters()).device.type
    except Exception:
        return "unknown"


def check_loaded(handler, device: str) -> dict:
    """Confirm ACE-Step kept every part on the GPU and on MLX. Returns the runtime record."""
    runtime = {
        "device": str(getattr(handler, "device", "unknown")),
        "mlx_dit": bool(getattr(handler, "use_mlx_dit", False)),
        "mlx_vae": bool(getattr(handler, "use_mlx_vae", False)),
        "text_encoder_device": _param_device(getattr(handler, "text_encoder", None)),
    }
    if device == "cpu":
        return runtime
    problems = []
    if runtime["device"] != device:
        problems.append(f"ACE-Step loaded on {runtime['device']} instead of {device}")
    if not runtime["mlx_dit"]:
        problems.append("the DiT fell back from MLX to PyTorch")
    if not runtime["mlx_vae"]:
        problems.append("the VAE fell back from MLX to PyTorch")
    if runtime["text_encoder_device"] != device:
        problems.append(f"the text encoder is on {runtime['text_encoder_device']}")
    if problems:
        raise Failure("gpu_unavailable", "; ".join(problems) + ". The log above shows why.")
    return runtime


def generation_kwargs(job: dict) -> dict:
    """GenerationParams for one bed: text-to-music, no language model, our own loudness."""
    return dict(
        task_type="text2music",
        caption=job["caption"],
        lyrics=job["lyrics"],
        instrumental=not job["vocals"],
        vocal_language="en" if job["vocals"] else "unknown",
        bpm=job.get("bpm"),
        keyscale=job.get("keyscale") or "",
        duration=float(job["seconds"]),
        inference_steps=int(job["steps"]),
        shift=TURBO_SHIFT,
        seed=int(job["seed"]),
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        enable_normalization=False,
    )


def run(request: dict, job_dir: Path) -> dict:
    found = probe()
    device = check_gpu_before_load(found, bool(request.get("allow_cpu")))

    import numpy as np
    import soundfile as sf
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    checkpoints = Path(request["checkpoints"])
    t0 = time.perf_counter()
    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(checkpoints.parent),
        config_path=DIT_MODEL,
        device=device,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
        use_mlx_dit=True,
    )
    if not ok:
        raise Failure("engine", f"ACE-Step failed to load: {status}")
    runtime = {
        **check_loaded(handler, device),
        "torch": found.get("torch"),
        "mlx": found.get("mlx"),
        "macos": found.get("macos"),
    }
    load_s = time.perf_counter() - t0

    results = []
    for i, job in enumerate(request["jobs"]):
        t1 = time.perf_counter()
        params = GenerationParams(**generation_kwargs(job))
        config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[int(job["seed"])], audio_format="wav")
        res = generate_music(handler, None, params, config, save_dir=str(job_dir / f"acestep-{i}"))
        if not getattr(res, "success", False):
            raise Failure("engine", getattr(res, "error", None) or getattr(res, "status_message", "generation failed"))
        audios = getattr(res, "audios", None) or []
        if not audios or audios[0].get("tensor") is None:
            raise Failure("engine", "ACE-Step reported success but returned no audio")
        audio = np.asarray(audios[0]["tensor"].detach().float().cpu().numpy(), dtype=np.float32)
        raw_path = job_dir / f"raw-{i}.wav"
        sf.write(str(raw_path), audio.T, int(audios[0].get("sample_rate", 48000)), subtype="FLOAT")
        results.append(
            {
                "raw_path": str(raw_path),
                "sample_rate": int(audios[0].get("sample_rate", 48000)),
                "seed": int(job["seed"]),
                "generate_s": round(time.perf_counter() - t1, 2),
            }
        )
    return {
        "ok": True,
        "error_kind": None,
        "error": None,
        "runtime": runtime,
        "load_s": round(load_s, 2),
        "jobs": results,
    }


def _write_result(job_dir: Path, result: dict) -> None:
    tmp = job_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result))
    tmp.replace(job_dir / "result.json")


def main(argv: list[str]) -> int:
    if argv == ["--probe"]:
        print(json.dumps(probe()))
        return 0
    if len(argv) != 1:
        print("usage: python -m mac_music_gen.acestep_worker <job_dir> | --probe", file=sys.stderr)
        return 2
    # ACE-Step and its libraries print freely; keep stdout clean for callers that parse it.
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    job_dir = Path(argv[0])
    try:
        result = run(json.loads((job_dir / "request.json").read_text()), job_dir)
    except Failure as exc:
        result = {"ok": False, "error_kind": exc.kind, "error": str(exc)}
    except Exception as exc:
        traceback.print_exc()
        result = {"ok": False, "error_kind": "engine", "error": f"{type(exc).__name__}: {exc}"}
    _write_result(job_dir, result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
