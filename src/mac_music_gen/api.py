"""The public Python API. The CLI is a thin layer over these four functions.

    from mac_music_gen import generate
    beds = generate("calm lo-fi piano", "bed.wav", seconds=30, seed=42)
    print(beds[0].measured.integrated_lufs)

Importing this module never imports torch: the model runs in a separate worker process.
"""

from __future__ import annotations

import json
import math
import secrets
import shlex
import tempfile
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import _runner, master, prompt, weights
from .errors import InvalidRequest
from .master import Mastering, Report

MIN_SECONDS, MAX_SECONDS = 10, 600  # what ACE-Step generates in one pass
DEFAULT_STEPS = 8  # the turbo checkpoint's value
MAX_VARIANTS = 16


def package_version() -> str:
    try:
        return version("mac-music-gen")
    except PackageNotFoundError:
        return "0+unknown"


@dataclass(frozen=True)
class Bed:
    path: Path
    sidecar: Path
    seed: int
    seconds: float
    sample_rate: int
    caption: str
    measured: Report
    runtime: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "sidecar": str(self.sidecar),
            "seed": self.seed,
            "seconds": self.seconds,
            "sample_rate": self.sample_rate,
            "caption": self.caption,
            "measured": self.measured.as_dict(),
            "runtime": self.runtime,
            "timings": self.timings,
        }


@dataclass(frozen=True)
class Diagnosis:
    gpu: bool
    mlx: bool
    engine_installed: bool
    weights_present: bool
    weights_path: Path
    versions: dict
    problems: list[str]

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict:
        return {**asdict(self), "weights_path": str(self.weights_path), "ok": self.ok}


def _variant_paths(out: Path, variants: int) -> list[Path]:
    if variants == 1:
        return [out]
    return [out.with_name(f"{out.stem}-{i}{out.suffix}") for i in range(1, variants + 1)]


def _reproduce(
    prompt_text: str,
    out: Path,
    seconds: float,
    seed: int,
    bpm: int | None,
    key: str | None,
    steps: int,
    lyrics_file: str | None,
    m: Mastering,
) -> str:
    argv = [
        "mac-music-gen",
        "generate",
        "--prompt",
        prompt_text,
        "--seconds",
        f"{seconds:g}",
        "--seed",
        str(seed),
        "--steps",
        str(steps),
    ]
    if bpm:
        argv += ["--bpm", str(bpm)]
    if key:
        argv += ["--key", key]
    if m.loop:
        argv += ["--loop", "--crossfade", f"{m.loop_crossfade:g}"]
    else:
        argv += ["--fade-in", f"{m.fade_in:g}", "--fade-out", f"{m.fade_out:g}"]
    argv += ["--lufs", f"{m.lufs:g}", "--ceiling", f"{m.ceiling_db:g}"]
    if lyrics_file:
        argv += ["--lyrics", lyrics_file]
    argv += ["--out", str(out)]
    return shlex.join(argv)


def _write_sidecar(path: Path, data: dict) -> Path:
    sidecar = path.with_suffix(path.suffix + ".json")
    sidecar.write_text(json.dumps(data, indent=2) + "\n")
    return sidecar


def generate(
    prompt_text: str,
    out: str | Path,
    *,
    seconds: float = 60.0,
    bpm: int | None = None,
    key: str | None = None,
    seed: int | None = None,
    variants: int = 1,
    lyrics: str | None = None,
    lyrics_file: str | None = None,
    steps: int = DEFAULT_STEPS,
    mastering: Mastering | None = None,
    allow_cpu: bool = False,
    timeout: float | None = 1800,
    verbose: bool = False,
    home: str | Path | None = None,
) -> list[Bed]:
    """Generate `variants` mastered beds. Variant i uses seed + i and is written as out-<i+1>.wav.

    Everything that can be checked is checked before the GPU worker starts, so a bad
    argument never costs a model load.
    """
    m = mastering or Mastering()
    out = Path(out)
    if out.suffix.lower() != ".wav":
        raise InvalidRequest(f"the output must be a .wav file, got {out.name}")
    if not 1 <= variants <= MAX_VARIANTS:
        raise InvalidRequest(f"variants must be between 1 and {MAX_VARIANTS}, got {variants}")
    if not 1 <= steps <= 100:
        raise InvalidRequest(f"steps must be between 1 and 100, got {steps}")
    master.validate(m, seconds)
    # A loop needs audio past the loop point to crossfade into the head.
    source_seconds = max(MIN_SECONDS, math.ceil(seconds + (m.loop_crossfade if m.loop else 0)))
    if source_seconds > MAX_SECONDS:
        raise InvalidRequest(f"the model generates at most {MAX_SECONDS} s per bed, asked for {source_seconds} s")
    if lyrics is None and lyrics_file:
        lyrics = Path(lyrics_file).read_text()
    cond = prompt.build(prompt_text, bpm=bpm, key=key or "", lyrics=lyrics)
    base_seed = seed if seed is not None else secrets.randbelow(2**31 - MAX_VARIANTS)
    checkpoints = weights.Store.default(home).require()
    paths = _variant_paths(out, variants)

    request = {
        "checkpoints": str(checkpoints),
        "allow_cpu": allow_cpu,
        "jobs": [
            {
                "caption": cond.caption,
                "lyrics": cond.lyrics,
                "vocals": cond.vocals,
                "bpm": cond.bpm,
                "keyscale": cond.keyscale,
                "seconds": source_seconds,
                "seed": base_seed + i,
                "steps": steps,
            }
            for i in range(variants)
        ],
    }
    beds = []
    with tempfile.TemporaryDirectory(prefix="mac-music-gen-") as job_dir:
        result = _runner.run_worker(
            request, Path(job_dir), weights.offline_env(checkpoints), timeout=timeout, verbose=verbose
        )
        runtime = {**result.get("runtime", {}), "acestep_commit": weights.ACESTEP_COMMIT}
        for path, job in zip(paths, result["jobs"], strict=True):
            t0 = time.perf_counter()
            raw, sr = master.read(job["raw_path"])
            audio, rep = master.master(raw, sr, seconds, m)
            master.write(path, audio, sr)
            timings = {
                "load_s": result.get("load_s"),
                "generate_s": job["generate_s"],
                "master_s": round(time.perf_counter() - t0, 2),
            }
            sidecar = _write_sidecar(
                path,
                {
                    "tool": "mac-music-gen",
                    "version": package_version(),
                    "model": {
                        "repo": weights.REPO_ID,
                        "revision": weights.REVISION,
                        "dit": weights.DIT_MODEL,
                        "acestep_commit": weights.ACESTEP_COMMIT,
                        "steps": steps,
                    },
                    "prompt": prompt_text,
                    "caption": cond.caption,
                    "lyrics": cond.lyrics,
                    "vocals": cond.vocals,
                    "bpm": cond.bpm,
                    "key": cond.keyscale,
                    "seed": job["seed"],
                    "seconds": seconds,
                    "mastering": asdict(m),
                    "runtime": runtime,
                    "timings": timings,
                    "measured": rep.as_dict(),
                    "reproduce": _reproduce(prompt_text, path, seconds, job["seed"], bpm, key, steps, lyrics_file, m),
                },
            )
            beds.append(Bed(path, sidecar, job["seed"], seconds, sr, cond.caption, rep, runtime, timings))
    return beds


def master_file(
    src: str | Path, out: str | Path, *, seconds: float | None = None, mastering: Mastering | None = None
) -> Bed:
    """Master an existing audio file as a bed. Needs no GPU."""
    m = mastering or Mastering()
    out = Path(out)
    t0 = time.perf_counter()
    audio, sr = master.read(src)
    length = seconds if seconds is not None else len(audio) / sr
    mastered, rep = master.master(audio, sr, length, m)
    master.write(out, mastered, sr)
    timings = {"master_s": round(time.perf_counter() - t0, 2)}
    sidecar = _write_sidecar(
        out,
        {
            "tool": "mac-music-gen",
            "version": package_version(),
            "source": str(src),
            "seconds": length,
            "mastering": asdict(m),
            "measured": rep.as_dict(),
        },
    )
    return Bed(out, sidecar, -1, length, sr, "", rep, {}, timings)


def fetch_weights(*, home: str | Path | None = None) -> Path:
    """Download the pinned weights (about 10.1 GB) unless present. Never asks: callers own consent."""
    store = weights.Store.default(home)
    return store.checkpoints if store.present() else store.fetch()


def doctor(*, home: str | Path | None = None) -> Diagnosis:
    """Check the GPU stack in a fresh worker process, and the weights cache."""
    found = _runner.probe()
    store = weights.Store.default(home)
    present = store.present()
    problems = []
    if not found.get("engine_installed"):
        problems.append(found.get("error") or "ACE-Step is not installed; install the `gpu` extra")
    elif not (found.get("mps") and found.get("mlx_metal")):
        from .errors import NO_GPU_HELP

        problems.append(NO_GPU_HELP)
    if not present:
        problems.append(f"the weights are missing; run `mac-music-gen fetch` (about {weights.DOWNLOAD_GB} GB)")
    versions = {k: found[k] for k in ("torch", "mlx", "macos") if k in found}
    versions["mac-music-gen"] = package_version()
    return Diagnosis(
        gpu=bool(found.get("mps")),
        mlx=bool(found.get("mlx_metal")),
        engine_installed=bool(found.get("engine_installed")),
        weights_present=present,
        weights_path=store.checkpoints,
        versions=versions,
        problems=problems,
    )
