"""Command line: `mac-music-gen generate | master | fetch | doctor`."""

from __future__ import annotations

import argparse
import json
import math
import secrets
import sys
import time
from dataclasses import asdict
from pathlib import Path

from . import __version__, master, prompt, weights


def _settings(args: argparse.Namespace) -> master.Settings:
    return master.Settings(
        lufs=args.lufs,
        ceiling_db=args.ceiling,
        fade_in=args.fade_in,
        fade_out=args.fade_out,
        loop=args.loop,
        loop_crossfade=args.crossfade,
    )


def _add_mastering(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("mastering")
    g.add_argument("--lufs", type=float, default=-14.0, help="integrated loudness target (default -14)")
    g.add_argument("--ceiling", type=float, default=-1.0, help="true-peak ceiling in dBTP (default -1)")
    g.add_argument("--fade-in", type=float, default=0.5, help="seconds (default 0.5; ignored with --loop)")
    g.add_argument("--fade-out", type=float, default=3.0, help="gentle tail, seconds (default 3; not with --loop)")
    g.add_argument("--loop", action="store_true", help="make the file loop seamlessly instead of fading")
    g.add_argument("--crossfade", type=float, default=2.0, help="loop crossfade in seconds (default 2)")


def _sidecar(out: Path, data: dict) -> Path:
    path = out.with_suffix(out.suffix + ".json")
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def _summary(out: Path, rep: master.Report) -> str:
    return (
        f"wrote {out}: {rep.seconds:.1f} s, {rep.integrated_lufs:.1f} LUFS, "
        f"{rep.true_peak_db:.1f} dBTP, short-term spread {rep.short_term_spread_db:.1f} dB"
    )


def cmd_generate(args: argparse.Namespace) -> int:
    from . import device, engine

    lyrics = Path(args.lyrics).read_text() if args.lyrics else None
    cond = prompt.build(args.prompt, bpm=args.bpm, key=args.key or "", lyrics=lyrics)
    settings = _settings(args)
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    # A loop needs extra audio after the loop point to crossfade into the head.
    source_seconds = args.seconds + (settings.loop_crossfade if settings.loop else 0)
    source_seconds = max(engine.MIN_SECONDS, math.ceil(source_seconds))

    checkpoints = weights.Store.default().require()
    dev = device.choose(allow_cpu=args.allow_cpu)
    print(f"generating {args.seconds:g} s on {dev}, seed {seed}", file=sys.stderr)
    raw = engine.generate(cond, source_seconds, seed, dev, checkpoints, steps=args.steps)

    t0 = time.perf_counter()
    audio, rep = master.master(raw.audio, raw.sample_rate, args.seconds, settings)
    out = Path(args.out)
    master.write(out, audio, raw.sample_rate)
    timings = {**raw.timings, "master_s": round(time.perf_counter() - t0, 2)}

    _sidecar(
        out,
        {
            "tool": "mac-music-gen",
            "version": __version__,
            "model": {
                "repo": weights.REPO_ID,
                "revision": weights.REVISION,
                "dit": weights.DIT_MODEL,
                "steps": args.steps,
                "shift": engine.TURBO_SHIFT,
            },
            "prompt": args.prompt,
            "caption": cond.caption,
            "lyrics": cond.lyrics,
            "vocals": cond.vocals,
            "bpm": cond.bpm,
            "key": cond.keyscale,
            "seed": raw.seed,
            "seconds": args.seconds,
            "mastering": asdict(settings),
            "runtime": raw.runtime,
            "timings": timings,
            "measured": rep.as_dict(),
            "reproduce": _reproduce(args, raw.seed),
        },
    )
    print(_summary(out, rep))
    backend = "MLX DiT" if raw.runtime.get("mlx_dit") else "PyTorch DiT"
    print(
        f"GPU time: load {timings['load_s']} s, generate {timings['generate_s']} s on "
        f"{raw.runtime['acestep_device']} ({backend}); seed {raw.seed}"
    )
    return 0


def _reproduce(args: argparse.Namespace, seed: int) -> str:
    parts = [
        "mac-music-gen generate",
        f"--prompt {json.dumps(args.prompt)}",
        f"--seconds {args.seconds:g}",
        f"--seed {seed}",
        f"--steps {args.steps}",
    ]
    if args.bpm:
        parts.append(f"--bpm {args.bpm}")
    if args.key:
        parts.append(f"--key {json.dumps(args.key)}")
    if args.loop:
        parts.append(f"--loop --crossfade {args.crossfade:g}")
    else:
        parts.append(f"--fade-in {args.fade_in:g} --fade-out {args.fade_out:g}")
    parts.append(f"--lufs {args.lufs:g} --ceiling {args.ceiling:g}")
    if args.lyrics:
        parts.append(f"--lyrics {json.dumps(args.lyrics)}")
    parts.append(f"--out {json.dumps(args.out)}")
    return " ".join(parts)


def cmd_master(args: argparse.Namespace) -> int:
    audio, sr = master.read(args.input)
    seconds = args.seconds if args.seconds is not None else len(audio) / sr
    out_audio, rep = master.master(audio, sr, seconds, _settings(args))
    out = Path(args.out)
    master.write(out, out_audio, sr)
    _sidecar(
        out,
        {
            "tool": "mac-music-gen",
            "version": __version__,
            "source": str(args.input),
            "seconds": seconds,
            "mastering": asdict(_settings(args)),
            "measured": rep.as_dict(),
        },
    )
    print(_summary(out, rep))
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    store = weights.Store.default()
    if store.present():
        print(f"weights already present in {store.checkpoints}")
        return 0
    question = (
        f"Download about {weights.DOWNLOAD_GB} GB of ACE-Step 1.5 weights "
        f"({weights.REPO_ID} at {weights.REVISION[:12]}) to {store.checkpoints}? [y/N] "
    )
    if not args.yes:
        if not sys.stdin.isatty():
            print("refusing to download without a terminal to confirm; pass --yes", file=sys.stderr)
            return 2
        if input(question).strip().lower() not in ("y", "yes"):
            print("nothing downloaded")
            return 1
    path = store.fetch()
    print(f"weights ready in {path}")
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    from . import device

    ok = True
    try:
        found = device.detect()
    except ImportError:
        found = "unavailable (PyTorch is not installed; run `uv sync --extra gpu`)"
        ok = False
    print(f"PyTorch device: {found}")
    if found == "cpu":
        print(f"  {device.NO_GPU_HELP}")
        ok = False
    print(f"MLX on Metal: {'yes' if device.mlx_available() else 'no'}")
    store = weights.Store.default()
    present = store.present()
    print(f"weights: {'present' if present else 'missing'} in {store.checkpoints}")
    if not present:
        print(f"  run `mac-music-gen fetch` (about {weights.DOWNLOAD_GB} GB)")
        ok = False
    return 0 if ok else 1


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mac-music-gen", description="Local text-to-music on your Mac's GPU.")
    p.add_argument("--version", action="version", version=f"mac-music-gen {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate a mastered instrumental bed")
    g.add_argument("--prompt", required=True, help="mood and instruments, for example 'calm lo-fi piano, soft drums'")
    g.add_argument("--seconds", type=float, default=60.0, help="final length (default 60)")
    g.add_argument("--bpm", type=int, help="tempo, 30 to 300")
    g.add_argument("--key", help="key, for example 'C major' or 'A minor'")
    g.add_argument("--seed", type=int, help="fix the seed to reproduce a bed (default random, printed and saved)")
    g.add_argument("--steps", type=int, default=8, help="diffusion steps (default 8, the turbo model's value)")
    g.add_argument("--lyrics", help="a lyrics file; only then does the bed get vocals")
    g.add_argument("--allow-cpu", action="store_true", help="run on the CPU when no GPU is visible (slow)")
    g.add_argument("--out", required=True, help="output .wav path; a .wav.json sidecar is written next to it")
    _add_mastering(g)
    g.set_defaults(func=cmd_generate)

    m = sub.add_parser("master", help="master an existing audio file as a bed (no GPU needed)")
    m.add_argument("input")
    m.add_argument("--seconds", type=float, help="final length (default: the input's length)")
    m.add_argument("--out", required=True)
    _add_mastering(m)
    m.set_defaults(func=cmd_master)

    f = sub.add_parser("fetch", help=f"download the pinned model weights once (about {weights.DOWNLOAD_GB} GB)")
    f.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    f.set_defaults(func=cmd_fetch)

    d = sub.add_parser("doctor", help="check the GPU, MLX and the weights")
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"mac-music-gen: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
