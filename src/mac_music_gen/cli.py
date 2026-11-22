"""Command line: `mac-music-gen generate | master | fetch | doctor`.

Exit codes: 0 ok, 1 unexpected error, 2 invalid request, 3 weights missing, 4 no GPU,
5 engine failure or engine not installed. With --json, stdout carries one JSON document
and every log line goes to stderr.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import api, weights
from .errors import MacMusicGenError
from .master import Mastering


def _mastering(args: argparse.Namespace) -> Mastering:
    return Mastering(
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
    g.add_argument("--fade-in", type=float, default=0.5, help="seconds (default 0.5; not with --loop)")
    g.add_argument("--fade-out", type=float, default=3.0, help="gentle tail, seconds (default 3; not with --loop)")
    g.add_argument("--loop", action="store_true", help="make the file loop seamlessly instead of fading")
    g.add_argument("--crossfade", type=float, default=2.0, help="loop crossfade in seconds (default 2)")


def _emit(args: argparse.Namespace, data: dict, lines: list[str]) -> None:
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        for line in lines:
            print(line)


def _bed_lines(bed: api.Bed) -> list[str]:
    r = bed.measured
    lines = [
        f"wrote {bed.path}: {r.seconds:.1f} s, {r.integrated_lufs:.1f} LUFS, {r.true_peak_db:.1f} dBTP, "
        f"short-term spread {r.short_term_spread_db:.1f} dB"
    ]
    if "generate_s" in bed.timings:
        backend = "MLX" if bed.runtime.get("mlx_dit") else "PyTorch"
        lines.append(
            f"GPU time: load {bed.timings['load_s']} s, generate {bed.timings['generate_s']} s on "
            f"{bed.runtime.get('device', '?')} ({backend}); seed {bed.seed}"
        )
    return lines


def cmd_generate(args: argparse.Namespace) -> int:
    print(
        f"generating {args.variants} x {args.seconds:g} s; the model loads first, which takes a while", file=sys.stderr
    )
    beds = api.generate(
        args.prompt,
        args.out,
        seconds=args.seconds,
        bpm=args.bpm,
        key=args.key,
        seed=args.seed,
        variants=args.variants,
        lyrics_file=args.lyrics,
        steps=args.steps,
        mastering=_mastering(args),
        allow_cpu=args.allow_cpu,
        verbose=args.verbose,
    )
    _emit(args, {"beds": [b.as_dict() for b in beds]}, [line for b in beds for line in _bed_lines(b)])
    return 0


def cmd_master(args: argparse.Namespace) -> int:
    bed = api.master_file(args.input, args.out, seconds=args.seconds, mastering=_mastering(args))
    _emit(args, bed.as_dict(), _bed_lines(bed))
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
    print(f"weights ready in {api.fetch_weights()}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    d = api.doctor()
    lines = [
        f"GPU (PyTorch MPS): {'yes' if d.gpu else 'no'}",
        f"MLX on Metal: {'yes' if d.mlx else 'no'}",
        f"ACE-Step installed: {'yes' if d.engine_installed else 'no'}",
        f"weights: {'present' if d.weights_present else 'missing'} in {d.weights_path}",
        "versions: " + ", ".join(f"{k} {v}" for k, v in d.versions.items()),
    ]
    lines += [f"problem: {p}" for p in d.problems]
    _emit(args, d.as_dict(), lines)
    if d.ok:
        return 0
    if not d.engine_installed:
        return 5
    if not (d.gpu and d.mlx):
        return 4
    return 3


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mac-music-gen", description="Local text-to-music on your Mac's GPU.")
    p.add_argument("--version", action="version", version=f"mac-music-gen {api.package_version()}")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="generate mastered instrumental beds")
    g.add_argument("--prompt", required=True, help="mood and instruments, for example 'calm lo-fi piano, soft drums'")
    g.add_argument("--seconds", type=float, default=60.0, help="final length (default 60)")
    g.add_argument("--bpm", type=int, help="tempo, 30 to 300")
    g.add_argument("--key", help="key, for example 'C major' or 'A minor'")
    g.add_argument("--seed", type=int, help="fix the seed to reproduce a bed (default random, printed and saved)")
    g.add_argument("--variants", type=int, default=1, help="how many beds to make with seeds seed, seed+1, ...")
    g.add_argument("--steps", type=int, default=8, help="diffusion steps (default 8, the turbo model's value)")
    g.add_argument("--lyrics", help="a lyrics file; only then does the bed get vocals")
    g.add_argument("--allow-cpu", action="store_true", help="run on the CPU when no GPU is visible (slow)")
    g.add_argument("--verbose", action="store_true", help="stream the model's own log")
    g.add_argument("--json", action="store_true", help="print one JSON document on stdout")
    g.add_argument("--out", required=True, help="output .wav path; a .wav.json sidecar is written next to it")
    _add_mastering(g)
    g.set_defaults(func=cmd_generate)

    m = sub.add_parser("master", help="master an existing audio file as a bed (no GPU needed)")
    m.add_argument("input")
    m.add_argument("--seconds", type=float, help="final length (default: the input's length)")
    m.add_argument("--json", action="store_true", help="print one JSON document on stdout")
    m.add_argument("--out", required=True)
    _add_mastering(m)
    m.set_defaults(func=cmd_master)

    f = sub.add_parser("fetch", help=f"download the pinned model weights once (about {weights.DOWNLOAD_GB} GB)")
    f.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    f.set_defaults(func=cmd_fetch)

    d = sub.add_parser("doctor", help="check the GPU, MLX, ACE-Step and the weights")
    d.add_argument("--json", action="store_true", help="print one JSON document on stdout")
    d.set_defaults(func=cmd_doctor)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except MacMusicGenError as exc:
        print(f"mac-music-gen: {exc}", file=sys.stderr)
        return exc.exit_code
    except OSError as exc:
        print(f"mac-music-gen: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
