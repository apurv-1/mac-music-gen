# mac-music-gen

[![tests](https://github.com/apurv-1/mac-music-gen/actions/workflows/test.yml/badge.svg)](https://github.com/apurv-1/mac-music-gen/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model: ACE-Step 1.5 (MIT)](https://img.shields.io/badge/model-ACE--Step%201.5%20(MIT)-blue.svg)](https://huggingface.co/ACE-Step/Ace-Step1.5)
![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)
![Python 3.11-3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)

**A local AI music generator for your Mac.**
Type a prompt and get a mastered WAV, generated on your Mac's GPU.

mac-music-gen runs the ACE-Step 1.5 music model on Apple Silicon through MLX and Metal.
Your prompts and your music never leave the machine.
There is no subscription, no upload and no per-song credit.
The model's weights are MIT licensed, so you can use what it makes in monetised work.

```bash
mac-music-gen generate --prompt "calm lo-fi piano, soft brushed drums" --seconds 60 --bpm 85 --out bed.wav
```

It is tuned for background music.
Every track is instrumental unless you pass lyrics, keeps a steady level, and is mastered to about -14 LUFS with a gentle fade at the end.
That makes it a good fit under a voice: reels, explainers, YouTube videos, podcasts and game loops.
Pass a lyrics file with `--lyrics` and it sings.

## Why local

| | Cloud music generators | mac-music-gen |
| --- | --- | --- |
| Where it runs | Someone else's servers | Your Mac's GPU, offline after one download |
| Cost per track | Credits or a subscription | Electricity |
| Your prompts | Uploaded | Stay on your machine |
| Rights to the output | Set by the service's terms and your plan | MIT-licensed weights, commercial use allowed |
| Reproducing a track | Usually not possible | Same seed and prompt give the same audio |

## What it does

- Generates on Apple Silicon with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5).
  ACE-Step runs its diffusion model and its audio decoder on Apple's MLX, and its text encoder on PyTorch MPS.
- Never falls back quietly.
  ACE-Step on its own drops to the CPU when the GPU is missing, and to PyTorch when MLX fails, with only a log line.
  mac-music-gen checks where every part actually loaded and stops if anything left the GPU or MLX.
  The only way to run without a GPU is `--allow-cpu`, which warns and records it.
- Keeps vocals out.
  The lyrics are set to ACE-Step's `[Instrumental]` tag, and the caption asks for no vocals.
  Vocals appear only when you pass a lyrics file with `--lyrics`.
  This is enforced through the model's conditioning, not checked afterwards, so listen before you publish.
- Makes runs reproducible.
  Every run has a seed, random unless you pass `--seed`.
  A JSON sidecar next to each WAV records the seed, the caption, the model revision, the ACE-Step commit, the PyTorch, MLX and macOS versions, and a shell-safe command that reproduces the bed.
- Masters for narration.
  A slow leveller evens out swells and drops by up to 6 dB.
  Loudness is then normalised to -14 LUFS integrated with a -1 dBTP true-peak ceiling, a 0.5 s fade-in and a 3 s fade-out.
- Controls length exactly.
  The output is exactly `--seconds` long.
  With `--loop`, the end crossfades into the start, so an editor can loop the file under a longer voice-over without a click.
- Caches weights once and never downloads during generation.
  The model downloads once, at a pinned revision, into `~/.cache/mac-music-gen`.
  Generation runs in a separate worker process with Hugging Face offline mode on, so it cannot start a download.
- Stays out of your process.
  The model runs in a child process, so its GPU memory is released when the job ends and a crash in the model becomes an error message instead of taking your program down.

## Install

You need an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/) and about 12 GB of free disk.

### Recommended: hash-locked install from a release tag

This installs exactly the dependency versions in `uv.lock`.

```bash
git clone --branch v0.2.0 https://github.com/apurv-1/mac-music-gen.git ~/mac-music-gen
```

```bash
cd ~/mac-music-gen && uv sync --locked --extra gpu
```

That second command downloads about 1 GB of Python packages, PyTorch and MLX among them.
Then put the command on your PATH:

```bash
ln -sf ~/mac-music-gen/.venv/bin/mac-music-gen ~/.local/bin/mac-music-gen
```

### Quick install with uv tool

This is one command, but it ignores `uv.lock`, so dependencies below the direct ones can float to newer versions.

```bash
uv tool install "mac-music-gen[gpu] @ git+https://github.com/apurv-1/mac-music-gen@v0.2.0"
```

### Then, either way

Download the model weights once.
This is about 10.1 GB from Hugging Face, and the command asks before it starts.

```bash
mac-music-gen fetch
```

Check that the GPU, MLX, ACE-Step and the weights are all visible.

```bash
mac-music-gen doctor
```

mac-music-gen is not on PyPI.
PyPI rejects packages that depend on a Git archive, and ACE-Step 1.5 is only published that way.
A PyPI package that could not generate music would mislead people, so it waits until ACE-Step is on PyPI.

## Usage

Generate a 60-second bed:

```bash
mac-music-gen generate --prompt "warm ambient pads, slow pulse, hopeful" --seconds 60 --out out/bed.wav
```

Generate three options in one model load, with seeds 100, 101 and 102, written as `options-1.wav` to `options-3.wav`:

```bash
mac-music-gen generate --prompt "minimal tech house, muted plucks" --seconds 30 --bpm 120 --seed 100 --variants 3 --out out/options.wav
```

Generate a seamless 30-second loop in a fixed key:

```bash
mac-music-gen generate --prompt "upbeat acoustic guitar, light percussion" --seconds 30 --key "G major" --loop --out out/loop.wav
```

To reproduce a bed exactly, run the `reproduce` line from its `.wav.json` sidecar.

Master a file you already have, such as a stock track, with the same chain.
This needs no GPU.

```bash
mac-music-gen master track.wav --seconds 45 --out out/track-bed.wav
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--prompt` | required | Mood, genre and instruments, in plain words. |
| `--seconds` | 60 | Final length. The model makes 10 to 600 seconds per bed. |
| `--bpm`, `--key` | model's choice | Tempo (30 to 300) and key, for example `C major`. |
| `--seed` | random | Fixes the result. The seed is always printed and saved. |
| `--variants` | 1 | How many beds to make, with seeds `seed`, `seed+1`, and so on. |
| `--steps` | 8 | Diffusion steps. 8 is right for the turbo model. |
| `--loop`, `--crossfade` | off, 2 s | Seamless loop instead of fades. |
| `--lufs`, `--ceiling` | -14, -1 | Loudness target and true-peak ceiling. |
| `--fade-in`, `--fade-out` | 0.5 s, 3 s | Fades, ignored with `--loop`. |
| `--lyrics FILE` | none | The only way to get vocals. |
| `--allow-cpu` | off | Run without a GPU. Slow, and it prints a warning. |
| `--verbose` | off | Stream the model's own log. Without it, the log appears only when something fails. |
| `--json` | off | Print one JSON document on stdout; every log line goes to stderr. Also on `master` and `doctor`. |

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success. |
| 2 | Invalid request, found before any GPU work started. |
| 3 | The weights are missing. Run `mac-music-gen fetch`. |
| 4 | No GPU, or part of the model left the GPU or MLX. |
| 5 | The model failed, or the `gpu` extra is not installed. |

### Prompts for beds

Describe texture and mood, not song structure.
Some starting points:

- `calm lo-fi piano, soft brushed drums, vinyl warmth`
- `minimal corporate electronic, light plucks, gentle pulse`
- `warm ambient pads, slow evolving texture, hopeful`
- `upbeat acoustic guitar, light percussion, positive`
- `dark cinematic drone, low strings, suspense`

mac-music-gen appends the bed descriptors itself, such as "instrumental", "steady energy" and "no sudden drops or builds", so you do not need to type them.

## Python API

```python
import mac_music_gen

beds = mac_music_gen.generate(
    "calm lo-fi piano, soft brushed drums",
    "out/bed.wav",
    seconds=30,
    seed=42,
    mastering=mac_music_gen.Mastering(lufs=-16, loop=True),
)
bed = beds[0]
print(bed.path, bed.seed, bed.measured.integrated_lufs, bed.timings["generate_s"])
```

The four functions are `generate`, `master_file`, `fetch_weights` and `doctor`.
Every deliberate error derives from `mac_music_gen.MacMusicGenError`: `InvalidRequest`, `WeightsMissing`, `GpuUnavailable`, `EngineNotInstalled` and `EngineError`.
`import mac_music_gen` never imports PyTorch, because the model runs in a worker process.
`fetch_weights` never asks for confirmation, so a program that calls it owns the decision to download 10.1 GB.

## Running from an agent sandbox

Agent sandboxes such as Claude Code's block Metal, so PyTorch and MLX cannot see the GPU there.
`mac-music-gen doctor` and `generate` exit with code 4 and say so, instead of running for many minutes on the CPU.
Run the GPU commands in your own terminal.
Everything else, including the tests and `mac-music-gen master`, runs fine inside a sandbox.

For agents, [skills/mac-music-gen/SKILL.md](skills/mac-music-gen/SKILL.md) is a Claude Code skill that explains when and how to use the tool.
Install it with:

```bash
cp -r skills/mac-music-gen ~/.claude/skills/
```

## Models and licences

The music goes into monetised reels, so every model must allow commercial use.

| Component | Licence | Used for |
| --- | --- | --- |
| [ACE-Step 1.5 code](https://github.com/ace-step/ACE-Step-1.5) | MIT | The generation pipeline, installed as a pinned dependency. |
| [ACE-Step 1.5 weights](https://huggingface.co/ACE-Step/Ace-Step1.5): `acestep-v15-turbo` DiT and VAE | MIT | The music itself. |
| Qwen3-Embedding-0.6B, bundled in the same repository | Apache 2.0 | Encoding the caption. |
| `acestep-5Hz-lm-1.7B`, bundled in the same repository | MIT | Downloaded because ACE-Step checks that it exists, but never loaded. |

Models that were ruled out:

- MusicGen and the other AudioCraft weights are CC BY-NC 4.0, which forbids commercial use.
- YuE2 was excluded because its weights were flagged as non-commercial. Its licence was not re-checked here.
- Stable Audio Open uses the Stability AI Community Licence. It allows commercial use only under a revenue cap and with registration, which is workable but less clean than MIT.

This is an engineering summary, not legal advice.
Read the licences yourself before relying on them.

## How it works

1. `api.py` validates the request, so a bad argument never costs a model load.
2. `prompt.py` turns the prompt into ACE-Step's caption and lyrics, adding the bed descriptors and the instrumental tag.
3. `weights.py` finds the pinned weights and builds the offline environment.
4. `_runner.py` starts `acestep_worker.py` in a child process with that environment, and turns its result or its crash into a return value or an exception.
5. `acestep_worker.py` checks the GPU, loads ACE-Step without its language model, confirms every part stayed on the GPU and MLX, and runs text-to-music with 8 steps and shift 3.0.
6. `master.py` levels the audio, fits the length, adds fades and normalises loudness, back in the parent process.

## Development

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

The tests replace the worker with a fake script, so they run anywhere.
Two optional groups need more:

- `uv sync --extra gpu && uv run pytest -m engine` checks that the installed ACE-Step still has the API the worker calls. It needs no GPU, and CI runs it.
- `uv run pytest -m gpu` generates a real bed twice with one seed and checks that the results match. Run it in your own terminal after `mac-music-gen fetch`.

See [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md).

## Licence

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
