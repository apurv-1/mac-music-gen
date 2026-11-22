# mac-music-gen

[![tests](https://github.com/apurv-1/mac-music-gen/actions/workflows/test.yml/badge.svg)](https://github.com/apurv-1/mac-music-gen/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Model: ACE-Step 1.5 (MIT)](https://img.shields.io/badge/model-ACE--Step%201.5%20(MIT)-blue.svg)](https://huggingface.co/ACE-Step/Ace-Step1.5)
![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey.svg)
![Python 3.11-3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)

mac-music-gen generates instrumental background music for short vertical explainer reels.
It runs on your Mac's GPU, offline, with a model whose licence allows commercial use.

A reel bed has a narrow job.
It has to sit under a voice for 30 to 90 seconds without pulling attention.
So every bed this tool makes is instrumental unless you ask for vocals, keeps a steady level, and is mastered to about -14 LUFS with a gentle fade at the end.

```bash
mac-music-gen generate --prompt "calm lo-fi piano, soft brushed drums" --seconds 60 --bpm 85 --out bed.wav
```

## What it does

- Generates on Apple Silicon with [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5).
  ACE-Step runs its diffusion model and its audio decoder on Apple's MLX, and its text encoder on PyTorch MPS.
- Refuses to run on the CPU unless you pass `--allow-cpu`.
  ACE-Step itself falls back to the CPU quietly when it cannot find a GPU.
  mac-music-gen therefore checks which device ACE-Step actually loaded on, and stops if it is not the one asked for.
- Keeps vocals out.
  The lyrics are set to ACE-Step's `[Instrumental]` tag, and the caption asks for no vocals.
  Vocals appear only when you pass a lyrics file with `--lyrics`.
  This is enforced through the model's conditioning, not checked afterwards, so listen before you publish.
- Makes runs reproducible.
  Every run has a seed, random unless you pass `--seed`.
  A JSON sidecar next to each WAV records the seed, the full caption, the model revision and a command that reproduces the bed.
- Masters for narration.
  A slow leveller evens out swells and drops.
  Loudness is then normalised to -14 LUFS integrated with a -1 dBTP true-peak ceiling, a 0.5 s fade-in and a 3 s fade-out.
- Controls length exactly.
  The output is exactly `--seconds` long.
  With `--loop`, the end crossfades into the start, so an editor can loop the file under a longer voice-over without a click.
- Caches weights once.
  The model downloads once, at a pinned revision, into `~/.cache/mac-music-gen`.
  Generation then runs with Hugging Face offline mode on, so it can never start a download.

## Install

You need an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/) and about 12 GB of free disk.

```bash
git clone https://github.com/apurv-1/mac-music-gen.git mac-music-gen
```

```bash
cd mac-music-gen
```

Install the tool with the model runtime.
This downloads about 1 GB of Python packages, PyTorch and MLX among them.

```bash
uv sync --extra gpu
```

Download the model weights once.
This is about 10.1 GB from Hugging Face, and the command asks before it starts.

```bash
bin/mac-music-gen fetch
```

Check that the GPU and the weights are visible.

```bash
bin/mac-music-gen doctor
```

`bin/mac-music-gen` runs the project's own virtual environment from any folder.
`uv run mac-music-gen` does the same from inside the repository.

## Usage

Generate a 60-second bed:

```bash
bin/mac-music-gen generate --prompt "warm ambient pads, slow pulse, hopeful" --seconds 60 --out out/bed.wav
```

Generate a 30-second loop at a fixed tempo and key:

```bash
bin/mac-music-gen generate --prompt "minimal tech house, muted plucks" --seconds 30 --bpm 120 --key "A minor" --loop --out out/loop.wav
```

To reproduce a bed exactly, copy the `reproduce` line from its `.wav.json` sidecar.
Passing the same prompt, length and `--seed` does the same thing.

Master a file you already have, such as a stock track, with the same chain.
This needs no GPU.

```bash
bin/mac-music-gen master track.wav --seconds 45 --out out/track-bed.wav
```

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--prompt` | required | Mood, genre and instruments, in plain words. |
| `--seconds` | 60 | Final length. The model makes 10 to 600 seconds at a time. |
| `--bpm`, `--key` | model's choice | Tempo (30 to 300) and key, for example `C major`. |
| `--seed` | random | Fixes the result. The seed is always printed and saved. |
| `--steps` | 8 | Diffusion steps. 8 is right for the turbo model. |
| `--loop`, `--crossfade` | off, 2 s | Seamless loop instead of fades. |
| `--lufs`, `--ceiling` | -14, -1 | Loudness target and true-peak ceiling. |
| `--fade-in`, `--fade-out` | 0.5 s, 3 s | Fades, ignored with `--loop`. |
| `--lyrics FILE` | none | The only way to get vocals. |
| `--allow-cpu` | off | Run without a GPU. Slow, and it prints a warning. |

### Prompts for beds

Describe texture and mood, not song structure.
Some starting points:

- `calm lo-fi piano, soft brushed drums, vinyl warmth`
- `minimal corporate electronic, light plucks, gentle pulse`
- `warm ambient pads, slow evolving texture, hopeful`
- `upbeat acoustic guitar, light percussion, positive`
- `dark cinematic drone, low strings, suspense`

mac-music-gen appends the bed descriptors itself, such as "instrumental", "steady energy" and "no sudden drops or builds", so you do not need to type them.

## Running from an agent sandbox

Agent sandboxes such as Claude Code's block Metal, so PyTorch and MLX cannot see the GPU there.
`mac-music-gen doctor` and `generate` say so plainly instead of running for many minutes on the CPU.
Run the GPU commands in your own terminal.
Everything else, including the tests and `mac-music-gen master`, runs fine inside a sandbox.

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

1. `prompt.py` turns your prompt into ACE-Step's caption and lyrics, adding the bed descriptors and the instrumental tag.
2. `weights.py` finds the pinned weights and sets the offline environment.
3. `device.py` picks MPS, or CUDA on other machines, and refuses the CPU.
4. `engine.py` loads ACE-Step without its language model and runs text-to-music with the turbo settings of 8 steps and shift 3.0.
5. `master.py` levels the audio, fits the length, adds fades and normalises loudness.
   It is plain NumPy and SciPy, and it is what the tests cover.

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

The tests cover everything except the GPU call, which they replace with a fake engine.
See [AGENTS.md](AGENTS.md) for the rules agents follow in this repository, and [skills/mac-music-gen/SKILL.md](skills/mac-music-gen/SKILL.md) for the agent skill.


## Licence

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
