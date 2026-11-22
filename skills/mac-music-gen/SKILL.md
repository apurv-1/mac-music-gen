---
name: mac-music-gen
description: Generate an instrumental background music bed for a short vertical reel, explainer or voice-over, locally on an Apple Silicon GPU with the mac-music-gen CLI (ACE-Step 1.5, MIT licensed, safe for monetised posts). Use when a reel or video needs original background music, a bed under narration, a loop of a given length, or when a licensed track should be replaced with self-generated music. Also use to master an existing track to -14 LUFS for use under a voice.
---

# mac-music-gen

mac-music-gen makes original instrumental beds that sit under narration.
The output is a 48 kHz stereo 24-bit WAV, mastered to -14 LUFS integrated with a -1 dBTP ceiling, plus a `.wav.json` sidecar.
The model's licence is MIT, so the music is fine for monetised Instagram and YouTube posts.

## Before you start

1. Find the repository.
   It is usually `~/opensource/mac-music-gen`.
   The launcher is `bin/mac-music-gen` inside it, which runs the project's own virtual environment.
2. Know where the GPU is.
   Agent sandboxes block Metal, so generation cannot run inside one.
   Run `bin/mac-music-gen doctor` first.
   If it reports `PyTorch device: cpu`, do not pass `--allow-cpu`.
   Instead, give the user the exact `generate` command to run in their own terminal, and wait for them to say it finished.
3. If `doctor` says the weights are missing, tell the user that `bin/mac-music-gen fetch` downloads about 10.1 GB, and ask before running it.

## Choosing the settings

- **Length.** Set `--seconds` to the voice-over length plus about 2 seconds for the fade-out.
  If the edit may change length later, generate with `--loop` and let the editor loop it.
- **Prompt.** Describe mood, texture and instruments in plain words, for example `calm lo-fi piano, soft brushed drums`.
  Do not describe song structure such as a verse or a drop.
  Do not add "instrumental" or "no vocals"; the tool adds those itself.
  Keep the energy one notch below the narration's.
- **Tempo.** `--bpm` is optional.
  Match it to the edit's cut rhythm when the reel cuts on a beat: 80 to 95 for calm explainers, 110 to 125 for upbeat ones.
- **Seed.** Leave it random for a first pass.
  To make several options, run the same command a few times; each prints its seed.
  To keep one, note its seed or copy the `reproduce` line from its sidecar.
- **Vocals.** Never pass `--lyrics` unless the user explicitly asks for a sung track.

## Commands

```bash
bin/mac-music-gen generate --prompt "warm ambient pads, slow pulse, hopeful" --seconds 62 --out out/bed.wav
```

```bash
bin/mac-music-gen generate --prompt "minimal electronic, light plucks" --seconds 30 --bpm 118 --loop --out out/loop.wav
```

Master an existing file so it sits under a voice.
This needs no GPU and runs inside a sandbox.

```bash
bin/mac-music-gen master track.wav --seconds 45 --out out/track-bed.wav
```

## After generating

1. Read the printed summary.
   Integrated loudness should be within 0.5 dB of -14 LUFS and true peak at or below -1 dBTP.
   A short-term spread above about 8 dB means the bed swells noticeably; try another seed.
2. Report the GPU time line to the user.
3. The music is instrumental by construction, not by detection.
   Ask the user to listen once before it ships, and regenerate with a new seed if a voice-like sound appears.
4. Keep the `.wav.json` sidecar with the WAV.
   It records the seed, caption and model revision, which is the provenance if a platform ever asks.
5. Never commit the WAV to git.

## When something fails

- `no GPU is visible`: you are in a sandbox or on a machine without Metal. Hand the command to the user.
- `weights are not in ...`: run `bin/mac-music-gen fetch` after the user agrees to the 10.1 GB download.
- `ACE-Step is not installed`: run `uv sync --extra gpu` in the repository, which downloads about 1 GB.
- `asked ACE-Step for mps but it loaded on cpu`: Metal is unavailable to that process. Do not retry with `--allow-cpu` without asking.
