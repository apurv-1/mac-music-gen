---
name: mac-music-gen
description: Generate original music locally on an Apple Silicon GPU with the mac-music-gen CLI, a local Suno-style text-to-music tool. Best for an instrumental background bed under a reel, explainer or voice-over (ACE-Step 1.5, MIT licensed, safe for monetised posts). Use when a reel or video needs original background music, a bed under narration, a loop of a given length, or when a licensed track should be replaced with self-generated music. Also use to master an existing track to -14 LUFS for use under a voice.
---

# mac-music-gen

mac-music-gen makes original instrumental beds that sit under narration.
The output is a 48 kHz stereo 24-bit WAV, mastered to -14 LUFS integrated with a -1 dBTP ceiling, plus a `.wav.json` sidecar.
The model's licence is MIT, so the music is fine for monetised Instagram and YouTube posts.

## Before you start

1. Check that the tool is installed: `mac-music-gen --version`.
   If the command is missing, do not install it yourself.
   Point the user to the install steps at https://github.com/apurv-1/mac-music-gen#install and ask them to install it.
2. Run `mac-music-gen doctor --json` and read the exit code.
   - 0: ready.
   - 3: the weights are missing. `mac-music-gen fetch` downloads about 10.1 GB. Ask the user before running it.
   - 4: no GPU is visible. Agent sandboxes block Metal, so this is expected inside one.
     Never add `--allow-cpu`.
     Give the user the exact `generate` command to run in their own terminal, and wait for them to say it finished.
   - 5: the model runtime is not installed. The install steps in the README fix it; ask the user.

## Choosing the settings

- **Length.** Set `--seconds` to the voice-over length plus about 2 seconds for the fade-out.
  If the edit may still change length, use `--loop` so the editor can loop the bed.
- **Prompt.** Describe mood, texture and instruments in plain words, for example `calm lo-fi piano, soft brushed drums`.
  Do not describe song structure such as a verse or a drop.
  Do not add "instrumental" or "no vocals"; the tool adds those itself.
  Keep the energy one notch below the narration's.
- **Tempo.** `--bpm` is optional.
  Match it to the edit's cut rhythm when the video cuts on a beat: 80 to 95 for calm explainers, 110 to 125 for upbeat ones.
- **Options.** `--variants 3` makes three beds with consecutive seeds in one model load, which is cheaper than three runs.
- **Seed.** Leave it random for a first pass.
  To keep a bed, keep its sidecar; its `reproduce` field is the exact command that recreates it.
- **Vocals.** Never pass `--lyrics` unless the user explicitly asks for a sung track.

## Commands

```bash
mac-music-gen generate --prompt "warm ambient pads, slow pulse, hopeful" --seconds 62 --json --out out/bed.wav
```

```bash
mac-music-gen generate --prompt "minimal electronic, light plucks" --seconds 30 --bpm 118 --loop --variants 3 --json --out out/loop.wav
```

Master an existing file so it sits under a voice.
This needs no GPU and runs inside a sandbox.

```bash
mac-music-gen master track.wav --seconds 45 --json --out out/track-bed.wav
```

With `--json`, stdout is one JSON document and logs go to stderr.
For `generate` it is `{"beds": [{"path", "sidecar", "seed", "measured", "runtime", "timings", ...}]}`.

## After generating

1. Check `measured`.
   `integrated_lufs` should be within 0.5 dB of -14 and `true_peak_db` at or below -1.
   A `short_term_spread_db` above about 8 means the bed swells noticeably; try another seed.
2. Report `timings.generate_s` and `timings.load_s` to the user.
3. The bed is instrumental by construction, not by detection.
   Ask the user to listen once before it ships, and regenerate with a new seed if a voice-like sound appears.
4. Keep the `.wav.json` sidecar with the WAV.
   It records the seed, caption and model revision, which is the provenance if a platform ever asks.
5. Do not commit generated audio to git.

## When something fails

Exit codes: 2 means a bad argument, found before any GPU work, so fix the argument and retry.
3, 4 and 5 are the cases under "Before you start".
Any other failure prints the last lines of the model's log; pass them on to the user rather than retrying blindly.
