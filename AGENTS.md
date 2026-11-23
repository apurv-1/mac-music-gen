# Agent instructions

mac-music-gen is a local AI music generator for the Mac: a standalone CLI and Python library that generates music from a text prompt on an Apple Silicon GPU.
It is tuned for instrumental background beds that sit under a voice, such as reels and explainers.
README.md explains the install, the options and the model licences.
AUDIT.md explains why the code looks the way it does around devices, downloads and licences.

## Using mac-music-gen from another project

Treat it like any outside tool.
Call the `mac-music-gen` command on PATH, or `import mac_music_gen` from an environment where it is installed.
Do not reach into this repository's files, virtual environment or cache layout.
[skills/mac-music-gen/SKILL.md](skills/mac-music-gen/SKILL.md) says when and how to generate a bed.

## Layout

- `src/mac_music_gen/api.py` is the public API: `generate`, `master_file`, `fetch_weights`, `doctor`. It validates before any GPU work.
- `src/mac_music_gen/errors.py` holds the exception tree and each kind's CLI exit code.
- `src/mac_music_gen/prompt.py` builds the caption and lyrics. Instrumental is the default.
- `src/mac_music_gen/master.py` is the NumPy and SciPy mastering chain.
- `src/mac_music_gen/weights.py` pins the ACE-Step commit and the weights revision, and owns the cache folder.
- `src/mac_music_gen/_runner.py` starts the worker process and maps its outcome to results or exceptions.
- `src/mac_music_gen/acestep_worker.py` is the only module that imports torch, mlx or acestep. The device guard lives here.
- `src/mac_music_gen/cli.py` is argparse over the API.
- `tests/fake_worker.py` stands in for the worker in tests.

## Rules

- Run the tests and lint before every commit: `uv run pytest`, then `uv run ruff check . && uv run ruff format --check .`.
- Keep `acestep_worker.py` the only module that imports `torch`, `mlx` or `acestep`. `tests/test_worker.py` checks that `import mac_music_gen` stays free of them.
- Never let generation fall back to the CPU or from MLX to PyTorch silently. The checks in `acestep_worker.check_loaded` are the guard; extend them, never loosen them.
- Never let generation download anything. Weights come only from `fetch_weights`, at the revision pinned in `weights.py`, and the worker always runs with the offline environment.
- Validate every argument in `api.generate` before the worker starts.
- Changing the ACE-Step commit or the weights revision is a security change and a minor release, because it changes what a seed produces.
  Re-read the new code's network, `trust_remote_code` and `torch.load` paths, update AUDIT.md, and update `pyproject.toml`, `weights.py` and NOTICE together. `tests/test_pins.py` fails if they disagree, and `pytest -m engine` checks the API still fits.
- Only add models whose weights allow commercial use, and list each one's licence in the README.
- Do not copy code from projects that are not under an open-source licence.
- Never commit audio, model weights or anything under `out/`; `.gitignore` enforces this.
- GPU work cannot run in an agent sandbox, because Metal is blocked there. Give the user the exact command to run instead.
- Ask before running `mac-music-gen fetch` (about 10.1 GB) or installing the `gpu` extra (about 1 GB).
- Do not use the em dash in docs, help text or messages.
