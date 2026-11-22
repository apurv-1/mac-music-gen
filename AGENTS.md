# Agent instructions

mac-music-gen is a CLI that generates instrumental background beds for short vertical reels on an Apple Silicon GPU.
To generate music, follow [skills/mac-music-gen/SKILL.md](skills/mac-music-gen/SKILL.md).
README.md explains the install, the options and the model licences.
AUDIT.md explains why the code looks the way it does around devices, downloads and licences.

## Layout

- `src/mac_music_gen/prompt.py` builds the caption and lyrics. Instrumental is the default.
- `src/mac_music_gen/weights.py` pins the weights revision and owns the cache folder.
- `src/mac_music_gen/device.py` picks the GPU and refuses the CPU.
- `src/mac_music_gen/engine.py` is the only module that imports ACE-Step or PyTorch.
- `src/mac_music_gen/master.py` is the NumPy and SciPy mastering chain.
- `src/mac_music_gen/cli.py` wires them into `generate`, `master`, `fetch` and `doctor`.
- `bin/mac-music-gen` runs the CLI from the project's virtual environment.

## Rules

- Run the tests and lint before every commit: `uv run pytest`, then `uv run ruff check . && uv run ruff format --check .`.
- Keep `engine.py` the only place that imports `torch` or `acestep`, and import them inside functions.
  That keeps the tests and `mac-music-gen master` working without the GPU extra.
- Never let generation fall back to the CPU silently. Any new code path that picks a device must go through `device.choose`.
- Never let generation download anything. Weights come only from `mac-music-gen fetch`, at the revision pinned in `weights.py`.
- Changing the ACE-Step commit or the weights revision is a security change.
  Re-read the new code's network, `trust_remote_code` and `torch.load` paths, update AUDIT.md, and update the pins in `pyproject.toml`, `weights.py` and NOTICE together.
- Only add models whose weights allow commercial use, and list each one's licence in the README.
- Do not copy code from projects that are not under an open-source licence.
- Never commit audio, model weights or anything under `out/`; `.gitignore` enforces this.
- GPU work cannot run in an agent sandbox, because Metal is blocked there. Give the user the exact command to run instead.
- Ask before running `mac-music-gen fetch` (about 10.1 GB) or `uv sync --extra gpu` (about 1 GB).
- Do not use the em dash in docs, help text or messages.
