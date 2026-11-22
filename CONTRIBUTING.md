# Contributing

Thanks for helping.
Issues and pull requests are welcome at https://github.com/apurv-1/mac-music-gen.

## Setup

```bash
uv sync
```

```bash
uv run pytest
```

```bash
uv run ruff check . && uv run ruff format --check .
```

The default tests need no GPU and no model.
If your change touches `acestep_worker.py`, also run the engine contract tests, which need the `gpu` extra but no GPU:

```bash
uv sync --extra gpu && uv run pytest -m engine
```

If you have an Apple Silicon Mac and the weights, run the real generation test in your own terminal:

```bash
uv run pytest -m gpu
```

## Ground rules

- Read AGENTS.md; its rules apply to people too.
- Use conventional commit messages, such as `feat:`, `fix:`, `docs:`, `chore:` or `refactor:`. Release notes are generated from them.
- Any new model must have weights that allow commercial use.
- Changes to the ACE-Step commit or the weights revision need an updated AUDIT.md in the same pull request.

## Releases

Maintainers bump `version` in `pyproject.toml`, commit, and push a `vX.Y.Z` tag.
The release workflow runs the tests, builds the wheel and sdist, and publishes a GitHub Release with generated notes.
Versions follow SemVer, and a change to the model pins is at least a minor release.
