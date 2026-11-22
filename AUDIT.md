# Audit

This file records what was checked before mac-music-gen was built, and what was decided because of it.
The review was done on 22 Nov 2026.

## ACE-Step 1.5

ACE-Step is the model mac-music-gen runs, so it was reviewed as a dependency.
It is installed from the tarball of commit `dce621408bee8c31b4fcf4811682eb9359e1bc94`, which is tag `v0.1.8`.
That tarball's SHA-256 is `66624bfbc375d06367d4bb62b0c3f55f07ad300228d774a2e2bd296b1bd1be7d`, and `uv.lock` pins the same hash.

Licence: MIT for the code, and MIT for the weights in `ACE-Step/Ace-Step1.5`.
The bundled Qwen3-Embedding-0.6B text encoder is Apache 2.0.
All of these allow commercial use of the generated music.

Findings, and what mac-music-gen does about each:

- **Silent CPU fallback.** If the requested device is missing, ACE-Step logs a warning and loads on the CPU.
  mac-music-gen checks the device ACE-Step reports after loading and stops if it differs from the one requested.
- **Unpinned auto-download.** If any of four model folders is missing, ACE-Step downloads the whole weights repository from its moving `main` branch.
  mac-music-gen downloads revision `19671f406d603126926c1b7e2adc169acbcade22` itself, and generates with `HF_HUB_OFFLINE=1`, so ACE-Step cannot download anything.
- **`trust_remote_code=True`.** ACE-Step loads its DiT through `transformers.AutoModel` with remote code enabled, which runs the `.py` files inside the checkpoint folder.
  Before loading, ACE-Step overwrites those files with copies from its own installed package when they differ.
  So the code that runs is the pinned, audited package code.
  The weights revision is pinned as well.
- **Pickle loading.** The only `torch.load` on the generation path reads `silence_latent.pt` with `weights_only=True`, which cannot run code.
- **Network calls.** On the generation path, the only network code is the model downloader, which offline mode blocks.
  An external language model discovery helper exists but is not called on this path.
  No telemetry was found.
- **Install scripts.** The repository ships `install_uv.sh` and update scripts that fetch moving references.
  mac-music-gen never runs them, because it installs ACE-Step as a package through `uv`.

## Dependencies

All direct dependencies are pinned to exact versions in `pyproject.toml`.
`uv.lock` pins every transitive dependency with hashes.
The lock resolves only for macOS on Apple Silicon, which is the platform this tool targets.
