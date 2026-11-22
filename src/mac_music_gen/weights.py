"""Where the model weights live, and how they get there.

The weights come from one Hugging Face repository at one pinned revision, so every run
uses exactly the bytes that were audited. They download once into a cache folder and are
read from there afterwards. Generation runs with Hugging Face offline mode switched on,
so it can never start a surprise download: if the weights are missing, it stops and says
to run `mac-music-gen fetch`.

ACE-Step's own loader would download the full repository from its moving main branch
when any component is missing. Fetching the pinned revision here first is what stops that.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ID = "ACE-Step/Ace-Step1.5"
REVISION = "19671f406d603126926c1b7e2adc169acbcade22"  # main on 22 Nov 2026
DOWNLOAD_GB = 10.1
# ACE-Step refuses to start unless all four exist. The 1.7B language model is not used for
# instrumental beds, but its presence check is part of ACE-Step, so it is fetched too.
COMPONENTS = ("acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B", "acestep-5Hz-lm-1.7B")
DIT_MODEL = "acestep-v15-turbo"
MARKER = ".mac-music-gen-complete"


class WeightsMissing(RuntimeError):
    pass


@dataclass(frozen=True)
class Store:
    root: Path

    @classmethod
    def default(cls) -> Store:
        home = os.environ.get("MAC_MUSIC_GEN_HOME")
        base = Path(home).expanduser() if home else Path.home() / ".cache" / "mac-music-gen"
        return cls(base)

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints" / f"ace-step-1.5-{REVISION[:12]}"

    def present(self) -> bool:
        marker = self.checkpoints / MARKER
        if not marker.is_file() or marker.read_text().strip() != REVISION:
            return False
        return all((self.checkpoints / c).is_dir() for c in COMPONENTS)

    def require(self) -> Path:
        if not self.present():
            raise WeightsMissing(
                f"the ACE-Step 1.5 weights are not in {self.checkpoints}. "
                f"Run `mac-music-gen fetch` once; it downloads about {DOWNLOAD_GB} GB."
            )
        return self.checkpoints

    def fetch(self) -> Path:
        """Download the pinned revision. Safe to re-run; finished files are skipped."""
        from huggingface_hub import snapshot_download

        self.checkpoints.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=REPO_ID,
            revision=REVISION,
            local_dir=str(self.checkpoints),
            allow_patterns=[f"{c}/*" for c in COMPONENTS] + ["config.json", "README.md"],
        )
        missing = [c for c in COMPONENTS if not (self.checkpoints / c).is_dir()]
        if missing:
            raise WeightsMissing(f"the download finished without {', '.join(missing)}")
        (self.checkpoints / MARKER).write_text(REVISION + "\n")
        return self.checkpoints


def offline_env(checkpoints: Path) -> dict[str, str]:
    """Environment that points ACE-Step at the pinned weights and forbids downloads."""
    return {
        "ACESTEP_CHECKPOINTS_DIR": str(checkpoints),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
