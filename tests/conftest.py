import sys
from pathlib import Path

import pytest

from mac_music_gen import _runner, weights

FAKE_WORKER = Path(__file__).with_name("fake_worker.py")


@pytest.fixture
def store(tmp_path, monkeypatch):
    """An isolated cache folder."""
    monkeypatch.setenv("MAC_MUSIC_GEN_HOME", str(tmp_path / "home"))
    return weights.Store.default()


@pytest.fixture
def weights_present(store):
    for c in weights.COMPONENTS:
        (store.checkpoints / c).mkdir(parents=True, exist_ok=True)
    (store.checkpoints / weights.MARKER).write_text(weights.REVISION + "\n")
    return store


@pytest.fixture
def fake_worker(monkeypatch):
    """Route the runner to tests/fake_worker.py. Set FAKE_WORKER_MODE to choose its behaviour."""
    monkeypatch.setattr(_runner, "WORKER_ARGV", [sys.executable, str(FAKE_WORKER)])
    monkeypatch.setenv("FAKE_WORKER_MODE", "ok")
    return monkeypatch
