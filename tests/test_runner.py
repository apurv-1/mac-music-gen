import pytest

from mac_music_gen import _runner
from mac_music_gen.errors import EngineError, EngineNotInstalled, GpuUnavailable

REQUEST = {
    "checkpoints": "/x",
    "allow_cpu": False,
    "jobs": [
        {
            "caption": "c",
            "lyrics": "[Instrumental]",
            "vocals": False,
            "bpm": None,
            "keyscale": "",
            "seconds": 10,
            "seed": 3,
            "steps": 8,
        }
    ],
}
OFFLINE = {"HF_HUB_OFFLINE": "1"}


def test_success_returns_jobs(fake_worker, tmp_path):
    result = _runner.run_worker(REQUEST, tmp_path, OFFLINE)
    assert result["ok"] and result["jobs"][0]["seed"] == 3


def test_offline_env_reaches_the_worker(fake_worker, tmp_path):
    with pytest.raises(EngineError, match="offline mode was not passed"):
        _runner.run_worker(REQUEST, tmp_path, {})


@pytest.mark.parametrize(
    "mode,exc", [("gpu", GpuUnavailable), ("engine_not_installed", EngineNotInstalled), ("engine", EngineError)]
)
def test_error_kinds_map_to_exceptions(fake_worker, tmp_path, mode, exc):
    fake_worker.setenv("FAKE_WORKER_MODE", mode)
    with pytest.raises(exc, match=f"fake {mode}"):
        _runner.run_worker(REQUEST, tmp_path, OFFLINE)


def test_crash_without_result_carries_stderr_tail(fake_worker, tmp_path):
    fake_worker.setenv("FAKE_WORKER_MODE", "crash")
    with pytest.raises(EngineError, match=r"code 134(.|\n)*Metal abort"):
        _runner.run_worker(REQUEST, tmp_path, OFFLINE)


def test_timeout_stops_the_worker(fake_worker, tmp_path):
    fake_worker.setenv("FAKE_WORKER_MODE", "hang")
    with pytest.raises(EngineError, match="did not finish within 2 s"):
        _runner.run_worker(REQUEST, tmp_path, OFFLINE, timeout=2)


def test_probe(fake_worker):
    assert _runner.probe()["mps"] is True
