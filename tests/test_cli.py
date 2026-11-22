import json
import sys
import types

import numpy as np
import pytest

from mac_music_gen import cli, device, engine, weights


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MAC_MUSIC_GEN_HOME", str(tmp_path / "home"))
    return weights.Store.default()


def make_present(store: weights.Store) -> None:
    for c in weights.COMPONENTS:
        (store.checkpoints / c).mkdir(parents=True, exist_ok=True)
    (store.checkpoints / weights.MARKER).write_text(weights.REVISION + "\n")


def test_weights_missing_message_names_fetch_and_size(store):
    assert not store.present()
    with pytest.raises(weights.WeightsMissing, match="mac-music-gen fetch.*10.1 GB"):
        store.require()


def test_marker_must_match_pinned_revision(store):
    make_present(store)
    assert store.present()
    (store.checkpoints / weights.MARKER).write_text("someotherrevision\n")
    assert not store.present()


def test_offline_env_forbids_downloads(tmp_path):
    env = weights.offline_env(tmp_path)
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["ACESTEP_CHECKPOINTS_DIR"] == str(tmp_path)


def fake_torch(mps: bool, cuda: bool):
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: mps)),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
    )


def test_device_refuses_cpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(False, False))
    with pytest.raises(device.GpuUnavailable, match="own terminal"):
        device.choose()


def test_device_cpu_only_when_allowed(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(False, False))
    assert device.choose(allow_cpu=True) == "cpu"
    assert "WARNING" in capsys.readouterr().err


def test_device_prefers_mps(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True, True))
    assert device.choose() == "mps"


def test_generate_end_to_end_with_fake_engine(store, tmp_path, monkeypatch, capsys):
    make_present(store)
    monkeypatch.setattr(device, "choose", lambda allow_cpu=False: "mps")
    calls = {}

    def fake_generate(cond, seconds, seed, dev, checkpoints, steps=8):
        calls.update(cond=cond, seconds=seconds, seed=seed, dev=dev)
        t = np.arange(int(seconds * 48000)) / 48000
        x = 0.3 * np.sin(2 * np.pi * 220 * t)
        return engine.Raw(
            np.stack([x, x]).astype(np.float32),
            48000,
            seed,
            {"load_s": 1.0, "generate_s": 2.0},
            {"acestep_device": "mps", "mlx_dit": True},
        )

    monkeypatch.setattr(engine, "generate", fake_generate)
    out = tmp_path / "bed.wav"
    code = cli.main(
        ["generate", "--prompt", "calm piano", "--seconds", "20", "--seed", "7", "--loop", "--out", str(out)]
    )
    assert code == 0
    assert calls["seconds"] == 22  # 20 s plus the 2 s loop crossfade
    assert calls["seed"] == 7 and calls["dev"] == "mps"
    assert calls["cond"].lyrics == "[Instrumental]"
    meta = json.loads(out.with_suffix(".wav.json").read_text())
    assert meta["seed"] == 7
    assert meta["model"]["revision"] == weights.REVISION
    assert "--seed 7" in meta["reproduce"] and "--loop" in meta["reproduce"]
    assert abs(meta["measured"]["integrated_lufs"] + 14) < 0.5
    assert "generate 2.0 s on mps (MLX DiT)" in capsys.readouterr().out


def test_generate_stops_without_weights(store, tmp_path, capsys):
    code = cli.main(["generate", "--prompt", "calm piano", "--out", str(tmp_path / "x.wav")])
    assert code == 1
    assert "mac-music-gen fetch" in capsys.readouterr().err


def test_fetch_refuses_without_terminal(store, monkeypatch, capsys):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli.main(["fetch"]) == 2
    assert "--yes" in capsys.readouterr().err


def test_master_command(tmp_path):
    from mac_music_gen import master

    t = np.arange(15 * 48000) / 48000
    src = tmp_path / "in.wav"
    master.write(src, np.stack([0.05 * np.sin(2 * np.pi * 330 * t)] * 2, axis=1), 48000)
    out = tmp_path / "out.wav"
    assert cli.main(["master", str(src), "--seconds", "12", "--out", str(out)]) == 0
    audio, sr = master.read(out)
    assert audio.shape[0] == 12 * sr
