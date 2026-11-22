import json
import shlex

import pytest

import mac_music_gen
from mac_music_gen import cli, weights
from mac_music_gen.errors import InvalidRequest, WeightsMissing


def test_generate_end_to_end_with_fake_worker(weights_present, fake_worker, tmp_path):
    beds = mac_music_gen.generate(
        "calm piano", tmp_path / "bed.wav", seconds=20, seed=7, mastering=mac_music_gen.Mastering(loop=True)
    )
    bed = beds[0]
    assert bed.seed == 7 and bed.path.is_file()
    assert abs(bed.measured.integrated_lufs + 14) < 0.5
    meta = json.loads(bed.sidecar.read_text())
    assert meta["model"]["revision"] == weights.REVISION
    assert meta["model"]["acestep_commit"] == weights.ACESTEP_COMMIT
    assert meta["lyrics"] == "[Instrumental]"
    assert meta["runtime"]["mlx_dit"] is True


def test_variants_use_consecutive_seeds_and_names(weights_present, fake_worker, tmp_path):
    beds = mac_music_gen.generate("calm piano", tmp_path / "bed.wav", seconds=12, seed=100, variants=3)
    assert [b.seed for b in beds] == [100, 101, 102]
    assert [b.path.name for b in beds] == ["bed-1.wav", "bed-2.wav", "bed-3.wav"]


def test_reproduce_line_is_shell_safe(weights_present, fake_worker, tmp_path):
    tricky = 'piano "$HOME" `ls`'
    bed = mac_music_gen.generate(tricky, tmp_path / "bed.wav", seconds=12, seed=9)[0]
    argv = shlex.split(json.loads(bed.sidecar.read_text())["reproduce"])
    assert argv[argv.index("--prompt") + 1] == tricky
    assert argv[argv.index("--seed") + 1] == "9"


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"out": "bed.mp3"}, ".wav"),
        ({"seconds": 700}, "at most 600"),
        ({"variants": 0}, "variants"),
        ({"steps": 0}, "steps"),
        ({"seconds": 3}, "do not fit"),
        ({"bpm": 5}, "bpm"),
    ],
)
def test_bad_requests_fail_before_the_worker_starts(weights_present, monkeypatch, tmp_path, kwargs, needle):
    from mac_music_gen import _runner

    monkeypatch.setattr(_runner, "run_worker", lambda *a, **k: pytest.fail("the worker must not start"))
    args = {"out": tmp_path / "bed.wav", "seconds": 20, **kwargs}
    with pytest.raises(InvalidRequest, match=needle):
        mac_music_gen.generate("calm piano", args.pop("out"), **args)


def test_weights_missing_names_fetch_and_size(store, tmp_path):
    with pytest.raises(WeightsMissing, match="mac-music-gen fetch.*10.1 GB"):
        mac_music_gen.generate("calm piano", tmp_path / "bed.wav", seconds=20)


def test_marker_must_match_pinned_revision(weights_present):
    (weights_present.checkpoints / weights.MARKER).write_text("other\n")
    assert not weights_present.present()


def test_doctor_reports_no_gpu(store, fake_worker):
    fake_worker.setenv("FAKE_WORKER_MODE", "gpu")
    d = mac_music_gen.doctor()
    assert not d.ok and not d.gpu
    assert any("own terminal" in p for p in d.problems)


def test_cli_json_and_exit_codes(weights_present, fake_worker, tmp_path, capsys):
    assert (
        cli.main(
            [
                "generate",
                "--prompt",
                "calm piano",
                "--seconds",
                "12",
                "--seed",
                "4",
                "--json",
                "--out",
                str(tmp_path / "b.wav"),
            ]
        )
        == 0
    )
    doc = json.loads(capsys.readouterr().out)
    assert doc["beds"][0]["seed"] == 4
    fake_worker.setenv("FAKE_WORKER_MODE", "gpu")
    assert cli.main(["generate", "--prompt", "x", "--seconds", "12", "--out", str(tmp_path / "c.wav")]) == 4
    assert cli.main(["generate", "--prompt", "x", "--seconds", "12", "--out", str(tmp_path / "c.mp3")]) == 2
    assert cli.main(["doctor", "--json"]) == 4
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_cli_weights_missing_exit_code(store, tmp_path):
    assert cli.main(["generate", "--prompt", "x", "--seconds", "12", "--out", str(tmp_path / "c.wav")]) == 3


def test_fetch_refuses_without_terminal(store, monkeypatch, capsys):
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli.main(["fetch"]) == 2
    assert "--yes" in capsys.readouterr().err


def test_master_file(tmp_path):
    import numpy as np

    from mac_music_gen import master

    t = np.arange(15 * 48000) / 48000
    src = tmp_path / "in.wav"
    master.write(src, np.stack([0.05 * np.sin(2 * np.pi * 330 * t)] * 2, axis=1), 48000)
    bed = mac_music_gen.master_file(src, tmp_path / "out.wav", seconds=12)
    assert bed.measured.seconds == 12 and bed.sidecar.is_file()
