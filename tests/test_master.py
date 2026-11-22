import numpy as np
import pytest

from mac_music_gen import master

SR = 48000


def bed(seconds: float, seed: int = 0, swell_db: float = 0.0) -> np.ndarray:
    """Stereo test signal: a chord over soft noise, optionally with a loud middle section."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    tone = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6)) / 3
    x = 0.2 * tone + 0.02 * rng.standard_normal(len(t))
    if swell_db:
        mid = (t > seconds * 0.4) & (t < seconds * 0.6)
        x = np.where(mid, x * 10 ** (swell_db / 20), x)
    return np.stack([x, 0.9 * x], axis=1)


def test_hits_loudness_target_and_ceiling():
    out, rep = master.master(bed(30), SR, 30)
    assert rep.integrated_lufs == pytest.approx(-14.0, abs=0.5)
    assert rep.true_peak_db <= -1.0 + 0.1
    assert out.shape == (30 * SR, 2)


def test_custom_target():
    _, rep = master.master(bed(20), SR, 20, master.Mastering(lufs=-18.0))
    assert rep.integrated_lufs == pytest.approx(-18.0, abs=0.5)


def test_leveller_tames_a_sudden_swell():
    raw = bed(30, swell_db=10)
    plain = master.master(raw, SR, 30, master.Mastering(level_range_db=0))[1]
    levelled = master.master(raw, SR, 30)[1]
    assert levelled.short_term_spread_db < plain.short_term_spread_db - 3


def test_fades_start_and_end_quiet():
    out, _ = master.master(bed(20), SR, 20)
    assert np.max(np.abs(out[:10])) < 1e-3
    assert np.max(np.abs(out[-10:])) < 1e-3
    tail = out[-3 * SR :]
    first, last = np.abs(tail[: SR // 2]).mean(), np.abs(tail[-SR // 2 :]).mean()
    assert last < first / 5


def test_exact_length_when_source_is_longer_or_shorter():
    assert master.master(bed(25), SR, 20)[0].shape[0] == 20 * SR
    assert master.master(bed(15), SR, 20)[0].shape[0] == 20 * SR


def test_loop_point_is_seamless():
    source = bed(23, seed=1)
    out, _ = master.master(source, SR, 20, master.Mastering(loop=True, loop_crossfade=2.0))
    assert out.shape[0] == 20 * SR
    # The jump from the last sample back to the first must look like any other step.
    wrap = np.abs(out[0] - out[-1]).max()
    steps = np.abs(np.diff(out, axis=0)).max(axis=1)
    assert wrap <= np.percentile(steps, 99.9)
    # No fade in loop mode: the start is at full level.
    assert np.abs(out[: SR // 10]).mean() > 0.01


def test_loop_needs_enough_source():
    with pytest.raises(ValueError, match="needs"):
        master.master(bed(20.5), SR, 20, master.Mastering(loop=True, loop_crossfade=2.0))


def test_silence_is_rejected():
    with pytest.raises(ValueError, match="silent"):
        master.master(np.zeros((10 * SR, 2)), SR, 10)


def test_mono_input_becomes_one_channel():
    out, rep = master.master(bed(12)[:, 0], SR, 12)
    assert out.shape == (12 * SR, 1)
    assert rep.channels == 1


def test_fades_longer_than_track_are_rejected():
    with pytest.raises(ValueError, match="do not fit"):
        master.master(bed(12), SR, 3, master.Mastering(fade_in=2, fade_out=2))


def test_write_and_read_roundtrip(tmp_path):
    out, _ = master.master(bed(12), SR, 12)
    path = tmp_path / "bed.wav"
    master.write(path, out, SR)
    back, sr = master.read(path)
    assert sr == SR
    assert back.shape == out.shape
    assert np.max(np.abs(back - out)) < 1e-4
