"""Real generation on the Mac's GPU. Deselected by default; run `uv run pytest -m gpu` in your own terminal.

It needs the `gpu` extra, the fetched weights and Metal, which agent sandboxes block.
"""

import numpy as np
import pytest

import mac_music_gen
from mac_music_gen import master

pytestmark = pytest.mark.gpu


def test_same_seed_gives_the_same_bed_on_mlx(tmp_path):
    a = mac_music_gen.generate("calm lo-fi piano", tmp_path / "a.wav", seconds=10, seed=1234)[0]
    b = mac_music_gen.generate("calm lo-fi piano", tmp_path / "b.wav", seconds=10, seed=1234)[0]
    assert a.runtime["device"] == "mps" and a.runtime["mlx_dit"] and a.runtime["mlx_vae"]
    assert abs(a.measured.integrated_lufs + 14) < 0.5
    assert a.measured.true_peak_db <= -0.9
    audio_a, _ = master.read(a.path)
    audio_b, _ = master.read(b.path)
    assert np.allclose(audio_a, audio_b, atol=1e-4)
