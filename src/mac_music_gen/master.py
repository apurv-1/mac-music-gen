"""Finish a raw generation into a bed that sits under narration.

The chain runs in this order, and each step exists for one reason:

1. A gentle high-pass at 30 Hz removes rumble and DC that eat headroom without being heard.
2. A slow leveller rides the gain over 3-second windows, so a swell or a drop in the
   generation does not poke out over the voice. It moves at most a few dB either way.
3. The length is fitted to the exact number of samples asked for. In loop mode the tail
   is crossfaded into the head, so the file loops without a click.
4. Fades go on before loudness is measured, so the measured number includes them. Loop
   mode skips fades, because a fade would make the loop point audible.
5. Loudness is normalised to the target (about -14 LUFS integrated, ITU-R BS.1770), and a
   look-ahead limiter keeps the true peak under the ceiling. The two alternate a few
   times, because limiting lowers loudness slightly and the next pass makes it up.

Everything here is plain NumPy and SciPy, so it runs and is tested without a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from scipy import ndimage, signal

SILENCE_DB = -60.0


@dataclass(frozen=True)
class Settings:
    lufs: float = -14.0
    ceiling_db: float = -1.0
    fade_in: float = 0.5
    fade_out: float = 3.0
    loop: bool = False
    loop_crossfade: float = 2.0
    level_window: float = 3.0
    level_range_db: float = 6.0
    highpass_hz: float = 30.0


@dataclass(frozen=True)
class Report:
    seconds: float
    sample_rate: int
    channels: int
    integrated_lufs: float
    true_peak_db: float
    short_term_spread_db: float  # 95th minus 10th percentile of 3 s loudness

    def as_dict(self) -> dict:
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def read(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float64", always_2d=True)
    return audio, sr


def write(path: str | Path, audio: np.ndarray, sr: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_24")


def as_2d(audio: np.ndarray) -> np.ndarray:
    """Return float64 audio shaped (samples, channels)."""
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        return a[:, None]
    if a.ndim != 2:
        raise ValueError(f"audio must be 1-D or 2-D, got shape {a.shape}")
    # ACE-Step returns (channels, samples); anything with few rows and many columns is that.
    if a.shape[0] <= 8 < a.shape[1]:
        a = a.T
    return np.ascontiguousarray(a)


def highpass(x: np.ndarray, sr: int, hz: float) -> np.ndarray:
    if hz <= 0:
        return x
    sos = signal.butter(2, hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def _window_levels_db(x: np.ndarray, sr: int, window_s: float, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """RMS level in dB of windows centred every hop. Returns (centres in samples, levels)."""
    power = np.mean(x**2, axis=1)
    csum = np.concatenate([[0.0], np.cumsum(power)])
    half = max(1, int(window_s * sr / 2))
    hop = max(1, int(hop_s * sr))
    centres = np.arange(0, len(power), hop)
    lo = np.clip(centres - half, 0, len(power))
    hi = np.clip(centres + half, 0, len(power))
    mean_power = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)
    return centres, 10 * np.log10(np.maximum(mean_power, 1e-12))


def level(x: np.ndarray, sr: int, window_s: float, range_db: float) -> np.ndarray:
    """Slowly ride the gain toward the median level, within plus or minus range_db."""
    if range_db <= 0 or len(x) < window_s * sr:
        return x
    hop_s = 0.1
    centres, levels = _window_levels_db(x, sr, window_s, hop_s)
    audible = levels > SILENCE_DB
    if not audible.any():
        return x
    target = float(np.median(levels[audible]))
    gain_db = np.where(audible, np.clip(target - levels, -range_db, range_db), 0.0)
    smooth = max(1, int(window_s / hop_s))
    gain_db = ndimage.uniform_filter1d(gain_db, size=smooth, mode="nearest")
    per_sample = np.interp(np.arange(len(x)), centres, gain_db)
    return x * (10 ** (per_sample / 20))[:, None]


def fit_length(x: np.ndarray, sr: int, seconds: float, loop: bool, crossfade_s: float) -> np.ndarray:
    """Cut or pad to exactly `seconds`. In loop mode, crossfade the overhang into the head."""
    n = int(round(seconds * sr))
    if not loop:
        if len(x) >= n:
            return x[:n].copy()
        return np.concatenate([x, np.zeros((n - len(x), x.shape[1]))])
    xf = int(round(crossfade_s * sr))
    if len(x) < n + xf:
        raise ValueError(
            f"a {seconds:g} s loop with a {crossfade_s:g} s crossfade needs {(n + xf) / sr:.2f} s of source, "
            f"got {len(x) / sr:.2f} s"
        )
    out = x[:n].copy()
    if xf:
        t = np.linspace(0.0, 1.0, xf, endpoint=False)[:, None]
        # Equal-power crossfade: the head fades in while the audio that follows the loop
        # point fades out, so sample n-1 runs straight into sample 0 on playback.
        out[:xf] = out[:xf] * np.sin(t * np.pi / 2) + x[n : n + xf] * np.cos(t * np.pi / 2)
    return out


def fades(x: np.ndarray, sr: int, fade_in: float, fade_out: float) -> np.ndarray:
    n_in, n_out = int(fade_in * sr), int(fade_out * sr)
    if n_in + n_out > len(x):
        raise ValueError(f"fades of {fade_in:g} s and {fade_out:g} s are longer than the {len(x) / sr:.2f} s track")
    y = x.copy()
    if n_in:
        y[:n_in] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_in)))[:, None]
    if n_out:
        y[-n_out:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n_out)))[:, None]
    return y


def integrated_lufs(x: np.ndarray, sr: int) -> float:
    return float(pyln.Meter(sr).integrated_loudness(x))


def true_peak_db(x: np.ndarray) -> float:
    """Peak of the 4x oversampled signal, a close estimate of BS.1770 true peak."""
    over = signal.resample_poly(x, 4, 1, axis=0)
    peak = float(np.max(np.abs(over)))
    return 20 * np.log10(max(peak, 1e-12))


def normalise(x: np.ndarray, sr: int, lufs: float) -> np.ndarray:
    current = integrated_lufs(x, sr)
    if not np.isfinite(current):
        raise ValueError("the audio is silent, so there is nothing to normalise")
    return x * 10 ** ((lufs - current) / 20)


def limit(x: np.ndarray, sr: int, ceiling_db: float, lookahead_ms: float = 5.0) -> np.ndarray:
    """Look-ahead peak limiter on the oversampled peak envelope.

    The gain each sample needs is the ceiling over its peak. A minimum filter spreads that
    gain `lookahead` either side, so the gain starts falling before a peak arrives, and a
    moving average no wider than the minimum filter smooths the curve without ever rising
    above what a peak needs.
    """
    ceiling = 10 ** (ceiling_db / 20)
    over = np.abs(signal.resample_poly(x, 4, 1, axis=0)).max(axis=1)
    peaks = over[: 4 * len(x)].reshape(len(x), 4).max(axis=1)
    gain = np.minimum(1.0, ceiling / np.maximum(peaks, 1e-12))
    if gain.min() >= 1.0:
        return x
    look = max(1, int(lookahead_ms * sr / 1000))
    held = ndimage.minimum_filter1d(gain, size=2 * look + 1, mode="nearest")
    smooth = ndimage.uniform_filter1d(held, size=look + 1, mode="nearest")
    return x * smooth[:, None]


def short_term_spread_db(x: np.ndarray, sr: int) -> float:
    """Loudness range proxy: spread of 3-second loudness readings, ignoring silence."""
    meter = pyln.Meter(sr)
    block, hop = int(3 * sr), int(1 * sr)
    readings = []
    for start in range(0, max(1, len(x) - block + 1), hop):
        chunk = x[start : start + block]
        if len(chunk) < block:
            break
        value = meter.integrated_loudness(chunk)
        if np.isfinite(value) and value > SILENCE_DB:
            readings.append(value)
    if len(readings) < 2:
        return 0.0
    return float(np.percentile(readings, 95) - np.percentile(readings, 10))


def report(x: np.ndarray, sr: int) -> Report:
    return Report(
        seconds=len(x) / sr,
        sample_rate=sr,
        channels=x.shape[1],
        integrated_lufs=integrated_lufs(x, sr),
        true_peak_db=true_peak_db(x),
        short_term_spread_db=short_term_spread_db(x, sr),
    )


def master(audio: np.ndarray, sr: int, seconds: float, settings: Settings | None = None) -> tuple[np.ndarray, Report]:
    """Run the whole chain and return the finished bed with its measurements."""
    settings = settings or Settings()
    x = as_2d(audio)
    x = highpass(x, sr, settings.highpass_hz)
    x = level(x, sr, settings.level_window, settings.level_range_db)
    x = fit_length(x, sr, seconds, settings.loop, settings.loop_crossfade)
    if not settings.loop:
        x = fades(x, sr, settings.fade_in, settings.fade_out)
    for _ in range(3):
        x = normalise(x, sr, settings.lufs)
        x = limit(x, sr, settings.ceiling_db)
    return x, report(x, sr)
