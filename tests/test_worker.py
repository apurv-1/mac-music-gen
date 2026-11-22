"""The worker's guard and parameters, tested without torch, mlx or ACE-Step installed."""

import subprocess
import sys
import types

import pytest

from mac_music_gen import acestep_worker as w

JOB = {
    "caption": "calm piano, instrumental",
    "lyrics": "[Instrumental]",
    "vocals": False,
    "bpm": 90,
    "keyscale": "C major",
    "seconds": 30,
    "seed": 5,
    "steps": 8,
}


def handler(device="mps", mlx_dit=True, mlx_vae=True, te_device="mps"):
    param = types.SimpleNamespace(device=types.SimpleNamespace(type=te_device))
    te = types.SimpleNamespace(parameters=lambda: iter([param]))
    return types.SimpleNamespace(device=device, use_mlx_dit=mlx_dit, use_mlx_vae=mlx_vae, text_encoder=te)


def test_guard_accepts_full_gpu_load():
    assert w.check_loaded(handler(), "mps")["mlx_dit"] is True


@pytest.mark.parametrize(
    "kwargs,needle",
    [
        ({"device": "cpu"}, "loaded on cpu"),
        ({"mlx_dit": False}, "DiT fell back"),
        ({"mlx_vae": False}, "VAE fell back"),
        ({"te_device": "cpu"}, "text encoder is on cpu"),
    ],
)
def test_guard_rejects_any_fallback(kwargs, needle):
    with pytest.raises(w.Failure, match=needle) as info:
        w.check_loaded(handler(**kwargs), "mps")
    assert info.value.kind == "gpu_unavailable"


def test_guard_skips_checks_when_cpu_allowed():
    assert w.check_loaded(handler(device="cpu", mlx_dit=False), "cpu")["device"] == "cpu"


def test_before_load_needs_mps_and_mlx():
    assert w.check_gpu_before_load({"engine_installed": True, "mps": True, "mlx_metal": True}, False) == "mps"
    with pytest.raises(w.Failure) as info:
        w.check_gpu_before_load({"engine_installed": True, "mps": True, "mlx_metal": False}, False)
    assert info.value.kind == "gpu_unavailable"
    assert w.check_gpu_before_load({"engine_installed": True, "mps": False, "mlx_metal": False}, True) == "cpu"


def test_before_load_reports_missing_engine():
    with pytest.raises(w.Failure) as info:
        w.check_gpu_before_load({"engine_installed": False, "error": "no acestep"}, True)
    assert info.value.kind == "engine_not_installed"


def test_generation_params_are_instrumental_lm_free_and_unnormalised():
    kw = w.generation_kwargs(JOB)
    assert kw["lyrics"] == "[Instrumental]" and kw["instrumental"] is True
    assert kw["thinking"] is False and kw["use_cot_caption"] is False
    assert kw["enable_normalization"] is False
    assert kw["inference_steps"] == 8 and kw["shift"] == 3.0 and kw["seed"] == 5
    assert kw["bpm"] == 90 and kw["keyscale"] == "C major" and kw["duration"] == 30.0


def test_importing_the_package_does_not_import_torch():
    code = "import sys, mac_music_gen; assert 'torch' not in sys.modules and 'acestep' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
