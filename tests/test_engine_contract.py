"""Check that the installed ACE-Step still has the API the worker calls. Needs the `gpu` extra, not a GPU."""

import inspect

import pytest

pytestmark = pytest.mark.engine

inference = pytest.importorskip("acestep.inference")


def test_generation_params_accept_our_fields():
    from mac_music_gen.acestep_worker import generation_kwargs

    kw = generation_kwargs(
        {
            "caption": "c",
            "lyrics": "[Instrumental]",
            "vocals": False,
            "bpm": 90,
            "keyscale": "",
            "seconds": 30,
            "seed": 1,
            "steps": 8,
        }
    )
    inference.GenerationParams(**kw)


def test_generation_config_accepts_our_fields():
    inference.GenerationConfig(batch_size=1, use_random_seed=False, seeds=[1], audio_format="wav")


def test_initialize_service_accepts_our_kwargs():
    from acestep.handler import AceStepHandler

    params = inspect.signature(AceStepHandler.initialize_service).parameters
    for name in (
        "project_root",
        "config_path",
        "device",
        "offload_to_cpu",
        "offload_dit_to_cpu",
        "quantization",
        "use_mlx_dit",
        "compile_model",
        "use_flash_attention",
    ):
        assert name in params, name


def test_main_components_match_what_we_fetch():
    from acestep.model_downloader import MAIN_MODEL_COMPONENTS

    from mac_music_gen import weights

    assert set(MAIN_MODEL_COMPONENTS) == set(weights.COMPONENTS)
