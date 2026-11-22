"""The ACE-Step commit and the weights revision must agree everywhere they are written down."""

from pathlib import Path

from mac_music_gen import weights

ROOT = Path(__file__).resolve().parents[1]


def test_acestep_commit_matches_everywhere():
    for name in ("pyproject.toml", "NOTICE", "AUDIT.md"):
        assert weights.ACESTEP_COMMIT in (ROOT / name).read_text(), name


def test_weights_revision_matches_everywhere():
    for name in ("NOTICE", "AUDIT.md"):
        assert weights.REVISION in (ROOT / name).read_text(), name


def test_worker_and_weights_agree_on_the_dit():
    from mac_music_gen import acestep_worker

    assert acestep_worker.DIT_MODEL == weights.DIT_MODEL
