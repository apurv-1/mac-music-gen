import pytest

from mac_music_gen import prompt


def test_instrumental_by_default():
    c = prompt.build("calm lo-fi piano")
    assert c.lyrics == prompt.INSTRUMENTAL_LYRICS
    assert not c.vocals
    assert c.caption.startswith("calm lo-fi piano, ")
    for d in prompt.BED_DESCRIPTORS:
        assert d in c.caption


def test_descriptors_not_repeated():
    c = prompt.build("instrumental ambient pads, no vocals")
    assert c.caption.count("instrumental") == 1
    assert c.caption.count("no vocals") == 1


def test_bpm_and_key():
    c = prompt.build("upbeat synthwave", bpm=110, key=" A minor ")
    assert c.caption.endswith("110 bpm")
    assert c.bpm == 110
    assert c.keyscale == "A minor"


def test_vocals_only_with_lyrics():
    c = prompt.build("indie pop", lyrics="[Verse]\nhello")
    assert c.vocals
    assert c.lyrics == "[Verse]\nhello"
    assert "no vocals" not in c.caption


def test_blank_lyrics_stay_instrumental():
    assert not prompt.build("jazz", lyrics="   \n").vocals


@pytest.mark.parametrize("bpm", [10, 400])
def test_bpm_range(bpm):
    with pytest.raises(ValueError, match="bpm"):
        prompt.build("jazz", bpm=bpm)


def test_empty_prompt():
    with pytest.raises(ValueError, match="empty"):
        prompt.build("   ")


def test_caption_length_limit():
    with pytest.raises(ValueError, match="shorten"):
        prompt.build("word " * 120)
