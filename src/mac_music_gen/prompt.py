"""Turn a short mood prompt into the text ACE-Step is conditioned on.

A bed sits under narration, so every caption gets the same descriptors that keep the
arrangement out of the voice's way: instrumental, steady energy, no lead vocal, and no
sudden builds or drops. ACE-Step reads the caption literally, so plain descriptive words
work better than instructions.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import InvalidRequest

# ACE-Step's own convention for "no vocals". Its `instrumental` flag is not read on the
# text-to-music path that runs without the language model, so the lyrics tag is what
# actually keeps the voice out.
INSTRUMENTAL_LYRICS = "[Instrumental]"

BED_DESCRIPTORS = (
    "instrumental",
    "no vocals",
    "background music for voice-over",
    "steady energy",
    "consistent volume",
    "sparse arrangement",
    "no sudden drops or builds",
)

MAX_CAPTION_CHARS = 512  # ACE-Step truncates longer captions
MAX_LYRICS_CHARS = 4096
MIN_BPM, MAX_BPM = 30, 300  # ACE-Step's documented range


@dataclass(frozen=True)
class Conditioning:
    caption: str
    lyrics: str
    bpm: int | None
    keyscale: str
    vocals: bool


def build(
    prompt: str,
    *,
    bpm: int | None = None,
    key: str = "",
    lyrics: str | None = None,
) -> Conditioning:
    """Build the caption and lyrics for one bed.

    Vocals appear only when the caller passes lyrics. Without lyrics the bed descriptors
    are appended and the lyrics are the instrumental tag.
    """
    prompt = " ".join(prompt.split())
    if not prompt:
        raise InvalidRequest("the prompt is empty; describe the mood, for example 'calm lo-fi piano'")
    if bpm is not None and not MIN_BPM <= bpm <= MAX_BPM:
        raise InvalidRequest(f"bpm must be between {MIN_BPM} and {MAX_BPM}, got {bpm}")

    vocals = bool(lyrics and lyrics.strip())
    if vocals:
        caption = prompt
        lyric_text = lyrics.strip()
        if len(lyric_text) > MAX_LYRICS_CHARS:
            raise InvalidRequest(f"lyrics are {len(lyric_text)} characters; the model reads at most {MAX_LYRICS_CHARS}")
    else:
        lowered = prompt.lower()
        extra = [d for d in BED_DESCRIPTORS if d not in lowered]
        caption = ", ".join([prompt, *extra])
        lyric_text = INSTRUMENTAL_LYRICS

    if bpm is not None and f"{bpm} bpm" not in caption.lower():
        caption = f"{caption}, {bpm} bpm"
    if len(caption) > MAX_CAPTION_CHARS:
        raise InvalidRequest(
            f"the caption is {len(caption)} characters after adding the bed descriptors; "
            f"shorten the prompt so it stays under {MAX_CAPTION_CHARS}"
        )
    return Conditioning(caption=caption, lyrics=lyric_text, bpm=bpm, keyscale=key.strip(), vocals=vocals)
