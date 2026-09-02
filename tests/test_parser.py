from pathlib import Path

import pytest

from caption_checker.parser import parse, serialize, tokenize

DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    "filename,format",
    [
        ("sample_lecture.srt", "srt"),
        ("sample_lecture.vtt", "vtt"),
    ],
)
def test_roundtrip_preserves_cues(filename: str, format: str) -> None:
    path = DATA_DIR / filename
    cues = parse(path)
    assert len(cues) == 7

    reserialized = serialize(cues, format=format)
    reparsed = parse_from_string(reserialized, format)

    assert len(reparsed) == len(cues)
    for original, roundtripped in zip(cues, reparsed):
        assert original.text == roundtripped.text
        assert original.start == roundtripped.start
        assert original.end == roundtripped.end


def parse_from_string(content: str, format: str):
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=f".{format}", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        temp_path = f.name
    return parse(temp_path)


def test_tokenize_tracks_global_word_positions() -> None:
    cues = parse(DATA_DIR / "sample_lecture.srt")
    words = tokenize(cues)

    assert words[0].text == "Welcome"
    assert words[0].cue_index == 1
    assert words[0].global_index == 0

    # global_index should be strictly increasing across cues
    for prev, curr in zip(words, words[1:]):
        assert curr.global_index == prev.global_index + 1

    # "cough ka" (mis-transcribed "Kafka") appears in cues 5 and 6
    kafka_mentions = [w for w in words if w.text.lower() == "ka,"] + [
        w for w in words if w.text.lower() == "ka"
    ]
    assert len(kafka_mentions) >= 2
