import sys

import numpy as np
import pytest

from chordpulse.chords import (
    ChordEngineUnavailable,
    ChromagramChordRecognizer,
    create_chord_recognizer,
)
from chordpulse.models import AudioData, BeatGrid


def test_template_recognizer_identifies_a_c_major_template() -> None:
    recognizer = ChromagramChordRecognizer(minimum_confidence=0.1)
    label, confidence = recognizer._best_label(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np,
    )

    assert label == "C"
    assert confidence == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("vector", "expected_label"),
    [
        ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], "C"),
        ([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], "Cm"),
        ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], "C7"),
        ([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], "Cm7"),
    ],
)
def test_template_recognizer_covers_supported_chord_types(vector, expected_label) -> None:
    recognizer = ChromagramChordRecognizer(minimum_confidence=0.1)
    label, _confidence = recognizer._best_label(np.array(vector, dtype=float), np)

    assert label == expected_label


def test_template_recognizer_merges_adjacent_equal_events(monkeypatch) -> None:
    recognizer = ChromagramChordRecognizer(minimum_confidence=0.1)
    audio = AudioData(samples=np.zeros(4_000), sample_rate=1_000, duration_seconds=4.0)
    beat_grid = BeatGrid(bpm=120.0, beat_times=(0.0, 1.0, 2.0), onset_times=())

    class StubLibrosa:
        class feature:
            @staticmethod
            def chroma_stft(**_kwargs):
                vector = np.array(
                    [
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ]
                )
                return np.tile(vector, (1, 8))

    monkeypatch.setitem(sys.modules, "librosa", StubLibrosa())
    events = recognizer.recognize(audio, beat_grid)

    assert len(events) == 1
    assert events[0].label == "C"
    assert events[0].start_seconds == 0.0
    assert events[0].end_seconds == 2.5


def test_unavailable_named_engine_fails_explicitly() -> None:
    assert create_chord_recognizer("template").name == "librosa-chroma-template"

    with pytest.raises(ChordEngineUnavailable, match="chordino"):
        create_chord_recognizer("chordino")
