from chordpulse.models import AnalysisResult, ChordEvent


def test_chord_event_rejects_invalid_interval() -> None:
    try:
        ChordEvent(start_seconds=1.0, end_seconds=1.0, label="C")
    except ValueError as error:
        assert "end_seconds" in str(error)
    else:
        raise AssertionError("invalid chord interval should be rejected")


def test_analysis_result_serializes_without_audio() -> None:
    result = AnalysisResult(
        bpm=120.0,
        duration_seconds=4.0,
        beat_times=(0.0, 0.5, 1.0, 1.5),
        onset_times=(0.0, 1.0),
        chords=(ChordEvent(0.0, 2.0, "C"),),
        chord_engine="test",
    )

    payload = result.to_dict()

    assert payload["bpm"] == 120.0
    assert payload["chords"][0]["label"] == "C"
    assert "samples" not in payload

