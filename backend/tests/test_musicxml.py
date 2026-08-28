from music21 import converter

from chordpulse.models import AnalysisResult, ChordEvent
from chordpulse.musicxml import MusicXmlGenerator, _ChordTimeline


def test_musicxml_generator_writes_a_score(tmp_path) -> None:
    output = tmp_path / "chart.musicxml"
    result = AnalysisResult(
        bpm=120.0,
        duration_seconds=4.0,
        beat_times=(0.0, 0.5, 1.0, 1.5),
        onset_times=(0.0, 1.0),
        chords=(ChordEvent(0.0, 4.0, "C"),),
        chord_engine="test",
    )

    MusicXmlGenerator().generate(result, output, rhythm_level=2)

    assert output.is_file()
    assert "score-partwise" in output.read_text(encoding="utf-8")
    parsed = converter.parse(output)
    assert len(parsed.parts) == 1
    assert len(parsed.parts[0].recurse().getElementsByClass("ChordSymbol")) >= 1


def test_level1_single_chord_produces_whole_note(tmp_path) -> None:
    """Level 1 with one chord covering the whole measure → one whole note per measure."""
    output = tmp_path / "chart.musicxml"
    result = AnalysisResult(
        bpm=120.0,
        duration_seconds=4.0,
        beat_times=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
        onset_times=(),
        chords=(ChordEvent(0.0, 4.0, "C"),),
        chord_engine="test",
    )

    MusicXmlGenerator().generate(result, output, rhythm_level=1)

    parsed = converter.parse(output)
    notes = list(parsed.parts[0].recurse().getElementsByClass("Note"))
    # Every note in a single-chord score should be a whole note (4 quarter lengths)
    assert all(n.quarterLength == 4.0 for n in notes), (
        f"Expected all whole notes, got: {[n.quarterLength for n in notes]}"
    )


def test_level1_chord_change_at_midpoint_produces_half_notes(tmp_path) -> None:
    """Level 1 with chord change at beat 3 (midpoint of 4/4) → two half notes."""
    output = tmp_path / "chart.musicxml"
    result = AnalysisResult(
        bpm=120.0,
        duration_seconds=4.0,
        beat_times=(0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5),
        onset_times=(),
        # C covers first 2 beats (0-2s at 120bpm→0.5s/beat→2 beats=1s), G covers next 2
        chords=(ChordEvent(0.0, 1.0, "C"), ChordEvent(1.0, 4.0, "G")),
        chord_engine="test",
    )

    MusicXmlGenerator().generate(result, output, rhythm_level=1)

    parsed = converter.parse(output)
    notes = list(parsed.parts[0].recurse().getElementsByClass("Note"))
    # The measure with chord change should use half notes (2.0 quarter lengths)
    half_notes = [n for n in notes if n.quarterLength == 2.0]
    assert len(half_notes) >= 2, (
        f"Expected at least 2 half notes, got note lengths: {[n.quarterLength for n in notes]}"
    )


def test_chord_timeline_bisect_lookup() -> None:
    """_ChordTimeline should return the correct label in O(log n) fashion."""
    events = (
        ChordEvent(0.0, 1.0, "C"),
        ChordEvent(1.0, 2.0, "Am"),
        ChordEvent(2.0, 3.0, "F"),
    )
    timeline = _ChordTimeline(events)

    assert timeline.label_at(0.0) == "C"
    assert timeline.label_at(0.99) == "C"
    assert timeline.label_at(1.0) == "Am"
    assert timeline.label_at(1.5) == "Am"
    assert timeline.label_at(2.0) == "F"
    assert timeline.label_at(3.0) == "N"   # past the last event
    assert timeline.label_at(-0.1) == "N"  # before any event


def test_chord_timeline_unsorted_input_is_handled() -> None:
    """_ChordTimeline must work even if events are passed out of time order."""
    events = (
        ChordEvent(2.0, 3.0, "F"),
        ChordEvent(0.0, 1.0, "C"),
        ChordEvent(1.0, 2.0, "Am"),
    )
    timeline = _ChordTimeline(events)

    assert timeline.label_at(0.5) == "C"
    assert timeline.label_at(1.5) == "Am"
    assert timeline.label_at(2.5) == "F"
