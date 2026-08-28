from music21 import converter

from chordpulse.models import AnalysisResult, ChordEvent
from chordpulse.musicxml import MusicXmlGenerator, _ChordTimeline, _to_music21_label


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


def test_to_music21_label_converts_flat_roots() -> None:
    """♭系ルートは music21 の '-' 表記に変換されなければならない。"""
    assert _to_music21_label("Bb") == "B-"
    assert _to_music21_label("Bbm") == "B-m"
    assert _to_music21_label("Bbm7") == "B-m7"
    assert _to_music21_label("Bb7") == "B-7"
    assert _to_music21_label("Eb") == "E-"
    assert _to_music21_label("Ebm7") == "E-m7"
    assert _to_music21_label("Ab") == "A-"
    assert _to_music21_label("Abm7") == "A-m7"
    assert _to_music21_label("Db") == "D-"
    assert _to_music21_label("Gb") == "G-"
    # シャープ系・N はそのまま
    assert _to_music21_label("C") == "C"
    assert _to_music21_label("C#m7") == "C#m7"
    assert _to_music21_label("Bm7") == "Bm7"
    assert _to_music21_label("N") == "N"


def test_musicxml_generator_handles_flat_root_chords(tmp_path) -> None:
    """♭系ルートのコード（Bb, Eb, Ab）を含む譜面が正常に生成されること。

    回帰テスト: music21 は 'Bb' を無効とし ValueError を投げるが、
    '_to_music21_label' による変換後は 'B-' として受け付けられる。
    """
    result = AnalysisResult(
        bpm=120.0,
        duration_seconds=8.0,
        beat_times=tuple(i * 0.5 for i in range(16)),
        onset_times=(),
        chords=(
            ChordEvent(0.0, 2.0, "Bb"),
            ChordEvent(2.0, 4.0, "Bbm7"),
            ChordEvent(4.0, 6.0, "Eb"),
            ChordEvent(6.0, 8.0, "Abm7"),
        ),
        chord_engine="test",
    )

    # rhythm_level=1, 2, 3 いずれも例外なく生成できること
    for level in (1, 2, 3):
        out = tmp_path / f"flat_level{level}.musicxml"
        MusicXmlGenerator().generate(result, out, rhythm_level=level)
        assert out.is_file(), f"rhythm_level={level} で MusicXML が生成されなかった"
        assert "score-partwise" in out.read_text(encoding="utf-8")


def test_adaptive_rhythm_quarter_notes(tmp_path) -> None:
    """各拍でコードが変わる場合、4つの4分音符スラッシュが生成されること。"""
    output = tmp_path / "quarter_chart.musicxml"
    result = AnalysisResult(
        bpm=120.0,
        duration_seconds=2.0,
        beat_times=(0.0, 0.5, 1.0, 1.5),
        onset_times=(),
        chords=(
            ChordEvent(0.0, 0.5, "C"),
            ChordEvent(0.5, 1.0, "Dm"),
            ChordEvent(1.0, 1.5, "G"),
            ChordEvent(1.5, 2.0, "C"),
        ),
        chord_engine="test",
    )

    MusicXmlGenerator().generate(result, output)

    parsed = converter.parse(output)
    notes = list(parsed.parts[0].recurse().getElementsByClass("Note"))
    assert len(notes) == 4
    assert all(n.quarterLength == 1.0 for n in notes)
    symbols = list(parsed.parts[0].recurse().getElementsByClass("ChordSymbol"))
    assert len(symbols) == 4


def test_adaptive_rhythm_syncopation(tmp_path) -> None:
    """裏拍（1.5拍）でコードが変わるシンコペーションで、付点4分音符（1.5）が生成されること。"""
    output = tmp_path / "syncopation_chart.musicxml"
    result = AnalysisResult(
        bpm=120.0,
        duration_seconds=2.0,
        beat_times=(0.0, 0.5, 1.0, 1.5),
        onset_times=(),
        # Cが1.5拍(0.75秒), Gが2.5拍(1.25秒)
        chords=(
            ChordEvent(0.0, 0.75, "C"),
            ChordEvent(0.75, 2.0, "G"),
        ),
        chord_engine="test",
    )

    MusicXmlGenerator().generate(result, output)

    parsed = converter.parse(output)
    notes = list(parsed.parts[0].recurse().getElementsByClass("Note"))
    # Cは1.5拍(付点4分音符)、Gは2.5拍
    assert notes[0].quarterLength == 1.5

