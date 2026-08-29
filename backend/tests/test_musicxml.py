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


def test_quarter_beat_anticipation_smoothed_away(tmp_path) -> None:
    """小節末尾の0.25拍（1スロット）のみの食いはフライングノイズとして平滑化されること。

    BPM=120, 4/4拍子: 1小節=2.0s, 0.25拍=0.125s。
    小節1末尾のsub 15(1.875s)のみDが検出され、小節2もDの場合、
    0.25拍は演奏の食い込みノイズとみなして平滑化する（不要なDを抑止）。
    """
    output = tmp_path / "quarter_anticipation.musicxml"
    bpm = 120.0
    spb = 60.0 / bpm
    beat_times = tuple(i * spb for i in range(16))

    result = AnalysisResult(
        bpm=bpm,
        duration_seconds=8.0,
        beat_times=beat_times,
        onset_times=(),
        chords=(
            ChordEvent(0.0, 1.85, "F#m"),  # sub 0..14 をカバー
            ChordEvent(1.85, 4.0, "D"),     # sub 15 (1.875s) のみD (0.25拍)
        ),
        chord_engine="test",
    )

    MusicXmlGenerator().generate(result, output)

    parsed = converter.parse(output)
    measure_1 = parsed.parts[0].getElementsByClass("Measure")[0]
    symbols = [s.figure for s in measure_1.recurse().getElementsByClass("ChordSymbol")]
    assert "D" not in symbols, f"0.25拍の食いDが平滑化されずに残っている: {symbols}"


def test_half_beat_anticipation_preserved(tmp_path) -> None:
    """小節末尾の0.5拍（2スロット）のアウフタクトは保持されること。

    BPM=120, 4/4拍子: 1小節=2.0s, 0.5拍=0.250s。
    小節1末尾のsub 14(1.750s), sub 15(1.875s)からDが検出され、小節2もDの場合、
    8分音符（0.5拍）のアウフタクトとして正しく保持されること。
    """
    output = tmp_path / "half_anticipation.musicxml"
    bpm = 120.0
    spb = 60.0 / bpm
    beat_times = tuple(i * spb for i in range(16))

    result = AnalysisResult(
        bpm=bpm,
        duration_seconds=8.0,
        beat_times=beat_times,
        onset_times=(),
        chords=(
            ChordEvent(0.0, 1.70, "F#m"),  # sub 0..13 をカバー
            ChordEvent(1.70, 4.0, "D"),     # sub 14, 15 (1.75s, 1.875s) でD (0.5拍)
        ),
        chord_engine="test",
    )

    MusicXmlGenerator().generate(result, output)

    parsed = converter.parse(output)
    measure_1 = parsed.parts[0].getElementsByClass("Measure")[0]
    symbols = [s.figure for s in measure_1.recurse().getElementsByClass("ChordSymbol")]
    assert "D" in symbols, f"0.5拍のアウフタクトDが平滑化されて消えてしまっている: {symbols}"


def test_unmatched_end_chord_smoothed_away(tmp_path) -> None:
    """小節末尾のコードが次の小節の頭と異なる場合はノイズとして平滑化されること。"""
    output = tmp_path / "unmatched_end.musicxml"
    bpm = 120.0
    spb = 60.0 / bpm
    beat_times = tuple(i * spb for i in range(16))

    result = AnalysisResult(
        bpm=bpm,
        duration_seconds=8.0,
        beat_times=beat_times,
        onset_times=(),
        chords=(
            ChordEvent(0.0, 1.70, "C"),
            ChordEvent(1.70, 2.0, "E"),     # 小節1末尾に0.5拍だけE
            ChordEvent(2.0, 4.0, "G"),      # 小節2の頭はG
        ),
        chord_engine="test",
    )

    MusicXmlGenerator().generate(result, output)

    parsed = converter.parse(output)
    measure_1 = parsed.parts[0].getElementsByClass("Measure")[0]
    symbols = [s.figure for s in measure_1.recurse().getElementsByClass("ChordSymbol")]
    assert "E" not in symbols, f"次の小節と不一致のノイズEが残っている: {symbols}"


def test_half_beat_subdivision_mode_smooths_half_beat(tmp_path) -> None:
    """beat_subdivision=0.5 モードでは0.5拍の食いが平滑化されること。"""
    output = tmp_path / "half_subdivision.musicxml"
    bpm = 120.0
    spb = 60.0 / bpm
    beat_times = tuple(i * spb for i in range(16))

    result = AnalysisResult(
        bpm=bpm,
        duration_seconds=8.0,
        beat_times=beat_times,
        onset_times=(),
        chords=(
            ChordEvent(0.0, 1.70, "F#m"),
            ChordEvent(1.70, 4.0, "D"),     # 0.5拍の食い
        ),
        chord_engine="test",
    )

    MusicXmlGenerator().generate(result, output, beat_subdivision=0.5)

    parsed = converter.parse(output)
    measure_1 = parsed.parts[0].getElementsByClass("Measure")[0]
    symbols = [s.figure for s in measure_1.recurse().getElementsByClass("ChordSymbol")]
    assert "D" not in symbols, f"0.5拍モードでDが平滑化されていない: {symbols}"



