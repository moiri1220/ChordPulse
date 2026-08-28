"""マスターコード譜用のMusicXML生成。"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable
from pathlib import Path

from .models import AnalysisResult, ChordEvent, RhythmLevel


class MusicXmlGenerationError(RuntimeError):
    """MusicXMLを生成できない場合に発生する例外。"""


class MusicXmlGenerator:
    """music21を使用して、スラッシュ表記とコードシンボルをレンダリングします。"""

    def generate(
        self,
        result: AnalysisResult,
        output_path: Path,
        *,
        rhythm_level: RhythmLevel,
        title: str = "ChordPulse Master Chord Chart",
    ) -> Path:
        try:
            from music21 import duration, harmony, metadata, meter, note, stream, tempo
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise MusicXmlGenerationError("music21がインストールされていません") from exc

        if rhythm_level not in (1, 2, 3):
            raise MusicXmlGenerationError("rhythm_levelは1、2、または3でなければなりません")

        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        beat_seconds = 60.0 / result.bpm
        origin = result.beat_times[0] if result.beat_times else 0.0
        total_beats = max(
            result.beats_per_measure,
            math.ceil(max(result.duration_seconds - origin, beat_seconds) / beat_seconds),
        )
        measure_count = math.ceil(total_beats / result.beats_per_measure)
        timeline = _ChordTimeline(result.chords)

        score = stream.Score(id="chordpulse-score")
        score.metadata = metadata.Metadata()
        score.metadata.title = title
        part = stream.Part(id="master-chord-part")
        part.partName = "Master Chord Chart"

        for measure_index in range(measure_count):
            measure = stream.Measure(number=measure_index + 1)
            if measure_index == 0:
                measure.insert(0, meter.TimeSignature(f"{result.beats_per_measure}/4"))
                measure.insert(0, tempo.MetronomeMark(number=round(result.bpm, 2)))
            if rhythm_level == 1:
                _render_simple_measure(
                    measure,
                    timeline,
                    origin,
                    beat_seconds,
                    result.beats_per_measure,
                    measure_index,
                    note,
                    harmony,
                    duration,
                )
            elif rhythm_level == 2:
                _render_quarter_measure(
                    measure,
                    timeline,
                    origin,
                    beat_seconds,
                    result.beats_per_measure,
                    measure_index,
                    note,
                    harmony,
                    duration,
                )
            else:
                _render_rhythm_measure(
                    measure,
                    timeline,
                    result,
                    origin,
                    beat_seconds,
                    measure_index,
                    note,
                    harmony,
                    duration,
                )
            part.append(measure)

        score.insert(0, part)
        try:
            score.write("musicxml", fp=str(output_path))
        except Exception as exc:
            raise MusicXmlGenerationError("music21がMusicXMLを書き込めませんでした") from exc
        return output_path


class _ChordTimeline:
    """時間位置によるO(log n)検索のために、コードイベントをインデックスします。"""

    def __init__(self, events: Iterable[ChordEvent]) -> None:
        # 防御的ソート：recognize()は時間順に隣接イベントをマージしますが、
        # 呼び出し元の順序に契約を依存させるべきではありません。
        sorted_events = sorted(events, key=lambda e: e.start_seconds)
        self._events = tuple(sorted_events)
        self._starts = [e.start_seconds for e in sorted_events]

    def label_at(self, seconds: float) -> str:
        """指定された秒数 *seconds* において有効なコードラベルを返します。ない場合は 'N' を返します。"""
        idx = bisect.bisect_right(self._starts, seconds) - 1
        if idx < 0:
            return "N"
        event = self._events[idx]
        return event.label if seconds < event.end_seconds else "N"


def _measure_time(
    origin: float,
    beat_seconds: float,
    beats_per_measure: int,
    measure: int,
    beat: int,
) -> float:
    return origin + (measure * beats_per_measure + beat) * beat_seconds


def _insert_chord_symbol(measure, label: str, offset: float, harmony, duration) -> None:
    if label == "N":
        return
    symbol = harmony.ChordSymbol(label)
    symbol.duration = duration.Duration(0)
    measure.insert(offset, symbol)


def _slash_note(note, duration, length: float):
    value = note.Note("C4", quarterLength=length)
    value.notehead = "slash"
    value.stemDirection = "up"
    return value


def _render_simple_measure(
    measure,
    timeline,
    origin,
    beat_seconds,
    beats_per_measure,
    measure_index,
    note,
    harmony,
    duration,
) -> None:
    """レベル1の表記をレンダリングします：デフォルトでは全音符、小節の中間点（拍数 // 2）で
    コードが変わる場合は2つの二分音符になります。

    これにより、出力を読みやすく保ちつつ（このレベルではより小さい音価は使用されません）、「全音符・二分音符を基本とする」という仕様要件を満たします。
    """
    half = beats_per_measure // 2
    start_seconds = _measure_time(origin, beat_seconds, beats_per_measure, measure_index, 0)
    half_seconds = _measure_time(origin, beat_seconds, beats_per_measure, measure_index, half)
    label_start = timeline.label_at(start_seconds)
    label_half = timeline.label_at(half_seconds)

    if label_start == label_half:
        # 単一のコードが小節全体をカバーしている場合 — 全音符1つを使用します。
        _insert_chord_symbol(measure, label_start, 0.0, harmony, duration)
        measure.insert(0.0, _slash_note(note, duration, float(beats_per_measure)))
    else:
        # 中間点でコードが変化する場合 — 二分音符2つを使用します。
        _insert_chord_symbol(measure, label_start, 0.0, harmony, duration)
        measure.insert(0.0, _slash_note(note, duration, float(half)))
        _insert_chord_symbol(measure, label_half, float(half), harmony, duration)
        measure.insert(float(half), _slash_note(note, duration, float(beats_per_measure - half)))


def _render_quarter_measure(
    measure,
    timeline,
    origin,
    beat_seconds,
    beats_per_measure,
    measure_index,
    note,
    harmony,
    duration,
) -> None:
    previous_label = None
    for beat in range(beats_per_measure):
        seconds = _measure_time(origin, beat_seconds, beats_per_measure, measure_index, beat)
        offset = float(beat)
        label = timeline.label_at(seconds)
        if label != previous_label:
            _insert_chord_symbol(measure, label, offset, harmony, duration)
            previous_label = label
        measure.insert(offset, _slash_note(note, duration, 1.0))


def _render_rhythm_measure(
    measure,
    timeline,
    result,
    origin,
    beat_seconds,
    measure_index,
    note,
    harmony,
    duration,
) -> None:
    previous_label = None
    subdivision_count = result.beats_per_measure * 2
    subdivision_seconds = beat_seconds / 2
    onset_tolerance = subdivision_seconds * 0.35
    for subdivision in range(subdivision_count):
        seconds = origin + (
            measure_index * result.beats_per_measure * beat_seconds
            + subdivision * subdivision_seconds
        )
        offset = subdivision * 0.5
        label = timeline.label_at(seconds)
        if label != previous_label:
            _insert_chord_symbol(measure, label, offset, harmony, duration)
            previous_label = label
        is_onset = any(abs(onset - seconds) <= onset_tolerance for onset in result.onset_times)
        if is_onset:
            measure.insert(offset, _slash_note(note, duration, 0.5))
        else:
            measure.insert(offset, note.Rest(quarterLength=0.5))

