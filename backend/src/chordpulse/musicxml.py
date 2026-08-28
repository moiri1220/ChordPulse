"""マスターコード譜用のMusicXML生成。"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable
from pathlib import Path

from .models import AnalysisResult, ChordEvent, RhythmLevel


class MusicXmlGenerationError(RuntimeError):
    """MusicXMLを生成できない場合に発生する例外。"""


# chords.py の PITCH_NAMES は "Bb"/"Eb"/"Ab" 等の慣用表記を使うが、
# music21 の ChordSymbol はフラット記号を "-"（ハイフン）で表現する。
# 例: "Bb" -> "B-", "Bbm7" -> "B-m7"
# この変換は表示層（MusicXML生成）に閉じ込め、内部表現を変更しない。
_FLAT_ROOTS = ("Bb", "Eb", "Ab", "Db", "Gb")
_FLAT_ROOT_TO_MUSIC21 = {root: root[0] + "-" for root in _FLAT_ROOTS}


def _to_music21_label(label: str) -> str:
    """コードラベルを music21 ChordSymbol が受け付ける形式に変換する。

    "Bb" -> "B-"、"Bbm7" -> "B-m7" のようにルート音のフラット記法を変換する。
    "N"（無音）はそのまま返す。
    """
    if label == "N":
        return label
    for flat_root, music21_root in _FLAT_ROOT_TO_MUSIC21.items():
        if label.startswith(flat_root):
            return music21_root + label[len(flat_root):]
    return label


class MusicXmlGenerator:
    """music21を使用して、スラッシュ表記とコードシンボルをレンダリングします。"""

    def generate(
        self,
        result: AnalysisResult,
        output_path: Path,
        *,
        rhythm_level: RhythmLevel = None,
        anticipation_smoothing: bool = True,
        title: str = "ChordPulse Master Chord Chart",
    ) -> Path:
        try:
            from music21 import duration, harmony, metadata, meter, note, stream, tempo
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise MusicXmlGenerationError("music21がインストールされていません") from exc

        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        beat_seconds = 60.0 / result.bpm if result.bpm > 0 else 0.5
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
            _render_adaptive_measure(
                measure,
                timeline,
                result,
                measure_index,
                note,
                harmony,
                duration,
                anticipation_smoothing=anticipation_smoothing,
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
    result: AnalysisResult,
    measure: int,
    beat: int,
) -> float:
    index = measure * result.beats_per_measure + beat
    beat_times = result.beat_times
    if not beat_times:
        bpm = result.bpm if result.bpm > 0 else 120.0
        return index * (60.0 / bpm)
    if index < len(beat_times):
        return beat_times[index]
    
    last_time = beat_times[-1]
    bpm = result.bpm if result.bpm > 0 else 120.0
    excess_beats = index - (len(beat_times) - 1)
    return last_time + excess_beats * (60.0 / bpm)


def _subdivision_time(
    result: AnalysisResult,
    measure: int,
    subdivision: int,
) -> float:
    beat = subdivision // 2
    is_off_beat = (subdivision % 2) != 0
    t1 = _measure_time(result, measure, beat)
    if not is_off_beat:
        return t1
    t2 = _measure_time(result, measure, beat + 1)
    return (t1 + t2) / 2.0


def _insert_chord_symbol(measure, label: str, offset: float, harmony, duration) -> None:
    if label == "N":
        return
    symbol = harmony.ChordSymbol(_to_music21_label(label))
    symbol.duration = duration.Duration(0)
    measure.insert(offset, symbol)


def _slash_note(note, duration, length: float):
    value = note.Note("C4", quarterLength=length)
    value.notehead = "slash"
    value.stemDirection = "up"
    return value


def _render_adaptive_measure(
    measure,
    timeline: _ChordTimeline,
    result: AnalysisResult,
    measure_index: int,
    note,
    harmony,
    duration,
    *,
    anticipation_smoothing: bool = True,
) -> None:
    """小節内のコードチェンジタイミングと長さに応じて、最適な音価（全音符、2分音符、4分音符、8分音符等）
    のスラッシュ音符およびコードシンボルをレンダリングします。
    """
    beats_per_measure = result.beats_per_measure
    subdivision_count = beats_per_measure * 2  # 8分音符スロット (各 0.5 拍)

    # 各スロット（0.5拍単位）のコードラベルを取得
    slot_labels: list[str] = []
    for sub in range(subdivision_count):
        t = _subdivision_time(result, measure_index, sub)
        slot_labels.append(timeline.label_at(t))

    # アンティシペーションおよびディレイ（スピルオーバー）の平滑化
    # 小節の境界付近での0.5拍のズレを吸収し、クオンタイズします。
    if anticipation_smoothing and len(slot_labels) >= 2:
        # 1. ディレイ（スピルオーバー）の平滑化
        # 小節の最初の0.5拍だけ前の小節の最後のコードが残っており、その後別のコードに変わる場合、
        # 最初の0.5拍を次のコードで上書きし、小節の頭から新しいコードにクオンタイズする。
        first_label = slot_labels[0]
        second_label = slot_labels[1]
        if first_label != second_label and measure_index > 0:
            t_prev = _subdivision_time(result, measure_index - 1, subdivision_count - 1)
            prev_measure_label = timeline.label_at(t_prev)
            if first_label == prev_measure_label:
                slot_labels[0] = second_label

        # 2. アンティシペーションの平滑化
        # 小節の最後の0.5拍だけコードが変わっており、かつそれが次の小節の頭のコードと同じ場合、
        # その小節内ではコードチェンジせず、直前のコードを維持する。
        last_label = slot_labels[-1]
        prev_label = slot_labels[-2]
        if last_label != prev_label:
            t_next = _subdivision_time(result, measure_index + 1, 0)
            next_measure_label = timeline.label_at(t_next)
            if last_label == next_measure_label:
                slot_labels[-1] = prev_label

    # 連続する同じコードラベルのスロットをセグメントにまとめる
    segments: list[tuple[str, float, float]] = []  # (label, offset, length)
    current_label = slot_labels[0]
    current_start = 0.0
    current_len = 0.5

    for sub in range(1, subdivision_count):
        label = slot_labels[sub]
        if label == current_label:
            current_len += 0.5
        else:
            segments.append((current_label, current_start, current_len))
            current_label = label
            current_start = sub * 0.5
            current_len = 0.5
    segments.append((current_label, current_start, current_len))

    # 各セグメントをレンダリング
    previous_label = None
    for label, offset, length in segments:
        if label != "N" and label != previous_label:
            _insert_chord_symbol(measure, label, offset, harmony, duration)
            previous_label = label

        if label == "N":
            measure.insert(offset, note.Rest(quarterLength=length))
        else:
            measure.insert(offset, _slash_note(note, duration, length))
