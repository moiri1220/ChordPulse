"""組み合わせ可能なフェーズ1解析パイプライン。"""

from __future__ import annotations

from pathlib import Path

from .audio import load_audio
from .beats import BeatAnalyzer, create_beat_analyzer
from .chords import ChordRecognizer, create_chord_recognizer
from .models import AnalysisResult, ChordEvent, RhythmLevel
from .musicxml import MusicXmlGenerator


def _align_downbeat_offset(
    beat_times: tuple[float, ...],
    downbeat_times: tuple[float, ...],
    beats_per_measure: int,
) -> int:
    """検出されたダウンビート群に最も近いビートを特定し、
    beat_timesの先頭から何拍目が小節の頭（ダウンビート）であるかを算出します。
    """
    if not beat_times or not downbeat_times:
        return 0

    votes = [0] * beats_per_measure
    for db in downbeat_times[:8]:
        closest_idx = -1
        min_diff = float("inf")
        for i, b_time in enumerate(beat_times):
            diff = abs(b_time - db)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
            elif diff > min_diff and min_diff < 0.5:
                break
        if closest_idx != -1 and min_diff < 0.35:
            votes[closest_idx % beats_per_measure] += 1

    max_votes = -1
    best_offset = 0
    for offset, count in enumerate(votes):
        if count > max_votes:
            max_votes = count
            best_offset = offset

    return best_offset


def _estimate_downbeat_offset(
    beat_times: tuple[float, ...],
    chords: tuple[ChordEvent, ...],
    beats_per_measure: int,
) -> int:
    """コードチェンジの頻度と最初の実音コードの発生位置を利用して、
    beat_timesの先頭から何拍目がダウンビート（1拍目）かを推定します。
    """
    if not beat_times or not chords:
        return 0

    scores = [0.0] * beats_per_measure

    # 1. 最初の実音コード（"N"以外）の開始位置にボーナスを付与
    first_real_chord = next((c for c in chords if c.label != "N"), None)
    if first_real_chord is not None:
        closest_idx = -1
        min_diff = float("inf")
        for i, b_time in enumerate(beat_times):
            diff = abs(b_time - first_real_chord.start_seconds)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
            elif diff > min_diff:
                break
        if closest_idx != -1 and min_diff < 0.3:
            offset = closest_idx % beats_per_measure
            scores[offset] += 2.0

    # 2. 各コードチェンジのタイミングを集計
    for chord in chords:
        if chord.label == "N":
            continue
        closest_beat_idx = -1
        min_diff = float("inf")
        for i, b_time in enumerate(beat_times):
            diff = abs(b_time - chord.start_seconds)
            if diff < min_diff:
                min_diff = diff
                closest_beat_idx = i
            elif diff > min_diff:
                break

        if closest_beat_idx != -1 and min_diff < 0.25:
            offset = closest_beat_idx % beats_per_measure
            scores[offset] += 1.0

    best_offset = 0
    max_score = -1.0
    for i, s in enumerate(scores):
        if s > max_score:
            max_score = s
            best_offset = i

    return best_offset


class AnalysisPipeline:
    """オーディオのロード、拍解析、コード認識、およびMusicXML生成を調整します。"""

    def __init__(
        self,
        *,
        beat_analyzer: BeatAnalyzer | None = None,
        beat_engine: str = "deep_learning",
        chord_recognizer: ChordRecognizer | None = None,
        chord_engine: str = "btc",
        musicxml_generator: MusicXmlGenerator | None = None,
    ) -> None:
        self.beat_analyzer = beat_analyzer or create_beat_analyzer(beat_engine)
        if chord_recognizer is not None and chord_engine != "btc":
            raise ValueError("chord_engineをchord_recognizerと組み合わせることはできません")
        self.chord_recognizer = chord_recognizer or create_chord_recognizer(chord_engine)
        self.musicxml_generator = musicxml_generator or MusicXmlGenerator()

    def analyze(
        self,
        audio_path: Path,
        *,
        beat_subdivision: float = 0.25,
    ) -> AnalysisResult:
        audio = load_audio(audio_path)
        beat_grid = self.beat_analyzer.analyze(audio)
        try:
            chords = self.chord_recognizer.recognize(audio, beat_grid, beat_subdivision=beat_subdivision)
        except TypeError:
            chords = self.chord_recognizer.recognize(audio, beat_grid)

        beat_times = beat_grid.beat_times
        beats_per_measure = beat_grid.beats_per_measure
        if beat_grid.downbeat_times:
            best_offset = _align_downbeat_offset(
                beat_times, beat_grid.downbeat_times, beats_per_measure
            )
        else:
            best_offset = _estimate_downbeat_offset(beat_times, chords, beats_per_measure)

        if best_offset > 0 and beat_times:
            pad_count = beats_per_measure - best_offset
            avg_beat_seconds = 60.0 / beat_grid.bpm if beat_grid.bpm > 0 else 0.5
            new_beats = []
            first_beat = beat_times[0]
            for i in range(pad_count, 0, -1):
                new_beats.append(first_beat - i * avg_beat_seconds)
            beat_times = tuple(new_beats) + beat_times

        return AnalysisResult(
            bpm=beat_grid.bpm,
            duration_seconds=audio.duration_seconds,
            beat_times=beat_times,
            onset_times=beat_grid.onset_times,
            chords=chords,
            beats_per_measure=beats_per_measure,
            chord_engine=getattr(
                self.chord_recognizer,
                "name",
                type(self.chord_recognizer).__name__,
            ),
        )

    def analyze_to_musicxml(
        self,
        audio_path: Path,
        output_path: Path,
        *,
        rhythm_level: RhythmLevel = None,
        anticipation_smoothing: bool = True,
        beat_subdivision: float = 0.25,
    ) -> AnalysisResult:
        result = self.analyze(audio_path, beat_subdivision=beat_subdivision)
        self.musicxml_generator.generate(
            result,
            output_path,
            rhythm_level=rhythm_level,
            anticipation_smoothing=anticipation_smoothing,
            beat_subdivision=beat_subdivision,
        )
        return result

