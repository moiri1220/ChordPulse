"""組み合わせ可能なフェーズ1解析パイプライン。"""

from __future__ import annotations

from pathlib import Path

from .audio import load_audio
from .beats import LibrosaBeatAnalyzer
from .chords import ChordRecognizer, create_chord_recognizer
from .models import AnalysisResult, RhythmLevel
from .musicxml import MusicXmlGenerator


class AnalysisPipeline:
    """オーディオのロード、拍解析、コード認識、およびMusicXML生成を調整します。"""

    def __init__(
        self,
        *,
        beat_analyzer: LibrosaBeatAnalyzer | None = None,
        chord_recognizer: ChordRecognizer | None = None,
        chord_engine: str = "template",
        musicxml_generator: MusicXmlGenerator | None = None,
    ) -> None:
        self.beat_analyzer = beat_analyzer or LibrosaBeatAnalyzer()
        if chord_recognizer is not None and chord_engine != "template":
            raise ValueError("chord_engineをchord_recognizerと組み合わせることはできません")
        self.chord_recognizer = chord_recognizer or create_chord_recognizer(chord_engine)
        self.musicxml_generator = musicxml_generator or MusicXmlGenerator()

    def analyze(self, audio_path: Path) -> AnalysisResult:
        audio = load_audio(audio_path)
        beat_grid = self.beat_analyzer.analyze(audio)
        chords = self.chord_recognizer.recognize(audio, beat_grid)
        return AnalysisResult(
            bpm=beat_grid.bpm,
            duration_seconds=audio.duration_seconds,
            beat_times=beat_grid.beat_times,
            onset_times=beat_grid.onset_times,
            chords=chords,
            beats_per_measure=beat_grid.beats_per_measure,
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
        rhythm_level: RhythmLevel,
    ) -> AnalysisResult:
        result = self.analyze(audio_path)
        self.musicxml_generator.generate(result, output_path, rhythm_level=rhythm_level)
        return result

