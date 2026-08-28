"""解析パイプライン用の、依存関係の少ない共有データモデル。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RhythmLevel = Literal[1, 2, 3]


@dataclass(frozen=True, slots=True)
class AudioData:
    """単一の解析リクエスト用にロードされたオーディオサンプル。

    サンプルはリクエストの間メモリ上に保持され、このモデルによって書き込まれることはありません。
    """

    samples: Any
    sample_rate: int
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class BeatGrid:
    """オーディオ開始時からの秒数で表された、拍（ビート）およびオンセットの位置。"""

    bpm: float
    beat_times: tuple[float, ...]
    onset_times: tuple[float, ...]
    beats_per_measure: int = 4

    @property
    def seconds_per_beat(self) -> float:
        if self.bpm <= 0:
            raise ValueError("BPMはゼロより大きくなければなりません")
        return 60.0 / self.bpm


@dataclass(frozen=True, slots=True)
class ChordEvent:
    """特定の時間区間に対するコードネーム。"""

    start_seconds: float
    end_seconds: float
    label: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.start_seconds < 0:
            raise ValueError("start_secondsは負の値であってはなりません")
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_secondsはstart_secondsより大きくなければなりません")
        if not self.label:
            raise ValueError("labelは空であってはなりません")


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """解析からMusicXML生成へ渡されるシリアライズ可能な結果。"""

    bpm: float
    duration_seconds: float
    beat_times: tuple[float, ...]
    onset_times: tuple[float, ...]
    chords: tuple[ChordEvent, ...]
    beats_per_measure: int = 4
    chord_engine: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "bpm": round(self.bpm, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "beat_times": list(self.beat_times),
            "onset_times": list(self.onset_times),
            "beats_per_measure": self.beats_per_measure,
            "chord_engine": self.chord_engine,
            "chords": [
                {
                    "start_seconds": event.start_seconds,
                    "end_seconds": event.end_seconds,
                    "label": event.label,
                    "confidence": event.confidence,
                }
                for event in self.chords
            ],
        }


