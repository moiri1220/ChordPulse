"""コード認識インターフェースと軽量なクロマベースライン。

MusicXML生成やAPIを変更することなく、Chordinoやmadmomがベースラインを置き換えられるよう、
認識エンジン境界は意図的にパイプラインから独立させています。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import AudioData, BeatGrid, ChordEvent

PITCH_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


class ChordRecognizer(Protocol):
    name: str

    def recognize(self, audio: AudioData, beat_grid: BeatGrid) -> tuple[ChordEvent, ...]:
        """検出された拍（ビート）の間隔をカバーする、マージされたコードイベントを返します。"""


class ChordEngineUnavailable(RuntimeError):
    """要求された認識エンジンがインストールされていないか、実装されていない場合に発生する例外。"""


@dataclass(frozen=True, slots=True)
class _ChordTemplate:
    suffix: str
    intervals: tuple[int, ...]


TEMPLATES = (
    _ChordTemplate("", (0, 4, 7)),
    _ChordTemplate("m", (0, 3, 7)),
    _ChordTemplate("7", (0, 4, 7, 10)),
    _ChordTemplate("m7", (0, 3, 7, 10)),
)


class ChromagramChordRecognizer:
    """拍同期クロマから基本的な三和音と七の和音を推定します。"""

    name = "librosa-chroma-template"

    def __init__(self, *, hop_length: int = 512, minimum_confidence: float = 0.18) -> None:
        self.hop_length = hop_length
        self.minimum_confidence = minimum_confidence

    def recognize(self, audio: AudioData, beat_grid: BeatGrid) -> tuple[ChordEvent, ...]:
        try:
            import librosa
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise RuntimeError("コード解析にはlibrosaとnumpyが必要です") from exc

        chroma = librosa.feature.chroma_stft(
            y=audio.samples,
            sr=audio.sample_rate,
            n_fft=4_096,
            hop_length=self.hop_length,
        )
        beat_times = beat_grid.beat_times
        beat_seconds = beat_grid.seconds_per_beat
        raw_events: list[ChordEvent] = []
        for index, start in enumerate(beat_times):
            if start >= audio.duration_seconds:
                continue
            end = beat_times[index + 1] if index + 1 < len(beat_times) else start + beat_seconds
            frame_start = int(round(start * audio.sample_rate / self.hop_length))
            frame_start = min(frame_start, max(0, chroma.shape[1] - 1))
            frame_end = int(round(end * audio.sample_rate / self.hop_length))
            frame_end = max(frame_start + 1, min(frame_end, chroma.shape[1]))
            end = min(float(end), audio.duration_seconds)
            if end <= start:
                continue
            vector = np.mean(chroma[:, frame_start:frame_end], axis=1)
            label, confidence = self._best_label(vector, np)
            raw_events.append(
                ChordEvent(
                    start_seconds=float(start),
                    end_seconds=end,
                    label=label,
                    confidence=confidence,
                )
            )
        return self._merge_adjacent(raw_events)

    def _best_label(self, vector, np) -> tuple[str, float]:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return "N", 0.0
        normalized = vector / norm
        best_label = "N"
        best_score = 0.0
        for root, root_name in enumerate(PITCH_NAMES):
            for template in TEMPLATES:
                values = np.zeros(12, dtype=float)
                values[(root + np.asarray(template.intervals)) % 12] = 1.0
                values /= np.linalg.norm(values)
                score = float(np.dot(normalized, values))
                if score > best_score:
                    best_score = score
                    best_label = f"{root_name}{template.suffix}"
        if best_score < self.minimum_confidence:
            return "N", best_score
        return best_label, best_score

    @staticmethod
    def _merge_adjacent(events: list[ChordEvent]) -> tuple[ChordEvent, ...]:
        if not events:
            return ()
        merged: list[ChordEvent] = [events[0]]
        for event in events[1:]:
            previous = merged[-1]
            if event.label == previous.label:
                merged[-1] = ChordEvent(
                    start_seconds=previous.start_seconds,
                    end_seconds=event.end_seconds,
                    label=previous.label,
                    confidence=min(
                        value
                        for value in (previous.confidence, event.confidence)
                        if value is not None
                    ),
                )
            else:
                merged.append(event)
        return tuple(merged)


def create_chord_recognizer(engine: str = "template") -> ChordRecognizer:
    """要求されたエンジンを暗黙的に変更することなく、名前付きコードエンジンを作成します。

    テンプレート認識エンジンはフェーズ1のベースラインです。Chordinoおよびmadmomは、
    ネイティブのランタイム依存関係がすべての開発環境で利用可能であるとは限らないため、
    明示的な拡張ポイントとして残されています。
    """

    normalized = engine.strip().lower()
    if normalized in {"template", "librosa-chroma-template"}:
        return ChromagramChordRecognizer()
    if normalized in {"chordino", "madmom"}:
        raise ChordEngineUnavailable(
            f"{normalized} コードエンジンはインストールされていません。テンプレートベースラインを使用するか、"
            "インジェクトされたChordRecognizerを提供してください"
        )
    raise ValueError(f"未知のコードエンジン: {engine}")

