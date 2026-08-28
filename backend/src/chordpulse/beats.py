from __future__ import annotations

import logging
from typing import Protocol

from .models import AudioData, BeatGrid

_logger = logging.getLogger(__name__)


class BeatAnalysisError(RuntimeError):
    """有効なビートグリッドを推定できない場合に発生する例外。"""


class BeatAnalyzer(Protocol):
    """ビート推定エンジンのインターフェースプロトコル。"""

    def analyze(self, audio: AudioData, *, beats_per_measure: int = 4) -> BeatGrid:
        ...


def complete_leading_beats(
    beat_times: tuple[float, ...],
    *,
    seconds_per_beat: float,
) -> tuple[float, ...]:
    """タイムゼロから最初の検出ビートまでのテンポグリッドを補完します。

    ビートトラッカーはイントロや最初の過渡応答（トランジェント）を見落とすことがよくあります。
    そのため、最初に検出されたビートがチャートの起点（ゼロ秒）であるとは仮定しません。
    フェーズ1では、推定されたテンポを使用して先行するグリッドを再構成し、
    タイムゼロを最初の小節の起点として扱います。
    """

    if not beat_times or seconds_per_beat <= 0:
        return beat_times

    first_detected = beat_times[0]
    if first_detected <= seconds_per_beat * 0.5:
        # 最初の拍がオーディオ開始位置に十分近いため、起点として扱います。
        anchor = 0.0
    else:
        # 最初に検出された拍から、ビート単位で逆方向に遡ります。
        # 次のステップでアンカーがタイムゼロより半拍以上前になってしまうまで遡ることで、
        # 録音開始前に余分な拍が挿入されるのを防ぎます。
        anchor = first_detected
        while anchor - seconds_per_beat >= -seconds_per_beat * 0.5:
            anchor -= seconds_per_beat
        anchor = max(0.0, anchor)

    # 最初に検出された拍に近すぎる（拍間隔の4分の1未満）場合は、
    # 重複を防ぐために合成拍を追加しません。
    tolerance = seconds_per_beat * 0.25
    leading: list[float] = []
    cursor = anchor
    while cursor < first_detected - tolerance:
        leading.append(round(cursor, 9))
        cursor += seconds_per_beat
    if first_detected > 0 and not leading:
        # フォールバック: 丸め誤差などで先行拍が生成されなかった場合でも、
        # 確実に0.0秒の拍が含まれるようにします。
        leading.append(0.0)
    return tuple(leading) + beat_times


class LibrosaBeatAnalyzer:
    """ロードされたオーディオからビートグリッドとオンセット位置を推定します。"""

    def __init__(self, *, hop_length: int = 512) -> None:
        self.hop_length = hop_length

    def analyze(self, audio: AudioData, *, beats_per_measure: int = 4) -> BeatGrid:
        try:
            import librosa
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise BeatAnalysisError("拍解析にはlibrosaとnumpyが必要です") from exc

        onset_envelope = librosa.onset.onset_strength(
            y=audio.samples,
            sr=audio.sample_rate,
            hop_length=self.hop_length,
        )
        tempo, beat_frames = librosa.beat.beat_track(
            onset_envelope=onset_envelope,
            sr=audio.sample_rate,
            hop_length=self.hop_length,
            units="frames",
        )
        tempo_values = np.asarray(tempo).reshape(-1)
        bpm = float(round(tempo_values[0])) if tempo_values.size else 0.0
        if bpm <= 0:
            raise BeatAnalysisError("正のBPMを推定できませんでした")

        beat_times = tuple(
            float(value)
            for value in librosa.frames_to_time(
                beat_frames,
                sr=audio.sample_rate,
                hop_length=self.hop_length,
            )
            if 0 <= value < audio.duration_seconds
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_envelope,
            sr=audio.sample_rate,
            hop_length=self.hop_length,
            units="frames",
            backtrack=False,
        )
        onset_times = tuple(
            float(value)
            for value in librosa.frames_to_time(
                onset_frames,
                sr=audio.sample_rate,
                hop_length=self.hop_length,
            )
            if 0 <= value < audio.duration_seconds
        )
        if not beat_times:
            raise BeatAnalysisError("拍位置を検出できませんでした")
        beat_times = complete_leading_beats(
            beat_times,
            seconds_per_beat=60.0 / bpm,
        )
        return BeatGrid(
            bpm=bpm,
            beat_times=beat_times,
            onset_times=onset_times,
            beats_per_measure=beats_per_measure,
            downbeat_times=(),
        )


def _calculate_bpm_from_beats(beat_times_list: list[float] | tuple[float, ...]) -> float:
    """拍時刻リストから線形回帰を用いてサブフレーム精度のBPMを算出します。

    beat-thisの50 FPS（20ms）量子化によって生じる離散化誤差（例: 160 BPMが157.89になる現象）を、
    拍インデックスに対する拍時刻の回帰直線の傾きから相殺・解消します。
    """
    if len(beat_times_list) < 2:
        return 120.0

    import numpy as np

    beats = np.asarray(beat_times_list, dtype=np.float64)
    diffs = np.diff(beats)
    positive_diffs = diffs[diffs > 0.05]
    if len(positive_diffs) == 0:
        return 120.0

    median_diff = float(np.median(positive_diffs))
    if median_diff <= 0:
        return 120.0

    # 拍抜け（休符など）を考慮し、連続的な拍インデックスを推定
    indices = [0]
    for d in diffs:
        step = max(1, int(round(d / median_diff))) if d > 0.05 else 0
        indices.append(indices[-1] + step)

    indices_arr = np.asarray(indices, dtype=np.float64)

    # 局所的なテンポ変動（リタルダンド等）による回帰直線の歪みを防ぐため、
    # スライディングウィンドウ（16拍）ごとに傾きを計算し、その中央値を採用します。
    # 16拍（4小節）あれば、20msの量子化誤差を十分吸収できます。
    K = min(16, len(beats) - 1)
    if K <= 0:
        return float(round(60.0 / median_diff, 2))

    slopes = []
    for i in range(len(beats) - K):
        dt = beats[i + K] - beats[i]
        d_idx = indices_arr[i + K] - indices_arr[i]
        if d_idx > 0:
            slopes.append(dt / d_idx)

    if not slopes:
        return float(round(60.0 / median_diff, 2))

    median_slope = float(np.median(slopes))
    if median_slope > 0:
        raw_bpm = 60.0 / median_slope
        return float(round(raw_bpm))

    return float(round(60.0 / median_diff))


class BeatThisAnalyzer:
    """beat-this（深層学習モデル）を用いた高精度ビート・ダウンビート推定器。"""

    def __init__(
        self,
        *,
        checkpoint_path: str = "final0",
        device: str = "cpu",
        dbn: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.dbn = dbn
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from beat_this.inference import Audio2Beats
            except ImportError as exc:
                raise BeatAnalysisError(
                    "深層学習拍解析にはbeat-thisが必要です: pip install beat-this"
                ) from exc
            self._model = Audio2Beats(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                dbn=self.dbn,
            )
        return self._model

    def analyze(self, audio: AudioData, *, beats_per_measure: int = 4) -> BeatGrid:
        try:
            import numpy as np
        except ImportError as exc:
            raise BeatAnalysisError("拍解析にはnumpyが必要です") from exc

        model = self._get_model()
        try:
            raw_beats, raw_downbeats = model(audio.samples, audio.sample_rate)
        except Exception as exc:
            raise BeatAnalysisError(f"beat-this による拍解析に失敗しました: {exc}") from exc

        # 0秒以上かつオーディオ長未満のビートを抽出
        beat_times_list = [
            float(b) for b in raw_beats if 0 <= b < audio.duration_seconds
        ]
        downbeat_times_list = [
            float(db) for db in raw_downbeats if 0 <= db < audio.duration_seconds
        ]

        if not beat_times_list:
            raise BeatAnalysisError("拍位置を検出できませんでした")

        # BPMの算出: 50 FPS量子化誤差を線形回帰で相殺し、サブフレーム精度のBPMを算出
        bpm = _calculate_bpm_from_beats(beat_times_list)

        # 先行拍の補完（タイムゼロからのグリッド確保）
        beat_times = complete_leading_beats(
            tuple(beat_times_list),
            seconds_per_beat=60.0 / bpm,
        )
        downbeat_times = tuple(downbeat_times_list)

        # オンセット位置の推定（補助情報）
        onset_times: tuple[float, ...] = ()
        try:
            import librosa

            onset_frames = librosa.onset.onset_detect(
                y=audio.samples,
                sr=audio.sample_rate,
                hop_length=512,
                units="frames",
                backtrack=False,
            )
            onset_times = tuple(
                float(value)
                for value in librosa.frames_to_time(
                    onset_frames,
                    sr=audio.sample_rate,
                    hop_length=512,
                )
                if 0 <= value < audio.duration_seconds
            )
        except Exception:
            onset_times = beat_times

        return BeatGrid(
            bpm=bpm,
            beat_times=beat_times,
            onset_times=onset_times,
            beats_per_measure=beats_per_measure,
            downbeat_times=downbeat_times,
        )


def create_beat_analyzer(engine: str = "deep_learning") -> BeatAnalyzer:
    """指定されたエンジンに応じたBeatAnalyzerインスタンスを作成します。"""
    normalized = engine.strip().lower()
    if normalized in {"deep_learning", "beat_this", "default"}:
        try:
            return BeatThisAnalyzer()
        except Exception:
            _logger.warning("beat-thisの初期化に失敗したため、librosaにフォールバックします")
            return LibrosaBeatAnalyzer()
    elif normalized == "librosa":
        return LibrosaBeatAnalyzer()
    raise ValueError(
        f"未知のビート解析エンジン: {engine}。'deep_learning' または 'librosa' を指定してください"
    )


