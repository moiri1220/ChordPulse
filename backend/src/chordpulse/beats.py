"""librosaを用いたBPM、拍（ビート）、およびオンセットの解析。"""

from __future__ import annotations

from .models import AudioData, BeatGrid


class BeatAnalysisError(RuntimeError):
    """有効なビートグリッドを推定できない場合に発生する例外。"""


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
        bpm = float(tempo_values[0]) if tempo_values.size else 0.0
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
        )

