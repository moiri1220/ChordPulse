"""コード認識インターフェースと軽量なクロマベースライン。

MusicXML生成やAPIを変更することなく、Chordinoやmadmomがベースラインを置き換えられるよう、
認識エンジン境界は意図的にパイプラインから独立させています。
"""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .models import AudioData, BeatGrid, ChordEvent

_logger = logging.getLogger(__name__)

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


class HarmonicChordRecognizer:
    """低音域重視・ハーモニック分離によるバッキング和音推定。

    処理フロー:
      1. librosa.effects.hpss でハーモニック成分（コード・メロ）とパーカッションを分離。
      2. ハーモニック成分に対して chroma_cqt（対数スケール）を計算。
         fmin=C2付近から始めることでギター・ピアノのバッキング帯域を重視する。
      3. 拍ごとのクロマ平均にテンプレートマッチングを適用してコードを決定。

    パラメータ:
        hop_length: STFT/CQTのホップサイズ（サンプル数）
        minimum_confidence: これ未満のコサイン類似度は "N"（不明）として扱う
        fmin: CQT最低周波数 [Hz]。C2=65.41 Hzがギター最低弦付近
        bins_per_octave: オクターブあたりのCQTビン数（12の倍数を推奨）
        hpss_margin: HPSSの分離マージン。大きいほどハーモニックとパーカッションを強く分離する
    """

    name = "librosa-harmonic-cqt"

    def __init__(
        self,
        *,
        hop_length: int = 512,
        minimum_confidence: float = 0.18,
        fmin: float = 65.41,
        bins_per_octave: int = 36,
        hpss_margin: float = 3.0,
    ) -> None:
        self.hop_length = hop_length
        self.minimum_confidence = minimum_confidence
        self.fmin = fmin
        self.bins_per_octave = bins_per_octave
        self.hpss_margin = hpss_margin

    def recognize(self, audio: AudioData, beat_grid: BeatGrid) -> tuple[ChordEvent, ...]:
        try:
            import librosa
            import numpy as np
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise RuntimeError("コード解析にはlibrosaとnumpyが必要です") from exc

        _logger.debug(
            "[HarmonicChordRecognizer] fmin=%.2f bins_per_octave=%d hpss_margin=%.1f",
            self.fmin,
            self.bins_per_octave,
            self.hpss_margin,
        )

        # Step 1: ハーモニック成分とパーカッション成分を分離
        harmonic, _ = librosa.effects.hpss(audio.samples, margin=self.hpss_margin)

        # Step 2: ハーモニック成分から低音域重視のクロマを計算
        chroma = librosa.feature.chroma_cqt(
            y=harmonic,
            sr=audio.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave,
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
            _logger.debug(
                "[HarmonicChordRecognizer] beat[%d] t=%.2f-%.2fs → %s (conf=%.3f)",
                index,
                start,
                end,
                label,
                confidence,
            )
            raw_events.append(
                ChordEvent(
                    start_seconds=float(start),
                    end_seconds=end,
                    label=label,
                    confidence=confidence,
                )
            )
        return ChromagramChordRecognizer._merge_adjacent(raw_events)

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


class ViterbiChordRecognizer:
    """Viterbiアルゴリズム（HMM）を用いた大域的コード進行推定。

    処理フロー:
      1. librosa.effects.hpss でハーモニック成分を分離（ドラム・打撃音抑制）。
      2. chroma_cqt で低音〜バッキング帯域（fmin=C2）のクロマを抽出。
      3. 拍ごとに平均クロマを計算し、全コード候補との類似度（観測尤度）を算出。
      4. 音楽理論的遷移確率行列（自己ループ、強進行、カノン進行、平行調など）に基づき、
         対数Viterbi動的計画法により系列全体で最も確率の高いコード列をデコード。
    """

    name = "viterbi-hmm"

    def __init__(
        self,
        *,
        hop_length: int = 512,
        fmin: float = 65.41,
        bins_per_octave: int = 36,
        hpss_margin: float = 3.0,
        self_transition_prob: float = 0.70,
        beta: float = 12.0,
        minimum_confidence: float = 0.18,
    ) -> None:
        self.hop_length = hop_length
        self.fmin = fmin
        self.bins_per_octave = bins_per_octave
        self.hpss_margin = hpss_margin
        self.self_transition_prob = self_transition_prob
        self.beta = beta
        self.minimum_confidence = minimum_confidence

        # 全48コード + "N" の定義
        self._chord_candidates: list[tuple[str, int, str]] = []
        for root_idx, root_name in enumerate(PITCH_NAMES):
            for template in TEMPLATES:
                self._chord_candidates.append(
                    (f"{root_name}{template.suffix}", root_idx, template.suffix)
                )

    def _build_log_transition_matrix(self, np: Any) -> Any:
        """49状態（48コード + N）の対数遷移確率行列を構築します。"""
        num_states = len(self._chord_candidates) + 1  # 49
        weights = np.ones((num_states, num_states), dtype=float)

        n_index = num_states - 1
        # "N" 状態との遷移は低めの重み
        weights[n_index, :] = 0.5
        weights[:, n_index] = 0.5

        for i, (_label_i, root_i, suffix_i) in enumerate(self._chord_candidates):
            for j, (_label_j, root_j, suffix_j) in enumerate(self._chord_candidates):
                if i == j:
                    continue
                root_diff = (root_j - root_i) % 12
                is_minor_i = "m" in suffix_i
                is_minor_j = "m" in suffix_j

                if root_i == root_j:
                    # 同一根音のクオリティ違い (例: C -> C7, Cm -> Cm7)
                    weights[i, j] = 4.0
                elif root_diff == 5:
                    # 強進行 (5度下降 / 4度上昇: 例 D -> G, E -> A, Bm -> E)
                    weights[i, j] = 5.0
                elif root_diff == 2:
                    # 全音上昇 (カノン進行 / サブドミナント進行: 例 D -> E)
                    weights[i, j] = 3.5
                elif root_diff == 7:
                    # 5度上昇 / 4度下降 (逆強進行: 例 A -> E)
                    weights[i, j] = 2.5
                elif (not is_minor_i and is_minor_j and root_diff == 9) or (
                    is_minor_i and not is_minor_j and root_diff == 3
                ):
                    # 平行調関係 (例: A -> F#m, C -> Am)
                    weights[i, j] = 4.0
                elif root_diff in {3, 4}:
                    # 近親調（3度関係）
                    weights[i, j] = 2.0
                else:
                    weights[i, j] = 1.0

        p_self = self.self_transition_prob
        trans = np.zeros((num_states, num_states), dtype=float)
        for i in range(num_states):
            row_weights = weights[i].copy()
            row_weights[i] = 0.0
            row_sum = row_weights.sum()
            if row_sum > 0:
                trans[i] = row_weights / row_sum * (1.0 - p_self)
            else:
                trans[i] = (1.0 - p_self) / (num_states - 1)
            trans[i, i] = p_self

        return np.log(np.maximum(trans, 1e-12))

    def _build_template_matrix(self, np: Any) -> Any:
        """48コードのL2正規化テンプレート行列（48 × 12）を作成します。"""
        matrix = np.zeros((len(self._chord_candidates), 12), dtype=float)
        for i, (_label, root_idx, suffix) in enumerate(self._chord_candidates):
            template = next(t for t in TEMPLATES if t.suffix == suffix)
            values = np.zeros(12, dtype=float)
            values[(root_idx + np.asarray(template.intervals)) % 12] = 1.0
            matrix[i] = values / np.linalg.norm(values)
        return matrix

    def recognize(self, audio: AudioData, beat_grid: BeatGrid) -> tuple[ChordEvent, ...]:
        try:
            import librosa
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("コード解析にはlibrosaとnumpyが必要です") from exc

        _logger.debug(
            "[ViterbiChordRecognizer] fmin=%.2f bins_per_octave=%d self_prob=%.2f beta=%.1f",
            self.fmin,
            self.bins_per_octave,
            self.self_transition_prob,
            self.beta,
        )

        beat_times = beat_grid.beat_times
        beat_seconds = beat_grid.seconds_per_beat
        valid_beats: list[tuple[float, float, int, int]] = []

        harmonic, _ = librosa.effects.hpss(audio.samples, margin=self.hpss_margin)
        chroma = librosa.feature.chroma_cqt(
            y=harmonic,
            sr=audio.sample_rate,
            hop_length=self.hop_length,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave,
        )

        for index, start in enumerate(beat_times):
            if start >= audio.duration_seconds:
                continue
            end = beat_times[index + 1] if index + 1 < len(beat_times) else start + beat_seconds
            end = min(float(end), audio.duration_seconds)
            if end <= start:
                continue
            frame_start = int(round(start * audio.sample_rate / self.hop_length))
            frame_start = min(frame_start, max(0, chroma.shape[1] - 1))
            frame_end = int(round(end * audio.sample_rate / self.hop_length))
            frame_end = max(frame_start + 1, min(frame_end, chroma.shape[1]))
            valid_beats.append((float(start), end, frame_start, frame_end))

        if not valid_beats:
            return ()

        num_beats = len(valid_beats)
        template_matrix = self._build_template_matrix(np)  # (48, 12)
        num_states = len(self._chord_candidates) + 1  # 49 (48 chords + "N")
        n_state_idx = num_states - 1

        # 各拍の観測対数尤度を算出
        emission = np.zeros((num_beats, num_states), dtype=float)
        confidences = np.zeros(num_beats, dtype=float)

        for t, (_start, _end, f_start, f_end) in enumerate(valid_beats):
            chroma_mean = np.mean(chroma[:, f_start:f_end], axis=1)
            norm = float(np.linalg.norm(chroma_mean))
            if norm <= 1e-8:
                emission[t, :] = -10.0
                emission[t, n_state_idx] = 0.0
                confidences[t] = 0.0
            else:
                norm_chroma = chroma_mean / norm
                sims = np.dot(template_matrix, norm_chroma)  # (48,)
                max_sim = float(np.max(sims))
                confidences[t] = max_sim
                emission[t, :n_state_idx] = self.beta * sims
                emission[t, n_state_idx] = self.beta * self.minimum_confidence

        log_trans = self._build_log_transition_matrix(np)  # (49, 49)
        log_init = np.full(num_states, -np.log(num_states))

        # 対数Viterbi動的計画法
        viterbi_dp = np.zeros((num_beats, num_states), dtype=float)
        backpointer = np.zeros((num_beats, num_states), dtype=int)

        viterbi_dp[0] = log_init + emission[0]

        for t in range(1, num_beats):
            # trans_scores[i, j] = viterbi_dp[t-1, i] + log_trans[i, j]
            trans_scores = viterbi_dp[t - 1, :, None] + log_trans
            backpointer[t] = np.argmax(trans_scores, axis=0)
            viterbi_dp[t] = np.max(trans_scores, axis=0) + emission[t]

        # バックトレース
        best_path = np.zeros(num_beats, dtype=int)
        best_path[-1] = int(np.argmax(viterbi_dp[-1]))
        for t in range(num_beats - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]

        # ChordEvent に変換
        raw_events: list[ChordEvent] = []
        for t, (start, end, _fs, _fe) in enumerate(valid_beats):
            state_idx = best_path[t]
            if state_idx == n_state_idx:
                label = "N"
            else:
                label = self._chord_candidates[state_idx][0]
            conf = float(confidences[t])
            raw_events.append(
                ChordEvent(
                    start_seconds=start,
                    end_seconds=end,
                    label=label,
                    confidence=conf,
                )
            )

        return ChromagramChordRecognizer._merge_adjacent(raw_events)


BTC_WEIGHT_URL = (
    "https://huggingface.co/spaces/amaai-lab/music2emo/resolve/main/inference/data/btc_model_large_voca.pt"
)

ROOT_LIST = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
QUALITY_LIST = [
    "min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7", "7",
    "dim7", "hdim7", "sus2", "sus4"
]
QUALITY_MAP = {
    "min": "m",
    "maj": "",
    "dim": "dim",
    "aug": "aug",
    "min6": "m6",
    "maj6": "6",
    "min7": "m7",
    "minmaj7": "mM7",
    "maj7": "maj7",
    "7": "7",
    "dim7": "dim7",
    "hdim7": "m7b5",
    "sus2": "sus2",
    "sus4": "sus4",
}


def _get_btc_chord_vocab() -> dict[int, str]:
    vocab: dict[int, str] = {168: "N", 169: "N"}
    for i in range(168):
        root = ROOT_LIST[i // 14]
        quality = QUALITY_LIST[i % 14]
        q_str = QUALITY_MAP.get(quality, quality)
        vocab[i] = f"{root}{q_str}"
    return vocab


class BtcChordRecognizer:
    """BTC (Bi-directional Transformer for Musical Chord Recognition) による深層学習コード認識。

    特徴:
      - 144次元 CQT スペクトログラムを入力とし、双方向 Self-Attention により和音を直接推定
      - 170種類のコード語彙（メジャー、マイナー、7th、maj7、sus4、dim、aug等）に対応
      - 事前学習済み重み（約12MB）を初回実行時に自動ダウンロード
      - 推論結果を BeatGrid（各拍）に同期・マッピングして ChordEvent 列を出力
    """

    name = "btc-transformer"

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        device: str | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.device_str = device
        self.confidence_threshold = confidence_threshold
        self._vocab = _get_btc_chord_vocab()
        self._model = None
        self._mean = -2.2279878897355596
        self._std = 1.7191329394436938

    def _ensure_weights(self) -> Path:
        if self.model_path is not None and self.model_path.is_file():
            return self.model_path

        cache_dir = Path.home() / ".cache" / "chordpulse"
        cache_dir.mkdir(parents=True, exist_ok=True)
        weight_file = cache_dir / "btc_model_large_voca.pt"

        if not weight_file.is_file() or weight_file.stat().st_size < 12_000_000:
            _logger.info("BTC事前学習済みモデルをダウンロード中 (%s)...", BTC_WEIGHT_URL)
            urllib.request.urlretrieve(BTC_WEIGHT_URL, weight_file)
            _logger.info("BTC事前学習済みモデルのダウンロードが完了しました")

        return weight_file

    def _get_or_load_model(self, torch):
        if self._model is not None:
            return self._model

        from .btc_model import BTC_model

        weight_path = self._ensure_weights()
        device = self.device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

        model = BTC_model()
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        self._mean = float(checkpoint.get("mean", self._mean))
        self._std = float(checkpoint.get("std", self._std))
        self._model = model
        return self._model

    def recognize(self, audio: AudioData, beat_grid: BeatGrid) -> tuple[ChordEvent, ...]:
        try:
            import librosa
            import numpy as np
            import torch
        except ImportError as exc:
            raise RuntimeError("BTCコード解析にはtorch, librosa, numpyが必要です") from exc

        model = self._get_or_load_model(torch)
        device = next(model.parameters()).device

        song_hz = 22050
        inst_len = 10.0
        hop_length = 2048
        n_bins = 144
        bins_per_octave = 24
        timestep = 108

        # サンプリングレートの確認とリサンプリング
        samples = audio.samples
        if audio.sample_rate != song_hz:
            samples = librosa.resample(samples, orig_sr=audio.sample_rate, target_sr=song_hz)

        # 10秒チャンクごとに CQT を計算
        step_samples = int(song_hz * inst_len)
        features = []
        curr_idx = 0
        while curr_idx + step_samples <= len(samples):
            chunk = samples[curr_idx : curr_idx + step_samples]
            cqt = librosa.cqt(
                chunk, sr=song_hz, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length
            )
            features.append(cqt)
            curr_idx += step_samples

        if curr_idx < len(samples):
            chunk = samples[curr_idx:]
            cqt = librosa.cqt(
                chunk, sr=song_hz, n_bins=n_bins, bins_per_octave=bins_per_octave, hop_length=hop_length
            )
            features.append(cqt)

        if not features:
            return ()

        feature = np.concatenate(features, axis=1)
        feature = np.log(np.abs(feature) + 1e-6)
        time_unit = inst_len / timestep  # 約 0.0926 秒 / フレーム

        # 正規化とパディング
        feature = feature.T
        feature = (feature - self._mean) / self._std

        num_pad = timestep - (feature.shape[0] % timestep)
        if num_pad < timestep:
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instances = feature.shape[0] // timestep

        # 推論
        all_preds = []
        with torch.no_grad():
            x_tensor = torch.tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)
            for t in range(num_instances):
                inst_x = x_tensor[:, t * timestep : (t + 1) * timestep, :]
                p, _ = model(inst_x)
                p = p.squeeze(0).cpu().numpy()
                all_preds.append(p)

        concat_preds = np.concatenate(all_preds)
        valid_frames = len(concat_preds) - num_pad
        frame_preds = concat_preds[:valid_frames]

        # 拍グリッドへのマッピング
        beat_times = beat_grid.beat_times
        beat_seconds = beat_grid.seconds_per_beat
        raw_events: list[ChordEvent] = []

        for index, start in enumerate(beat_times):
            if start >= audio.duration_seconds:
                continue
            end = beat_times[index + 1] if index + 1 < len(beat_times) else start + beat_seconds
            end = min(float(end), audio.duration_seconds)
            if end <= start:
                continue

            f_start = int(round(start / time_unit))
            f_end = int(round(end / time_unit))
            f_start = max(0, min(f_start, len(frame_preds) - 1))
            f_end = max(f_start + 1, min(f_end, len(frame_preds)))

            slice_preds = frame_preds[f_start:f_end]
            if len(slice_preds) == 0:
                label = "N"
            else:
                # 拍区間内での最頻コードを採用
                counts = np.bincount(slice_preds)
                best_idx = int(np.argmax(counts))
                label = self._vocab.get(best_idx, "N")

            raw_events.append(
                ChordEvent(
                    start_seconds=float(start),
                    end_seconds=end,
                    label=label,
                    confidence=0.95 if label != "N" else 0.0,
                )
            )

        return ChromagramChordRecognizer._merge_adjacent(raw_events)


def create_chord_recognizer(engine: str = "btc") -> ChordRecognizer:
    """要求されたエンジンを暗黙的に変更することなく、名前付きコードエンジンを作成します。

    エンジン一覧:
        "btc" (デフォルト): BTC (Bi-directional Transformer) によるAI深層学習コード認識（最高精度）。
        "viterbi": HMM + Viterbiアルゴリズムによる大域的平滑化・コード認識。
        "harmonic": HPSS + chroma_cqt による低音域重視・バッキング和音推定。
        "template": 全帯域 chroma_stft によるシンプルなテンプレートマッチング（フェーズ1ベースライン）。
        "chordino", "madmom": 将来の拡張ポイント（現時点では未実装）。
    """

    normalized = engine.strip().lower()
    if normalized in {"btc", "btc-transformer", "transformer"}:
        return BtcChordRecognizer()
    if normalized in {"viterbi", "viterbi-hmm", "librosa-viterbi-hmm"}:
        return ViterbiChordRecognizer()
    if normalized in {"harmonic", "librosa-harmonic-cqt"}:
        return HarmonicChordRecognizer()
    if normalized in {"template", "librosa-chroma-template"}:
        return ChromagramChordRecognizer()
    if normalized in {"chordino", "madmom"}:
        raise ChordEngineUnavailable(
            f"{normalized} コードエンジンはインストールされていません。テンプレートベースラインを使用するか、"
            "インジェクトされたChordRecognizerを提供してください"
        )
    raise ValueError(f"未知のコードエンジン: {engine}")


