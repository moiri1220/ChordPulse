import sys

import numpy as np
import pytest

from chordpulse.chords import (
    BtcChordRecognizer,
    ChordEngineUnavailable,
    ChromagramChordRecognizer,
    HarmonicChordRecognizer,
    ViterbiChordRecognizer,
    _get_btc_chord_vocab,
    create_chord_recognizer,
)
from chordpulse.models import AudioData, BeatGrid


def test_template_recognizer_identifies_a_c_major_template() -> None:
    recognizer = ChromagramChordRecognizer(minimum_confidence=0.1)
    label, confidence = recognizer._best_label(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np,
    )

    assert label == "C"
    assert confidence == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("vector", "expected_label"),
    [
        ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], "C"),
        ([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], "Cm"),
        ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], "C7"),
        ([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], "Cm7"),
    ],
)
def test_template_recognizer_covers_supported_chord_types(vector, expected_label) -> None:
    recognizer = ChromagramChordRecognizer(minimum_confidence=0.1)
    label, _confidence = recognizer._best_label(np.array(vector, dtype=float), np)

    assert label == expected_label


def test_template_recognizer_merges_adjacent_equal_events(monkeypatch) -> None:
    recognizer = ChromagramChordRecognizer(minimum_confidence=0.1)
    audio = AudioData(samples=np.zeros(4_000), sample_rate=1_000, duration_seconds=4.0)
    beat_grid = BeatGrid(bpm=120.0, beat_times=(0.0, 1.0, 2.0), onset_times=())

    class StubLibrosa:
        class feature:
            @staticmethod
            def chroma_stft(**_kwargs):
                vector = np.array(
                    [
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ]
                )
                return np.tile(vector, (1, 8))

    monkeypatch.setitem(sys.modules, "librosa", StubLibrosa())
    events = recognizer.recognize(audio, beat_grid)

    assert len(events) == 1
    assert events[0].label == "C"
    assert events[0].start_seconds == 0.0
    assert events[0].end_seconds == 2.5


def test_unavailable_named_engine_fails_explicitly() -> None:
    assert create_chord_recognizer("template").name == "librosa-chroma-template"
    assert create_chord_recognizer("harmonic").name == "librosa-harmonic-cqt"
    assert create_chord_recognizer("viterbi").name == "viterbi-hmm"
    assert create_chord_recognizer("btc").name == "btc-transformer"

    with pytest.raises(ChordEngineUnavailable, match="chordino"):
        create_chord_recognizer("chordino")


# ---------- HarmonicChordRecognizer ----------


def test_harmonic_recognizer_identifies_c_major() -> None:
    recognizer = HarmonicChordRecognizer(minimum_confidence=0.1)
    label, confidence = recognizer._best_label(
        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        np,
    )

    assert label == "C"
    assert confidence == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("vector", "expected_label"),
    [
        ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], "C"),
        ([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], "Cm"),
        ([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], "C7"),
        ([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], "Cm7"),
    ],
)
def test_harmonic_recognizer_covers_chord_types(vector, expected_label) -> None:
    recognizer = HarmonicChordRecognizer(minimum_confidence=0.1)
    label, _confidence = recognizer._best_label(np.array(vector, dtype=float), np)

    assert label == expected_label


def test_harmonic_recognizer_uses_hpss_and_chroma_cqt(monkeypatch) -> None:
    """HPSSとchroma_cqtが呼び出され、結果がCメジャーとしてマージされることを確認する。"""
    recognizer = HarmonicChordRecognizer(minimum_confidence=0.1)
    audio = AudioData(samples=np.zeros(4_000), sample_rate=1_000, duration_seconds=4.0)
    beat_grid = BeatGrid(bpm=120.0, beat_times=(0.0, 1.0, 2.0), onset_times=())

    hpss_called = {"count": 0}
    cqt_called = {"count": 0}

    class StubLibrosa:
        class effects:
            @staticmethod
            def hpss(y, **_kwargs):
                hpss_called["count"] += 1
                return y, np.zeros_like(y)

        class feature:
            @staticmethod
            def chroma_cqt(**_kwargs):
                cqt_called["count"] += 1
                # Cメジャー: 12音 × 8フレーム
                vector = np.array(
                    [
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ]
                )
                return np.tile(vector, (1, 8))

    monkeypatch.setitem(sys.modules, "librosa", StubLibrosa())
    events = recognizer.recognize(audio, beat_grid)

    assert hpss_called["count"] == 1, "HPSSが呼び出されていない"
    assert cqt_called["count"] == 1, "chroma_cqtが呼び出されていない"
    assert len(events) == 1
    assert events[0].label == "C"
    assert events[0].start_seconds == 0.0
    assert events[0].end_seconds == 2.5


def test_harmonic_recognizer_engine_name() -> None:
    recognizer = HarmonicChordRecognizer()
    assert recognizer.name == "librosa-harmonic-cqt"


# ---------- ViterbiChordRecognizer ----------


def test_viterbi_recognizer_engine_name() -> None:
    recognizer = ViterbiChordRecognizer()
    assert recognizer.name == "viterbi-hmm"


def test_viterbi_recognizer_identifies_c_major(monkeypatch) -> None:
    """Cメジャーのクロマが入力された時に Viterbi で C と認識されることを確認する。"""
    recognizer = ViterbiChordRecognizer()
    audio = AudioData(samples=np.zeros(4_000), sample_rate=1_000, duration_seconds=4.0)
    beat_grid = BeatGrid(bpm=120.0, beat_times=(0.0, 1.0, 2.0), onset_times=())

    class StubLibrosa:
        class effects:
            @staticmethod
            def hpss(y, **_kwargs):
                return y, np.zeros_like(y)

        class feature:
            @staticmethod
            def chroma_cqt(**_kwargs):
                # Cメジャー: C, E, G
                vector = np.array(
                    [
                        [1.0],  # C
                        [0.0],
                        [0.0],
                        [0.0],
                        [1.0],  # E
                        [0.0],
                        [0.0],
                        [1.0],  # G
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ]
                )
                return np.tile(vector, (1, 8))

    monkeypatch.setitem(sys.modules, "librosa", StubLibrosa())
    events = recognizer.recognize(audio, beat_grid)

    assert len(events) == 1
    assert events[0].label == "C"
    assert events[0].start_seconds == 0.0
    assert events[0].end_seconds == 2.5


def test_viterbi_recognizer_smooths_transient_noise(monkeypatch) -> None:
    """自己遷移確率により、1拍だけ混ざった弱いノイズが平滑化されることを確認する。"""
    recognizer = ViterbiChordRecognizer(self_transition_prob=0.80, beta=5.0)
    audio = AudioData(samples=np.zeros(8_000), sample_rate=1_000, duration_seconds=4.0)
    beat_grid = BeatGrid(bpm=120.0, beat_times=(0.0, 0.5, 1.0, 1.5), onset_times=())

    # 4拍中、0, 2, 3拍目は明確なCメジャー、1拍目のみわずかに別のノイズが乗る
    c_vector = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float)[:, None]
    noisy_vector = np.array([0.5, 0.6, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0, 0], dtype=float)[:, None]
    chroma_mock = np.concatenate([c_vector, noisy_vector, c_vector, c_vector], axis=1)

    class StubLibrosa:
        class effects:
            @staticmethod
            def hpss(y, **_kwargs):
                return y, np.zeros_like(y)

        class feature:
            @staticmethod
            def chroma_cqt(**_kwargs):
                # 4拍分に複製 (各拍2フレーム)
                return np.repeat(chroma_mock, 2, axis=1)

    monkeypatch.setitem(sys.modules, "librosa", StubLibrosa())
    events = recognizer.recognize(audio, beat_grid)

    # 1拍ごとのチラつきが平滑化され、単一の C にマージされることを確認
    assert len(events) == 1
    assert events[0].label == "C"


# ---------- BtcChordRecognizer ----------


def test_btc_chord_vocab() -> None:
    vocab = _get_btc_chord_vocab()
    assert len(vocab) == 170
    assert vocab[0] == "Cm"
    assert vocab[1] == "C"
    assert vocab[6] == "Cm7"
    assert vocab[8] == "Cmaj7"
    assert vocab[9] == "C7"
    assert vocab[13] == "Csus4"
    assert vocab[169] == "N"


def test_btc_recognizer_engine_name() -> None:
    recognizer = BtcChordRecognizer()
    assert recognizer.name == "btc-transformer"


def test_btc_recognizer_mock_inference(monkeypatch) -> None:
    """モックモデルを用いて BtcChordRecognizer が拍同期コードイベントを返すことを確認する。"""
    recognizer = BtcChordRecognizer()
    audio = AudioData(samples=np.zeros(22050 * 2), sample_rate=22050, duration_seconds=2.0)
    beat_grid = BeatGrid(bpm=120.0, beat_times=(0.0, 0.5, 1.0, 1.5), onset_times=())

    class MockModel:
        def __call__(self, x):
            # batch=1, time=108 -> 全てインデックス1 ('C') を予測
            import torch
            preds = torch.ones(x.shape[0], x.shape[1], dtype=torch.long)
            return preds, preds

        def parameters(self):
            import torch
            yield torch.empty(1)

    monkeypatch.setattr(recognizer, "_get_or_load_model", lambda torch: MockModel())
    events = recognizer.recognize(audio, beat_grid)

    assert len(events) == 1
    assert events[0].label == "C"
    assert events[0].start_seconds == 0.0
    assert events[0].end_seconds == 2.0
