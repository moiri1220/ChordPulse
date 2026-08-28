import pytest

from chordpulse.beats import complete_leading_beats


def test_complete_leading_beats_reconstructs_the_chart_origin() -> None:
    completed = complete_leading_beats(
        (1.02, 1.53, 2.04),
        seconds_per_beat=0.51,
    )

    assert completed[0] == 0.0
    assert completed[1] == pytest.approx(0.51)
    assert completed[2:] == pytest.approx((1.02, 1.53, 2.04))


def test_complete_leading_beats_does_not_duplicate_an_origin_beat() -> None:
    completed = complete_leading_beats((0.0, 0.5, 1.0), seconds_per_beat=0.5)

    assert completed == (0.0, 0.5, 1.0)


def test_complete_leading_beats_adds_origin_for_an_early_first_beat() -> None:
    completed = complete_leading_beats((0.1, 0.6, 1.1), seconds_per_beat=0.5)

    assert completed[0] == 0.0
    assert completed[1:] == (0.1, 0.6, 1.1)


def test_complete_leading_beats_returns_unchanged_for_empty_input() -> None:
    """Empty beat_times must pass through without error."""
    assert complete_leading_beats((), seconds_per_beat=0.5) == ()


def test_complete_leading_beats_returns_unchanged_for_zero_bpm() -> None:
    """Non-positive seconds_per_beat must pass through without division by zero."""
    beats = (1.0, 2.0)
    assert complete_leading_beats(beats, seconds_per_beat=0.0) == beats
    assert complete_leading_beats(beats, seconds_per_beat=-0.5) == beats


def test_complete_leading_beats_first_beat_at_zero_needs_no_prepend() -> None:
    """When the first beat is already at 0.0, no leading beats should be added."""
    beats = (0.0, 0.5, 1.0)
    completed = complete_leading_beats(beats, seconds_per_beat=0.5)
    assert completed == beats


def test_complete_leading_beats_prepends_multiple_leading_beats() -> None:
    """A first beat far from the origin should produce multiple leading beats."""
    # BPM 120 → 0.5 s/beat; first detected at 2.0 → three leading beats at 0, 0.5, 1.0, 1.5
    completed = complete_leading_beats((2.0, 2.5, 3.0), seconds_per_beat=0.5)
    assert completed[0] == 0.0
    assert completed[-3:] == (2.0, 2.5, 3.0)
    # There must be at least 3 synthetic leading beats (0.0, 0.5, 1.0, 1.5)
    assert len(completed) >= 3 + 3


def test_complete_leading_beats_tolerance_prevents_duplicate_near_first_beat() -> None:
    """A beat that would fall within the tolerance window of the first detected
    beat must not be appended (it would duplicate the detected beat)."""
    # seconds_per_beat=1.0, tolerance=0.25; first_detected=0.9
    # anchor=0.0, cursor starts at 0.0 → 0.0 < 0.9 - 0.25 = 0.65 → appended
    # next cursor=1.0 → 1.0 >= 0.65 → loop stops
    completed = complete_leading_beats((0.9, 1.9, 2.9), seconds_per_beat=1.0)
    assert completed[0] == 0.0
    assert completed[1] == 0.9  # no duplicate 1.0 between 0.0 and 0.9


def test_create_beat_analyzer_factory() -> None:
    from chordpulse.beats import BeatThisAnalyzer, LibrosaBeatAnalyzer, create_beat_analyzer

    default_analyzer = create_beat_analyzer()
    assert isinstance(default_analyzer, BeatThisAnalyzer)

    librosa_analyzer = create_beat_analyzer("librosa")
    assert isinstance(librosa_analyzer, LibrosaBeatAnalyzer)

    with pytest.raises(ValueError, match="未知のビート解析エンジン"):
        create_beat_analyzer("unknown_engine")


def test_beat_this_analyzer_estimates_160_bpm_accurately() -> None:
    import numpy as np
    from chordpulse.beats import BeatThisAnalyzer
    from chordpulse.models import AudioData

    # 160 BPM の合成オーディオを生成 (0.375秒間隔のビート)
    sr = 22050
    duration = 10.0
    bpm_target = 160.0
    beat_period = 60.0 / bpm_target

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    samples = np.zeros_like(t, dtype=np.float32)

    for i, beat_t in enumerate(np.arange(0.2, duration, beat_period)):
        idx = int(beat_t * sr)
        amp = 1.0 if (i % 4 == 0) else 0.5
        decay = np.exp(-np.linspace(0, 10, int(0.05 * sr), dtype=np.float32))
        end_idx = min(idx + len(decay), len(samples))
        samples[idx:end_idx] += amp * decay[: end_idx - idx]

    audio = AudioData(samples=samples, sample_rate=sr, duration_seconds=duration)
    analyzer = BeatThisAnalyzer()
    grid = analyzer.analyze(audio)

    # 160 BPMに対して107などの誤判定にならず、150〜170の範囲で検出されることを検証
    assert 150.0 <= grid.bpm <= 170.0
    assert len(grid.beat_times) >= 20
    assert grid.beat_times[0] == 0.0
    assert len(grid.downbeat_times) > 0


def test_calculate_bpm_from_beats_resolves_50fps_quantization() -> None:
    import numpy as np
    from chordpulse.beats import _calculate_bpm_from_beats

    # 160 BPM の真の拍位置 (0.375秒刻み)
    true_beats = np.arange(0.2, 30.0, 0.375)
    # 50 FPS（0.02秒刻み）で量子化されたタイムスタンプ
    quantized_beats = np.round(true_beats * 50) / 50

    # 従来の中央値計算では 60 / 0.38 = 157.89 となる
    median_bpm = 60.0 / float(np.median(np.diff(quantized_beats)))
    assert median_bpm == pytest.approx(157.89, abs=0.01)

    # 線形回帰による算出では 160.0 に補正されることを検証
    calculated_bpm = _calculate_bpm_from_beats(quantized_beats)
    assert calculated_bpm == 160.0


def test_calculate_bpm_from_beats_rounds_decimals_to_integer() -> None:
    import numpy as np
    from chordpulse.beats import _calculate_bpm_from_beats

    # 120.4 BPM (拍間隔 60/120.4 ≈ 0.498338s) -> 四捨五入で 120.0
    beats_120_4 = np.arange(0.0, 20.0, 60.0 / 120.4)
    assert _calculate_bpm_from_beats(beats_120_4) == 120.0

    # 120.6 BPM (拍間隔 60/120.6 ≈ 0.497512s) -> 四捨五入で 121.0
    beats_120_6 = np.arange(0.0, 20.0, 60.0 / 120.6)
    assert _calculate_bpm_from_beats(beats_120_6) == 121.0



