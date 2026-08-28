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
