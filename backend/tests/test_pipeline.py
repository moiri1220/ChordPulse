import numpy as np
import soundfile as sf
from music21 import converter

from chordpulse.pipeline import AnalysisPipeline


def test_pipeline_analyzes_a_synthetic_click_and_writes_musicxml(tmp_path) -> None:
    sample_rate = 22_050
    duration_seconds = 8.0
    time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    samples = 0.03 * (
        np.sin(2 * np.pi * 261.63 * time)
        + np.sin(2 * np.pi * 329.63 * time)
        + np.sin(2 * np.pi * 392.00 * time)
    )
    for onset in np.arange(0.0, duration_seconds, 0.5):
        start = int(onset * sample_rate)
        length = min(int(0.04 * sample_rate), samples.size - start)
        if length > 0:
            samples[start : start + length] += 0.5 * np.hanning(length)

    source = tmp_path / "synthetic.wav"
    output = tmp_path / "synthetic.musicxml"
    sf.write(source, samples, sample_rate)

    result = AnalysisPipeline().analyze_to_musicxml(source, output, rhythm_level=2)

    assert result.beat_times
    assert result.beat_times[0] == 0.0
    assert result.chords
    parsed = converter.parse(output)
    assert len(parsed.parts) == 1
