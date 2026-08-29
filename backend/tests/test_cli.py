import json

from chordpulse.cli import build_parser, run
from chordpulse.models import AnalysisResult, ChordEvent


def test_cli_requires_lawful_use_confirmation(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--input",
            str(tmp_path / "source.wav"),
            "--output",
            str(tmp_path / "result.musicxml"),
        ]
    )

    assert run(args) == 2


def test_cli_requires_one_source() -> None:
    parser = build_parser()
    try:
        parser.parse_args(["--output", "result.musicxml"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("CLI should require a source")


def test_cli_rejects_overwriting_input(tmp_path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"placeholder")
    args = build_parser().parse_args(
        [
            "--input",
            str(source),
            "--output",
            str(source),
            "--lawful-use-confirmation",
        ]
    )

    assert run(args) == 2


def test_cli_success_removes_request_audio_and_writes_metadata(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"placeholder")
    output = tmp_path / "result.musicxml"
    metadata = tmp_path / "result.json"
    observed = {}

    class FakePipeline:
        def __init__(self, chord_engine: str = "viterbi", **kwargs) -> None:
            observed["chord_engine"] = chord_engine

        def analyze_to_musicxml(self, audio_path, output_path, *, rhythm_level, beat_subdivision=0.25):
            observed["audio_path"] = audio_path
            observed["beat_subdivision"] = beat_subdivision
            assert audio_path.is_file()
            assert rhythm_level == 2
            assert beat_subdivision in {0.25, 0.5}
            output_path.write_text("<score-partwise version='4.0' />", encoding="utf-8")
            return AnalysisResult(
                bpm=120.0,
                duration_seconds=1.0,
                beat_times=(0.0, 0.5),
                onset_times=(0.0,),
                chords=(ChordEvent(0.0, 1.0, "C"),),
                chord_engine="test",
            )

    monkeypatch.setattr("chordpulse.cli.AnalysisPipeline", FakePipeline)
    args = build_parser().parse_args(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--analysis-json",
            str(metadata),
            "--lawful-use-confirmation",
        ]
    )

    assert run(args) == 0
    assert output.is_file()
    assert observed["beat_subdivision"] == 0.25
    assert json.loads(metadata.read_text(encoding="utf-8"))["chord_engine"] == "test"
    assert not observed["audio_path"].exists()


def test_cli_youtube_requires_both_environment_and_flag(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CHORDPULSE_ENABLE_YOUTUBE", raising=False)
    args = build_parser().parse_args(
        [
            "--youtube-url",
            "https://www.youtube.com/watch?v=example",
            "--output",
            str(tmp_path / "result.musicxml"),
            "--enable-youtube",
            "--lawful-use-confirmation",
        ]
    )

    assert run(args) == 1
