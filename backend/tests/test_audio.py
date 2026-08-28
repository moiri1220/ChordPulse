import sys
import types
from pathlib import Path

import pytest

from chordpulse.audio import AudioAcquisitionError, acquire_youtube_audio


def test_youtube_acquisition_fails_closed_when_disabled() -> None:
    with pytest.raises(AudioAcquisitionError, match="無効"):
        with acquire_youtube_audio(
            "https://www.youtube.com/watch?v=example",
            enabled=False,
            lawful_use_confirmation=True,
        ):
            pass


def test_local_source_rejects_unknown_extension(tmp_path: Path) -> None:
    from chordpulse.audio import acquire_local_audio

    source = tmp_path / "source.txt"
    source.write_text("not audio", encoding="utf-8")
    with pytest.raises(AudioAcquisitionError, match="サポートされていないオーディオ形式"):
        with acquire_local_audio(source):
            pass


def test_youtube_acquisition_requires_ffmpeg(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace())
    monkeypatch.setattr("chordpulse.audio.shutil.which", lambda _name: None)

    with pytest.raises(AudioAcquisitionError, match="FFmpeg"):
        with acquire_youtube_audio(
            "https://www.youtube.com/watch?v=example",
            enabled=True,
            lawful_use_confirmation=True,
        ):
            pass


def test_youtube_acquisition_rejects_long_source(monkeypatch) -> None:
    class FakeYoutubeDL:
        def __init__(self, _options):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def extract_info(self, _url, download):
            assert download is False
            return {"duration": 901}

        def download(self, _urls):
            raise AssertionError("download must not start for a long source")

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    monkeypatch.setattr("chordpulse.audio.shutil.which", lambda _name: "ffmpeg")

    with pytest.raises(AudioAcquisitionError, match="時間制限"):
        with acquire_youtube_audio(
            "https://www.youtube.com/watch?v=example",
            enabled=True,
            lawful_use_confirmation=True,
        ):
            pass

