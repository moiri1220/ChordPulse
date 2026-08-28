import pytest
from fastapi.testclient import TestClient

from chordpulse.api import AnalysisBusyError, AnalysisGate, create_app
from chordpulse.models import AnalysisResult, ChordEvent


class FakePipeline:
    def __init__(self) -> None:
        self.audio_paths: list = []
        self.rhythm_levels: list[int] = []

    def analyze_to_musicxml(self, audio_path, output_path, *, rhythm_level):
        self.audio_paths.append(audio_path)
        self.rhythm_levels.append(rhythm_level)
        assert audio_path.is_file()
        assert rhythm_level in {1, 2, 3}
        output_path.write_text("<score-partwise version='4.0' />", encoding="utf-8")
        return AnalysisResult(
            bpm=120.0,
            duration_seconds=1.0,
            beat_times=(0.0, 0.5),
            onset_times=(0.0,),
            chords=(ChordEvent(0.0, 1.0, "C"),),
            chord_engine="test",
        )


def test_health_endpoint() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "youtube_enabled" in body
    assert isinstance(body["youtube_enabled"], bool)


def test_analyze_upload_returns_musicxml_and_deletes_audio(tmp_path) -> None:
    pipeline = FakePipeline()
    client = TestClient(create_app(pipeline=pipeline))

    response = client.post(
        "/analyze",
        files={"file": ("source.wav", b"RIFF-test", "audio/wav")},
        data={"rhythm_level": "2", "lawful_use_confirmation": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.recordare.musicxml+xml"
    assert response.headers["content-disposition"] == 'attachment; filename="chordpulse.musicxml"'
    assert response.content.startswith(b"<score-partwise")
    assert response.headers["x-chordpulse-bpm"] == "120.000"
    assert len(pipeline.audio_paths) == 1
    assert not pipeline.audio_paths[0].exists()


def test_analyze_requires_one_source_and_lawful_confirmation() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    missing_source = client.post(
        "/analyze",
        data={"lawful_use_confirmation": "true"},
    )
    missing_confirmation = client.post(
        "/analyze",
        files={"file": ("source.wav", b"RIFF-test", "audio/wav")},
    )

    assert missing_source.status_code == 400
    assert "いずれか一方のみ" in missing_source.json()["detail"]
    assert missing_confirmation.status_code == 403
    assert "lawful_use_confirmation" in missing_confirmation.json()["detail"]


def test_analyze_rejects_file_and_url_together() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/analyze",
        files={"file": ("source.wav", b"RIFF-test", "audio/wav")},
        data={
            "youtube_url": "https://www.youtube.com/watch?v=example",
            "lawful_use_confirmation": "true",
        },
    )

    assert response.status_code == 400
    assert "いずれか一方のみ" in response.json()["detail"]


def test_analyze_rejects_invalid_rhythm_level() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/analyze",
        files={"file": ("source.wav", b"RIFF-test", "audio/wav")},
        data={"rhythm_level": "4", "lawful_use_confirmation": "true"},
    )

    assert response.status_code == 400
    assert "rhythm_level" in response.json()["detail"]


def test_analyze_rejects_invalid_chord_engine() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/analyze",
        files={"file": ("source.wav", b"RIFF-test", "audio/wav")},
        data={"chord_engine": "invalid_engine", "lawful_use_confirmation": "true"},
    )

    assert response.status_code == 400
    assert "chord_engine" in response.json()["detail"]


def test_analyze_accepts_valid_chord_engines() -> None:
    pipeline = FakePipeline()
    client = TestClient(create_app(pipeline=pipeline))

    for engine in ("btc", "viterbi", "harmonic", "template"):
        response = client.post(
            "/analyze",
            files={"file": ("source.wav", b"RIFF-test", "audio/wav")},
            data={"chord_engine": engine, "lawful_use_confirmation": "true"},
        )
        assert response.status_code == 200


def test_analyze_rejects_invalid_file_extension() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/analyze",
        files={"file": ("source.txt", b"not audio", "text/plain")},
        data={"lawful_use_confirmation": "true"},
    )

    assert response.status_code == 422
    assert "サポートされていないオーディオ形式" in response.json()["detail"]


def test_analyze_rejects_oversized_body_before_pipeline() -> None:
    pipeline = FakePipeline()
    client = TestClient(create_app(pipeline=pipeline, max_request_body_bytes=128))

    response = client.post(
        "/analyze",
        files={"file": ("source.wav", b"R" * 256, "audio/wav")},
        data={"lawful_use_confirmation": "true"},
    )

    assert response.status_code == 413
    assert pipeline.audio_paths == []


def test_analyze_returns_generic_internal_error() -> None:
    class FailingPipeline:
        def analyze_to_musicxml(self, *_args, **_kwargs):
            raise RuntimeError("internal implementation detail")

    client = TestClient(create_app(pipeline=FailingPipeline()))

    response = client.post(
        "/analyze",
        files={"file": ("source.wav", b"RIFF-test", "audio/wav")},
        data={"lawful_use_confirmation": "true"},
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "解析に失敗しました"}


def test_analyze_does_not_enable_youtube_from_request(monkeypatch) -> None:
    monkeypatch.delenv("CHORDPULSE_ENABLE_YOUTUBE", raising=False)
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/analyze",
        data={
            "youtube_url": "https://www.youtube.com/watch?v=example",
            "lawful_use_confirmation": "true",
        },
    )

    assert response.status_code == 403
    assert "無効" in response.json()["detail"]


def test_analyze_rejects_non_youtube_url(monkeypatch) -> None:
    monkeypatch.setenv("CHORDPULSE_ENABLE_YOUTUBE", "1")
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.post(
        "/analyze",
        data={
            "youtube_url": "https://example.com/audio",
            "lawful_use_confirmation": "true",
        },
    )

    assert response.status_code == 422
    assert "youtube.com" in response.json()["detail"]


def test_cors_allows_the_local_frontend() -> None:
    client = TestClient(create_app(pipeline=FakePipeline()))

    response = client.options(
        "/analyze",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_analysis_gate_rejects_an_excess_request() -> None:
    gate = AnalysisGate(1)

    with gate.acquire():
        with pytest.raises(AnalysisBusyError):
            with gate.acquire():
                pass


def test_analyze_routes_all_rhythm_levels_to_pipeline() -> None:
    """rhythm_level 1, 2, 3がそれぞれそのままパイプラインへ転送されなければなりません。"""
    pipeline = FakePipeline()
    client = TestClient(create_app(pipeline=pipeline))

    for level in (1, 2, 3):
        response = client.post(
            "/analyze",
            files={"file": ("source.wav", b"RIFF-test", "audio/wav")},
            data={"rhythm_level": str(level), "lawful_use_confirmation": "true"},
        )
        assert response.status_code == 200

    assert pipeline.rhythm_levels == [1, 2, 3]

