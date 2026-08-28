"""ワンショットのChordPulse解析リクエストを処理するFastAPIの境界層。"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import queue
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from .audio import (
    DEFAULT_MAX_FILE_BYTES,
    SUPPORTED_SUFFIXES,
    AudioAcquisitionError,
    YouTubeAcquisitionDisabledError,
    acquire_youtube_audio,
    youtube_enabled_from_environment,
)
from .pipeline import AnalysisPipeline

_logger = logging.getLogger(__name__)

UPLOAD_CHUNK_BYTES = 1024 * 1024
MUSICXML_MEDIA_TYPE = "application/vnd.recordare.musicxml+xml"
MAX_REQUEST_BODY_BYTES = DEFAULT_MAX_FILE_BYTES + 2 * UPLOAD_CHUNK_BYTES
DEFAULT_MAX_CONCURRENT_ANALYSES = 2
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 300
DEFAULT_CORS_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


class UploadTooLargeError(RuntimeError):
    """アップロードされたファイルが設定されたサイズ制限を超えた場合に発生する例外。"""


class RequestBodyTooLargeError(RuntimeError):
    """受信したHTTPボディが設定された制限サイズを超えた場合に発生する例外。"""


class AnalysisBusyError(RuntimeError):
    """解析の実行枠（スロット）がすべて塞がっている場合に発生する例外。"""


class AnalysisTimeoutError(RuntimeError):
    """実行制限時間を超えた解析ワーカープロセスを強制終了した後に発生する例外。"""


class AnalysisWorkerError(RuntimeError):
    """隔離された解析ワーカープロセスが、利用可能な結果を出力せずに終了した場合に発生する例外。"""


class RequestBodySizeLimitMiddleware:
    """FastAPIがマルチパートアップロードをパースする前に、サイズ制限を超えたリクエストボディを拒否するミドルウェア。"""

    def __init__(self, app, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._send_too_large(scope, receive, send)
                    return
            except ValueError:
                pass

        received_bytes = 0
        response_started = False

        async def limited_receive():
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_body_bytes:
                    raise RequestBodyTooLargeError
            return message

        async def tracked_send(message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLargeError:
            if not response_started:
                await self._send_too_large(scope, receive, send)

    async def _send_too_large(self, scope, receive, send) -> None:
        response = JSONResponse(
            {"detail": "リクエストボディが設定されたサイズ制限を超えています"},
            status_code=413,
        )
        await response(scope, receive, send)


class AnalysisGate:
    """単一のAPIプロセスで同時に処理するCPU負荷の高いリクエスト数を制限するセマフォ。"""

    def __init__(self, max_concurrent_analyses: int) -> None:
        if max_concurrent_analyses < 1:
            raise ValueError("max_concurrent_analysesは1以上でなければなりません")
        self._semaphore = threading.BoundedSemaphore(max_concurrent_analyses)

    @contextlib.contextmanager
    def acquire(self) -> Iterator[None]:
        if not self._semaphore.acquire(blocking=False):
            raise AnalysisBusyError("解析の実行容量が現在上限に達しています")
        try:
            yield
        finally:
            self._semaphore.release()


@contextlib.contextmanager
def _store_upload(
    upload: UploadFile,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> Iterator[Path]:
    """アップロードされたファイルを現在のリクエスト処理中のみ一時的に保存します。

    書き込み時にチャンク単位でサイズを検証するため、Content-Lengthヘッダーがないストリームの場合でも、
    ボディ全体がバッファリングされる前に拒否されます。
    audio._validate_audio_fileも書き込み後に同じ検証を行い、CLI実行パスをカバーします。
    両方の制限値は DEFAULT_MAX_FILE_BYTES を共有しているため、同期が維持されます。
    """

    filename = Path(upload.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise AudioAcquisitionError(f"サポートされていないオーディオ形式です。次のいずれかを指定してください: {allowed}")

    with tempfile.TemporaryDirectory(prefix="chordpulse-api-") as temp_dir:
        destination = Path(temp_dir) / f"source{suffix}"
        total_bytes = 0
        with destination.open("wb") as temporary_file:
            while chunk := upload.file.read(UPLOAD_CHUNK_BYTES):
                total_bytes += len(chunk)
                if total_bytes > max_file_bytes:
                    raise UploadTooLargeError("オーディオファイルが設定されたサイズ制限を超えています")
                temporary_file.write(chunk)
        if total_bytes == 0:
            raise AudioAcquisitionError("オーディオファイルにデータが含まれていません")
        try:
            yield destination
        finally:
            destination.unlink(missing_ok=True)


def _result_metadata(result) -> dict[str, str]:
    return {
        "bpm": f"{result.bpm:.3f}",
        "chord_engine": result.chord_engine,
    }


ALLOWED_CHORD_ENGINES = {"btc", "viterbi", "harmonic", "template"}


def _analysis_worker(
    audio_path: str,
    output_path: str,
    rhythm_level: int | None,
    chord_engine: str,
    result_queue,
) -> None:
    """タイムアウト時に強制終了できるプロセス内で指定されたコードエンジンのパイプラインを実行します。"""

    try:
        result = AnalysisPipeline(chord_engine=chord_engine).analyze_to_musicxml(
            Path(audio_path),
            Path(output_path),
            rhythm_level=rhythm_level,
        )
        result_queue.put(("success", _result_metadata(result)))
    except Exception:
        _logger.exception("解析ワーカープロセス内で例外が発生しました (chord_engine=%s)", chord_engine)
        # パス、ソースメタデータ、または生の例外メッセージは親プロセスに送信しません。
        result_queue.put(("failure", None))


def _run_isolated_analysis(
    audio_path: Path,
    output_path: Path,
    *,
    rhythm_level: int | None,
    chord_engine: str,
    timeout_seconds: float,
) -> dict[str, str]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_analysis_worker,
        args=(str(audio_path), str(output_path), rhythm_level, chord_engine, result_queue),
    )
    process.start()
    try:
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            raise AnalysisTimeoutError("解析が設定された制限時間を超えました")
        if process.exitcode != 0:
            raise AnalysisWorkerError("解析ワーカーが正常に終了しませんでした")
        try:
            status, metadata = result_queue.get(timeout=1)
        except queue.Empty as exc:
            raise AnalysisWorkerError("解析ワーカーから結果が返されませんでした") from exc
        if status != "success" or not isinstance(metadata, dict):
            raise AnalysisWorkerError("解析ワーカーが失敗しました")
        return metadata
    finally:
        result_queue.close()
        result_queue.join_thread()


def _validate_request(
    *,
    upload: UploadFile | None,
    youtube_url: str | None,
    rhythm_level: int | None,
    chord_engine: str,
    lawful_use_confirmation: bool,
) -> str | None:
    # MVP仕様§7: 入力不足・形式不正→400、利用条件違反→403、解析不能→422
    normalized_youtube_url = youtube_url.strip() if youtube_url else None
    if (upload is None) == (normalized_youtube_url is None):
        raise HTTPException(
            status_code=400,
            detail="file または youtube_url のいずれか一方のみを指定する必要があります",
        )
    if rhythm_level is not None and rhythm_level not in {1, 2, 3}:
        raise HTTPException(status_code=400, detail="rhythm_level は 1, 2, 3 のいずれかでなければなりません")
    if chord_engine.strip().lower() not in ALLOWED_CHORD_ENGINES:
        allowed = ", ".join(sorted(ALLOWED_CHORD_ENGINES))
        raise HTTPException(
            status_code=400,
            detail=f"chord_engine は {allowed} のいずれかでなければなりません",
        )
    if not lawful_use_confirmation:
        raise HTTPException(
            status_code=403,
            detail="lawful_use_confirmation（適法利用の確認）が必要です",
        )
    return normalized_youtube_url


def cors_origins_from_environment() -> tuple[str, ...]:
    value = os.environ.get("CHORDPULSE_CORS_ORIGINS")
    if not value:
        return DEFAULT_CORS_ORIGINS
    origins = tuple(origin.strip() for origin in value.split(",") if origin.strip())
    return origins or DEFAULT_CORS_ORIGINS


def _positive_int_from_environment(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def create_app(
    *,
    pipeline: AnalysisPipeline | None = None,
    max_request_body_bytes: int = MAX_REQUEST_BODY_BYTES,
    max_concurrent_analyses: int = DEFAULT_MAX_CONCURRENT_ANALYSES,
    analysis_timeout_seconds: int = DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    cors_origins: tuple[str, ...] | None = None,
) -> FastAPI:
    """単一の解析に適した制限を設定したAPIアプリケーションを作成します。"""

    if max_request_body_bytes < 1:
        raise ValueError("max_request_body_bytesは1以上でなければなりません")
    if analysis_timeout_seconds < 1:
        raise ValueError("analysis_timeout_secondsは1以上でなければなりません")

    app = FastAPI(
        title="ChordPulse API",
        version="0.1.0",
        description="単一のオーディオソースから一時的なMusicXMLマスターコード譜を生成します。",
    )
    app.add_middleware(
        RequestBodySizeLimitMiddleware,
        max_body_bytes=max_request_body_bytes,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins or cors_origins_from_environment()),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
        expose_headers=["X-ChordPulse-BPM", "X-ChordPulse-Chord-Engine"],
    )
    analysis_gate = AnalysisGate(max_concurrent_analyses)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            # フロントエンドがビルド時の環境変数ではなく実際のバックエンド状態を反映できるよう、
            # オペレータが管理するスイッチを公開します。
            "youtube_enabled": youtube_enabled_from_environment(),
        }

    @app.post("/analyze", response_class=Response)
    def analyze(
        file: Annotated[UploadFile | None, File()] = None,
        youtube_url: Annotated[str | None, Form()] = None,
        rhythm_level: Annotated[int, Form()] = 2,
        chord_engine: Annotated[str, Form()] = "btc",
        lawful_use_confirmation: Annotated[bool, Form()] = False,
    ) -> Response:
        normalized_youtube_url = _validate_request(
            upload=file,
            youtube_url=youtube_url,
            rhythm_level=rhythm_level,
            chord_engine=chord_engine,
            lawful_use_confirmation=lawful_use_confirmation,
        )

        try:
            with analysis_gate.acquire():
                with contextlib.ExitStack() as stack:
                    if file is not None:
                        audio_path = stack.enter_context(_store_upload(file))
                    else:
                        audio_path = stack.enter_context(
                            acquire_youtube_audio(
                                normalized_youtube_url or "",
                                enabled=youtube_enabled_from_environment(),
                                lawful_use_confirmation=lawful_use_confirmation,
                            )
                        )
                    output_dir = stack.enter_context(
                        tempfile.TemporaryDirectory(prefix="chordpulse-api-output-")
                    )
                    output_path = Path(output_dir) / "result.musicxml"
                    if pipeline is None:
                        metadata = _run_isolated_analysis(
                            audio_path,
                            output_path,
                            rhythm_level=rhythm_level,
                            chord_engine=chord_engine.strip().lower(),
                            timeout_seconds=analysis_timeout_seconds,
                        )
                    else:
                        result = pipeline.analyze_to_musicxml(
                            audio_path,
                            output_path,
                            rhythm_level=rhythm_level,
                        )
                        metadata = _result_metadata(result)
                    musicxml = output_path.read_bytes()
        except UploadTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except YouTubeAcquisitionDisabledError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except AudioAcquisitionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AnalysisBusyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except AnalysisTimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except AnalysisWorkerError as exc:
            _logger.exception("解析ワーカーが正常に終了しませんでした")
            raise HTTPException(status_code=500, detail="解析に失敗しました") from exc
        except (RuntimeError, ValueError, OSError) as exc:
            _logger.exception("解析リクエストの処理中に予期しない例外が発生しました")
            raise HTTPException(status_code=500, detail="解析に失敗しました") from exc

        return Response(
            content=musicxml,
            media_type=MUSICXML_MEDIA_TYPE,
            headers={
                "Content-Disposition": 'attachment; filename="chordpulse.musicxml"',
                "X-ChordPulse-BPM": metadata["bpm"],
                "X-ChordPulse-Chord-Engine": metadata["chord_engine"],
            },
        )

    return app


app = create_app(
    max_concurrent_analyses=_positive_int_from_environment(
        "CHORDPULSE_MAX_CONCURRENT_ANALYSES",
        DEFAULT_MAX_CONCURRENT_ANALYSES,
    ),
    analysis_timeout_seconds=_positive_int_from_environment(
        "CHORDPULSE_ANALYSIS_TIMEOUT_SECONDS",
        DEFAULT_ANALYSIS_TIMEOUT_SECONDS,
    ),
)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "chordpulse.api:app",
        host=os.environ.get("CHORDPULSE_API_HOST", "127.0.0.1"),
        port=int(os.environ.get("CHORDPULSE_API_PORT", "8000")),
    )

