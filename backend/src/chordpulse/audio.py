"""リクエストスコープの一時ストレージを使用したオーディオの取得とロード。"""

from __future__ import annotations

import contextlib
import math
import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

from .models import AudioData

SUPPORTED_SUFFIXES = {".mp3", ".wav", ".flac", ".m4a", ".ogg"}
DEFAULT_MAX_FILE_BYTES = 250 * 1024 * 1024
DEFAULT_MAX_DURATION_SECONDS = 15 * 60
DEFAULT_YOUTUBE_SOCKET_TIMEOUT_SECONDS = 30


class AudioAcquisitionError(RuntimeError):
    """分析用の入力を安全に取得できない場合に発生する例外。"""


class YouTubeAcquisitionDisabledError(AudioAcquisitionError):
    """オペレータがオプショナルのYouTube取得を無効にしている場合に発生する例外。"""


def _validate_audio_file(path: Path, max_file_bytes: int) -> None:
    if not path.is_file():
        raise AudioAcquisitionError(f"オーディオファイルが存在しません: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise AudioAcquisitionError(f"サポートされていないオーディオ形式です。次のいずれかを指定してください: {allowed}")
    if path.stat().st_size > max_file_bytes:
        raise AudioAcquisitionError("オーディオファイルが設定されたサイズ制限を超えています")


@contextlib.contextmanager
def acquire_local_audio(
    source: Path,
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> Iterator[Path]:
    """ローカルソースを隔離されたリクエストスコープの一時ディレクトリにコピーします。"""

    source = source.expanduser().resolve()
    _validate_audio_file(source, max_file_bytes)
    with tempfile.TemporaryDirectory(prefix="chordpulse-") as temp_dir:
        temporary_path = Path(temp_dir) / source.name
        shutil.copy2(source, temporary_path)
        try:
            yield temporary_path
        finally:
            # TemporaryDirectoryがディレクトリを削除します。これを明示的に残すことで、
            # 将来のAPIやワーカー実装に対して削除規約を明確にします。
            temporary_path.unlink(missing_ok=True)


def _validate_youtube_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "www.youtu.be",
    }:
        raise AudioAcquisitionError("youtube.comおよびyoutu.beのURLのみ受け付けられます")


@contextlib.contextmanager
def acquire_youtube_audio(
    url: str,
    *,
    enabled: bool,
    lawful_use_confirmation: bool,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
) -> Iterator[Path]:
    """許可されたYouTubeソースを一時ディレクトリにダウンロードします。

    この関数は意図的に「安全側に倒して失敗（fail closed）」するように設計されています。
    この機能を有効にするのはオペレータの決定であり、YouTubeの利用規約、著作権法、
    またはソース所有者の権利を上書きするものではありません。
    """

    if not enabled:
        raise YouTubeAcquisitionDisabledError("YouTubeソースの取得は無効になっています")
    if not lawful_use_confirmation:
        raise AudioAcquisitionError("YouTube入力には適法利用の確認が必要です")
    _validate_youtube_url(url)

    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise AudioAcquisitionError(
            "YouTubeサポートがインストールされていません。オプションのyoutube依存関係をインストールしてください"
        ) from exc

    with tempfile.TemporaryDirectory(prefix="chordpulse-youtube-") as temp_dir:
        output_template = str(Path(temp_dir) / "source.%(ext)s")
        options = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": DEFAULT_YOUTUBE_SOCKET_TIMEOUT_SECONDS,
            "retries": 1,
            "fragment_retries": 1,
            "max_filesize": max_file_bytes,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
        }
        if shutil.which("ffmpeg") is None:
            raise AudioAcquisitionError(
                "YouTubeオーディオ変換にはFFmpegが必要です。インストールするか、YouTube入力を無効にしてください"
            )
        try:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=False)
                duration = info.get("duration") if isinstance(info, dict) else None
                if duration is not None:
                    duration_seconds = float(duration)
                    if not math.isfinite(duration_seconds) or duration_seconds > max_duration_seconds:
                        raise AudioAcquisitionError(
                            "YouTubeソースが設定された時間制限を超えています"
                        )
                downloader.download([url])
        except AudioAcquisitionError:
            raise
        except Exception as exc:  # yt-dlp exposes several exception types by version
            raise AudioAcquisitionError("YouTubeオーディオの取得に失敗しました") from exc

        wav_path = Path(temp_dir) / "source.wav"
        if not wav_path.is_file():
            raise AudioAcquisitionError("YouTubeオーディオ変換でWAVファイルが生成されませんでした")
        if wav_path.stat().st_size > max_file_bytes:
            raise AudioAcquisitionError("ダウンロードされたオーディオが設定されたサイズ制限を超えています")
        try:
            yield wav_path
        finally:
            wav_path.unlink(missing_ok=True)


def load_audio(
    path: Path,
    *,
    sample_rate: int = 22_050,
    max_duration_seconds: float = DEFAULT_MAX_DURATION_SECONDS,
) -> AudioData:
    """単一の解析リクエスト用にモノラルオーディオをメモリにロードします。"""

    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise AudioAcquisitionError("librosaがインストールされていません") from exc

    try:
        source_duration = float(librosa.get_duration(path=path))
    except Exception as exc:  # decoder errors vary by installed backend
        raise AudioAcquisitionError(f"オーディオファイルを検査できませんでした: {path.name}") from exc
    if not math.isfinite(source_duration) or source_duration > max_duration_seconds:
        raise AudioAcquisitionError("オーディオファイルが設定された時間制限を超えています")

    try:
        samples, actual_sample_rate = librosa.load(path, sr=sample_rate, mono=True)
    except Exception as exc:  # librosa backend errors vary by installed decoder
        raise AudioAcquisitionError(f"オーディオファイルをデコードできませんでした: {path.name}") from exc
    if samples.size == 0:
        raise AudioAcquisitionError("オーディオファイルにサンプルが含まれていません")
    duration_seconds = float(samples.shape[0] / actual_sample_rate)
    if not math.isfinite(duration_seconds) or duration_seconds > max_duration_seconds:
        raise AudioAcquisitionError("オーディオファイルが設定された時間制限を超えています")
    return AudioData(
        samples=samples,
        sample_rate=int(actual_sample_rate),
        duration_seconds=duration_seconds,
    )


def youtube_enabled_from_environment() -> bool:
    """オプショナルのYouTube取得に関する明示的なオペレータスイッチを読み取ります。"""

    return os.environ.get("CHORDPULSE_ENABLE_YOUTUBE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

