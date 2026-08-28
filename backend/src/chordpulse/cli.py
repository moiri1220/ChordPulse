"""フェーズ1ローカルパイプライン用のコマンドラインエントリポイント。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .audio import (
    AudioAcquisitionError,
    acquire_local_audio,
    acquire_youtube_audio,
    youtube_enabled_from_environment,
)
from .pipeline import AnalysisPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chordpulse-analyze",
        description="オーディオソースを解析し、ChordPulseのMusicXMLチャートを生成します。",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="ローカルのMP3/WAV/FLAC音源")
    source.add_argument("--youtube-url", help="YouTube音源（設定および規約が適用されます）")
    parser.add_argument("--output", type=Path, required=True, help="出力先 .musicxml パス")
    parser.add_argument(
        "--rhythm-level",
        type=int,
        choices=(1, 2, 3),
        default=2,
        help="1=シンプル, 2=四分音符スラッシュ, 3=オンセットベースのリズムスラッシュ",
    )
    parser.add_argument(
        "--lawful-use-confirmation",
        action="store_true",
        help="音源が適法に取得され、この用途での利用が許可されていることを確認します",
    )
    parser.add_argument(
        "--enable-youtube",
        action="store_true",
        help="このプロセスでYouTubeの取得を明示的に有効にします",
    )
    parser.add_argument(
        "--analysis-json",
        type=Path,
        help="解析メタデータの出力先オプションパス（オーディオデータは含みません）",
    )
    return parser


def _write_analysis_json(path: Path, result) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    if not args.lawful_use_confirmation:
        print("エラー: --lawful-use-confirmation が必須です", file=sys.stderr)
        return 2
    if args.input is not None:
        input_path = args.input.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        if input_path == output_path:
            print("エラー: 出力先が入力オーディオファイルを上書きしてはなりません", file=sys.stderr)
            return 2

    pipeline = AnalysisPipeline()
    try:
        if args.input is not None:
            source_context = acquire_local_audio(args.input)
        else:
            source_context = acquire_youtube_audio(
                args.youtube_url,
                enabled=args.enable_youtube and youtube_enabled_from_environment(),
                lawful_use_confirmation=args.lawful_use_confirmation,
            )
        with source_context as audio_path:
            result = pipeline.analyze_to_musicxml(
                audio_path,
                args.output,
                rhythm_level=args.rhythm_level,
            )
        if args.analysis_json:
            _write_analysis_json(args.analysis_json, result)
    except (AudioAcquisitionError, RuntimeError, ValueError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    print(f"MusicXML: {args.output.resolve()}")
    print(f"BPM: {result.bpm:.2f}")
    print(f"拍数: {len(result.beat_times)}")
    print(f"コードエンジン: {result.chord_engine}")
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

