# ChordPulse backend（Phase 1）

ローカル音源を解析し、マスターコード譜をMusicXMLへ変換するコアエンジンです。FastAPIやフロントエンドから独立して利用できます。

## セットアップ

Python 3.11以上を用意し、`backend`ディレクトリで実行します。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,youtube]"
```

YouTube input additionally requires the `ffmpeg` executable to be installed and
available on `PATH`. The backend checks this before starting `yt-dlp`; local file
analysis does not require FFmpeg.

YouTube入力を使わないローカル音源だけなら、`.[dev]`でインストールできます。YouTube取得は利用規約・権利条件を確認したうえで、明示的に有効化してください。

The Phase 1 recognizer is explicitly named `librosa-chroma-template` in the
analysis result. Chordino and madmom are extension points, not silently selected
fallbacks; a future adapter can be injected without changing MusicXML generation.

## API (Phase 2)

Install the API extra and start the development server:

```powershell
python -m pip install -e ".[dev,api,youtube]"
chordpulse-api
```

`POST /analyze` accepts one `file` or one `youtube_url`, `rhythm_level` (`1` to
`3`), and the required `lawful_use_confirmation` form field. It returns a
request-scoped MusicXML download. Uploaded and downloaded audio, as well as the
generated MusicXML temporary file, are removed when the request finishes.

## CLI

```powershell
chordpulse-analyze `
  --input .\path\to\source.wav `
  --output .\analysis-output\song.musicxml `
  --rhythm-level 2 `
  --lawful-use-confirmation `
  --analysis-json .\analysis-output\song.json
```

リズムレベルは以下のとおりです。

- `1`: 1小節にコードネームとシンプルなスラッシュ
- `2`: 4分音符スラッシュ
- `3`: Onset検出に基づく8分音符レベルのスラッシュ

YouTube入力は、オペレーターが有効化したプロセスで、かつ利用条件を確認した場合だけ実行します。

```powershell
$env:CHORDPULSE_ENABLE_YOUTUBE = "1"
chordpulse-analyze `
  --youtube-url "https://www.youtube.com/watch?v=..." `
  --enable-youtube `
  --lawful-use-confirmation `
  --output .\analysis-output\song.musicxml
```

入力音源はリクエスト単位の一時ディレクトリで処理され、処理終了時に削除されます。音源ファイルや生成MusicXMLはGitへ追加しないでください。

## 現在のコード認識

現在は、`librosa`のクロマ特徴量とコードテンプレートを用いたベースラインを使用します。`ChordRecognizer`プロトコルを境界にしているため、Chordinoまたはmadmomの導入時もMusicXML生成・API層を変更せず差し替えられます。
