"use client";

import { ChangeEvent, DragEvent, FormEvent, useEffect, useId, useMemo, useRef, useState } from "react";

import { ScoreViewer } from "../components/score-viewer";

const MAX_UPLOAD_BYTES = 250 * 1024 * 1024;

type SourceMode = "file" | "youtube";
type AnalysisStatus = "idle" | "analyzing" | "success" | "error";

type AnalysisOutput = {
  musicXml: string;
  bpm: string | null;
  chordEngine: string | null;
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function errorMessageFrom(response: Response, body: unknown): string {
  // APIの例外詳細メッセージ（英語や実装詳細を含む場合あり）にフォールバックする前に、
  // 既知のステータスコードを適切な日本語メッセージで処理します。
  if (response.status === 400) {
    return "入力内容に誤りがあります。ファイルまたはURLを確認して再試行してください。";
  }
  if (response.status === 403) {
    return "この操作は許可されていません。利用条件の確認が必要です。";
  }
  if (response.status === 413) {
    return "音源ファイルが許可されたサイズ（250 MiB）を超えています。";
  }
  if (response.status === 503) {
    return "現在ほかの解析を処理中です。少し待ってから再試行してください。";
  }
  if (response.status === 504) {
    return "解析が制限時間を超えました。より短い音源で再試行してください。";
  }
  if (
    body &&
    typeof body === "object" &&
    "detail" in body &&
    typeof body.detail === "string"
  ) {
    return body.detail;
  }
  return "解析に失敗しました。入力と設定を確認して再試行してください。";
}

function createSyntheticWav(): File {
  const sampleRate = 22050;
  const duration = 6.0; // 6秒
  const totalSamples = Math.floor(sampleRate * duration);
  const buffer = new ArrayBuffer(44 + totalSamples * 2);
  const view = new DataView(buffer);

  // WAVヘッダーの書き込み
  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, "RIFF");
  view.setUint32(4, 36 + totalSamples * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // サブチャンク1サイズ
  view.setUint16(20, 1, true); // 音声フォーマット (PCM)
  view.setUint16(22, 1, true); // チャンネル数 (モノラル)
  view.setUint32(24, sampleRate, true); // サンプリングレート
  view.setUint32(28, sampleRate * 2, true); // バイトレート
  view.setUint16(32, 2, true); // ブロックアライン
  view.setUint16(34, 16, true); // サンプルあたりのビット数
  writeString(36, "data");
  view.setUint32(40, totalSamples * 2, true);

  // 音声サンプルの書き込み
  for (let i = 0; i < totalSamples; i++) {
    const t = i / sampleRate;
    let sample = 0;
    // 最初の3秒間はハ長調（C Major: C4=261.63, E4=329.63, G4=392.00）
    // 次の3秒間はト長調（G Major: G3=196.00, B3=246.94, D4=293.66）
    if (t < 3.0) {
      sample = 0.25 * (
        Math.sin(2 * Math.PI * 261.63 * t) +
        Math.sin(2 * Math.PI * 329.63 * t) +
        Math.sin(2 * Math.PI * 392.00 * t)
      );
    } else {
      sample = 0.25 * (
        Math.sin(2 * Math.PI * 196.00 * t) +
        Math.sin(2 * Math.PI * 246.94 * t) +
        Math.sin(2 * Math.PI * 293.66 * t)
      );
    }

    // 0.5秒ごとに120 BPMのクリック音を追加
    const beatPos = t % 0.5;
    if (beatPos < 0.03) {
      const clickWindow = 0.5 * (1 - Math.cos((2 * Math.PI * beatPos) / 0.03));
      sample += 0.4 * clickWindow;
    }

    const int16 = Math.max(-32768, Math.min(32767, Math.floor(sample * 32767)));
    view.setInt16(44 + i * 2, int16, true);
  }

  const blob = new Blob([buffer], { type: "audio/wav" });
  return new File([blob], "sample_c_g_progression.wav", { type: "audio/wav" });
}

export default function HomePage() {
  const [sourceMode, setSourceMode] = useState<SourceMode>("file");
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [chordEngine, setChordEngine] = useState("btc");
  const [lawfulUseConfirmation, setLawfulUseConfirmation] = useState(false);
  const [status, setStatus] = useState<AnalysisStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [output, setOutput] = useState<AnalysisOutput | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  // マウント時に /health から取得し、UIが実際のバックエンド設定を反映するようにします。
  const [youtubeEnabled, setYoutubeEnabled] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const fileInputId = useId();
  const chordEngineSelectId = useId();
  const consentCheckboxId = useId();

  const apiBaseUrl = useMemo(
    () =>
      (process.env.NEXT_PUBLIC_CHORDPULSE_API_URL ?? "http://127.0.0.1:8000").replace(
        /\/$/,
        "",
      ),
    [],
  );

  // マウント時にバックエンドからオペレータ制御の youtube_enabled フラグを取得し、
  // UIが常に実際のバックエンド設定を反映するようにします。
  useEffect(() => {
    let active = true;
    fetch(`${apiBaseUrl}/health`)
      .then((r) => r.json() as Promise<{ youtube_enabled?: boolean }>)
      .then((data) => {
        if (active && typeof data?.youtube_enabled === "boolean") {
          setYoutubeEnabled(data.youtube_enabled);
        }
      })
      .catch(() => {
        // マウント時にバックエンドに接続できない場合は、YouTube入力を無効のままにします。
      });
    return () => {
      active = false;
    };
  }, [apiBaseUrl]);

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selectedFile = event.target.files?.[0] ?? null;
    setFile(selectedFile);
    setError(null);
  }

  function handleDragOver(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave() {
    setIsDragging(false);
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setIsDragging(false);
    const droppedFile = event.dataTransfer.files?.[0] ?? null;
    if (droppedFile) {
      setFile(droppedFile);
      setError(null);
    }
  }

  function clearFile() {
    setFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  /**
   * 開発用ヘルパー: ブラウザ上で合成されたWAV音源をロードし、適法利用の同意確認を自動で設定します。
   * サンプルは完全にクライアント側で生成されます（ネットワーク要求やサードパーティの音声は含まれません）。
   * 本番環境へデプロイする前に、削除するか環境変数などで保護してください。
   */
  function loadSampleAudio() {
    const sampleFile = createSyntheticWav();
    setFile(sampleFile);
    setLawfulUseConfirmation(true);
    setError(null);
  }

  function onReset() {
    clearFile();
    setYoutubeUrl("");
    setStatus("idle");
    setError(null);
    setOutput(null);
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    setOutput(null);

    if (!lawfulUseConfirmation) {
      setStatus("error");
      setError("適法に利用できる音源であることへの確認が必要です。");
      return;
    }
    if (sourceMode === "file") {
      if (!file) {
        setStatus("error");
        setError("解析する音源ファイルを選択してください。");
        return;
      }
      if (file.size > MAX_UPLOAD_BYTES) {
        setStatus("error");
        setError("音源ファイルが250 MiBの上限を超えています。");
        return;
      }
    } else if (!youtubeUrl.trim()) {
      setStatus("error");
      setError("YouTube URLを入力してください。");
      return;
    }

    const formData = new FormData();
    formData.set("rhythm_level", "2");
    formData.set("chord_engine", chordEngine);
    formData.set("lawful_use_confirmation", "true");
    if (sourceMode === "file" && file) {
      formData.set("file", file);
    } else {
      formData.set("youtube_url", youtubeUrl.trim());
    }


    setStatus("analyzing");
    try {
      const response = await fetch(`${apiBaseUrl}/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const body: unknown = await response.json().catch(() => null);
        throw new Error(errorMessageFrom(response, body));
      }
      const musicXml = await response.text();
      // XMLの妥当性検証はOSMDの load() に委ねます。
      // パースエラーは ScoreViewer のエラー状態を介して報告され、
      // score-partwise と score-timewise の両方の形式を正しく処理できます。
      setOutput({
        musicXml,
        bpm: response.headers.get("X-ChordPulse-BPM"),
        chordEngine: response.headers.get("X-ChordPulse-Chord-Engine"),
      });
      setStatus("success");

    } catch (caught) {
      setStatus("error");
      setError(caught instanceof Error ? caught.message : "解析に失敗しました。");
    }
  }

  function downloadMusicXml() {
    if (!output) {
      return;
    }
    const blob = new Blob([output.musicXml], {
      type: "application/vnd.recordare.musicxml+xml",
    });
    const downloadUrl = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = downloadUrl;
    anchor.download = "chordpulse.musicxml";
    anchor.click();
    URL.revokeObjectURL(downloadUrl);
  }

  return (
    <main className="page-shell">
      <header className="intro" aria-labelledby="page-title">
        <div className="badge">ChordPulse MVP</div>
        <h1 id="page-title">音源からマスターコード譜を生成</h1>
        <p className="subtitle">
          MP3/WAV音源を解析し、BPM・ビート・コードネーム・リズムスラッシュを含むMusicXMLを自動生成・ブラウザ描画します。
        </p>
      </header>

      <section className="panel" aria-labelledby="analysis-title">
        <div className="panel-header">
          <h2 id="analysis-title">音源の解析設定</h2>
          {output && (
            <button className="text-button" onClick={onReset} type="button">
              ✕ 設定をリセット
            </button>
          )}
        </div>

        <form onSubmit={onSubmit}>
          <fieldset disabled={status === "analyzing"}>
            <div className="form-group">
              <label className="field-label">入力方法</label>
              <div className="source-tabs" role="radiogroup" aria-label="入力方法">
                <label className={`tab-label ${sourceMode === "file" ? "active" : ""}`}>
                  <input
                    checked={sourceMode === "file"}
                    name="source-mode"
                    onChange={() => {
                      setSourceMode("file");
                      setError(null);
                    }}
                    type="radio"
                    value="file"
                  />
                  音源ファイル（MP3 / WAV）
                </label>
                {youtubeEnabled && (
                  <label className={`tab-label ${sourceMode === "youtube" ? "active" : ""}`}>
                    <input
                      checked={sourceMode === "youtube"}
                      name="source-mode"
                      onChange={() => {
                        setSourceMode("youtube");
                        setError(null);
                      }}
                      type="radio"
                      value="youtube"
                    />
                    YouTube URL
                  </label>
                )}
              </div>
            </div>

            {sourceMode === "file" ? (
              <div className="form-group">
                <div className="field-label-row">
                  <label className="field-label" htmlFor={fileInputId}>
                    音源ファイル
                  </label>
                  <button
                    className="sample-load-button"
                    onClick={loadSampleAudio}
                    type="button"
                  >
                    💡 テスト用サンプル音源をセット
                  </button>
                </div>
                <div className="file-drop-area-container">
                  <label
                    className={`file-drop-area ${isDragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
                    htmlFor={fileInputId}
                    onDragLeave={handleDragLeave}
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                  >
                    <input
                      accept=".mp3,.wav,.flac,.m4a,.ogg,audio/*"
                      id={fileInputId}
                      onChange={onFileChange}
                      ref={fileInputRef}
                      type="file"
                    />
                    {file ? (
                      <div className="selected-file-info">
                        <span className="file-icon">🎵</span>
                        <div className="file-meta">
                          <strong className="file-name">{file.name}</strong>
                          <span className="file-size">{formatBytes(file.size)}</span>
                        </div>
                        <button
                          className="file-clear-button"
                          onClick={(e) => {
                            e.preventDefault();
                            clearFile();
                          }}
                          title="ファイルを解除"
                          type="button"
                        >
                          ✕
                        </button>
                      </div>
                    ) : (
                      <div className="drop-prompt">
                        <span className="upload-icon">📂</span>
                        <span>
                          クリックしてファイルを選択、またはここにドラッグ＆ドロップ
                        </span>
                        <span className="file-hint">
                          MP3 / WAV / FLAC / M4A / OGG（最大 250 MiB）
                        </span>
                      </div>
                    )}
                  </label>
                </div>
              </div>
            ) : (
              <div className="form-group">
                <label className="field-label" htmlFor="youtube-input">
                  YouTube URL
                </label>
                <input
                  className="text-input"
                  id="youtube-input"
                  onChange={(event) => setYoutubeUrl(event.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                  type="url"
                  value={youtubeUrl}
                />
              </div>
            )}

            <div className="form-group">
              <label className="field-label" htmlFor={chordEngineSelectId}>
                コード解析モード（エンジン）
              </label>
              <select
                className="select-input"
                id={chordEngineSelectId}
                onChange={(event) => setChordEngine(event.target.value)}
                value={chordEngine}
              >
                <option value="btc">BTC（AI深層学習・最高精度・推奨）</option>
                <option value="viterbi">Viterbi（HMM平滑化）</option>
                <option value="harmonic">Harmonic（低音重視・HPSS）</option>
                <option value="template">Template（標準クロマ）</option>
              </select>
            </div>

            <div className="consent-container">
              <label className="consent-label" htmlFor={consentCheckboxId}>
                <input
                  checked={lawfulUseConfirmation}
                  id={consentCheckboxId}
                  onChange={(event) => setLawfulUseConfirmation(event.target.checked)}
                  type="checkbox"
                />
                <span>
                  適法に入手・利用できる音源であり、著作権法第30条の4等に基づく情報解析目的であることを確認しました。
                </span>
              </label>
              <p className="helper-text">
                ※ 音源は解析処理中のみ隔離された一時領域に保存され、完了・失敗時に直ちに削除されます。永続保存や再配信は行いません。
              </p>
            </div>

            <div className="form-actions">
              <button
                className={`primary-button ${status === "analyzing" ? "loading" : ""}`}
                disabled={status === "analyzing"}
                type="submit"
              >
                {status === "analyzing" ? (
                  <>
                    <span className="spinner" aria-hidden="true" />
                    音源を解析中…
                  </>
                ) : (
                  "解析して譜面を生成"
                )}
              </button>
            </div>
          </fieldset>
        </form>

        {status === "analyzing" && (
          <div aria-live="polite" className="status-banner analyzing">
            <span className="spinner" aria-hidden="true" />
            <div>
              <strong>音源を解析しています…</strong>
              <p>BPM・ビート位置およびコード進行を推定しています。音源の長さに応じて数十秒程度かかる場合があります。</p>
            </div>
          </div>
        )}

        {status === "error" && error && (
          <div aria-live="assertive" className="status-banner error">
            <span className="status-icon">⚠️</span>
            <div>
              <strong>エラーが発生しました</strong>
              <p>{error}</p>
            </div>
          </div>
        )}
      </section>

      {output && (
        <section className="panel score-panel" aria-labelledby="score-title">
          <div className="score-heading">
            <div>
              <div className="badge badge-success">解析完了</div>
              <h2 id="score-title">生成されたマスターコード譜</h2>
              <div className="score-meta">
                <span className="meta-item">
                  <strong>推定 BPM:</strong> {output.bpm ? Math.round(Number(output.bpm)) : "取得不可"}
                </span>
                {output.chordEngine && (
                  <span className="meta-item">
                    <strong>コード認識:</strong> {output.chordEngine}
                  </span>
                )}
              </div>
            </div>
            <div className="score-actions">
              <button className="secondary-button" onClick={downloadMusicXml} type="button">
                📥 MusicXMLをダウンロード
              </button>
              <button className="text-button reset-btn" onClick={onReset} type="button">
                別の音源を解析
              </button>
            </div>
          </div>

          <p className="score-disclaimer">
            ※ 解析結果およびコードシンボルは音源からの推定値です。必要に応じてMusicXMLを対応譜面エディタにインポートしてご活用ください。
          </p>

          <ScoreViewer musicXml={output.musicXml} />
        </section>
      )}
    </main>
  );
}
