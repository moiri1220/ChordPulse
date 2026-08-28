"use client";

import { useEffect, useRef, useState } from "react";

type ScoreViewerProps = {
  musicXml: string;
};

/** クリーンアップ用に呼び出す最小限のOSMDインターフェース。 */
type OsmdInstance = {
  clear: () => void;
};

export function ScoreViewer({ musicXml }: ScoreViewerProps) {
  const container = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<OsmdInstance | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);

  useEffect(() => {
    const target = container.current;
    if (!target) {
      return;
    }

    let cancelled = false;
    target.replaceChildren();
    setRenderError(null);

    async function renderScore(element: HTMLElement) {
      try {
        const { OpenSheetMusicDisplay } = await import("opensheetmusicdisplay");
        if (cancelled) {
          return;
        }
        const renderer = new OpenSheetMusicDisplay(element, {
          autoResize: true,
          drawTitle: false,
          followCursor: false,
        });
        // クリーンアップ処理がXMLパース中に実行された場合でもclear()を呼び出せるよう、
        // 非同期ロードの前に退避します。
        rendererRef.current = renderer;
        await renderer.load(musicXml);
        if (!cancelled) {
          renderer.render();
        }
      } catch {
        if (!cancelled) {
          setRenderError("譜面を表示できませんでした。MusicXMLをダウンロードして確認してください。");
        }
      }
    }

    void renderScore(target);
    return () => {
      cancelled = true;
      if (rendererRef.current) {
        try {
          rendererRef.current.clear();
        } catch {
          // クリーンアップエラーは無視します — いずれにせよDOMは以下でクリアされます。
        }
        rendererRef.current = null;
      }
      target.replaceChildren();
    };
  }, [musicXml]);

  return (
    <div className="score-viewer">
      {renderError && <p className="status error">{renderError}</p>}
      <div aria-label="生成された楽譜" ref={container} />
    </div>
  );
}
