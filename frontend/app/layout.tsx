import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "ChordPulse",
  description: "音源からマスターコード譜を生成するMusicXMLアプリケーション",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ja">
      <body>{children}</body>
    </html>
  );
}
