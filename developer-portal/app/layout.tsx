import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'VoiceHelper Developer Portal',
  description: 'VoiceHelper AI 开发者门户',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  )
}
