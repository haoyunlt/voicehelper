import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import ErrorBoundary from '@/components/ErrorBoundary'
import { LoggerProvider } from '@/components/LoggerProvider'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'VoiceHelper - 智能语音助手',
  description: '基于RAG技术的企业级智能语音助手',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh-CN">
      <body className={inter.className}>
        <LoggerProvider>
          <ErrorBoundary>
            {children}
          </ErrorBoundary>
        </LoggerProvider>
      </body>
    </html>
  )
}
