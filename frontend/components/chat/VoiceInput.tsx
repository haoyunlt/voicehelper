'use client'

import { useState, useEffect, useRef } from 'react'
import { Mic, MicOff, Volume2, VolumeX } from 'lucide-react'
import { Button } from '../ui/button'

interface VoiceInputProps {
  conversationId?: string
  onTranscript?: (text: string, isFinal: boolean) => void
  onResponse?: (text: string) => void
  onReferences?: (refs: Array<{chunk_id: string, source: string, score: number}>) => void
  disabled?: boolean
}

export default function VoiceInput({
  conversationId,
  onTranscript,
  onResponse,
  onReferences,
  disabled = false
}: VoiceInputProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isConnected, setIsConnected] = useState(true) // 简化为始终连接
  const [transcript, setTranscript] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [volume, setVolume] = useState(1)

  const toggleRecording = () => {
    if (isRecording) {
      setIsRecording(false)
      // 模拟语音转写结果
      setTimeout(() => {
        const mockText = "这是模拟的语音转写结果"
        onTranscript?.(mockText, true)
      }, 1000)
    } else {
      setIsRecording(true)
      setError(null)
    }
  }

  const toggleMute = () => {
    const newVolume = volume > 0 ? 0 : 1
    setVolume(newVolume)
  }

  return (
    <div className="flex items-center space-x-2">
      {/* 录音按钮 */}
      <Button
        variant={isRecording ? "destructive" : "outline"}
        size="sm"
        onClick={toggleRecording}
        disabled={disabled || !isConnected}
        className={`${isRecording ? 'animate-pulse' : ''}`}
      >
        {isRecording ? (
          <MicOff className="w-4 h-4" />
        ) : (
          <Mic className="w-4 h-4" />
        )}
      </Button>

      {/* 音量控制 */}
      <Button
        variant="ghost"
        size="sm"
        onClick={toggleMute}
        disabled={disabled}
      >
        {volume > 0 ? (
          <Volume2 className="w-4 h-4" />
        ) : (
          <VolumeX className="w-4 h-4" />
        )}
      </Button>

      {/* 状态指示 */}
      <div className="flex items-center space-x-2 text-sm">
        {!isConnected && (
          <span className="text-red-500">● 连接中...</span>
        )}
        {isConnected && !isRecording && !isPlaying && (
          <span className="text-green-500">● 就绪</span>
        )}
        {isRecording && (
          <span className="text-blue-500">● 录音中</span>
        )}
        {isPlaying && (
          <span className="text-purple-500">● 播放中</span>
        )}
      </div>

      {/* 实时转写显示 */}
      {transcript && (
        <div className="flex-1 text-sm text-gray-600 italic">
          "{transcript}"
        </div>
      )}

      {/* 错误提示 */}
      {error && (
        <div className="text-sm text-red-500">
          {error}
        </div>
      )}
    </div>
  )
}
