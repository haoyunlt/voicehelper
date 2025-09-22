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

  const toggleRecording = async () => {
    if (isRecording) {
      setIsRecording(false)
      // 停止录音并处理音频
      try {
        // 集成语音识别API
        const response = await fetch('/api/v1/voice/transcribe', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('token')}`,
          },
          body: JSON.stringify({
            audio_data: '', // 实际音频数据将由MediaRecorder提供
            language: 'zh-CN',
            is_final: true
          })
        })
        
        if (response.ok) {
          const result = await response.json()
          if (result.text) {
            onTranscript?.(result.text, true)
          }
        } else {
          throw new Error('语音识别失败')
        }
      } catch (error) {
        console.error('语音识别错误:', error)
        setError('语音识别失败，请重试')
        // 降级到模拟结果
        const fallbackText = "语音识别服务暂不可用，这是降级响应"
        onTranscript?.(fallbackText, true)
      }
    } else {
      setIsRecording(true)
      setError(null)
      // 开始录音
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
        // 这里可以集成MediaRecorder API进行实际录音
        console.log('开始录音，音频流已获取')
      } catch (error) {
        console.error('获取麦克风权限失败:', error)
        setError('无法访问麦克风，请检查权限设置')
        setIsRecording(false)
      }
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
