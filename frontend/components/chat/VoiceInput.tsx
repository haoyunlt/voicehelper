'use client'

import { useState, useEffect, useRef } from 'react'
import { Mic, MicOff, Volume2, VolumeX } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { AudioRecorder, AudioPlayer, VoiceWebSocket, VoiceMessage } from '@/lib/audio'
import { BargeInManager } from '@/lib/barge-in'

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
  const [isConnected, setIsConnected] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [volume, setVolume] = useState(1)

  const recorderRef = useRef<AudioRecorder | null>(null)
  const playerRef = useRef<AudioPlayer | null>(null)
  const wsRef = useRef<VoiceWebSocket | null>(null)
  const bargeInRef = useRef<BargeInManager | null>(null)
  const seqRef = useRef(0)
  const currentRequestIdRef = useRef<string | null>(null)

  useEffect(() => {
    initializeAudio()
    return () => {
      cleanup()
    }
  }, [])

  const initializeAudio = async () => {
    try {
      // 初始化录音器
      recorderRef.current = new AudioRecorder()
      await recorderRef.current.initialize()

      // 初始化播放器
      playerRef.current = new AudioPlayer()
      await playerRef.current.initialize()

      // 初始化语音打断管理器
      bargeInRef.current = new BargeInManager()
      // 需要在获取到音频流后初始化
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      await bargeInRef.current.initialize(stream)

      // 初始化 WebSocket
      const wsUrl = `ws://localhost:8080/api/voice/stream`
      wsRef.current = new VoiceWebSocket(wsUrl)
      
      wsRef.current.onMessage(handleWebSocketMessage)
      wsRef.current.onError((error) => {
        setError('语音连接错误')
        console.error('WebSocket error:', error)
      })
      wsRef.current.onClose(() => {
        setIsConnected(false)
      })

      await wsRef.current.connect()
      setIsConnected(true)

    } catch (error) {
      setError('语音功能初始化失败，请检查麦克风权限')
      console.error('Audio initialization error:', error)
    }
  }

  const handleWebSocketMessage = (message: VoiceMessage) => {
    switch (message.type) {
      case 'asr_partial':
        if (message.text) {
          setTranscript(message.text)
          onTranscript?.(message.text, false)
        }
        break

      case 'asr_final':
        if (message.text) {
          setTranscript('')
          onTranscript?.(message.text, true)
        }
        break

      case 'llm_delta':
        if (message.text) {
          onResponse?.(message.text)
        }
        break

      case 'tts_chunk':
        if (message.pcm && playerRef.current) {
          try {
            const audioData = Uint8Array.from(atob(message.pcm), c => c.charCodeAt(0))
            playerRef.current.queueAudio(audioData.buffer)
            setIsPlaying(true)
            
            // 开始 TTS 播放时启动语音打断监控
            if (bargeInRef.current && !bargeInRef.current.getIsPlayingTTS()) {
              const requestId = `req_${Date.now()}`
              currentRequestIdRef.current = requestId
              bargeInRef.current.startTTSPlayback(requestId, handleBargeInCancel)
            }
          } catch (error) {
            console.error('Failed to play TTS chunk:', error)
          }
        }
        break

      case 'refs':
        if (message.items) {
          onReferences?.(message.items)
        }
        break

      case 'done':
        setIsPlaying(false)
        // 停止语音打断监控
        if (bargeInRef.current) {
          bargeInRef.current.stopTTSPlayback()
        }
        currentRequestIdRef.current = null
        break

      case 'error':
        setError(message.error || '语音处理错误')
        break
    }
  }

  const startRecording = async () => {
    if (!recorderRef.current || !wsRef.current || !isConnected) {
      setError('语音功能未就绪')
      return
    }

    try {
      // 停止当前播放
      if (playerRef.current) {
        playerRef.current.stop()
        setIsPlaying(false)
      }

      // 发送开始消息
      wsRef.current.send({
        type: 'start',
        codec: 'opus',
        sample_rate: 16000,
        conversation_id: conversationId
      })

      // 开始录音
      recorderRef.current.startRecording(
        (audioData) => {
          if (wsRef.current) {
            const base64Audio = btoa(String.fromCharCode(...new Uint8Array(audioData)))
            wsRef.current.send({
              type: 'audio',
              seq: ++seqRef.current,
              chunk: base64Audio
            })
          }
        },
        (error) => {
          setError('录音错误')
          console.error('Recording error:', error)
        }
      )

      setIsRecording(true)
      setError(null)

    } catch (error) {
      setError('开始录音失败')
      console.error('Start recording error:', error)
    }
  }

  const stopRecording = () => {
    if (recorderRef.current) {
      recorderRef.current.stopRecording()
    }

    if (wsRef.current) {
      wsRef.current.send({ type: 'stop' })
    }

    setIsRecording(false)
  }

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }

  const handleBargeInCancel = async (requestId: string) => {
    try {
      // 立即停止播放
      if (playerRef.current) {
        playerRef.current.stop()
        setIsPlaying(false)
      }

      // 发送取消请求到后端
      await fetch('/api/chat/cancel', {
        method: 'POST',
        headers: {
          'X-Request-ID': requestId,
          'Content-Type': 'application/json'
        }
      })

      console.log(`Cancelled request: ${requestId}`)
    } catch (error) {
      console.error('Failed to cancel request:', error)
    }
  }

  const toggleMute = () => {
    const newVolume = volume > 0 ? 0 : 1
    setVolume(newVolume)
    if (playerRef.current) {
      playerRef.current.setVolume(newVolume)
    }
  }

  const cleanup = () => {
    if (recorderRef.current) {
      recorderRef.current.cleanup()
    }
    if (playerRef.current) {
      playerRef.current.cleanup()
    }
    if (wsRef.current) {
      wsRef.current.close()
    }
    if (bargeInRef.current) {
      bargeInRef.current.cleanup()
    }
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
