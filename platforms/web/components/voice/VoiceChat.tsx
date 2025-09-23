'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX, 
  Phone, 
  PhoneOff,
  Wifi,
  WifiOff,
  Activity
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface VoiceMessage {
  id: string
  type: 'user' | 'assistant'
  text: string
  timestamp: Date
  confidence?: number
  isFinal?: boolean
  audioUrl?: string
}

interface VoiceChatProps {
  conversationId: string
  onTranscript?: (text: string, isFinal: boolean) => void
  onResponse?: (text: string) => void
  onReferences?: (refs: any[]) => void
  className?: string
}

interface AudioWorkletNode extends AudioNode {
  port: MessagePort
}

export default function VoiceChat({
  conversationId,
  onTranscript,
  onResponse,
  onReferences,
  className
}: VoiceChatProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [connectionQuality, setConnectionQuality] = useState<'good' | 'fair' | 'poor'>('good')
  const [volume, setVolume] = useState(0)
  const [messages, setMessages] = useState<VoiceMessage[]>([])
  const [currentTranscript, setCurrentTranscript] = useState('')
  const [latencyStats, setLatencyStats] = useState({
    asr: 0,
    llm: 0,
    tts: 0,
    e2e: 0
  })

  // Refs
  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const workletNodeRef = useRef<AudioWorkletNode | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioQueueRef = useRef<ArrayBuffer[]>([])
  const playbackNodeRef = useRef<AudioBufferSourceNode | null>(null)
  const sequenceNumberRef = useRef(0)
  const latencyTrackerRef = useRef<Map<string, number>>(new Map())

  // 初始化音频上下文
  const initAudioContext = useCallback(async () => {
    try {
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)({
          sampleRate: 16000
        })
      }

      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume()
      }

      // 加载AudioWorklet
      if (!workletNodeRef.current) {
        try {
          await audioContextRef.current.audioWorklet.addModule('/audio/voice-processor.js')
          workletNodeRef.current = new AudioWorkletNode(
            audioContextRef.current,
            'voice-processor'
          ) as AudioWorkletNode

          // 监听处理后的音频数据
          workletNodeRef.current.port.onmessage = (event) => {
            const { type, data } = event.data
            
            if (type === 'audio-data') {
              sendAudioFrame(data)
            } else if (type === 'volume') {
              setVolume(data)
            }
          }
        } catch (error) {
          console.warn('AudioWorklet not supported, using fallback')
          // 降级到ScriptProcessorNode
        }
      }

      return true
    } catch (error) {
      console.error('Failed to initialize audio context:', error)
      return false
    }
  }, [])

  // 连接WebSocket
  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }

    const token = localStorage.getItem('token')
    if (!token) {
      console.error('No authentication token found')
      return
    }

    const wsUrl = new URL('/api/v1/voice/stream', window.location.origin)
    wsUrl.protocol = wsUrl.protocol === 'https:' ? 'wss:' : 'ws:'
    wsUrl.searchParams.set('token', token)

    const ws = new WebSocket(wsUrl.toString())
    wsRef.current = ws

    ws.onopen = () => {
      console.log('WebSocket connected')
      setIsConnected(true)
      
      // 发送初始化消息
      ws.send(JSON.stringify({
        type: 'start',
        codec: 'pcm16',
        sample_rate: 16000,
        conversation_id: conversationId,
        lang: 'zh-CN',
        vad: {
          enable: true,
          min_speech_ms: 200,
          min_silence_ms: 250
        }
      }))
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        handleWebSocketMessage(data)
      } catch (error) {
        // 可能是二进制音频数据
        if (event.data instanceof ArrayBuffer) {
          handleAudioData(event.data)
        } else {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
    }

    ws.onerror = (error) => {
      console.error('WebSocket error:', error)
      setIsConnected(false)
    }

    ws.onclose = () => {
      console.log('WebSocket closed')
      setIsConnected(false)
      
      // 重连逻辑
      setTimeout(() => {
        if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
          connectWebSocket()
        }
      }, 3000)
    }
  }, [conversationId])

  // 处理WebSocket消息
  const handleWebSocketMessage = useCallback((data: any) => {
    const now = Date.now()
    
    switch (data.type) {
      case 'connected':
        console.log('Voice session connected:', data.session_id)
        break
        
      case 'asr_partial':
        setCurrentTranscript(data.text || '')
        onTranscript?.(data.text || '', false)
        break
        
      case 'asr_final':
        const userMessage: VoiceMessage = {
          id: Date.now().toString(),
          type: 'user',
          text: data.text || '',
          timestamp: new Date(),
          confidence: data.confidence,
          isFinal: true
        }
        
        setMessages(prev => [...prev, userMessage])
        setCurrentTranscript('')
        onTranscript?.(data.text || '', true)
        
        // 记录ASR延迟
        if (data.trace_id && latencyTrackerRef.current.has(data.trace_id)) {
          const startTime = latencyTrackerRef.current.get(data.trace_id)!
          setLatencyStats(prev => ({ ...prev, asr: now - startTime }))
        }
        break
        
      case 'llm_delta':
        onResponse?.(data.delta || '')
        break
        
      case 'tts_chunk':
        // 处理TTS音频块
        if (data.audio) {
          // TODO: 实现音频播放队列
          console.log('Received TTS audio chunk:', data.audio.length)
        }
        break
        
      case 'references':
        onReferences?.(data.references || [])
        break
        
      case 'throttle':
        console.warn('Connection throttled:', data.reason)
        setConnectionQuality('poor')
        break
        
      case 'heartbeat':
        // 心跳响应，更新连接质量
        if (data.latency_ms) {
          if (data.latency_ms < 100) {
            setConnectionQuality('good')
          } else if (data.latency_ms < 300) {
            setConnectionQuality('fair')
          } else {
            setConnectionQuality('poor')
          }
        }
        break
        
      case 'error':
        console.error('Voice service error:', data)
        break
        
      default:
        console.log('Unknown message type:', data.type)
    }
  }, [onTranscript, onResponse, onReferences])

  // 发送音频帧
  const sendAudioFrame = useCallback((audioData: Float32Array) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      return
    }

    // 转换为PCM16
    const pcm16 = new Int16Array(audioData.length)
    for (let i = 0; i < audioData.length; i++) {
      pcm16[i] = Math.max(-32768, Math.min(32767, (audioData[i] || 0) * 32767))
    }

    // 构建二进制帧头部（20字节）
    const header = new ArrayBuffer(20)
    const headerView = new DataView(header)
    
    headerView.setUint32(0, sequenceNumberRef.current++, true) // sequence
    headerView.setUint32(4, 16000, true) // sample_rate
    headerView.setUint8(8, 1) // channels
    headerView.setUint16(9, pcm16.length * 2, true) // frame_size
    headerView.setBigUint64(12, BigInt(Date.now()), true) // timestamp

    // 合并头部和音频数据
    const frame = new ArrayBuffer(header.byteLength + pcm16.byteLength)
    new Uint8Array(frame).set(new Uint8Array(header), 0)
    new Uint8Array(frame).set(new Uint8Array(pcm16.buffer), header.byteLength)

    wsRef.current.send(frame)
    
    // 记录发送时间用于延迟计算
    latencyTrackerRef.current.set(`frame_${sequenceNumberRef.current}`, Date.now())
  }, [])

  // 处理接收到的音频数据
  const handleAudioData = useCallback((audioBuffer: ArrayBuffer) => {
    audioQueueRef.current.push(audioBuffer)
    
    if (!isPlaying) {
      playNextAudio()
    }
  }, [isPlaying])

  // 播放音频队列
  const playNextAudio = useCallback(async () => {
    if (audioQueueRef.current.length === 0 || !audioContextRef.current) {
      setIsPlaying(false)
      return
    }

    setIsPlaying(true)
    
    try {
      const audioBuffer = audioQueueRef.current.shift()!
      const decodedBuffer = await audioContextRef.current.decodeAudioData(audioBuffer)
      
      const source = audioContextRef.current.createBufferSource()
      source.buffer = decodedBuffer
      source.connect(audioContextRef.current.destination)
      
      source.onended = () => {
        playNextAudio()
      }
      
      source.start()
      playbackNodeRef.current = source
      
    } catch (error) {
      console.error('Failed to play audio:', error)
      setIsPlaying(false)
    }
  }, [])

  // 开始录音
  const startRecording = useCallback(async () => {
    try {
      const audioInitialized = await initAudioContext()
      if (!audioInitialized) {
        throw new Error('Failed to initialize audio')
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      })

      mediaStreamRef.current = stream

      if (workletNodeRef.current && audioContextRef.current) {
        const source = audioContextRef.current.createMediaStreamSource(stream)
        source.connect(workletNodeRef.current)
        
        // 启动音频处理
        workletNodeRef.current.port.postMessage({ type: 'start' })
      }

      setIsRecording(true)
      
      // 发送开始录音消息
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'start_recording',
          timestamp: Date.now()
        }))
      }

    } catch (error) {
      console.error('Failed to start recording:', error)
    }
  }, [initAudioContext])

  // 停止录音
  const stopRecording = useCallback(() => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop())
      mediaStreamRef.current = null
    }

    if (workletNodeRef.current) {
      workletNodeRef.current.port.postMessage({ type: 'stop' })
    }

    setIsRecording(false)
    setVolume(0)

    // 发送停止录音消息
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'stop_recording',
        timestamp: Date.now()
      }))
    }
  }, [])

  // 切换录音状态
  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording()
    } else {
      startRecording()
    }
  }, [isRecording, startRecording, stopRecording])

  // 停止播放
  const stopPlayback = useCallback(() => {
    if (playbackNodeRef.current) {
      playbackNodeRef.current.stop()
      playbackNodeRef.current = null
    }
    
    audioQueueRef.current = []
    setIsPlaying(false)
  }, [])

  // 初始化
  useEffect(() => {
    connectWebSocket()
    initAudioContext()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop())
      }
      if (audioContextRef.current) {
        audioContextRef.current.close()
      }
    }
  }, [connectWebSocket, initAudioContext])

  // 渲染连接状态
  const renderConnectionStatus = () => {
    const statusConfig = {
      good: { color: 'text-green-500', icon: Wifi, label: '连接良好' },
      fair: { color: 'text-yellow-500', icon: Wifi, label: '连接一般' },
      poor: { color: 'text-red-500', icon: WifiOff, label: '连接较差' }
    }

    const config = statusConfig[connectionQuality]
    const Icon = config.icon

    return (
      <div className="flex items-center gap-2">
        <div
          className={cn(
            "w-2 h-2 rounded-full",
            isConnected ? "bg-green-500" : "bg-red-500"
          )}
        />
        <Icon className={cn("h-4 w-4", config.color)} />
        <span className="text-sm text-muted-foreground">
          {isConnected ? config.label : '未连接'}
        </span>
      </div>
    )
  }

  // 渲染延迟统计
  const renderLatencyStats = () => (
    <div className="grid grid-cols-4 gap-2 text-xs">
      <div className="text-center">
        <div className="font-medium">ASR</div>
        <div className="text-muted-foreground">{latencyStats.asr}ms</div>
      </div>
      <div className="text-center">
        <div className="font-medium">LLM</div>
        <div className="text-muted-foreground">{latencyStats.llm}ms</div>
      </div>
      <div className="text-center">
        <div className="font-medium">TTS</div>
        <div className="text-muted-foreground">{latencyStats.tts}ms</div>
      </div>
      <div className="text-center">
        <div className="font-medium">E2E</div>
        <div className="text-muted-foreground">{latencyStats.e2e}ms</div>
      </div>
    </div>
  )

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* 状态栏 */}
      <div className="p-4 border-b space-y-3">
        <div className="flex items-center justify-between">
          {renderConnectionStatus()}
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-muted-foreground" />
            <span className="text-sm text-muted-foreground">延迟统计</span>
          </div>
        </div>
        
        {renderLatencyStats()}
        
        {/* 音量指示器 */}
        {isRecording && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <span>音量</span>
              <span>{Math.round(volume * 100)}%</span>
            </div>
            <Progress value={volume * 100} className="h-1" />
          </div>
        )}
      </div>

      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={cn(
              "flex gap-3",
              message.type === 'user' ? "justify-end" : "justify-start"
            )}
          >
            <Card
              className={cn(
                "max-w-[80%] p-3",
                message.type === 'user'
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted"
              )}
            >
              <CardContent className="p-0">
                <div className="whitespace-pre-wrap break-words">
                  {message.text}
                </div>
                
                <div className="flex items-center gap-2 mt-2 text-xs opacity-70">
                  <span>{message.timestamp.toLocaleTimeString()}</span>
                  {message.confidence && (
                    <Badge variant="outline" className="text-xs">
                      {Math.round(message.confidence * 100)}%
                    </Badge>
                  )}
                  <Badge variant="outline" className="text-xs">
                    语音
                  </Badge>
                </div>
              </CardContent>
            </Card>
          </div>
        ))}
        
        {/* 实时转写显示 */}
        {currentTranscript && (
          <div className="flex justify-end">
            <Card className="max-w-[80%] p-3 bg-primary/50 text-primary-foreground">
              <CardContent className="p-0">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-current rounded-full animate-pulse" />
                  <span className="text-sm opacity-75">识别中...</span>
                </div>
                <div className="mt-1">{currentTranscript}</div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      {/* 控制按钮 */}
      <div className="p-4 border-t">
        <div className="flex items-center justify-center gap-4">
          <Button
            variant={isRecording ? "destructive" : "default"}
            size="lg"
            className="rounded-full w-16 h-16"
            onClick={toggleRecording}
            disabled={!isConnected}
          >
            {isRecording ? (
              <MicOff className="h-6 w-6" />
            ) : (
              <Mic className="h-6 w-6" />
            )}
          </Button>
          
          <Button
            variant={isPlaying ? "secondary" : "outline"}
            size="lg"
            className="rounded-full w-12 h-12"
            onClick={stopPlayback}
            disabled={!isPlaying}
          >
            {isPlaying ? (
              <VolumeX className="h-5 w-5" />
            ) : (
              <Volume2 className="h-5 w-5" />
            )}
          </Button>
        </div>
        
        <div className="text-center mt-2 text-sm text-muted-foreground">
          {isRecording ? "正在录音..." : "点击开始语音对话"}
        </div>
      </div>
    </div>
  )
}
