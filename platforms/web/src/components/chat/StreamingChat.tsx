'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Send, Mic, MicOff, Volume2, VolumeX, Copy, ExternalLink } from 'lucide-react'
import { cn } from '@/lib/utils'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  modality?: 'text' | 'asr' | 'tts'
  references?: Reference[]
  isStreaming?: boolean
}

interface Reference {
  id: string
  source: string
  title?: string
  content?: string
  score?: number
}

interface StreamingChatProps {
  conversationId: string
  onVoiceTranscript?: (text: string, isFinal: boolean) => void
  onVoiceResponse?: (text: string) => void
  onVoiceReferences?: (refs: Reference[]) => void
  className?: string
}

export default function StreamingChat({
  conversationId,
  onVoiceTranscript,
  onVoiceResponse,
  onVoiceReferences,
  className
}: StreamingChatProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState<Message | null>(null)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected' | 'reconnecting'>('disconnected')
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const eventSourceRef = useRef<EventSource | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const requestIdRef = useRef<string | null>(null)

  // 滚动到底部
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, currentStreamingMessage, scrollToBottom])

  // 生成请求ID
  const generateRequestId = useCallback(() => {
    return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }, [])

  // 建立SSE连接
  const connectSSE = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    // 清除重连定时器
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    const token = localStorage.getItem('token')
    if (!token) {
      console.error('No authentication token found')
      setConnectionStatus('disconnected')
      return
    }

    setConnectionStatus('connecting')
    
    const url = new URL('/api/v1/chat/stream', window.location.origin)
    url.searchParams.set('token', token)
    
    // 添加重连标识
    if (reconnectAttempts > 0) {
      url.searchParams.set('reconnect', 'true')
      url.searchParams.set('attempt', reconnectAttempts.toString())
    }

    const eventSource = new EventSource(url.toString())
    eventSourceRef.current = eventSource

    eventSource.onopen = () => {
      console.log('SSE connection established')
      setIsConnected(true)
      setConnectionStatus('connected')
      setReconnectAttempts(0)
    }

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        handleSSEMessage(data)
      } catch (error) {
        console.error('Failed to parse SSE message:', error)
      }
    }

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error)
      setIsConnected(false)
      setConnectionStatus('disconnected')
      
      // 智能重连逻辑
      if (reconnectAttempts < 5) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000) // 指数退避，最大30秒
        setConnectionStatus('reconnecting')
        
        reconnectTimeoutRef.current = setTimeout(() => {
          setReconnectAttempts(prev => prev + 1)
          connectSSE()
        }, delay)
      } else {
        console.error('Max reconnection attempts reached')
        setConnectionStatus('disconnected')
      }
    }

    return () => {
      eventSource.close()
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [reconnectAttempts])

  // 处理SSE消息
  const handleSSEMessage = useCallback((data: any) => {
    switch (data.type) {
      case 'connected':
        console.log('SSE stream connected:', data.stream_id)
        break
        
      case 'llm_delta':
        if (currentStreamingMessage) {
          setCurrentStreamingMessage(prev => prev ? {
            ...prev,
            content: prev.content + (data.delta || ''),
            timestamp: new Date()
          } : null)
        }
        break
        
      case 'references':
        if (currentStreamingMessage && data.references) {
          setCurrentStreamingMessage(prev => prev ? {
            ...prev,
            references: data.references
          } : null)
          onVoiceReferences?.(data.references)
        }
        break
        
      case 'llm_done':
        if (currentStreamingMessage) {
          setMessages(prev => [...prev, {
            ...currentStreamingMessage,
            isStreaming: false
          }])
          setCurrentStreamingMessage(null)
        }
        setIsLoading(false)
        break
        
      case 'error':
        console.error('SSE error:', data)
        setIsLoading(false)
        setCurrentStreamingMessage(null)
        break
        
      case 'keep_alive':
        // 心跳消息，无需处理
        break
        
      default:
        console.log('Unknown SSE message type:', data.type)
    }
  }, [currentStreamingMessage, onVoiceReferences])

  // 初始化连接
  useEffect(() => {
    const cleanup = connectSSE()
    return cleanup
  }, [connectSSE])

  // 发送消息
  const sendMessage = async () => {
    if (!input.trim() || isLoading || !isConnected) return

    // 生成幂等请求ID
    const requestId = generateRequestId()
    requestIdRef.current = requestId

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
      modality: 'text'
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // 创建流式响应消息占位符
    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      modality: 'text',
      isStreaming: true
    }
    setCurrentStreamingMessage(assistantMessage)

    try {
      // 创建新的AbortController
      abortControllerRef.current = new AbortController()

      const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`,
          'X-Request-ID': requestId, // 幂等性请求ID
        },
        body: JSON.stringify({
          message: userMessage.content,
          conversation_id: conversationId,
          request_id: requestId, // 幂等性请求ID
          stream_id: eventSourceRef.current ? 'current_stream' : undefined,
          context: {
            modality: 'text',
            timestamp: userMessage.timestamp.toISOString()
          }
        }),
        signal: abortControllerRef.current.signal
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      console.log('Chat request submitted:', result)

    } catch (error: any) {
      if (error.name !== 'AbortError') {
        console.error('Failed to send message:', error)
        setIsLoading(false)
        setCurrentStreamingMessage(null)
        
        // 显示错误消息
        const errorMessage: Message = {
          id: (Date.now() + 2).toString(),
          role: 'assistant',
          content: '抱歉，发送消息时出现错误，请重试。',
          timestamp: new Date(),
          modality: 'text'
        }
        setMessages(prev => [...prev, errorMessage])
      }
    }
  }

  // 取消当前请求
  const cancelRequest = async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    
    // 调用后端取消接口
    if (requestIdRef.current) {
      try {
        await fetch('/api/v1/chat/cancel', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${localStorage.getItem('token')}`,
            'X-Request-ID': requestIdRef.current,
          },
          body: JSON.stringify({
            request_id: requestIdRef.current,
            conversation_id: conversationId
          })
        })
      } catch (error) {
        console.error('Failed to cancel request:', error)
      }
    }
    
    setIsLoading(false)
    setCurrentStreamingMessage(null)
    requestIdRef.current = null
  }

  // 处理键盘事件
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  // 复制消息内容
  const copyMessage = (content: string) => {
    navigator.clipboard.writeText(content)
  }

  // 渲染引用
  const renderReferences = (references: Reference[]) => {
    if (!references || references.length === 0) return null

    return (
      <div className="mt-3 space-y-2">
        <div className="text-sm font-medium text-muted-foreground">参考资料：</div>
        <div className="space-y-1">
          {references.map((ref, index) => (
            <Card key={ref.id} className="p-2 bg-muted/50">
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <Badge variant="secondary" className="text-xs">
                      {index + 1}
                    </Badge>
                    {ref.title && (
                      <span className="text-sm font-medium truncate">
                        {ref.title}
                      </span>
                    )}
                    {ref.score && (
                      <Badge variant="outline" className="text-xs">
                        {(ref.score * 100).toFixed(0)}%
                      </Badge>
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground truncate">
                    {ref.source}
                  </div>
                  {ref.content && (
                    <div className="text-xs mt-1 line-clamp-2">
                      {ref.content}
                    </div>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => window.open(ref.source, '_blank')}
                >
                  <ExternalLink className="h-3 w-3" />
                </Button>
              </div>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  // 渲染消息
  const renderMessage = (message: Message) => {
    const isUser = message.role === 'user'
    
    return (
      <div
        key={message.id}
        className={cn(
          "flex gap-3 p-4",
          isUser ? "justify-end" : "justify-start"
        )}
      >
        <div
          className={cn(
            "max-w-[80%] rounded-lg p-3 relative group",
            isUser
              ? "bg-primary text-primary-foreground"
              : "bg-muted"
          )}
        >
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1">
              <div className="whitespace-pre-wrap break-words">
                {message.content}
              </div>
              
              {message.references && renderReferences(message.references)}
              
              <div className="flex items-center gap-2 mt-2 text-xs opacity-70">
                <span>{message.timestamp.toLocaleTimeString()}</span>
                {message.modality && (
                  <Badge variant="outline" className="text-xs">
                    {message.modality}
                  </Badge>
                )}
                {message.isStreaming && (
                  <div className="flex items-center gap-1">
                    <div className="w-1 h-1 bg-current rounded-full animate-pulse" />
                    <span>生成中...</span>
                  </div>
                )}
              </div>
            </div>
            
            <Button
              variant="ghost"
              size="sm"
              className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 transition-opacity"
              onClick={() => copyMessage(message.content)}
            >
              <Copy className="h-3 w-3" />
            </Button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className={cn("flex flex-col h-full", className)}>
      {/* 连接状态指示器 */}
      <div className="flex items-center justify-between p-4 border-b">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "w-2 h-2 rounded-full",
              connectionStatus === 'connected' ? "bg-green-500" : 
              connectionStatus === 'connecting' ? "bg-yellow-500 animate-pulse" :
              connectionStatus === 'reconnecting' ? "bg-orange-500 animate-pulse" :
              "bg-red-500"
            )}
          />
          <span className="text-sm text-muted-foreground">
            {connectionStatus === 'connected' ? "已连接" :
             connectionStatus === 'connecting' ? "连接中..." :
             connectionStatus === 'reconnecting' ? `重连中... (${reconnectAttempts}/5)` :
             "连接断开"}
          </span>
          
          {connectionStatus === 'disconnected' && reconnectAttempts >= 5 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setReconnectAttempts(0)
                connectSSE()
              }}
              className="text-xs"
            >
              重新连接
            </Button>
          )}
        </div>
        
        {isLoading && (
          <Button
            variant="outline"
            size="sm"
            onClick={cancelRequest}
          >
            取消
          </Button>
        )}
      </div>

      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto">
        {messages.map(renderMessage)}
        {currentStreamingMessage && renderMessage(currentStreamingMessage)}
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="p-4 border-t">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="输入消息..."
            disabled={isLoading || !isConnected}
            className="flex-1"
          />
          <Button
            onClick={sendMessage}
            disabled={!input.trim() || isLoading || !isConnected}
            size="icon"
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
