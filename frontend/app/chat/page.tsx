'use client'

import { useState, useRef, useEffect } from 'react'
import { Send, Upload, Paperclip } from 'lucide-react'
import { Button } from '../../components/ui/button'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import VoiceInput from '../../components/chat/VoiceInput'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  references?: Reference[]
  modality?: 'text' | 'audio' | 'asr' | 'tts'
}

interface Reference {
  chunk_id: string
  source: string
  score: number
  content?: string
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId] = useState(() => `conv_${Date.now()}`)
  const [currentAssistantMessage, setCurrentAssistantMessage] = useState<Message | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // 语音转写处理
  const handleVoiceTranscript = (text: string, isFinal: boolean) => {
    if (isFinal) {
      // 最终转写结果，创建用户消息
      const userMessage: Message = {
        id: Date.now().toString(),
        role: 'user',
        content: text,
        timestamp: new Date(),
        modality: 'asr'
      }
      setMessages(prev => [...prev, userMessage])
      
      // 创建助手消息占位符
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: '',
        timestamp: new Date(),
        modality: 'tts'
      }
      setMessages(prev => [...prev, assistantMessage])
      setCurrentAssistantMessage(assistantMessage)
    }
  }

  // 语音响应处理
  const handleVoiceResponse = (text: string) => {
    if (currentAssistantMessage) {
      setMessages(prev => prev.map(msg => 
        msg.id === currentAssistantMessage.id 
          ? { ...msg, content: msg.content + text }
          : msg
      ))
    }
  }

  // 语音引用处理
  const handleVoiceReferences = (refs: Reference[]) => {
    if (currentAssistantMessage) {
      setMessages(prev => prev.map(msg => 
        msg.id === currentAssistantMessage.id 
          ? { ...msg, references: refs }
          : msg
      ))
    }
  }

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
      timestamp: new Date(),
      modality: 'text'
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // 创建助手消息
    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      modality: 'text'
    }

    setMessages(prev => [...prev, assistantMessage])

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          conversation_id: conversationId,
          messages: [
            ...messages.map(m => ({ role: m.role, content: m.content })),
            { role: 'user', content: input }
          ]
        })
      })

      if (!response.ok) {
        throw new Error('Failed to send message')
      }

      const reader = response.body?.getReader()
      if (!reader) throw new Error('No reader available')

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = new TextDecoder().decode(value)
        const lines = chunk.split('\n')

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              if (data.type === 'delta' && data.content) {
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessage.id 
                    ? { ...msg, content: msg.content + data.content }
                    : msg
                ))
              } else if (data.type === 'refs' && data.refs) {
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessage.id 
                    ? { ...msg, references: data.refs }
                    : msg
                ))
              }
            } catch (e) {
              // 忽略解析错误
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error)
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessage.id 
          ? { ...msg, content: '抱歉，发生了错误，请稍后重试。' }
          : msg
      ))
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900">智能助手</h1>
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">
              <Upload className="w-4 h-4 mr-2" />
              上传文档
            </Button>
            <VoiceInput
              conversationId={conversationId}
              onTranscript={handleVoiceTranscript}
              onResponse={handleVoiceResponse}
              onReferences={handleVoiceReferences}
              disabled={isLoading}
            />
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4" data-testid="chat-container">
        <div className="max-w-4xl mx-auto space-y-6">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="text-gray-500 mb-4">
                <svg className="w-12 h-12 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">开始对话</h3>
              <p className="text-gray-500">向我提问任何问题，我会基于知识库为您提供准确的答案。</p>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div 
                className={`max-w-3xl ${message.role === 'user' ? 'bg-blue-600 text-white' : 'bg-white'} rounded-lg px-4 py-3 shadow-sm`}
                data-testid={message.role === 'user' ? 'user-message' : 'assistant-message'}
              >
                {/* 消息类型指示器 */}
                {message.modality && (
                  <div className="text-xs opacity-60 mb-1">
                    {message.modality === 'asr' && '🎤 语音输入'}
                    {message.modality === 'tts' && '🔊 语音回复'}
                    {message.modality === 'text' && '💬 文本'}
                  </div>
                )}
                <div className="prose prose-sm max-w-none">
                  {message.role === 'user' ? (
                    <p className="text-white m-0">{message.content}</p>
                  ) : (
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      className="text-gray-900"
                    >
                      {message.content || '正在思考...'}
                    </ReactMarkdown>
                  )}
                </div>
                
                {/* References */}
                {message.references && message.references.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-xs text-gray-500 mb-2">参考来源：</p>
                    <div className="space-y-1">
                      {message.references.map((ref, index) => (
                        <div key={ref.chunk_id} className="text-xs text-gray-600 bg-gray-50 rounded px-2 py-1">
                          <span className="font-medium">[{index + 1}]</span> {ref.source} (相似度: {(ref.score * 100).toFixed(1)}%)
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                
                <div className="text-xs text-gray-400 mt-2">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="bg-white border-t border-gray-200 px-6 py-4">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end space-x-3">
            <div className="flex-1">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="输入您的问题..."
                className="w-full resize-none border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={1}
                style={{ minHeight: '40px', maxHeight: '120px' }}
                data-testid="message-input"
              />
            </div>
            <Button 
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className="shrink-0"
              data-testid="send-button"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
