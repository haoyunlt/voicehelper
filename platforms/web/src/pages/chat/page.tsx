'use client'

import { useState, useEffect } from 'react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { MessageSquare, Mic, Settings, Activity } from 'lucide-react'

import StreamingChat from '@/components/chat/StreamingChat'
import VoiceChat from '@/components/voice/VoiceChat'

interface Reference {
  id: string
  source: string
  title?: string
  content?: string
  score?: number
}

export default function ChatPage() {
  const [conversationId] = useState(() => `conv_${Date.now()}`)
  const [activeTab, setActiveTab] = useState('text')
  const [references, setReferences] = useState<Reference[]>([])
  const [transcript, setTranscript] = useState('')
  const [isVoiceActive, setIsVoiceActive] = useState(false)

  // 语音转写处理
  const handleVoiceTranscript = (text: string, isFinal: boolean) => {
    setTranscript(text)
    setIsVoiceActive(!isFinal)
  }

  // 语音响应处理
  const handleVoiceResponse = (text: string) => {
    // 可以在这里处理语音响应，比如显示实时转写
    console.log('Voice response:', text)
  }

  // 语音引用处理
  const handleVoiceReferences = (refs: Reference[]) => {
    setReferences(refs)
  }

  // 渲染引用面板
  const renderReferencesPanel = () => {
    if (references.length === 0) return null

    return (
      <Card className="mt-4">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Activity className="h-4 w-4" />
            参考资料 ({references.length})
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          {references.map((ref, index) => (
            <div
              key={ref.id}
              className="p-3 rounded-lg border bg-muted/50 hover:bg-muted/70 transition-colors"
            >
              <div className="flex items-start justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <Badge variant="secondary" className="text-xs">
                    {index + 1}
                  </Badge>
                  {ref.title && (
                    <span className="font-medium text-sm truncate">
                      {ref.title}
                    </span>
                  )}
                  {ref.score && (
                    <Badge variant="outline" className="text-xs">
                      {(ref.score * 100).toFixed(0)}%
                    </Badge>
                  )}
                </div>
              </div>
              
              <div className="text-xs text-muted-foreground mb-1 truncate">
                {ref.source}
              </div>
              
              {ref.content && (
                <div className="text-sm line-clamp-2">
                  {ref.content}
                </div>
              )}
            </div>
          ))}
        </CardContent>
      </Card>
    )
  }

  // 渲染实时转写
  const renderTranscriptPanel = () => {
    if (!transcript && !isVoiceActive) return null

    return (
      <Card className="mt-4">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm flex items-center gap-2">
            <Mic className="h-4 w-4" />
            实时转写
            {isVoiceActive && (
              <div className="flex items-center gap-1">
                <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                <span className="text-xs text-muted-foreground">识别中</span>
              </div>
            )}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-3 rounded-lg bg-muted/50 text-sm">
            {transcript || '等待语音输入...'}
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="container mx-auto p-4 h-screen flex flex-col">
      <div className="mb-4">
        <h1 className="text-2xl font-bold mb-2">语音增强聊天助手</h1>
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <span>会话ID: {conversationId}</span>
          <Badge variant="outline">v1.0.0</Badge>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* 主聊天区域 */}
        <div className="lg:col-span-3">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="text" className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                文本聊天
              </TabsTrigger>
              <TabsTrigger value="voice" className="flex items-center gap-2">
                <Mic className="h-4 w-4" />
                语音对话
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="text" className="h-full mt-4">
              <Card className="h-full">
                <StreamingChat
                  conversationId={conversationId}
                  onVoiceTranscript={handleVoiceTranscript}
                  onVoiceResponse={handleVoiceResponse}
                  onVoiceReferences={handleVoiceReferences}
                  className="h-full"
                />
              </Card>
            </TabsContent>
            
            <TabsContent value="voice" className="h-full mt-4">
              <Card className="h-full">
                <VoiceChat
                  conversationId={conversationId}
                  onTranscript={handleVoiceTranscript}
                  onResponse={handleVoiceResponse}
                  onReferences={handleVoiceReferences}
                  className="h-full"
                />
              </Card>
            </TabsContent>
          </Tabs>
        </div>

        {/* 侧边栏 */}
        <div className="space-y-4">
          {/* 实时转写面板 */}
          {renderTranscriptPanel()}
          
          {/* 引用面板 */}
          {renderReferencesPanel()}
          
          {/* 设置面板 */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Settings className="h-4 w-4" />
                设置
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span>语音识别</span>
                <Badge variant="secondary">中文</Badge>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span>语音合成</span>
                <Badge variant="secondary">标准</Badge>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span>模型</span>
                <Badge variant="secondary">豆包</Badge>
              </div>
              <div className="flex items-center justify-between text-sm">
                <span>RAG检索</span>
                <Badge variant="secondary">BGE+FAISS</Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}