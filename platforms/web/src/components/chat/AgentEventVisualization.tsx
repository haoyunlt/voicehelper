'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Button } from '@/components/ui/button'
import { 
  Brain, 
  Search, 
  Wrench, 
  CheckCircle, 
  AlertCircle, 
  Clock,
  ChevronDown,
  ChevronRight,
  Copy
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface AgentEvent {
  id: string
  type: 'plan' | 'step' | 'tool_result' | 'summary' | 'error'
  timestamp: Date
  title: string
  description?: string
  data?: any
  status: 'pending' | 'running' | 'completed' | 'failed'
  duration?: number
  metadata?: {
    step_index?: number
    total_steps?: number
    tool_name?: string
    confidence?: number
  }
}

interface AgentEventVisualizationProps {
  events: AgentEvent[]
  className?: string
  showDetails?: boolean
  onEventClick?: (event: AgentEvent) => void
}

export default function AgentEventVisualization({
  events,
  className,
  showDetails = true,
  onEventClick
}: AgentEventVisualizationProps) {
  const [expandedEvents, setExpandedEvents] = useState<Set<string>>(new Set())
  const [filter, setFilter] = useState<string>('all')

  // 切换事件展开状态
  const toggleExpanded = (eventId: string) => {
    setExpandedEvents(prev => {
      const newSet = new Set(prev)
      if (newSet.has(eventId)) {
        newSet.delete(eventId)
      } else {
        newSet.add(eventId)
      }
      return newSet
    })
  }

  // 获取事件图标
  const getEventIcon = (type: string, status: string) => {
    const iconClass = cn(
      "w-4 h-4",
      status === 'running' && "animate-spin",
      status === 'completed' && "text-green-600",
      status === 'failed' && "text-red-600",
      status === 'pending' && "text-gray-400"
    )

    switch (type) {
      case 'plan':
        return <Brain className={iconClass} />
      case 'step':
        return <Clock className={iconClass} />
      case 'tool_result':
        return <Wrench className={iconClass} />
      case 'summary':
        return <CheckCircle className={iconClass} />
      case 'error':
        return <AlertCircle className={cn(iconClass, "text-red-600")} />
      default:
        return <Clock className={iconClass} />
    }
  }

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-500'
      case 'running': return 'bg-blue-500'
      case 'failed': return 'bg-red-500'
      case 'pending': return 'bg-gray-400'
      default: return 'bg-gray-400'
    }
  }

  // 获取状态文本
  const getStatusText = (status: string) => {
    switch (status) {
      case 'completed': return '已完成'
      case 'running': return '执行中'
      case 'failed': return '失败'
      case 'pending': return '等待中'
      default: return '未知'
    }
  }

  // 复制事件数据
  const copyEventData = (event: AgentEvent) => {
    const data = JSON.stringify(event, null, 2)
    navigator.clipboard.writeText(data)
  }

  // 过滤事件
  const filteredEvents = events.filter(event => {
    if (filter === 'all') return true
    return event.type === filter
  })

  // 计算总体进度
  const totalProgress = events.length > 0 
    ? (events.filter(e => e.status === 'completed').length / events.length) * 100 
    : 0

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Agent 执行过程
          </CardTitle>
          
          <div className="flex items-center gap-2">
            {/* 过滤器 */}
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="text-sm border rounded px-2 py-1"
            >
              <option value="all">全部</option>
              <option value="plan">规划</option>
              <option value="step">步骤</option>
              <option value="tool_result">工具</option>
              <option value="summary">总结</option>
            </select>
          </div>
        </div>
        
        {/* 总体进度 */}
        {events.length > 0 && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>总体进度</span>
              <span>{Math.round(totalProgress)}%</span>
            </div>
            <Progress value={totalProgress} className="h-2" />
          </div>
        )}
      </CardHeader>
      
      <CardContent className="space-y-3">
        {filteredEvents.length === 0 ? (
          <div className="text-center text-muted-foreground py-8">
            <Brain className="w-12 h-12 mx-auto mb-2 opacity-50" />
            <p>暂无Agent执行事件</p>
          </div>
        ) : (
          <div className="space-y-2">
            {filteredEvents.map((event, index) => (
              <div
                key={event.id}
                className="border rounded-lg p-3 hover:bg-muted/50 transition-colors"
              >
                {/* 事件头部 */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {/* 时间线连接线 */}
                    <div className="relative">
                      <div
                        className={cn(
                          "w-3 h-3 rounded-full border-2 border-white",
                          getStatusColor(event.status)
                        )}
                      />
                      {index < filteredEvents.length - 1 && (
                        <div className="absolute top-3 left-1/2 transform -translate-x-1/2 w-px h-6 bg-border" />
                      )}
                    </div>
                    
                    {/* 事件图标和标题 */}
                    <div className="flex items-center gap-2">
                      {getEventIcon(event.type, event.status)}
                      <span className="font-medium">{event.title}</span>
                    </div>
                    
                    {/* 状态和元数据 */}
                    <div className="flex items-center gap-2">
                      <Badge variant={event.status === 'completed' ? 'default' : 
                                   event.status === 'failed' ? 'destructive' : 'secondary'}>
                        {getStatusText(event.status)}
                      </Badge>
                      
                      {event.duration && (
                        <Badge variant="outline">
                          {event.duration}ms
                        </Badge>
                      )}
                      
                      {event.metadata?.step_index && event.metadata?.total_steps && (
                        <Badge variant="outline">
                          {event.metadata.step_index}/{event.metadata.total_steps}
                        </Badge>
                      )}
                      
                      {event.metadata?.confidence && (
                        <Badge variant="outline">
                          置信度: {(event.metadata.confidence * 100).toFixed(0)}%
                        </Badge>
                      )}
                    </div>
                  </div>
                  
                  {/* 操作按钮 */}
                  <div className="flex items-center gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyEventData(event)}
                      className="h-6 w-6 p-0"
                    >
                      <Copy className="w-3 h-3" />
                    </Button>
                    
                    {showDetails && (event.description || event.data) && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => toggleExpanded(event.id)}
                        className="h-6 w-6 p-0"
                      >
                        {expandedEvents.has(event.id) ? (
                          <ChevronDown className="w-3 h-3" />
                        ) : (
                          <ChevronRight className="w-3 h-3" />
                        )}
                      </Button>
                    )}
                  </div>
                </div>
                
                {/* 时间戳 */}
                <div className="ml-6 text-xs text-muted-foreground">
                  {event.timestamp.toLocaleTimeString()}
                </div>
                
                {/* 展开的详细信息 */}
                {expandedEvents.has(event.id) && (
                  <div className="ml-6 mt-2 space-y-2">
                    {event.description && (
                      <div className="text-sm text-muted-foreground">
                        {event.description}
                      </div>
                    )}
                    
                    {event.data && (
                      <div className="bg-muted rounded p-2">
                        <pre className="text-xs overflow-x-auto">
                          {JSON.stringify(event.data, null, 2)}
                        </pre>
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// 工具函数：创建Agent事件
export const createAgentEvent = (
  type: AgentEvent['type'],
  title: string,
  options: Partial<AgentEvent> = {}
): AgentEvent => {
  return {
    id: `event_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    type,
    title,
    timestamp: new Date(),
    status: 'pending',
    ...options
  }
}

// 工具函数：更新事件状态
export const updateEventStatus = (
  events: AgentEvent[],
  eventId: string,
  status: AgentEvent['status'],
  duration?: number
): AgentEvent[] => {
  return events.map(event => 
    event.id === eventId 
      ? { ...event, status, duration }
      : event
  )
}
