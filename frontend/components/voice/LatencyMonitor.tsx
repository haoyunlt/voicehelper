'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'

interface LatencyMetrics {
  capture: number      // 音频采集延迟
  asr: number         // ASR识别延迟
  llm: number         // LLM生成延迟
  tts: number         // TTS合成延迟
  play: number        // 音频播放延迟
  e2e: number         // 端到端延迟
}

interface LatencyMonitorProps {
  className?: string
  onMetricsUpdate?: (metrics: LatencyMetrics) => void
}

export default function LatencyMonitor({
  className,
  onMetricsUpdate
}: LatencyMonitorProps) {
  const [metrics, setMetrics] = useState<LatencyMetrics>({
    capture: 0,
    asr: 0,
    llm: 0,
    tts: 0,
    play: 0,
    e2e: 0
  })

  const [movingAverages, setMovingAverages] = useState<LatencyMetrics>({
    capture: 0,
    asr: 0,
    llm: 0,
    tts: 0,
    play: 0,
    e2e: 0
  })

  const [isMonitoring, setIsMonitoring] = useState(false)
  const [history, setHistory] = useState<LatencyMetrics[]>([])

  // 移动平均窗口大小
  const WINDOW_SIZE = 10

  // 延迟阈值配置
  const THRESHOLDS = {
    capture: { good: 50, warning: 100, critical: 200 },
    asr: { good: 300, warning: 600, critical: 1000 },
    llm: { good: 800, warning: 1500, critical: 3000 },
    tts: { good: 500, warning: 1000, critical: 2000 },
    play: { good: 100, warning: 200, critical: 500 },
    e2e: { good: 2000, warning: 3500, critical: 5000 }
  }

  // 更新指标
  const updateMetrics = useCallback((newMetrics: Partial<LatencyMetrics>) => {
    setMetrics(prev => {
      const updated = { ...prev, ...newMetrics }
      
      // 更新历史记录
      setHistory(prevHistory => {
        const newHistory = [...prevHistory, updated].slice(-WINDOW_SIZE)
        
        // 计算移动平均
        const averages = Object.keys(updated).reduce((acc, key) => {
          const values = newHistory.map(h => h[key as keyof LatencyMetrics])
          acc[key as keyof LatencyMetrics] = values.reduce((sum, val) => sum + val, 0) / values.length
          return acc
        }, {} as LatencyMetrics)
        
        setMovingAverages(averages)
        return newHistory
      })
      
      onMetricsUpdate?.(updated)
      return updated
    })
  }, [onMetricsUpdate])

  // 获取延迟状态
  const getLatencyStatus = (value: number, type: keyof LatencyMetrics) => {
    const threshold = THRESHOLDS[type]
    if (value <= threshold.good) return 'good'
    if (value <= threshold.warning) return 'warning'
    return 'critical'
  }

  // 获取状态颜色
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good': return 'bg-green-500'
      case 'warning': return 'bg-yellow-500'
      case 'critical': return 'bg-red-500'
      default: return 'bg-gray-500'
    }
  }

  // 获取状态文本
  const getStatusText = (status: string) => {
    switch (status) {
      case 'good': return '良好'
      case 'warning': return '警告'
      case 'critical': return '严重'
      default: return '未知'
    }
  }

  // 监听性能事件
  useEffect(() => {
    const handlePerformanceEvent = (event: CustomEvent) => {
      const { type, latency } = event.detail
      updateMetrics({ [type]: latency })
    }

    // 监听自定义性能事件
    window.addEventListener('voiceLatency', handlePerformanceEvent as EventListener)
    
    return () => {
      window.removeEventListener('voiceLatency', handlePerformanceEvent as EventListener)
    }
  }, [updateMetrics])

  // 渲染延迟指标
  const renderMetric = (key: keyof LatencyMetrics, label: string) => {
    const current = metrics[key]
    const average = movingAverages[key]
    const status = getLatencyStatus(average, key)
    const threshold = THRESHOLDS[key]
    
    return (
      <div key={key} className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">{label}</span>
          <div className="flex items-center gap-2">
            <div
              className={cn(
                "w-2 h-2 rounded-full",
                getStatusColor(status)
              )}
            />
            <Badge variant={status === 'good' ? 'default' : status === 'warning' ? 'secondary' : 'destructive'}>
              {getStatusText(status)}
            </Badge>
          </div>
        </div>
        
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>当前: {current.toFixed(0)}ms</span>
            <span>平均: {average.toFixed(0)}ms</span>
          </div>
          
          <Progress 
            value={Math.min((average / threshold.critical) * 100, 100)} 
            className="h-2"
          />
          
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>0ms</span>
            <span className="text-green-600">{threshold.good}ms</span>
            <span className="text-yellow-600">{threshold.warning}ms</span>
            <span className="text-red-600">{threshold.critical}ms</span>
          </div>
        </div>
      </div>
    )
  }

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center justify-between">
          延迟监控
          <div className="flex items-center gap-2">
            <div
              className={cn(
                "w-2 h-2 rounded-full",
                isMonitoring ? "bg-green-500 animate-pulse" : "bg-gray-500"
              )}
            />
            <span className="text-sm font-normal text-muted-foreground">
              {isMonitoring ? "监控中" : "未监控"}
            </span>
          </div>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* 端到端延迟 */}
        <div className="p-3 bg-muted rounded-lg">
          {renderMetric('e2e', '端到端延迟')}
        </div>
        
        {/* 各阶段延迟 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {renderMetric('capture', '音频采集')}
          {renderMetric('asr', 'ASR识别')}
          {renderMetric('llm', 'LLM生成')}
          {renderMetric('tts', 'TTS合成')}
          {renderMetric('play', '音频播放')}
        </div>
        
        {/* 延迟分布图 */}
        {history.length > 0 && (
          <div className="space-y-2">
            <h4 className="text-sm font-medium">延迟趋势</h4>
            <div className="h-20 flex items-end gap-1">
              {history.slice(-20).map((h, index) => (
                <div
                  key={index}
                  className="flex-1 bg-blue-200 rounded-t"
                  style={{
                    height: `${Math.min((h.e2e / 5000) * 100, 100)}%`
                  }}
                  title={`${h.e2e.toFixed(0)}ms`}
                />
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// 工具函数：发送性能事件
export const emitLatencyEvent = (type: keyof LatencyMetrics, latency: number) => {
  const event = new CustomEvent('voiceLatency', {
    detail: { type, latency }
  })
  window.dispatchEvent(event)
}

// 性能计时器类
export class LatencyTimer {
  private startTimes: Map<string, number> = new Map()
  
  start(stage: keyof LatencyMetrics) {
    this.startTimes.set(stage, performance.now())
  }
  
  end(stage: keyof LatencyMetrics) {
    const startTime = this.startTimes.get(stage)
    if (startTime) {
      const latency = performance.now() - startTime
      emitLatencyEvent(stage, latency)
      this.startTimes.delete(stage)
      return latency
    }
    return 0
  }
  
  measure(stage: keyof LatencyMetrics, fn: () => Promise<any>) {
    return async () => {
      this.start(stage)
      try {
        const result = await fn()
        return result
      } finally {
        this.end(stage)
      }
    }
  }
}
