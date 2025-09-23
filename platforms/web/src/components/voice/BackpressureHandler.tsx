'use client'

import { useState, useEffect, useCallback, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  Wifi, 
  WifiOff, 
  AlertTriangle, 
  TrendingDown, 
  TrendingUp,
  Signal
} from 'lucide-react'
import { cn } from '@/lib/utils'

interface NetworkQuality {
  rtt: number           // 往返时间 (ms)
  bandwidth: number     // 带宽估计 (kbps)
  packetLoss: number    // 丢包率 (0-1)
  jitter: number        // 抖动 (ms)
  quality: 'excellent' | 'good' | 'fair' | 'poor' | 'critical'
}

interface BackpressureConfig {
  maxQueueSize: number
  throttleThreshold: number
  adaptiveRate: boolean
  qualityThresholds: {
    excellent: { rtt: number; loss: number }
    good: { rtt: number; loss: number }
    fair: { rtt: number; loss: number }
    poor: { rtt: number; loss: number }
  }
}

interface BackpressureHandlerProps {
  onThrottle?: (enabled: boolean, reason: string) => void
  onQualityChange?: (quality: NetworkQuality) => void
  className?: string
}

export default function BackpressureHandler({
  onThrottle,
  onQualityChange,
  className
}: BackpressureHandlerProps) {
  const [networkQuality, setNetworkQuality] = useState<NetworkQuality>({
    rtt: 0,
    bandwidth: 0,
    packetLoss: 0,
    jitter: 0,
    quality: 'good'
  })

  const [isThrottling, setIsThrottling] = useState(false)
  const [queueSize, setQueueSize] = useState(0)
  const [throttleReason, setThrottleReason] = useState('')
  const [adaptiveRate, setAdaptiveRate] = useState(1.0) // 1.0 = 正常速率

  const measurementRef = useRef<{
    startTime: number
    sentPackets: number
    receivedPackets: number
    rttSamples: number[]
    lastMeasurement: number
  }>({
    startTime: Date.now(),
    sentPackets: 0,
    receivedPackets: 0,
    rttSamples: [],
    lastMeasurement: Date.now()
  })

  const config: BackpressureConfig = {
    maxQueueSize: 50,
    throttleThreshold: 40,
    adaptiveRate: true,
    qualityThresholds: {
      excellent: { rtt: 50, loss: 0.001 },
      good: { rtt: 150, loss: 0.01 },
      fair: { rtt: 300, loss: 0.05 },
      poor: { rtt: 500, loss: 0.1 }
    }
  }

  // 网络质量评估
  const assessNetworkQuality = useCallback((quality: Partial<NetworkQuality>) => {
    const { rtt = 0, packetLoss = 0 } = quality
    
    let qualityLevel: NetworkQuality['quality'] = 'critical'
    
    if (rtt <= config.qualityThresholds.excellent.rtt && packetLoss <= config.qualityThresholds.excellent.loss) {
      qualityLevel = 'excellent'
    } else if (rtt <= config.qualityThresholds.good.rtt && packetLoss <= config.qualityThresholds.good.loss) {
      qualityLevel = 'good'
    } else if (rtt <= config.qualityThresholds.fair.rtt && packetLoss <= config.qualityThresholds.fair.loss) {
      qualityLevel = 'fair'
    } else if (rtt <= config.qualityThresholds.poor.rtt && packetLoss <= config.qualityThresholds.poor.loss) {
      qualityLevel = 'poor'
    }
    
    return qualityLevel
  }, [config.qualityThresholds])

  // 计算自适应速率
  const calculateAdaptiveRate = useCallback((quality: NetworkQuality) => {
    if (!config.adaptiveRate) return 1.0
    
    switch (quality.quality) {
      case 'excellent': return 1.0
      case 'good': return 0.9
      case 'fair': return 0.7
      case 'poor': return 0.5
      case 'critical': return 0.3
      default: return 1.0
    }
  }, [config.adaptiveRate])

  // 检查是否需要节流
  const checkThrottling = useCallback((quality: NetworkQuality, currentQueueSize: number) => {
    let shouldThrottle = false
    let reason = ''
    
    // 队列大小检查
    if (currentQueueSize >= config.throttleThreshold) {
      shouldThrottle = true
      reason = `队列过载 (${currentQueueSize}/${config.maxQueueSize})`
    }
    
    // 网络质量检查
    if (quality.quality === 'critical' || quality.quality === 'poor') {
      shouldThrottle = true
      reason = reason ? `${reason}, 网络质量差` : '网络质量差'
    }
    
    // 高丢包率检查
    if (quality.packetLoss > 0.1) {
      shouldThrottle = true
      reason = reason ? `${reason}, 高丢包率` : `高丢包率 (${(quality.packetLoss * 100).toFixed(1)}%)`
    }
    
    return { shouldThrottle, reason }
  }, [config.throttleThreshold, config.maxQueueSize])

  // 更新网络质量
  const updateNetworkQuality = useCallback((measurements: Partial<NetworkQuality>) => {
    setNetworkQuality(prev => {
      const updated = { ...prev, ...measurements }
      updated.quality = assessNetworkQuality(updated)
      
      // 计算自适应速率
      const newRate = calculateAdaptiveRate(updated)
      setAdaptiveRate(newRate)
      
      // 检查节流
      const { shouldThrottle, reason } = checkThrottling(updated, queueSize)
      
      if (shouldThrottle !== isThrottling) {
        setIsThrottling(shouldThrottle)
        setThrottleReason(reason)
        onThrottle?.(shouldThrottle, reason)
      }
      
      onQualityChange?.(updated)
      return updated
    })
  }, [assessNetworkQuality, calculateAdaptiveRate, checkThrottling, queueSize, isThrottling, onThrottle, onQualityChange])

  // 测量RTT
  const measureRTT = useCallback(() => {
    const startTime = performance.now()
    
    // 模拟ping测试（实际应用中可能需要WebSocket ping/pong）
    return new Promise<number>((resolve) => {
      // 这里应该发送实际的ping消息并等待pong响应
      // 现在使用模拟数据
      setTimeout(() => {
        const rtt = performance.now() - startTime + Math.random() * 100
        resolve(rtt)
      }, Math.random() * 50 + 10)
    })
  }, [])

  // 定期网络质量检测
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const rtt = await measureRTT()
        const measurement = measurementRef.current
        
        // 更新RTT样本
        measurement.rttSamples.push(rtt)
        if (measurement.rttSamples.length > 10) {
          measurement.rttSamples.shift()
        }
        
        // 计算平均RTT和抖动
        const avgRTT = measurement.rttSamples.reduce((a, b) => a + b, 0) / measurement.rttSamples.length
        const jitter = measurement.rttSamples.length > 1 
          ? Math.sqrt(measurement.rttSamples.reduce((sum, sample) => sum + Math.pow(sample - avgRTT, 2), 0) / measurement.rttSamples.length)
          : 0
        
        // 计算丢包率（模拟）
        const packetLoss = Math.max(0, (measurement.sentPackets - measurement.receivedPackets) / Math.max(measurement.sentPackets, 1))
        
        // 估算带宽（简化计算）
        const bandwidth = Math.max(0, 1000 - avgRTT) * 10 // 简化的带宽估算
        
        updateNetworkQuality({
          rtt: avgRTT,
          jitter,
          packetLoss,
          bandwidth
        })
        
      } catch (error) {
        console.error('Network quality measurement failed:', error)
      }
    }, 2000) // 每2秒测量一次
    
    return () => clearInterval(interval)
  }, [measureRTT, updateNetworkQuality])

  // 监听WebSocket事件
  useEffect(() => {
    const handleWebSocketEvent = (event: CustomEvent) => {
      const { type, data } = event.detail
      
      switch (type) {
        case 'queue_size':
          setQueueSize(data.size)
          break
        case 'packet_sent':
          measurementRef.current.sentPackets++
          break
        case 'packet_received':
          measurementRef.current.receivedPackets++
          break
        case 'throttle':
          setIsThrottling(data.enabled)
          setThrottleReason(data.reason)
          break
      }
    }
    
    window.addEventListener('voiceWebSocketEvent', handleWebSocketEvent as EventListener)
    
    return () => {
      window.removeEventListener('voiceWebSocketEvent', handleWebSocketEvent as EventListener)
    }
  }, [])

  // 获取质量颜色
  const getQualityColor = (quality: string) => {
    switch (quality) {
      case 'excellent': return 'text-green-600'
      case 'good': return 'text-blue-600'
      case 'fair': return 'text-yellow-600'
      case 'poor': return 'text-orange-600'
      case 'critical': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  // 获取质量图标
  const getQualityIcon = (quality: string) => {
    switch (quality) {
      case 'excellent': return <Signal className="w-4 h-4 text-green-600" />
      case 'good': return <Wifi className="w-4 h-4 text-blue-600" />
      case 'fair': return <Wifi className="w-4 h-4 text-yellow-600" />
      case 'poor': return <WifiOff className="w-4 h-4 text-orange-600" />
      case 'critical': return <WifiOff className="w-4 h-4 text-red-600" />
      default: return <Wifi className="w-4 h-4 text-gray-600" />
    }
  }

  return (
    <Card className={cn("w-full", className)}>
      <CardHeader className="pb-3">
        <CardTitle className="text-lg flex items-center justify-between">
          网络质量监控
          <div className="flex items-center gap-2">
            {getQualityIcon(networkQuality.quality)}
            <span className={cn("text-sm font-normal", getQualityColor(networkQuality.quality))}>
              {networkQuality.quality === 'excellent' ? '优秀' :
               networkQuality.quality === 'good' ? '良好' :
               networkQuality.quality === 'fair' ? '一般' :
               networkQuality.quality === 'poor' ? '较差' : '严重'}
            </span>
          </div>
        </CardTitle>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* 节流警告 */}
        {isThrottling && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              正在进行流量控制: {throttleReason}
            </AlertDescription>
          </Alert>
        )}
        
        {/* 网络指标 */}
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>延迟 (RTT)</span>
              <span>{networkQuality.rtt.toFixed(0)}ms</span>
            </div>
            <Progress value={Math.min((networkQuality.rtt / 500) * 100, 100)} />
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>抖动</span>
              <span>{networkQuality.jitter.toFixed(0)}ms</span>
            </div>
            <Progress value={Math.min((networkQuality.jitter / 100) * 100, 100)} />
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>丢包率</span>
              <span>{(networkQuality.packetLoss * 100).toFixed(1)}%</span>
            </div>
            <Progress value={networkQuality.packetLoss * 100} />
          </div>
          
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>带宽估计</span>
              <span>{(networkQuality.bandwidth / 1000).toFixed(1)}Mbps</span>
            </div>
            <Progress value={Math.min((networkQuality.bandwidth / 10000) * 100, 100)} />
          </div>
        </div>
        
        {/* 自适应控制 */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">自适应速率</span>
            <div className="flex items-center gap-2">
              {adaptiveRate < 1.0 ? (
                <TrendingDown className="w-4 h-4 text-orange-500" />
              ) : (
                <TrendingUp className="w-4 h-4 text-green-500" />
              )}
              <Badge variant={adaptiveRate >= 0.8 ? 'default' : adaptiveRate >= 0.5 ? 'secondary' : 'destructive'}>
                {(adaptiveRate * 100).toFixed(0)}%
              </Badge>
            </div>
          </div>
          <Progress value={adaptiveRate * 100} />
        </div>
        
        {/* 队列状态 */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>发送队列</span>
            <span>{queueSize}/{config.maxQueueSize}</span>
          </div>
          <Progress 
            value={(queueSize / config.maxQueueSize) * 100}
            className={cn(
              queueSize >= config.throttleThreshold && "bg-red-100"
            )}
          />
        </div>
      </CardContent>
    </Card>
  )
}

// 工具函数：发送WebSocket事件
export const emitWebSocketEvent = (type: string, data: any) => {
  const event = new CustomEvent('voiceWebSocketEvent', {
    detail: { type, data }
  })
  window.dispatchEvent(event)
}
