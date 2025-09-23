'use client'

import React, { useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface VoiceVisualizerProps {
  isRecording: boolean
  isProcessing: boolean
  audioLevel: number
  className?: string
}

interface WaveformData {
  frequencies: number[]
  timestamp: number
}

export const VoiceVisualizer: React.FC<VoiceVisualizerProps> = ({
  isRecording,
  isProcessing,
  audioLevel,
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>()
  const [waveformData, setWaveformData] = useState<WaveformData[]>([])
  const [currentFrequencies, setCurrentFrequencies] = useState<number[]>([])

  // 生成模拟频谱数据
  useEffect(() => {
    if (isRecording) {
      const interval = setInterval(() => {
        const frequencies = Array.from({ length: 32 }, (_, i) => {
          const baseFreq = Math.sin(Date.now() * 0.001 + i * 0.2) * 0.5 + 0.5
          const levelMultiplier = audioLevel * 2
          return Math.min(baseFreq * levelMultiplier, 1)
        })
        
        setCurrentFrequencies(frequencies)
        setWaveformData(prev => [
          ...prev.slice(-100), // 保留最近100个数据点
          { frequencies, timestamp: Date.now() }
        ])
      }, 50) // 20fps更新

      return () => clearInterval(interval)
    } else {
      setCurrentFrequencies([])
    }
    return undefined;
  }, [isRecording, audioLevel])

  // Canvas绘制
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const draw = () => {
      const { width, height } = canvas
      ctx.clearRect(0, 0, width, height)

      if (isRecording && currentFrequencies.length > 0) {
        // 绘制频谱条
        const barWidth = width / currentFrequencies.length
        const maxBarHeight = height * 0.8

        currentFrequencies.forEach((freq, index) => {
          const barHeight = freq * maxBarHeight
          const x = index * barWidth
          const y = height - barHeight

          // 渐变色
          const gradient = ctx.createLinearGradient(0, height, 0, y)
          gradient.addColorStop(0, '#3b82f6') // blue-500
          gradient.addColorStop(0.5, '#06b6d4') // cyan-500
          gradient.addColorStop(1, '#10b981') // emerald-500

          ctx.fillStyle = gradient
          ctx.fillRect(x, y, barWidth - 1, barHeight)
        })

        // 绘制波形线
        if (waveformData.length > 1) {
          ctx.strokeStyle = '#ffffff'
          ctx.lineWidth = 2
          ctx.beginPath()

          const waveHeight = height * 0.3
          const waveY = height * 0.5

          waveformData.forEach((data, index) => {
            const x = (index / waveformData.length) * width
            const avgFreq = data.frequencies.reduce((sum, f) => sum + f, 0) / data.frequencies.length
            const y = waveY + (avgFreq - 0.5) * waveHeight

            if (index === 0) {
              ctx.moveTo(x, y)
            } else {
              ctx.lineTo(x, y)
            }
          })

          ctx.stroke()
        }
      } else if (isProcessing) {
        // 处理中的动画
        const time = Date.now() * 0.005
        const centerX = width / 2
        const centerY = height / 2
        const radius = Math.min(width, height) * 0.3

        for (let i = 0; i < 3; i++) {
          const angle = time + (i * Math.PI * 2) / 3
          const x = centerX + Math.cos(angle) * radius * 0.5
          const y = centerY + Math.sin(angle) * radius * 0.5
          const size = 10 + Math.sin(time * 2 + i) * 5

          const gradient = ctx.createRadialGradient(x, y, 0, x, y, size)
          gradient.addColorStop(0, '#3b82f6')
          gradient.addColorStop(1, 'transparent')

          ctx.fillStyle = gradient
          ctx.beginPath()
          ctx.arc(x, y, size, 0, Math.PI * 2)
          ctx.fill()
        }
      }

      animationRef.current = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
    }
  }, [isRecording, isProcessing, currentFrequencies, waveformData])

  return (
    <div className={`relative ${className}`}>
      <canvas
        ref={canvasRef}
        width={400}
        height={200}
        className="w-full h-full rounded-lg bg-gray-900/50 backdrop-blur-sm"
      />
      
      {/* 状态指示器 */}
      <AnimatePresence>
        {isRecording && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="absolute top-2 right-2 flex items-center space-x-2"
          >
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-xs text-white font-medium">录音中</span>
          </motion.div>
        )}
        
        {isProcessing && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="absolute top-2 right-2 flex items-center space-x-2"
          >
            <div className="w-2 h-2 bg-blue-500 rounded-full animate-spin" />
            <span className="text-xs text-white font-medium">处理中</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 音量级别指示器 */}
      {isRecording && (
        <div className="absolute bottom-2 left-2 right-2">
          <div className="flex items-center space-x-2">
            <span className="text-xs text-white/70">音量</span>
            <div className="flex-1 h-1 bg-gray-700 rounded-full overflow-hidden">
              <motion.div
                className="h-full bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"
                initial={{ width: 0 }}
                animate={{ width: `${audioLevel * 100}%` }}
                transition={{ duration: 0.1 }}
              />
            </div>
            <span className="text-xs text-white/70 w-8">
              {Math.round(audioLevel * 100)}%
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default VoiceVisualizer
