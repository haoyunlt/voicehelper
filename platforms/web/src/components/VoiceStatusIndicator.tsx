'use client'

import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  MicrophoneIcon, 
  SpeakerWaveIcon, 
  Cog6ToothIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline'

export enum VoiceStatus {
  IDLE = 'idle',
  RECORDING = 'recording',
  PROCESSING = 'processing',
  SPEAKING = 'speaking',
  ERROR = 'error',
  SUCCESS = 'success'
}

interface VoiceStatusIndicatorProps {
  status: VoiceStatus
  message?: string
  progress?: number
  className?: string
}

const statusConfig = {
  [VoiceStatus.IDLE]: {
    icon: MicrophoneIcon,
    color: 'text-gray-400',
    bgColor: 'bg-gray-100',
    label: '待机',
    description: '点击开始录音'
  },
  [VoiceStatus.RECORDING]: {
    icon: MicrophoneIcon,
    color: 'text-red-500',
    bgColor: 'bg-red-50',
    label: '录音中',
    description: '正在录制语音...'
  },
  [VoiceStatus.PROCESSING]: {
    icon: Cog6ToothIcon,
    color: 'text-blue-500',
    bgColor: 'bg-blue-50',
    label: '处理中',
    description: '正在识别语音...'
  },
  [VoiceStatus.SPEAKING]: {
    icon: SpeakerWaveIcon,
    color: 'text-green-500',
    bgColor: 'bg-green-50',
    label: '播放中',
    description: '正在播放语音...'
  },
  [VoiceStatus.ERROR]: {
    icon: ExclamationTriangleIcon,
    color: 'text-red-500',
    bgColor: 'bg-red-50',
    label: '错误',
    description: '处理失败'
  },
  [VoiceStatus.SUCCESS]: {
    icon: CheckCircleIcon,
    color: 'text-green-500',
    bgColor: 'bg-green-50',
    label: '完成',
    description: '处理成功'
  }
}

export const VoiceStatusIndicator: React.FC<VoiceStatusIndicatorProps> = ({
  status,
  message,
  progress,
  className = ''
}) => {
  const config = statusConfig[status]
  const IconComponent = config.icon

  return (
    <div className={`flex items-center space-x-3 p-3 rounded-lg ${config.bgColor} ${className}`}>
      {/* 状态图标 */}
      <div className="relative">
        <motion.div
          animate={status === VoiceStatus.RECORDING ? { scale: [1, 1.1, 1] } : {}}
          transition={{ duration: 1, repeat: status === VoiceStatus.RECORDING ? Infinity : 0 }}
        >
          <IconComponent className={`w-6 h-6 ${config.color}`} />
        </motion.div>
        
        {/* 录音动画 */}
        {status === VoiceStatus.RECORDING && (
          <motion.div
            className="absolute inset-0 rounded-full border-2 border-red-500"
            animate={{ scale: [1, 1.5], opacity: [0.7, 0] }}
            transition={{ duration: 1, repeat: Infinity }}
          />
        )}
        
        {/* 处理动画 */}
        {status === VoiceStatus.PROCESSING && (
          <motion.div
            className="absolute inset-0"
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
          >
            <Cog6ToothIcon className={`w-6 h-6 ${config.color}`} />
          </motion.div>
        )}
      </div>

      {/* 状态信息 */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center space-x-2">
          <span className={`text-sm font-medium ${config.color}`}>
            {config.label}
          </span>
          
          {/* 进度指示 */}
          {progress !== undefined && (
            <span className="text-xs text-gray-500">
              {Math.round(progress * 100)}%
            </span>
          )}
        </div>
        
        <p className="text-xs text-gray-600 truncate">
          {message || config.description}
        </p>
        
        {/* 进度条 */}
        {progress !== undefined && (
          <div className="mt-1 w-full bg-gray-200 rounded-full h-1">
            <motion.div
              className={`h-1 rounded-full ${
                status === VoiceStatus.ERROR ? 'bg-red-500' : 'bg-blue-500'
              }`}
              initial={{ width: 0 }}
              animate={{ width: `${progress * 100}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
        )}
      </div>

      {/* 状态指示灯 */}
      <div className="flex flex-col items-center space-y-1">
        <motion.div
          className={`w-2 h-2 rounded-full ${
            status === VoiceStatus.RECORDING ? 'bg-red-500' :
            status === VoiceStatus.PROCESSING ? 'bg-blue-500' :
            status === VoiceStatus.SPEAKING ? 'bg-green-500' :
            status === VoiceStatus.ERROR ? 'bg-red-500' :
            status === VoiceStatus.SUCCESS ? 'bg-green-500' :
            'bg-gray-300'
          }`}
          animate={
            status === VoiceStatus.RECORDING || status === VoiceStatus.PROCESSING
              ? { opacity: [1, 0.3, 1] }
              : {}
          }
          transition={{ duration: 1, repeat: Infinity }}
        />
      </div>
    </div>
  )
}

// 语音状态管理Hook
export const useVoiceStatus = () => {
  const [status, setStatus] = React.useState<VoiceStatus>(VoiceStatus.IDLE)
  const [message, setMessage] = React.useState<string>('')
  const [progress, setProgress] = React.useState<number | undefined>(undefined)

  const updateStatus = (
    newStatus: VoiceStatus, 
    newMessage?: string, 
    newProgress?: number
  ) => {
    setStatus(newStatus)
    if (newMessage !== undefined) setMessage(newMessage)
    if (newProgress !== undefined) setProgress(newProgress)
  }

  const startRecording = () => {
    updateStatus(VoiceStatus.RECORDING, '正在录制语音，请说话...')
  }

  const startProcessing = () => {
    updateStatus(VoiceStatus.PROCESSING, '正在识别语音内容...', 0)
  }

  const updateProcessingProgress = (progressValue: number, progressMessage?: string) => {
    updateStatus(VoiceStatus.PROCESSING, progressMessage, progressValue)
  }

  const startSpeaking = () => {
    updateStatus(VoiceStatus.SPEAKING, '正在播放语音回复...')
  }

  const setError = (errorMessage: string) => {
    updateStatus(VoiceStatus.ERROR, errorMessage)
  }

  const setSuccess = (successMessage?: string) => {
    updateStatus(VoiceStatus.SUCCESS, successMessage || '处理完成')
  }

  const reset = () => {
    updateStatus(VoiceStatus.IDLE, '', undefined)
  }

  return {
    status,
    message,
    progress,
    updateStatus,
    startRecording,
    startProcessing,
    updateProcessingProgress,
    startSpeaking,
    setError,
    setSuccess,
    reset
  }
}

export default VoiceStatusIndicator
