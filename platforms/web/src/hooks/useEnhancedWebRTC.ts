'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { useWebSocket } from './useWebSocket'

export interface WebRTCConfig {
  iceServers: RTCIceServer[]
  maxRetries: number
  retryDelay: number
  connectionTimeout: number
  heartbeatInterval: number
  audioConstraints: MediaTrackConstraints
  enableAutoReconnect: boolean
  enableNetworkAdaptation: boolean
}

export interface WebRTCMetrics {
  connectionState: RTCPeerConnectionState
  iceConnectionState: RTCIceConnectionState
  bytesReceived: number
  bytesSent: number
  packetsLost: number
  jitter: number
  roundTripTime: number
  audioLevel: number
  networkQuality: 'excellent' | 'good' | 'fair' | 'poor'
}

export interface WebRTCCallbacks {
  onConnectionStateChange?: (state: RTCPeerConnectionState) => void
  onIceConnectionStateChange?: (state: RTCIceConnectionState) => void
  onAudioReceived?: (audioData: ArrayBuffer) => void
  onError?: (error: Error) => void
  onMetricsUpdate?: (metrics: WebRTCMetrics) => void
  onNetworkQualityChange?: (quality: string) => void
}

const DEFAULT_CONFIG: WebRTCConfig = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
    {
      urls: 'turn:your-turn-server.com:3478',
      username: 'your-username',
      credential: 'your-password'
    }
  ],
  maxRetries: 3,
  retryDelay: 2000,
  connectionTimeout: 10000,
  heartbeatInterval: 5000,
  audioConstraints: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    sampleRate: 16000,
    channelCount: 1
  },
  enableAutoReconnect: true,
  enableNetworkAdaptation: true
}

export const useEnhancedWebRTC = (
  signalingUrl: string,
  config: Partial<WebRTCConfig> = {},
  callbacks: WebRTCCallbacks = {}
) => {
  const fullConfig = { ...DEFAULT_CONFIG, ...config }
  
  // State
  const [isConnected, setIsConnected] = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [metrics, setMetrics] = useState<WebRTCMetrics>({
    connectionState: 'new',
    iceConnectionState: 'new',
    bytesReceived: 0,
    bytesSent: 0,
    packetsLost: 0,
    jitter: 0,
    roundTripTime: 0,
    audioLevel: 0,
    networkQuality: 'excellent'
  })
  const [error, setError] = useState<Error | null>(null)

  // Refs
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const remoteStreamRef = useRef<MediaStream | null>(null)
  const dataChannelRef = useRef<RTCDataChannel | null>(null)
  const retryCountRef = useRef(0)
  const heartbeatIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const metricsIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const lastNetworkQualityRef = useRef<string>('excellent')

  // WebSocket for signaling
  const {
    isConnected: wsConnected,
    send: sendSignalingMessage
  } = useWebSocket(signalingUrl, {
    onMessage: handleSignalingMessage,
    onError: (error) => {
      console.error('Signaling error:', error)
      setError(new Error(`Signaling error: ${error.type || 'Unknown error'}`))
    }
  })

  // Initialize peer connection
  const initializePeerConnection = useCallback(() => {
    try {
      const pc = new RTCPeerConnection({
        iceServers: fullConfig.iceServers,
        iceCandidatePoolSize: 10
      })

      // Connection state handlers
      pc.onconnectionstatechange = () => {
        const state = pc.connectionState
        setMetrics(prev => ({ ...prev, connectionState: state }))
        callbacks.onConnectionStateChange?.(state)

        if (state === 'connected') {
          setIsConnected(true)
          setIsConnecting(false)
          retryCountRef.current = 0
          startHeartbeat()
          startMetricsCollection()
        } else if (state === 'disconnected' || state === 'failed') {
          setIsConnected(false)
          handleConnectionFailure()
        }
      }

      pc.oniceconnectionstatechange = () => {
        const state = pc.iceConnectionState
        setMetrics(prev => ({ ...prev, iceConnectionState: state }))
        callbacks.onIceConnectionStateChange?.(state)

        if (state === 'disconnected' || state === 'failed') {
          handleConnectionFailure()
        }
      }

      // ICE candidate handler
      pc.onicecandidate = (event) => {
        if (event.candidate) {
          sendSignalingMessage({
            type: 'ice-candidate',
            candidate: event.candidate
          })
        }
      }

      // Remote stream handler
      pc.ontrack = (event) => {
        const stream = event.streams[0]
        if (stream) {
          remoteStreamRef.current = stream
          // Handle remote audio
          const audioContext = new AudioContext()
          const source = audioContext.createMediaStreamSource(stream)
          const analyser = audioContext.createAnalyser()
          source.connect(analyser)
          
          // Monitor audio level
          const dataArray = new Uint8Array(analyser.frequencyBinCount)
          const updateAudioLevel = () => {
            analyser.getByteFrequencyData(dataArray)
            const average = dataArray.reduce((sum, value) => sum + value, 0) / dataArray.length
            const audioLevel = average / 255
            setMetrics(prev => ({ ...prev, audioLevel }))
            
            if (isConnected) {
              requestAnimationFrame(updateAudioLevel)
            }
          }
          updateAudioLevel()
        }
      }

      // Data channel for heartbeat and metrics
      const dataChannel = pc.createDataChannel('control', {
        ordered: true
      })
      
      dataChannel.onopen = () => {
        console.log('Data channel opened')
      }
      
      dataChannel.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          handleDataChannelMessage(message)
        } catch (error) {
          console.error('Failed to parse data channel message:', error)
        }
      }

      dataChannelRef.current = dataChannel
      peerConnectionRef.current = pc

      return pc
    } catch (error) {
      console.error('Failed to initialize peer connection:', error)
      setError(error as Error)
      return null
    }
  }, [fullConfig.iceServers, callbacks, sendSignalingMessage, isConnected])

  // Handle signaling messages
  async function handleSignalingMessage(message: any) {
    const pc = peerConnectionRef.current
    if (!pc) return

    try {
      switch (message.type) {
        case 'offer':
          await pc.setRemoteDescription(new RTCSessionDescription(message.offer))
          const answer = await pc.createAnswer()
          await pc.setLocalDescription(answer)
          sendSignalingMessage({
            type: 'answer',
            answer: answer
          })
          break

        case 'answer':
          await pc.setRemoteDescription(new RTCSessionDescription(message.answer))
          break

        case 'ice-candidate':
          await pc.addIceCandidate(new RTCIceCandidate(message.candidate))
          break

        default:
          console.warn('Unknown signaling message type:', message.type)
      }
    } catch (error) {
      console.error('Failed to handle signaling message:', error)
      setError(error as Error)
    }
  }

  // Handle data channel messages
  const handleDataChannelMessage = useCallback((message: any) => {
    switch (message.type) {
      case 'heartbeat':
        // Respond to heartbeat
        if (dataChannelRef.current?.readyState === 'open') {
          dataChannelRef.current.send(JSON.stringify({
            type: 'heartbeat-response',
            timestamp: Date.now()
          }))
        }
        break

      case 'heartbeat-response':
        // Calculate RTT
        const rtt = Date.now() - message.timestamp
        setMetrics(prev => ({ ...prev, roundTripTime: rtt }))
        break

      case 'network-quality':
        const quality = message.quality
        if (quality !== lastNetworkQualityRef.current) {
          lastNetworkQualityRef.current = quality
          setMetrics(prev => ({ ...prev, networkQuality: quality }))
          callbacks.onNetworkQualityChange?.(quality)
        }
        break
    }
  }, [callbacks])

  // Start connection
  const connect = useCallback(async () => {
    if (isConnecting || isConnected) return

    setIsConnecting(true)
    setError(null)

    try {
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: fullConfig.audioConstraints,
        video: false
      })
      
      localStreamRef.current = stream

      // Initialize peer connection
      const pc = initializePeerConnection()
      if (!pc) {
        throw new Error('Failed to initialize peer connection')
      }

      // Add local stream
      stream.getTracks().forEach(track => {
        pc.addTrack(track, stream)
      })

      // Create offer
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)

      // Send offer through signaling
      sendSignalingMessage({
        type: 'offer',
        offer: offer
      })

      // Set connection timeout
      setTimeout(() => {
        if (!isConnected) {
          handleConnectionFailure()
        }
      }, fullConfig.connectionTimeout)

    } catch (error) {
      console.error('Failed to connect:', error)
      setError(error as Error)
      setIsConnecting(false)
    }
  }, [isConnecting, isConnected, fullConfig, initializePeerConnection, sendSignalingMessage])

  // Disconnect
  const disconnect = useCallback(() => {
    // Stop heartbeat
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current)
      heartbeatIntervalRef.current = null
    }

    // Stop metrics collection
    if (metricsIntervalRef.current) {
      clearInterval(metricsIntervalRef.current)
      metricsIntervalRef.current = null
    }

    // Close peer connection
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close()
      peerConnectionRef.current = null
    }

    // Stop local stream
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop())
      localStreamRef.current = null
    }

    // Reset state
    setIsConnected(false)
    setIsConnecting(false)
    setError(null)
    retryCountRef.current = 0
  }, [])

  // Handle connection failure
  const handleConnectionFailure = useCallback(() => {
    setIsConnected(false)
    setIsConnecting(false)

    if (fullConfig.enableAutoReconnect && retryCountRef.current < fullConfig.maxRetries) {
      retryCountRef.current++
      console.log(`Connection failed, retrying (${retryCountRef.current}/${fullConfig.maxRetries})...`)
      
      reconnectTimeoutRef.current = setTimeout(() => {
        connect()
      }, fullConfig.retryDelay * retryCountRef.current)
    } else {
      setError(new Error('Connection failed after maximum retries'))
      callbacks.onError?.(new Error('Connection failed'))
    }
  }, [fullConfig, connect, callbacks])

  // Start heartbeat
  const startHeartbeat = useCallback(() => {
    if (heartbeatIntervalRef.current) return

    heartbeatIntervalRef.current = setInterval(() => {
      if (dataChannelRef.current?.readyState === 'open') {
        dataChannelRef.current.send(JSON.stringify({
          type: 'heartbeat',
          timestamp: Date.now()
        }))
      }
    }, fullConfig.heartbeatInterval)
  }, [fullConfig.heartbeatInterval])

  // Start metrics collection
  const startMetricsCollection = useCallback(() => {
    if (metricsIntervalRef.current) return

    metricsIntervalRef.current = setInterval(async () => {
      const pc = peerConnectionRef.current
      if (!pc) return

      try {
        const stats = await pc.getStats()
        let bytesReceived = 0
        let bytesSent = 0
        let packetsLost = 0
        let jitter = 0

        stats.forEach((report) => {
          if (report.type === 'inbound-rtp' && report.mediaType === 'audio') {
            bytesReceived += report.bytesReceived || 0
            packetsLost += report.packetsLost || 0
            jitter += report.jitter || 0
          } else if (report.type === 'outbound-rtp' && report.mediaType === 'audio') {
            bytesSent += report.bytesSent || 0
          }
        })

        // Calculate network quality based on metrics
        const networkQuality = calculateNetworkQuality(packetsLost, jitter, metrics.roundTripTime)

        const newMetrics = {
          ...metrics,
          bytesReceived,
          bytesSent,
          packetsLost,
          jitter,
          networkQuality
        }

        setMetrics(newMetrics)
        callbacks.onMetricsUpdate?.(newMetrics)

        // Adapt to network conditions
        if (fullConfig.enableNetworkAdaptation) {
          adaptToNetworkConditions(networkQuality)
        }

      } catch (error) {
        console.error('Failed to collect metrics:', error)
      }
    }, 1000) // Collect metrics every second
  }, [metrics, callbacks, fullConfig.enableNetworkAdaptation])

  // Calculate network quality
  const calculateNetworkQuality = (packetsLost: number, jitter: number, rtt: number): 'excellent' | 'good' | 'fair' | 'poor' => {
    const lossRate = packetsLost / 1000 // Assume 1000 packets baseline
    
    if (lossRate < 0.01 && jitter < 20 && rtt < 100) return 'excellent'
    if (lossRate < 0.03 && jitter < 50 && rtt < 200) return 'good'
    if (lossRate < 0.05 && jitter < 100 && rtt < 500) return 'fair'
    return 'poor'
  }

  // Adapt to network conditions
  const adaptToNetworkConditions = useCallback((quality: string) => {
    const pc = peerConnectionRef.current
    if (!pc) return

    // Adjust audio bitrate based on network quality
    const sender = pc.getSenders().find(s => s.track?.kind === 'audio')
    if (sender) {
      const params = sender.getParameters()
      if (params.encodings && params.encodings.length > 0 && params.encodings[0]) {
        switch (quality) {
          case 'poor':
            params.encodings[0].maxBitrate = 32000 // 32 kbps
            break
          case 'fair':
            params.encodings[0].maxBitrate = 64000 // 64 kbps
            break
          case 'good':
            params.encodings[0].maxBitrate = 128000 // 128 kbps
            break
          case 'excellent':
            params.encodings[0].maxBitrate = 256000 // 256 kbps
            break
        }
        sender.setParameters(params)
      }
    }
  }, [])

  // Send audio data
  const sendAudio = useCallback((audioData: ArrayBuffer) => {
    if (dataChannelRef.current?.readyState === 'open') {
      dataChannelRef.current.send(audioData)
    }
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      disconnect()
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
    }
  }, [disconnect])

  return {
    isConnected,
    isConnecting,
    metrics,
    error,
    connect,
    disconnect,
    sendAudio,
    localStream: localStreamRef.current,
    remoteStream: remoteStreamRef.current
  }
}

export default useEnhancedWebRTC
