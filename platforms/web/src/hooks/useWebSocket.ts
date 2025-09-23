import { useCallback, useEffect, useRef, useState } from 'react'

export interface WebSocketConfig {
  reconnectAttempts?: number
  reconnectInterval?: number
  onOpen?: () => void
  onClose?: () => void
  onError?: (error: Event) => void
  onMessage?: (data: any) => void
}

export interface WebSocketReturn {
  isConnected: boolean
  send: (data: any) => void
  close: () => void
  reconnect: () => void
}

export function useWebSocket(url: string, config: WebSocketConfig = {}): WebSocketReturn {
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)

  const {
    reconnectAttempts = 3,
    reconnectInterval = 3000,
    onOpen,
    onClose,
    onError,
    onMessage
  } = config

  const connect = useCallback(() => {
    try {
      wsRef.current = new WebSocket(url)

      wsRef.current.onopen = () => {
        setIsConnected(true)
        reconnectAttemptsRef.current = 0
        onOpen?.()
      }

      wsRef.current.onclose = () => {
        setIsConnected(false)
        onClose?.()

        // 自动重连
        if (reconnectAttemptsRef.current < reconnectAttempts) {
          reconnectAttemptsRef.current++
          reconnectTimeoutRef.current = setTimeout(() => {
            connect()
          }, reconnectInterval)
        }
      }

      wsRef.current.onerror = (error) => {
        onError?.(error)
      }

      wsRef.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          onMessage?.(data)
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
    }
  }, [url, reconnectAttempts, reconnectInterval, onOpen, onClose, onError, onMessage])

  const send = useCallback((data: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const close = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }
    
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    
    setIsConnected(false)
  }, [])

  const reconnect = useCallback(() => {
    close()
    reconnectAttemptsRef.current = 0
    connect()
  }, [close, connect])

  useEffect(() => {
    connect()
    
    return () => {
      close()
    }
  }, [connect, close])

  return {
    isConnected,
    send,
    close,
    reconnect
  }
}
