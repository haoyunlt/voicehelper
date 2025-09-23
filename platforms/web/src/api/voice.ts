/**
 * V2架构语音WebSocket客户端
 * 基于BaseStreamClient实现语音流式通信
 */

import { BaseStreamClient, EventSink, StreamOptions } from './base';

export interface VoiceConfig {
  sample_rate: number;
  channels: number;
  language: string;
}

export interface VoiceStartMessage {
  type: 'start';
  session_id: string;
  config: VoiceConfig;
}

export interface VoiceAudioMessage {
  type: 'audio';
  data: string; // base64编码的音频数据
}

export interface VoiceStopMessage {
  type: 'stop';
  session_id: string;
}

export type VoiceMessage = VoiceStartMessage | VoiceAudioMessage | VoiceStopMessage;

export class VoiceWSClient extends BaseStreamClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private isConnecting = false;
  private eventSink: EventSink | null = null;
  private currentSessionId: string | null = null;
  
  async connect(sink: EventSink, options: StreamOptions = {}): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }
    
    this.eventSink = sink;
    this.isConnecting = true;
    
    try {
      await this.retryWithBackoff(async () => {
        await this.establishConnection();
      }, options.retries || this.maxReconnectAttempts);
    } catch (error) {
      this.isConnecting = false;
      throw error;
    }
  }
  
  private async establishConnection(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = this.getWebSocketURL();
      this.ws = new WebSocket(wsUrl);
      
      const timeout = setTimeout(() => {
        if (this.ws) {
          this.ws.close();
        }
        reject(new Error('WebSocket connection timeout'));
      }, 10000);
      
      this.ws.onopen = () => {
        clearTimeout(timeout);
        this.isConnecting = false;
        this.reconnectAttempts = 0;
        this.eventSink?.on('connected', { status: 'connected' });
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        this.handleMessage(event);
      };
      
      this.ws.onclose = (event) => {
        clearTimeout(timeout);
        this.handleClose(event);
      };
      
      this.ws.onerror = (error) => {
        clearTimeout(timeout);
        this.handleError(error);
        reject(new Error('WebSocket connection failed'));
      };
    });
  }
  
  private getWebSocketURL(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = this.baseURL.replace(/^https?:\/\//, '');
    return `${protocol}//${host}/api/v2/voice/stream`;
  }
  
  private handleMessage(event: MessageEvent): void {
    try {
      const message = JSON.parse(event.data);
      const eventType = message.event || 'message';
      const eventData = message.data || {};
      
      // 处理不同类型的语音事件
      switch (eventType) {
        case 'connected':
          this.eventSink?.on('connected', eventData);
          break;
        case 'session_started':
          this.eventSink?.on('session_started', eventData);
          break;
        case 'session_stopped':
          this.eventSink?.on('session_stopped', eventData);
          this.currentSessionId = null;
          break;
        case 'asr_partial':
          this.eventSink?.on('asr_partial', eventData);
          break;
        case 'asr_final':
          this.eventSink?.on('asr_final', eventData);
          break;
        case 'agent_response':
          this.eventSink?.on('agent_response', eventData);
          break;
        case 'tts_audio':
          this.eventSink?.on('tts_audio', eventData);
          break;
        case 'error':
          this.eventSink?.on('error', eventData);
          break;
        default:
          this.eventSink?.on(eventType, eventData);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      this.eventSink?.on('error', { error: 'Failed to parse message' });
    }
  }
  
  private handleClose(event: CloseEvent): void {
    this.ws = null;
    
    if (event.wasClean) {
      this.eventSink?.on('disconnected', { reason: 'clean_close' });
    } else {
      this.eventSink?.on('disconnected', { reason: 'unexpected_close' });
      
      // 尝试重连
      if (this.reconnectAttempts < this.maxReconnectAttempts && !this.isConnecting) {
        this.scheduleReconnect();
      }
    }
  }
  
  private handleError(error: Event): void {
    console.error('WebSocket error:', error);
    this.eventSink?.on('error', { error: 'WebSocket connection error' });
  }
  
  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    setTimeout(() => {
      if (this.eventSink && !this.isConnecting) {
        this.eventSink.on('reconnecting', { attempt: this.reconnectAttempts });
        this.establishConnection().catch(() => {
          // 重连失败，继续尝试
          if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          } else {
            this.eventSink?.on('reconnect_failed', { attempts: this.reconnectAttempts });
          }
        });
      }
    }, delay);
  }
  
  startVoiceSession(sessionId: string, config: VoiceConfig): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }
    
    this.currentSessionId = sessionId;
    
    const message: VoiceStartMessage = {
      type: 'start',
      session_id: sessionId,
      config: config
    };
    
    this.ws.send(JSON.stringify(message));
  }
  
  sendAudioData(audioData: ArrayBuffer): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected');
    }
    
    if (!this.currentSessionId) {
      throw new Error('No active voice session');
    }
    
    // 将音频数据转换为base64
    const base64Data = this.arrayBufferToBase64(audioData);
    
    const message: VoiceAudioMessage = {
      type: 'audio',
      data: base64Data
    };
    
    this.ws.send(JSON.stringify(message));
  }
  
  stopVoiceSession(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }
    
    if (!this.currentSessionId) {
      return;
    }
    
    const message: VoiceStopMessage = {
      type: 'stop',
      session_id: this.currentSessionId
    };
    
    this.ws.send(JSON.stringify(message));
  }
  
  disconnect(): void {
    if (this.currentSessionId) {
      this.stopVoiceSession();
    }
    
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    
    this.eventSink = null;
    this.currentSessionId = null;
    this.reconnectAttempts = 0;
  }
  
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
  
  hasActiveSession(): boolean {
    return this.currentSessionId !== null;
  }
  
  getCurrentSessionId(): string | null {
    return this.currentSessionId;
  }
  
  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i] || 0);
    }
    return btoa(binary);
  }
}
