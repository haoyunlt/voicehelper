/**
 * VoiceHelper SDK - 实时语音客户端
 * 支持WebRTC、LiveKit、OpenAI Realtime等多种连接方式
 */

export interface VoiceClientConfig {
  apiKey: string;
  baseUrl?: string;
  provider?: 'webrtc' | 'livekit' | 'openai_realtime' | 'websocket';
  sampleRate?: number;
  channels?: number;
  enableVAD?: boolean;
  enableBargeIn?: boolean;
  autoConnect?: boolean;
  debug?: boolean;
}

export interface SessionConfig {
  user_id?: string;
  language?: string;
  voice_id?: string;
  context?: Record<string, any>;
  routing_policy?: {
    stt_provider?: string;
    tts_provider?: string;
    llm_model?: string;
    max_latency_ms?: number;
  };
}

export interface VoiceMetrics {
  e2e_latency_ms: number;
  stt_latency_ms: number;
  llm_latency_ms: number;
  tts_latency_ms: number;
  packet_loss: number;
  jitter_ms: number;
  audio_level: number;
  connection_state: string;
}

export interface VoiceEvents {
  onConnected?: () => void;
  onDisconnected?: () => void;
  onError?: (error: Error) => void;
  onTranscription?: (text: string, isFinal: boolean) => void;
  onResponse?: (text: string) => void;
  onAudioReceived?: (audioData: ArrayBuffer) => void;
  onMetricsUpdate?: (metrics: VoiceMetrics) => void;
  onVADStateChange?: (state: 'silence' | 'speech') => void;
  onBargeIn?: () => void;
}

export class VoiceClient {
  private config: Required<VoiceClientConfig>;
  private events: VoiceEvents;
  private sessionId: string | null = null;
  private connectionInfo: any = null;
  private isConnected = false;
  private isRecording = false;
  
  // WebRTC相关
  private peerConnection: RTCPeerConnection | null = null;
  private localStream: MediaStream | null = null;
  private remoteStream: MediaStream | null = null;
  
  // LiveKit相关
  private livekitRoom: any = null;
  
  // OpenAI Realtime相关
  private realtimeWs: WebSocket | null = null;
  
  // 音频处理
  private audioContext: AudioContext | null = null;
  private audioWorklet: AudioWorkletNode | null = null;
  
  // 指标收集
  private metrics: VoiceMetrics = {
    e2e_latency_ms: 0,
    stt_latency_ms: 0,
    llm_latency_ms: 0,
    tts_latency_ms: 0,
    packet_loss: 0,
    jitter_ms: 0,
    audio_level: 0,
    connection_state: 'new'
  };
  
  constructor(config: VoiceClientConfig, events: VoiceEvents = {}) {
    this.config = {
      baseUrl: 'http://localhost:8000',
      provider: 'webrtc',
      sampleRate: 16000,
      channels: 1,
      enableVAD: true,
      enableBargeIn: true,
      autoConnect: false,
      debug: false,
      ...config
    };
    
    this.events = events;
    
    if (this.config.debug) {
      console.log('VoiceClient initialized:', this.config);
    }
  }
  
  /**
   * 连接到语音服务
   */
  async connect(sessionConfig: SessionConfig = {}): Promise<void> {
    try {
      if (this.isConnected) {
        throw new Error('Already connected');
      }
      
      // 创建会话
      const sessionData = await this.createSession(sessionConfig);
      this.sessionId = sessionData.session_id;
      this.connectionInfo = sessionData.connection_info;
      
      // 根据提供商类型建立连接
      switch (sessionData.connection_info.type) {
        case 'webrtc':
          if (sessionData.connection_info.url.includes('livekit')) {
            await this.connectLiveKit(sessionData.connection_info);
          } else if (sessionData.connection_info.url.includes('openai')) {
            await this.connectOpenAIRealtime(sessionData.connection_info);
          } else {
            await this.connectWebRTC(sessionData.connection_info);
          }
          break;
          
        case 'websocket':
          await this.connectWebSocket(sessionData.connection_info);
          break;
          
        default:
          throw new Error(`Unsupported connection type: ${sessionData.connection_info.type}`);
      }
      
      this.isConnected = true;
      this.events.onConnected?.();
      
      if (this.config.debug) {
        console.log('Connected successfully:', this.sessionId);
      }
      
    } catch (error) {
      this.events.onError?.(error as Error);
      throw error;
    }
  }
  
  /**
   * 断开连接
   */
  async disconnect(): Promise<void> {
    try {
      // 停止录音
      await this.stopRecording();
      
      // 关闭连接
      if (this.peerConnection) {
        this.peerConnection.close();
        this.peerConnection = null;
      }
      
      if (this.livekitRoom) {
        await this.livekitRoom.disconnect();
        this.livekitRoom = null;
      }
      
      if (this.realtimeWs) {
        this.realtimeWs.close();
        this.realtimeWs = null;
      }
      
      // 清理音频资源
      if (this.localStream) {
        this.localStream.getTracks().forEach(track => track.stop());
        this.localStream = null;
      }
      
      if (this.audioContext) {
        await this.audioContext.close();
        this.audioContext = null;
      }
      
      // 关闭会话
      if (this.sessionId) {
        await this.closeSession();
      }
      
      this.isConnected = false;
      this.sessionId = null;
      this.connectionInfo = null;
      
      this.events.onDisconnected?.();
      
      if (this.config.debug) {
        console.log('Disconnected successfully');
      }
      
    } catch (error) {
      this.events.onError?.(error as Error);
      throw error;
    }
  }
  
  /**
   * 开始录音
   */
  async startRecording(): Promise<void> {
    if (this.isRecording) return;
    
    try {
      // 获取麦克风权限
      this.localStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channels,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
      
      // 初始化音频处理
      await this.initializeAudioProcessing();
      
      this.isRecording = true;
      
      if (this.config.debug) {
        console.log('Recording started');
      }
      
    } catch (error) {
      this.events.onError?.(error as Error);
      throw error;
    }
  }
  
  /**
   * 停止录音
   */
  async stopRecording(): Promise<void> {
    if (!this.isRecording) return;
    
    if (this.localStream) {
      this.localStream.getTracks().forEach(track => track.stop());
    }
    
    if (this.audioWorklet) {
      this.audioWorklet.disconnect();
      this.audioWorklet = null;
    }
    
    this.isRecording = false;
    
    if (this.config.debug) {
      console.log('Recording stopped');
    }
  }
  
  /**
   * 发送文本消息
   */
  async sendMessage(text: string): Promise<void> {
    if (!this.isConnected || !this.sessionId) {
      throw new Error('Not connected');
    }
    
    // 根据连接类型发送消息
    if (this.realtimeWs) {
      // OpenAI Realtime
      this.realtimeWs.send(JSON.stringify({
        type: 'conversation.item.create',
        item: {
          type: 'message',
          role: 'user',
          content: [{ type: 'input_text', text }]
        }
      }));
      
      this.realtimeWs.send(JSON.stringify({
        type: 'response.create'
      }));
    } else {
      // 其他提供商通过HTTP API
      await fetch(`${this.config.baseUrl}/api/chat/message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`
        },
        body: JSON.stringify({
          session_id: this.sessionId,
          message: text
        })
      });
    }
  }
  
  /**
   * 触发barge-in（打断TTS）
   */
  async triggerBargeIn(): Promise<void> {
    if (!this.isConnected || !this.sessionId) {
      throw new Error('Not connected');
    }
    
    try {
      const response = await fetch(`${this.config.baseUrl}/api/realtime/cancel-tts`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.apiKey}`
        },
        body: JSON.stringify({
          session_id: this.sessionId,
          reason: 'manual_barge_in'
        })
      });
      
      if (!response.ok) {
        throw new Error(`Failed to trigger barge-in: ${response.statusText}`);
      }
      
      this.events.onBargeIn?.();
      
    } catch (error) {
      this.events.onError?.(error as Error);
      throw error;
    }
  }
  
  /**
   * 获取当前指标
   */
  getMetrics(): VoiceMetrics {
    return { ...this.metrics };
  }
  
  /**
   * 获取连接状态
   */
  getConnectionState(): string {
    if (!this.isConnected) return 'disconnected';
    
    if (this.peerConnection) {
      return this.peerConnection.connectionState;
    }
    
    if (this.realtimeWs) {
      return this.realtimeWs.readyState === WebSocket.OPEN ? 'connected' : 'connecting';
    }
    
    return 'connected';
  }
  
  // 私有方法
  
  private async createSession(config: SessionConfig): Promise<any> {
    const response = await fetch(`${this.config.baseUrl}/api/realtime/session`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`
      },
      body: JSON.stringify({
        client_capabilities: {
          webrtc: true,
          opus_codec: true,
          echo_cancellation: true,
          noise_suppression: true,
          auto_gain_control: true,
          sample_rate: this.config.sampleRate,
          channels: this.config.channels
        },
        user_preferences: config,
        context: {
          sdk_version: '2.0.0',
          user_agent: navigator.userAgent
        }
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`);
    }
    
    return response.json();
  }
  
  private async closeSession(): Promise<void> {
    if (!this.sessionId) return;
    
    try {
      await fetch(`${this.config.baseUrl}/api/realtime/session/${this.sessionId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${this.config.apiKey}`
        }
      });
    } catch (error) {
      if (this.config.debug) {
        console.warn('Failed to close session:', error);
      }
    }
  }
  
  private async connectWebRTC(connectionInfo: any): Promise<void> {
    const configuration: RTCConfiguration = {
      iceServers: connectionInfo.ice_servers || [
        { urls: ['stun:stun.l.google.com:19302'] }
      ]
    };
    
    this.peerConnection = new RTCPeerConnection(configuration);
    
    // 设置事件监听器
    this.peerConnection.onconnectionstatechange = () => {
      this.metrics.connection_state = this.peerConnection!.connectionState;
      this.events.onMetricsUpdate?.(this.metrics);
    };
    
    this.peerConnection.ontrack = (event) => {
      if (event.track.kind === 'audio') {
        this.remoteStream = event.streams[0];
        this.playRemoteAudio(this.remoteStream);
      }
    };
    
    // 添加本地音频轨道
    if (this.localStream) {
      this.localStream.getAudioTracks().forEach(track => {
        this.peerConnection!.addTrack(track, this.localStream!);
      });
    }
    
    // WebRTC信令过程（简化版）
    // 实际实现需要通过信令服务器交换offer/answer和ICE candidates
  }
  
  private async connectLiveKit(connectionInfo: any): Promise<void> {
    try {
      // 动态导入LiveKit
      const { Room, RoomEvent, Track } = await import('livekit-client');
      
      this.livekitRoom = new Room();
      
      this.livekitRoom.on(RoomEvent.TrackSubscribed, (track: any, publication: any, participant: any) => {
        if (track.kind === Track.Kind.Audio) {
          const audioElement = track.attach();
          document.body.appendChild(audioElement);
        }
      });
      
      this.livekitRoom.on(RoomEvent.Connected, () => {
        this.metrics.connection_state = 'connected';
        this.events.onMetricsUpdate?.(this.metrics);
      });
      
      await this.livekitRoom.connect(connectionInfo.url, connectionInfo.token);
      
      // 发布本地音频
      if (this.localStream) {
        await this.livekitRoom.localParticipant.publishTrack(
          this.localStream.getAudioTracks()[0]
        );
      }
      
    } catch (error) {
      throw new Error(`LiveKit connection failed: ${error}`);
    }
  }
  
  private async connectOpenAIRealtime(connectionInfo: any): Promise<void> {
    return new Promise((resolve, reject) => {
      this.realtimeWs = new WebSocket(
        `${connectionInfo.url}?model=${connectionInfo.model}`,
        ['realtime', 'json']
      );
      
      this.realtimeWs.onopen = () => {
        // 发送会话配置
        this.realtimeWs!.send(JSON.stringify({
          type: 'session.update',
          session: {
            modalities: ['text', 'audio'],
            instructions: 'You are a helpful assistant.',
            voice: 'alloy',
            input_audio_format: 'pcm16',
            output_audio_format: 'pcm16',
            turn_detection: {
              type: 'server_vad',
              threshold: 0.5,
              prefix_padding_ms: 300,
              silence_duration_ms: 200
            }
          }
        }));
        
        this.metrics.connection_state = 'connected';
        resolve();
      };
      
      this.realtimeWs.onmessage = (event) => {
        this.handleRealtimeMessage(JSON.parse(event.data));
      };
      
      this.realtimeWs.onerror = (error) => {
        reject(error);
      };
      
      this.realtimeWs.onclose = () => {
        this.metrics.connection_state = 'disconnected';
        this.events.onDisconnected?.();
      };
    });
  }
  
  private async connectWebSocket(connectionInfo: any): Promise<void> {
    // WebSocket回退连接实现
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(connectionInfo.url);
      
      ws.onopen = () => {
        this.metrics.connection_state = 'connected';
        resolve();
      };
      
      ws.onerror = reject;
      ws.onclose = () => {
        this.metrics.connection_state = 'disconnected';
        this.events.onDisconnected?.();
      };
    });
  }
  
  private async initializeAudioProcessing(): Promise<void> {
    if (!this.localStream) return;
    
    this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    
    try {
      // 加载AudioWorklet处理器
      await this.audioContext.audioWorklet.addModule('/audio-worklet-processor.js');
      
      this.audioWorklet = new AudioWorkletNode(this.audioContext, 'audio-processor');
      
      // 监听音频数据
      this.audioWorklet.port.onmessage = (event) => {
        const { type, audioData, energy, vadState } = event.data;
        
        if (type === 'audioData') {
          this.handleAudioData(audioData);
          this.metrics.audio_level = energy;
          
          if (this.config.enableVAD && vadState) {
            this.events.onVADStateChange?.(vadState);
          }
          
          this.events.onMetricsUpdate?.(this.metrics);
        }
      };
      
      // 连接音频流
      const source = this.audioContext.createMediaStreamSource(this.localStream);
      source.connect(this.audioWorklet);
      
    } catch (error) {
      if (this.config.debug) {
        console.warn('AudioWorklet not supported, using fallback');
      }
      // 回退到基本音频处理
    }
  }
  
  private handleAudioData(audioData: ArrayBuffer): void {
    // 根据连接类型发送音频数据
    if (this.realtimeWs && this.realtimeWs.readyState === WebSocket.OPEN) {
      // OpenAI Realtime
      const base64Audio = this.arrayBufferToBase64(audioData);
      this.realtimeWs.send(JSON.stringify({
        type: 'input_audio_buffer.append',
        audio: base64Audio
      }));
    }
    // 其他连接类型的音频数据处理...
  }
  
  private handleRealtimeMessage(message: any): void {
    switch (message.type) {
      case 'response.audio.delta':
        // 处理音频响应
        const audioData = this.base64ToArrayBuffer(message.delta);
        this.events.onAudioReceived?.(audioData);
        this.playAudioBuffer(audioData);
        break;
        
      case 'response.text.delta':
        this.events.onResponse?.(message.delta);
        break;
        
      case 'conversation.item.input_audio_transcription.completed':
        this.events.onTranscription?.(message.transcript, true);
        break;
        
      case 'error':
        this.events.onError?.(new Error(message.error.message));
        break;
    }
  }
  
  private playRemoteAudio(stream: MediaStream): void {
    const audioElement = new Audio();
    audioElement.srcObject = stream;
    audioElement.play().catch(console.error);
  }
  
  private async playAudioBuffer(arrayBuffer: ArrayBuffer): Promise<void> {
    if (!this.audioContext) return;
    
    try {
      const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.audioContext.destination);
      source.start();
    } catch (error) {
      if (this.config.debug) {
        console.error('Failed to play audio buffer:', error);
      }
    }
  }
  
  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }
  
  private base64ToArrayBuffer(base64: string): ArrayBuffer {
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
  }
}
