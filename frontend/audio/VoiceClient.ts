/**
 * 优化版语音客户端 - 集成AudioWorklet + JitterBuffer
 * 功能: 低延迟语音采集播放 + 实时监控 + 打断支持
 */

import { JitterBuffer, AudioFrame, JitterBufferConfig } from './buffers/JitterBuffer';

export interface VoiceClientConfig {
    sampleRate?: number;
    frameSize?: number;
    channels?: number;
    bufferConfig?: Partial<JitterBufferConfig>;
    enableVAD?: boolean;
    vadThreshold?: number;
    enableMonitoring?: boolean;
}

export interface VoiceMetrics {
    e2eLatency: number;
    captureLatency: number;
    playLatency: number;
    jitterBufferHealth: number;
    audioLevel: number;
    vadState: 'silence' | 'speech';
    droppedFrames: number;
    networkLatency: number;
}

export interface VoiceEvents {
    onConnected?: () => void;
    onDisconnected?: () => void;
    onAudioFrame?: (frame: AudioFrame) => void;
    onVADStateChange?: (state: 'silence' | 'speech') => void;
    onMetricsUpdate?: (metrics: VoiceMetrics) => void;
    onError?: (error: Error) => void;
    onPlaybackStart?: () => void;
    onPlaybackEnd?: () => void;
}

export class VoiceClient {
    private config: Required<VoiceClientConfig>;
    private events: VoiceEvents;
    
    // Audio Context
    private audioContext: AudioContext | null = null;
    private micProcessor: AudioWorkletNode | null = null;
    private playerProcessor: AudioWorkletNode | null = null;
    private mediaStream: MediaStream | null = null;
    
    // Buffers
    private jitterBuffer: JitterBuffer;
    
    // State
    private isRecording = false;
    private isPlaying = false;
    private isConnected = false;
    private sessionId = '';
    
    // Metrics
    private metrics: VoiceMetrics = {
        e2eLatency: 0,
        captureLatency: 0,
        playLatency: 0,
        jitterBufferHealth: 100,
        audioLevel: 0,
        vadState: 'silence',
        droppedFrames: 0,
        networkLatency: 0
    };
    
    // Timing
    private lastFrameTime = 0;
    private frameSequence = 0;
    
    constructor(config: VoiceClientConfig = {}, events: VoiceEvents = {}) {
        this.config = {
            sampleRate: 16000,
            frameSize: 320,
            channels: 1,
            bufferConfig: {},
            enableVAD: true,
            vadThreshold: 0.01,
            enableMonitoring: true,
            ...config
        };
        
        this.events = events;
        this.jitterBuffer = new JitterBuffer(this.config.bufferConfig);
        this.sessionId = this.generateSessionId();
        
        console.log('VoiceClient initialized:', this.config);
    }
    
    /**
     * 初始化音频系统
     */
    async initialize(): Promise<void> {
        try {
            // 创建音频上下文
            this.audioContext = new AudioContext({
                sampleRate: this.config.sampleRate,
                latencyHint: 'interactive'
            });
            
            // 加载AudioWorklet处理器
            await Promise.all([
                this.audioContext.audioWorklet.addModule('/audio/worklets/MicProcessor.js'),
                this.audioContext.audioWorklet.addModule('/audio/worklets/PlayerProcessor.js')
            ]);
            
            // 创建处理器节点
            this.micProcessor = new AudioWorkletNode(this.audioContext, 'mic-processor');
            this.playerProcessor = new AudioWorkletNode(this.audioContext, 'player-processor');
            
            // 设置事件监听
            this.setupAudioEventListeners();
            
            // 连接播放器到输出
            this.playerProcessor.connect(this.audioContext.destination);
            
            this.isConnected = true;
            this.events.onConnected?.();
            
            console.log('VoiceClient initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize VoiceClient:', error);
            this.events.onError?.(error as Error);
            throw error;
        }
    }
    
    /**
     * 开始录音
     */
    async startRecording(): Promise<void> {
        if (this.isRecording || !this.audioContext || !this.micProcessor) {
            return;
        }
        
        try {
            // 获取麦克风权限
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.config.sampleRate,
                    channelCount: this.config.channels,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true,
                    latency: 0.01
                }
            });
            
            // 连接音频流
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            source.connect(this.micProcessor);
            
            this.isRecording = true;
            this.frameSequence = 0;
            
            console.log('Recording started');
            
        } catch (error) {
            console.error('Failed to start recording:', error);
            this.events.onError?.(error as Error);
            throw error;
        }
    }
    
    /**
     * 停止录音
     */
    stopRecording(): void {
        if (!this.isRecording) return;
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
        
        if (this.micProcessor) {
            this.micProcessor.disconnect();
        }
        
        this.isRecording = false;
        console.log('Recording stopped');
    }
    
    /**
     * 开始播放
     */
    startPlayback(): void {
        if (this.isPlaying || !this.playerProcessor) return;
        
        this.playerProcessor.port.postMessage({ type: 'play' });
        this.isPlaying = true;
        this.events.onPlaybackStart?.();
        
        console.log('Playback started');
    }
    
    /**
     * 停止播放
     */
    stopPlayback(): void {
        if (!this.isPlaying || !this.playerProcessor) return;
        
        this.playerProcessor.port.postMessage({ type: 'stop' });
        this.isPlaying = false;
        this.jitterBuffer.flush();
        this.events.onPlaybackEnd?.();
        
        console.log('Playback stopped');
    }
    
    /**
     * 添加接收到的音频帧
     */
    addReceivedFrame(audioData: ArrayBuffer, timestamp: number): void {
        const frame: AudioFrame = {
            audio: audioData,
            timestamp,
            sequenceNum: this.frameSequence++,
            receivedAt: performance.now(),
            frameSize: this.config.frameSize,
            sampleRate: this.config.sampleRate
        };
        
        // 添加到jitter buffer
        this.jitterBuffer.addFrame(frame);
        
        // 发送到播放器处理器
        if (this.playerProcessor) {
            this.playerProcessor.port.postMessage({
                type: 'audioFrame',
                data: {
                    audio: audioData,
                    timestamp,
                    sequenceNum: frame.sequenceNum
                }
            });
        }
        
        // 自动开始播放
        if (!this.isPlaying && this.jitterBuffer.canStartPlayback()) {
            this.startPlayback();
        }
        
        this.events.onAudioFrame?.(frame);
    }
    
    /**
     * 打断当前播放
     */
    interrupt(): void {
        if (this.isPlaying) {
            this.stopPlayback();
            this.jitterBuffer.flush();
            
            if (this.playerProcessor) {
                this.playerProcessor.port.postMessage({ type: 'flush' });
            }
            
            console.log('Playback interrupted');
        }
    }
    
    /**
     * 获取当前指标
     */
    getMetrics(): VoiceMetrics {
        const bufferStats = this.jitterBuffer.getStats();
        
        this.metrics.jitterBufferHealth = this.jitterBuffer.getHealthScore();
        this.metrics.droppedFrames = bufferStats.droppedFrames;
        
        return { ...this.metrics };
    }
    
    /**
     * 更新配置
     */
    updateConfig(newConfig: Partial<VoiceClientConfig>): void {
        this.config = { ...this.config, ...newConfig };
        
        if (newConfig.bufferConfig) {
            this.jitterBuffer.updateConfig(newConfig.bufferConfig);
        }
        
        console.log('Config updated:', this.config);
    }
    
    /**
     * 断开连接
     */
    disconnect(): void {
        this.stopRecording();
        this.stopPlayback();
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.isConnected = false;
        this.events.onDisconnected?.();
        
        console.log('VoiceClient disconnected');
    }
    
    private setupAudioEventListeners(): void {
        if (!this.micProcessor || !this.playerProcessor) return;
        
        // 麦克风处理器事件
        this.micProcessor.port.onmessage = (event) => {
            const { type, data } = event.data;
            
            switch (type) {
                case 'audioFrame':
                    this.handleMicFrame(data);
                    break;
                    
                case 'vadStateChange':
                    this.metrics.vadState = data.state;
                    this.events.onVADStateChange?.(data.state);
                    break;
                    
                case 'performance':
                    this.metrics.captureLatency = data.processingTime;
                    this.metrics.audioLevel = data.audioLevel;
                    break;
            }
        };
        
        // 播放器处理器事件
        this.playerProcessor.port.onmessage = (event) => {
            const { type, data } = event.data;
            
            switch (type) {
                case 'bufferStatus':
                    this.updateBufferMetrics(data);
                    break;
                    
                case 'playbackStarted':
                    console.log('Playback started in processor');
                    break;
                    
                case 'playbackStopped':
                    console.log('Playback stopped in processor');
                    break;
            }
        };
        
        // 定期更新指标
        if (this.config.enableMonitoring) {
            setInterval(() => {
                this.updateMetrics();
            }, 100); // 每100ms更新一次
        }
    }
    
    private handleMicFrame(frameData: any): void {
        // 这里可以发送到服务器或进行其他处理
        const now = performance.now();
        
        if (this.lastFrameTime > 0) {
            this.metrics.captureLatency = now - this.lastFrameTime;
        }
        
        this.lastFrameTime = now;
        
        // 触发音频帧事件
        const frame: AudioFrame = {
            audio: frameData.audio,
            timestamp: frameData.timestamp,
            sequenceNum: frameData.sequenceNum,
            receivedAt: now,
            frameSize: frameData.frameSize,
            sampleRate: frameData.sampleRate
        };
        
        this.events.onAudioFrame?.(frame);
    }
    
    private updateBufferMetrics(bufferData: any): void {
        this.metrics.playLatency = bufferData.size * 20; // 每帧20ms
        this.metrics.droppedFrames = bufferData.droppedFrames;
    }
    
    private updateMetrics(): void {
        const bufferStats = this.jitterBuffer.getStats();
        
        this.metrics.e2eLatency = this.metrics.captureLatency + 
                                 this.metrics.networkLatency + 
                                 this.metrics.playLatency;
        
        this.metrics.jitterBufferHealth = this.jitterBuffer.getHealthScore();
        
        this.events.onMetricsUpdate?.(this.metrics);
    }
    
    private generateSessionId(): string {
        return 'voice_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
}
