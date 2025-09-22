/**
 * 音频传输层抽象接口
 * 功能: WebSocket/WebRTC统一接口 + 连接管理 + 事件处理
 */

import { AudioFrame } from '../buffers/JitterBuffer';

export enum TransportType {
    WEBSOCKET = 'websocket',
    WEBRTC = 'webrtc',
    SSE = 'sse'
}

export enum ConnectionState {
    DISCONNECTED = 'disconnected',
    CONNECTING = 'connecting',
    CONNECTED = 'connected',
    RECONNECTING = 'reconnecting',
    FAILED = 'failed'
}

export interface TransportConfig {
    type: TransportType;
    url: string;
    options?: {
        // WebSocket选项
        protocols?: string[];
        headers?: Record<string, string>;
        
        // WebRTC选项
        iceServers?: RTCIceServer[];
        dataChannelConfig?: RTCDataChannelInit;
        
        // 通用选项
        reconnectInterval?: number;
        maxReconnectAttempts?: number;
        heartbeatInterval?: number;
        timeout?: number;
        
        // 音频选项
        sampleRate?: number;
        channels?: number;
        frameSize?: number;
    };
}

export interface TransportStats {
    connectionState: ConnectionState;
    connectedAt?: number;
    lastActivity?: number;
    bytesSent: number;
    bytesReceived: number;
    messagesSent: number;
    messagesReceived: number;
    reconnectCount: number;
    latency?: number;
    jitter?: number;
    packetLoss?: number;
}

export interface TransportEvents {
    onConnected?: () => void;
    onDisconnected?: (reason?: string) => void;
    onReconnecting?: (attempt: number) => void;
    onError?: (error: Error) => void;
    onAudioFrame?: (frame: AudioFrame) => void;
    onTextMessage?: (message: any) => void;
    onStatsUpdate?: (stats: TransportStats) => void;
}

export abstract class AudioTransport {
    protected config: TransportConfig;
    protected events: TransportEvents;
    protected state: ConnectionState = ConnectionState.DISCONNECTED;
    protected stats: TransportStats;
    protected reconnectTimer?: NodeJS.Timeout;
    protected heartbeatTimer?: NodeJS.Timeout;
    protected reconnectAttempts = 0;

    constructor(config: TransportConfig, events: TransportEvents = {}) {
        this.config = config;
        this.events = events;
        this.stats = {
            connectionState: ConnectionState.DISCONNECTED,
            bytesSent: 0,
            bytesReceived: 0,
            messagesSent: 0,
            messagesReceived: 0,
            reconnectCount: 0
        };
    }

    // 抽象方法 - 子类必须实现
    abstract connect(): Promise<void>;
    abstract disconnect(): Promise<void>;
    abstract sendAudioFrame(frame: AudioFrame): Promise<void>;
    abstract sendTextMessage(message: any): Promise<void>;
    abstract isConnected(): boolean;

    // 通用方法
    getState(): ConnectionState {
        return this.state;
    }

    getStats(): TransportStats {
        return { ...this.stats };
    }

    updateConfig(newConfig: Partial<TransportConfig>): void {
        this.config = { ...this.config, ...newConfig };
    }

    protected setState(newState: ConnectionState): void {
        if (this.state !== newState) {
            const oldState = this.state;
            this.state = newState;
            this.stats.connectionState = newState;
            
            console.log(`Transport state changed: ${oldState} -> ${newState}`);
            
            // 触发相应事件
            switch (newState) {
                case ConnectionState.CONNECTED:
                    this.stats.connectedAt = Date.now();
                    this.reconnectAttempts = 0;
                    this.startHeartbeat();
                    this.events.onConnected?.();
                    break;
                    
                case ConnectionState.DISCONNECTED:
                    this.stopHeartbeat();
                    this.events.onDisconnected?.();
                    break;
                    
                case ConnectionState.RECONNECTING:
                    this.events.onReconnecting?.(this.reconnectAttempts);
                    break;
                    
                case ConnectionState.FAILED:
                    this.stopReconnect();
                    this.events.onError?.(new Error('Connection failed'));
                    break;
            }
        }
    }

    protected updateStats(update: Partial<TransportStats>): void {
        this.stats = { ...this.stats, ...update };
        this.stats.lastActivity = Date.now();
        this.events.onStatsUpdate?.(this.stats);
    }

    protected startReconnect(): void {
        if (this.reconnectTimer) return;
        
        const maxAttempts = this.config.options?.maxReconnectAttempts ?? 5;
        const interval = this.config.options?.reconnectInterval ?? 3000;
        
        if (this.reconnectAttempts >= maxAttempts) {
            this.setState(ConnectionState.FAILED);
            return;
        }
        
        this.setState(ConnectionState.RECONNECTING);
        this.reconnectAttempts++;
        
        this.reconnectTimer = setTimeout(async () => {
            this.reconnectTimer = undefined;
            
            try {
                await this.connect();
            } catch (error) {
                console.error(`Reconnect attempt ${this.reconnectAttempts} failed:`, error);
                this.startReconnect(); // 继续重连
            }
        }, interval * this.reconnectAttempts); // 指数退避
    }

    protected stopReconnect(): void {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = undefined;
        }
    }

    protected startHeartbeat(): void {
        const interval = this.config.options?.heartbeatInterval ?? 30000;
        
        this.heartbeatTimer = setInterval(() => {
            this.sendHeartbeat();
        }, interval);
    }

    protected stopHeartbeat(): void {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = undefined;
        }
    }

    protected async sendHeartbeat(): Promise<void> {
        try {
            await this.sendTextMessage({
                type: 'heartbeat',
                timestamp: Date.now()
            });
        } catch (error) {
            console.error('Heartbeat failed:', error);
            // 心跳失败可能表示连接问题，开始重连
            if (this.isConnected()) {
                await this.disconnect();
                this.startReconnect();
            }
        }
    }

    protected handleError(error: Error): void {
        console.error('Transport error:', error);
        this.events.onError?.(error);
        
        // 如果是连接错误，尝试重连
        if (this.state === ConnectionState.CONNECTED) {
            this.setState(ConnectionState.DISCONNECTED);
            this.startReconnect();
        }
    }

    // 清理资源
    destroy(): void {
        this.stopReconnect();
        this.stopHeartbeat();
        this.disconnect().catch(console.error);
    }
}

// 传输工厂
export class TransportFactory {
    static create(config: TransportConfig, events: TransportEvents = {}): AudioTransport {
        switch (config.type) {
            case TransportType.WEBSOCKET:
                return new WebSocketTransport(config, events);
            case TransportType.WEBRTC:
                return new WebRTCTransport(config, events);
            case TransportType.SSE:
                return new SSETransport(config, events);
            default:
                throw new Error(`Unsupported transport type: ${config.type}`);
        }
    }
}

// WebSocket实现
export class WebSocketTransport extends AudioTransport {
    private ws?: WebSocket;
    private pingInterval?: NodeJS.Timeout;

    async connect(): Promise<void> {
        if (this.ws?.readyState === WebSocket.OPEN) {
            return;
        }

        this.setState(ConnectionState.CONNECTING);

        return new Promise((resolve, reject) => {
            try {
                this.ws = new WebSocket(this.config.url, this.config.options?.protocols);
                
                const timeout = setTimeout(() => {
                    reject(new Error('Connection timeout'));
                }, this.config.options?.timeout ?? 10000);

                this.ws.onopen = () => {
                    clearTimeout(timeout);
                    this.setState(ConnectionState.CONNECTED);
                    this.startPing();
                    resolve();
                };

                this.ws.onclose = (event) => {
                    clearTimeout(timeout);
                    this.stopPing();
                    
                    if (this.state === ConnectionState.CONNECTED) {
                        this.setState(ConnectionState.DISCONNECTED);
                        this.startReconnect();
                    }
                };

                this.ws.onerror = (error) => {
                    clearTimeout(timeout);
                    this.handleError(new Error('WebSocket error'));
                    reject(error);
                };

                this.ws.onmessage = (event) => {
                    this.handleMessage(event.data);
                };

            } catch (error) {
                reject(error);
            }
        });
    }

    async disconnect(): Promise<void> {
        this.stopPing();
        
        if (this.ws) {
            this.ws.close();
            this.ws = undefined;
        }
        
        this.setState(ConnectionState.DISCONNECTED);
    }

    async sendAudioFrame(frame: AudioFrame): Promise<void> {
        if (!this.isConnected() || !this.ws) {
            throw new Error('WebSocket not connected');
        }

        // 构造二进制帧
        const header = new ArrayBuffer(20);
        const headerView = new DataView(header);
        
        headerView.setUint32(0, frame.sequenceNum, true);
        headerView.setUint32(4, frame.sampleRate, true);
        headerView.setUint8(8, 1); // channels
        headerView.setUint16(9, frame.frameSize, true);
        headerView.setBigUint64(12, BigInt(frame.timestamp), true);

        const audioData = frame.audio instanceof ArrayBuffer ? 
            new Uint8Array(frame.audio) : 
            new Uint8Array(frame.audio.buffer);

        const frameData = new Uint8Array(20 + audioData.length);
        frameData.set(new Uint8Array(header), 0);
        frameData.set(audioData, 20);

        this.ws.send(frameData);
        
        this.updateStats({
            bytesSent: this.stats.bytesSent + frameData.length,
            messagesSent: this.stats.messagesSent + 1
        });
    }

    async sendTextMessage(message: any): Promise<void> {
        if (!this.isConnected() || !this.ws) {
            throw new Error('WebSocket not connected');
        }

        const data = JSON.stringify(message);
        this.ws.send(data);
        
        this.updateStats({
            bytesSent: this.stats.bytesSent + data.length,
            messagesSent: this.stats.messagesSent + 1
        });
    }

    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }

    private handleMessage(data: any): void {
        this.updateStats({
            bytesReceived: this.stats.bytesReceived + (data.length || data.byteLength || 0),
            messagesReceived: this.stats.messagesReceived + 1
        });

        if (data instanceof ArrayBuffer) {
            // 二进制音频数据
            this.handleAudioFrame(data);
        } else {
            // 文本消息
            try {
                const message = JSON.parse(data);
                this.events.onTextMessage?.(message);
            } catch (error) {
                console.error('Failed to parse message:', error);
            }
        }
    }

    private handleAudioFrame(data: ArrayBuffer): void {
        if (data.byteLength < 20) return;

        const headerView = new DataView(data, 0, 20);
        const audioData = data.slice(20);

        const frame: AudioFrame = {
            audio: audioData,
            timestamp: Number(headerView.getBigUint64(12, true)),
            sequenceNum: headerView.getUint32(0, true),
            receivedAt: Date.now(),
            frameSize: headerView.getUint16(9, true),
            sampleRate: headerView.getUint32(4, true)
        };

        this.events.onAudioFrame?.(frame);
    }

    private startPing(): void {
        this.pingInterval = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.ping?.();
            }
        }, 30000);
    }

    private stopPing(): void {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = undefined;
        }
    }
}

// WebRTC实现 (基础框架)
export class WebRTCTransport extends AudioTransport {
    private peerConnection?: RTCPeerConnection;
    private dataChannel?: RTCDataChannel;
    private signalingClient?: WebSocket;

    async connect(): Promise<void> {
        this.setState(ConnectionState.CONNECTING);
        
        // 建立信令连接
        await this.connectSignaling();
        
        // 创建PeerConnection
        await this.createPeerConnection();
        
        // 创建数据通道
        this.createDataChannel();
        
        this.setState(ConnectionState.CONNECTED);
    }

    async disconnect(): Promise<void> {
        if (this.dataChannel) {
            this.dataChannel.close();
            this.dataChannel = undefined;
        }
        
        if (this.peerConnection) {
            this.peerConnection.close();
            this.peerConnection = undefined;
        }
        
        if (this.signalingClient) {
            this.signalingClient.close();
            this.signalingClient = undefined;
        }
        
        this.setState(ConnectionState.DISCONNECTED);
    }

    async sendAudioFrame(frame: AudioFrame): Promise<void> {
        if (!this.dataChannel || this.dataChannel.readyState !== 'open') {
            throw new Error('DataChannel not ready');
        }

        // 实现音频帧发送
        const data = new Uint8Array(frame.audio instanceof ArrayBuffer ? frame.audio : frame.audio.buffer);
        this.dataChannel.send(data);
        
        this.updateStats({
            bytesSent: this.stats.bytesSent + data.length,
            messagesSent: this.stats.messagesSent + 1
        });
    }

    async sendTextMessage(message: any): Promise<void> {
        if (!this.dataChannel || this.dataChannel.readyState !== 'open') {
            throw new Error('DataChannel not ready');
        }

        const data = JSON.stringify(message);
        this.dataChannel.send(data);
        
        this.updateStats({
            bytesSent: this.stats.bytesSent + data.length,
            messagesSent: this.stats.messagesSent + 1
        });
    }

    isConnected(): boolean {
        return this.dataChannel?.readyState === 'open';
    }

    private async connectSignaling(): Promise<void> {
        // 信令服务器连接实现
        const signalingUrl = this.config.url.replace('webrtc://', 'ws://') + '/signaling';
        
        return new Promise((resolve, reject) => {
            this.signalingClient = new WebSocket(signalingUrl);
            
            this.signalingClient.onopen = () => resolve();
            this.signalingClient.onerror = (error) => reject(error);
            this.signalingClient.onmessage = (event) => {
                this.handleSignalingMessage(JSON.parse(event.data));
            };
        });
    }

    private async createPeerConnection(): Promise<void> {
        const config: RTCConfiguration = {
            iceServers: this.config.options?.iceServers ?? [
                { urls: 'stun:stun.l.google.com:19302' }
            ]
        };

        this.peerConnection = new RTCPeerConnection(config);
        
        this.peerConnection.onicecandidate = (event) => {
            if (event.candidate && this.signalingClient) {
                this.signalingClient.send(JSON.stringify({
                    type: 'ice-candidate',
                    candidate: event.candidate
                }));
            }
        };

        this.peerConnection.onconnectionstatechange = () => {
            console.log('PeerConnection state:', this.peerConnection?.connectionState);
        };
    }

    private createDataChannel(): void {
        if (!this.peerConnection) return;

        const config = this.config.options?.dataChannelConfig ?? {
            ordered: false,
            maxRetransmits: 0
        };

        this.dataChannel = this.peerConnection.createDataChannel('audio', config);
        
        this.dataChannel.onopen = () => {
            console.log('DataChannel opened');
        };

        this.dataChannel.onmessage = (event) => {
            this.handleDataChannelMessage(event.data);
        };

        this.dataChannel.onerror = (error) => {
            console.error('DataChannel error:', error);
        };
    }

    private handleSignalingMessage(message: any): void {
        // 处理信令消息 (offer/answer/ice-candidate)
        switch (message.type) {
            case 'offer':
                this.handleOffer(message.offer);
                break;
            case 'answer':
                this.handleAnswer(message.answer);
                break;
            case 'ice-candidate':
                this.handleIceCandidate(message.candidate);
                break;
        }
    }

    private async handleOffer(offer: RTCSessionDescriptionInit): Promise<void> {
        if (!this.peerConnection) return;

        await this.peerConnection.setRemoteDescription(offer);
        const answer = await this.peerConnection.createAnswer();
        await this.peerConnection.setLocalDescription(answer);

        if (this.signalingClient) {
            this.signalingClient.send(JSON.stringify({
                type: 'answer',
                answer: answer
            }));
        }
    }

    private async handleAnswer(answer: RTCSessionDescriptionInit): Promise<void> {
        if (!this.peerConnection) return;
        await this.peerConnection.setRemoteDescription(answer);
    }

    private async handleIceCandidate(candidate: RTCIceCandidateInit): Promise<void> {
        if (!this.peerConnection) return;
        await this.peerConnection.addIceCandidate(candidate);
    }

    private handleDataChannelMessage(data: any): void {
        this.updateStats({
            bytesReceived: this.stats.bytesReceived + (data.length || data.byteLength || 0),
            messagesReceived: this.stats.messagesReceived + 1
        });

        if (data instanceof ArrayBuffer) {
            this.handleAudioFrame(data);
        } else {
            try {
                const message = JSON.parse(data);
                this.events.onTextMessage?.(message);
            } catch (error) {
                console.error('Failed to parse DataChannel message:', error);
            }
        }
    }

    private handleAudioFrame(data: ArrayBuffer): void {
        // 解析音频帧 (简化版)
        const frame: AudioFrame = {
            audio: data,
            timestamp: Date.now(),
            sequenceNum: 0,
            receivedAt: Date.now(),
            frameSize: data.byteLength,
            sampleRate: 16000
        };

        this.events.onAudioFrame?.(frame);
    }
}

// SSE实现 (仅用于文本流)
export class SSETransport extends AudioTransport {
    private eventSource?: EventSource;

    async connect(): Promise<void> {
        this.setState(ConnectionState.CONNECTING);

        return new Promise((resolve, reject) => {
            this.eventSource = new EventSource(this.config.url);

            this.eventSource.onopen = () => {
                this.setState(ConnectionState.CONNECTED);
                resolve();
            };

            this.eventSource.onerror = (error) => {
                this.handleError(new Error('SSE connection error'));
                reject(error);
            };

            this.eventSource.onmessage = (event) => {
                this.handleMessage(event.data);
            };
        });
    }

    async disconnect(): Promise<void> {
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = undefined;
        }
        this.setState(ConnectionState.DISCONNECTED);
    }

    async sendAudioFrame(frame: AudioFrame): Promise<void> {
        throw new Error('SSE does not support audio frame sending');
    }

    async sendTextMessage(message: any): Promise<void> {
        // SSE是单向的，需要通过HTTP发送
        const response = await fetch(this.config.url.replace('/stream', '/send'), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(message)
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
    }

    isConnected(): boolean {
        return this.eventSource?.readyState === EventSource.OPEN;
    }

    private handleMessage(data: string): void {
        try {
            const message = JSON.parse(data);
            this.events.onTextMessage?.(message);
            
            this.updateStats({
                bytesReceived: this.stats.bytesReceived + data.length,
                messagesReceived: this.stats.messagesReceived + 1
            });
        } catch (error) {
            console.error('Failed to parse SSE message:', error);
        }
    }
}
