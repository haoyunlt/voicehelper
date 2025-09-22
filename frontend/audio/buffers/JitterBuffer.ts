/**
 * Jitter Buffer - 音频抖动缓冲器
 * 功能: 时间戳排序 + 自适应缓冲 + 追帧策略
 */

export interface AudioFrame {
    audio: Float32Array | ArrayBuffer;
    timestamp: number;
    sequenceNum: number;
    receivedAt: number;
    frameSize: number;
    sampleRate: number;
}

export interface JitterBufferConfig {
    bufferSize: number;       // 最大缓冲帧数
    targetSize: number;       // 目标缓冲帧数
    minSize: number;          // 最小缓冲帧数
    maxDropFrames: number;    // 最多丢弃帧数
    adaptiveMode: boolean;    // 自适应追帧模式
    catchupThreshold: number; // 追帧阈值
    catchupRate: number;      // 追帧播放速率
}

export interface BufferStats {
    size: number;
    targetSize: number;
    droppedFrames: number;
    lateFrames: number;
    duplicatedFrames: number;
    underruns: number;
    overruns: number;
    avgJitter: number;
    currentRate: number;
}

export class JitterBuffer {
    private buffer: AudioFrame[] = [];
    private config: JitterBufferConfig;
    private stats: BufferStats;
    private lastTimestamp = 0;
    private jitterSum = 0;
    private jitterCount = 0;
    private currentPlaybackRate = 1.0;
    
    constructor(config: Partial<JitterBufferConfig> = {}) {
        this.config = {
            bufferSize: 10,
            targetSize: 5,
            minSize: 2,
            maxDropFrames: 3,
            adaptiveMode: true,
            catchupThreshold: 7,
            catchupRate: 1.1,
            ...config
        };
        
        this.stats = {
            size: 0,
            targetSize: this.config.targetSize,
            droppedFrames: 0,
            lateFrames: 0,
            duplicatedFrames: 0,
            underruns: 0,
            overruns: 0,
            avgJitter: 0,
            currentRate: 1.0
        };
    }
    
    /**
     * 添加音频帧到缓冲区
     */
    addFrame(frame: AudioFrame): boolean {
        // 检查重复帧
        if (this.isDuplicateFrame(frame)) {
            this.stats.duplicatedFrames++;
            return false;
        }
        
        // 计算抖动
        this.updateJitterStats(frame);
        
        // 按时间戳排序插入
        const insertIndex = this.findInsertPosition(frame.timestamp);
        this.buffer.splice(insertIndex, 0, frame);
        
        // 检查迟到帧
        if (insertIndex < this.buffer.length - 1) {
            this.stats.lateFrames++;
        }
        
        // 缓冲区溢出处理
        if (this.buffer.length > this.config.bufferSize) {
            this.handleOverflow();
        }
        
        this.stats.size = this.buffer.length;
        
        // 更新播放速率
        if (this.config.adaptiveMode) {
            this.updatePlaybackRate();
        }
        
        return true;
    }
    
    /**
     * 获取下一个播放帧
     */
    getFrame(): AudioFrame | null {
        if (this.buffer.length < this.config.minSize) {
            this.stats.underruns++;
            return null;
        }
        
        const frame = this.buffer.shift();
        if (frame) {
            this.stats.size = this.buffer.length;
            return frame;
        }
        
        return null;
    }
    
    /**
     * 检查是否可以开始播放
     */
    canStartPlayback(): boolean {
        return this.buffer.length >= this.config.targetSize;
    }
    
    /**
     * 获取当前播放速率
     */
    getPlaybackRate(): number {
        return this.currentPlaybackRate;
    }
    
    /**
     * 获取缓冲区统计信息
     */
    getStats(): BufferStats {
        return { ...this.stats };
    }
    
    /**
     * 清空缓冲区
     */
    flush(): void {
        this.buffer = [];
        this.stats.size = 0;
        this.currentPlaybackRate = 1.0;
        this.lastTimestamp = 0;
    }
    
    /**
     * 更新配置
     */
    updateConfig(newConfig: Partial<JitterBufferConfig>): void {
        this.config = { ...this.config, ...newConfig };
        this.stats.targetSize = this.config.targetSize;
    }
    
    private isDuplicateFrame(frame: AudioFrame): boolean {
        return this.buffer.some(bufferedFrame => 
            bufferedFrame.sequenceNum === frame.sequenceNum ||
            Math.abs(bufferedFrame.timestamp - frame.timestamp) < 1
        );
    }
    
    private updateJitterStats(frame: AudioFrame): void {
        if (this.lastTimestamp > 0) {
            const expectedInterval = 20; // 20ms per frame
            const actualInterval = frame.timestamp - this.lastTimestamp;
            const jitter = Math.abs(actualInterval - expectedInterval);
            
            this.jitterSum += jitter;
            this.jitterCount++;
            this.stats.avgJitter = this.jitterSum / this.jitterCount;
        }
        
        this.lastTimestamp = frame.timestamp;
    }
    
    private findInsertPosition(timestamp: number): number {
        for (let i = 0; i < this.buffer.length; i++) {
            if (this.buffer[i].timestamp > timestamp) {
                return i;
            }
        }
        return this.buffer.length;
    }
    
    private handleOverflow(): void {
        // 优先丢弃最旧的帧
        const droppedFrame = this.buffer.shift();
        if (droppedFrame) {
            this.stats.droppedFrames++;
            this.stats.overruns++;
            console.warn(`JitterBuffer overflow: dropped frame ${droppedFrame.sequenceNum}`);
        }
    }
    
    private updatePlaybackRate(): void {
        const bufferLength = this.buffer.length;
        
        if (bufferLength > this.config.catchupThreshold) {
            // 缓冲区积压，加速播放
            this.currentPlaybackRate = this.config.catchupRate;
        } else if (bufferLength < this.config.minSize) {
            // 缓冲区不足，减速播放
            this.currentPlaybackRate = 0.95;
        } else {
            // 正常播放
            this.currentPlaybackRate = 1.0;
        }
        
        this.stats.currentRate = this.currentPlaybackRate;
    }
    
    /**
     * 获取缓冲区健康度评分 (0-100)
     */
    getHealthScore(): number {
        const sizeScore = Math.min(100, (this.buffer.length / this.config.targetSize) * 50);
        const jitterScore = Math.max(0, 100 - this.stats.avgJitter * 2);
        const dropScore = Math.max(0, 100 - this.stats.droppedFrames);
        
        return Math.round((sizeScore + jitterScore + dropScore) / 3);
    }
    
    /**
     * 获取延迟估计 (毫秒)
     */
    getLatencyEstimate(): number {
        return this.buffer.length * 20; // 每帧20ms
    }
}
