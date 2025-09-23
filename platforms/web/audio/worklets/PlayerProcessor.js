/**
 * 播放器音频处理器 - 优化版AudioWorklet
 * 功能: Jitter Buffer + 追帧策略 + 平滑播放
 */

class PlayerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        
        // 音频配置
        this.sampleRate = 16000;
        this.frameSize = 320;     // 20ms @ 16kHz
        this.channels = 1;
        
        // Jitter Buffer配置
        this.bufferSize = 8;      // 最大缓冲帧数 (160ms)
        this.targetSize = 4;      // 目标缓冲帧数 (80ms)
        this.minSize = 2;         // 最小缓冲帧数 (40ms)
        this.maxDropFrames = 3;   // 最多丢弃帧数
        
        // 缓冲区
        this.jitterBuffer = [];
        this.outputBuffer = new Float32Array(this.frameSize);
        this.outputIndex = 0;
        
        // 播放状态
        this.isPlaying = false;
        this.playbackStarted = false;
        this.underrunCount = 0;
        this.overrunCount = 0;
        
        // 追帧策略
        this.adaptiveMode = true;
        this.catchupThreshold = 6;  // 超过6帧开始追帧
        this.catchupRate = 1.1;     // 追帧播放速率
        this.normalRate = 1.0;      // 正常播放速率
        this.currentRate = 1.0;
        
        // 音频质量监控
        this.frameCount = 0;
        this.droppedFrames = 0;
        this.lateFrames = 0;
        this.duplicatedFrames = 0;
        
        // 平滑处理
        this.fadeInSamples = 32;    // 淡入样本数
        this.fadeOutSamples = 32;   // 淡出样本数
        this.lastSample = 0;
        
        console.log('PlayerProcessor initialized: Jitter buffer, adaptive catchup enabled');
        
        // 监听主线程消息
        this.port.onmessage = (event) => {
            this.handleMessage(event.data);
        };
    }
    
    process(inputs, outputs, parameters) {
        const output = outputs[0];
        
        if (output && output.length > 0) {
            const outputChannel = output[0];
            this.generateOutput(outputChannel);
        }
        
        this.frameCount++;
        
        // 每100帧报告一次状态
        if (this.frameCount % 100 === 0) {
            this.reportStatus();
        }
        
        return true;
    }
    
    handleMessage(message) {
        switch (message.type) {
            case 'audioFrame':
                this.addAudioFrame(message.data);
                break;
                
            case 'play':
                this.startPlayback();
                break;
                
            case 'pause':
                this.pausePlayback();
                break;
                
            case 'stop':
                this.stopPlayback();
                break;
                
            case 'flush':
                this.flushBuffer();
                break;
                
            case 'setConfig':
                this.updateConfig(message.config);
                break;
        }
    }
    
    addAudioFrame(frameData) {
        const frame = {
            audio: new Float32Array(frameData.audio),
            timestamp: frameData.timestamp,
            sequenceNum: frameData.sequenceNum,
            receivedAt: currentTime * 1000
        };
        
        // 按时间戳排序插入
        const insertIndex = this.findInsertPosition(frame.timestamp);
        this.jitterBuffer.splice(insertIndex, 0, frame);
        
        // 检查是否为迟到帧
        if (insertIndex < this.jitterBuffer.length - 1) {
            this.lateFrames++;
        }
        
        // 缓冲区溢出处理
        if (this.jitterBuffer.length > this.bufferSize) {
            this.handleBufferOverflow();
        }
        
        // 自动开始播放
        if (!this.playbackStarted && this.jitterBuffer.length >= this.targetSize) {
            this.startPlayback();
        }
        
        // 自适应追帧
        if (this.adaptiveMode) {
            this.updatePlaybackRate();
        }
    }
    
    findInsertPosition(timestamp) {
        for (let i = 0; i < this.jitterBuffer.length; i++) {
            if (this.jitterBuffer[i].timestamp > timestamp) {
                return i;
            }
        }
        return this.jitterBuffer.length;
    }
    
    handleBufferOverflow() {
        // 优先丢弃最旧的帧
        const droppedFrame = this.jitterBuffer.shift();
        this.droppedFrames++;
        this.overrunCount++;
        
        console.warn(`Buffer overflow: dropped frame ${droppedFrame?.sequenceNum}, buffer size: ${this.jitterBuffer.length}`);
    }
    
    updatePlaybackRate() {
        const bufferLength = this.jitterBuffer.length;
        
        if (bufferLength > this.catchupThreshold) {
            // 缓冲区积压，加速播放
            this.currentRate = this.catchupRate;
        } else if (bufferLength < this.minSize) {
            // 缓冲区不足，减速播放
            this.currentRate = 0.9;
        } else {
            // 正常播放
            this.currentRate = this.normalRate;
        }
    }
    
    generateOutput(outputChannel) {
        if (!this.isPlaying) {
            // 静音输出
            outputChannel.fill(0);
            return;
        }
        
        for (let i = 0; i < outputChannel.length; i++) {
            if (this.outputIndex >= this.outputBuffer.length) {
                // 需要新的音频帧
                if (!this.loadNextFrame()) {
                    // 缓冲区下溢，输出静音
                    outputChannel[i] = 0;
                    this.underrunCount++;
                    continue;
                }
                this.outputIndex = 0;
            }
            
            // 应用播放速率调整
            const sampleIndex = Math.floor(this.outputIndex * this.currentRate);
            if (sampleIndex < this.outputBuffer.length) {
                let sample = this.outputBuffer[sampleIndex];
                
                // 应用平滑处理
                sample = this.applySmoothTransition(sample, i);
                
                outputChannel[i] = sample;
                this.lastSample = sample;
            } else {
                outputChannel[i] = 0;
            }
            
            this.outputIndex += this.currentRate;
        }
    }
    
    loadNextFrame() {
        if (this.jitterBuffer.length < this.minSize) {
            return false;
        }
        
        const frame = this.jitterBuffer.shift();
        if (!frame) {
            return false;
        }
        
        // 转换PCM16到Float32
        if (frame.audio instanceof ArrayBuffer) {
            this.outputBuffer = this.pcm16ToFloat(frame.audio);
        } else {
            this.outputBuffer.set(frame.audio);
        }
        
        return true;
    }
    
    pcm16ToFloat(arrayBuffer) {
        const view = new DataView(arrayBuffer);
        const output = new Float32Array(arrayBuffer.byteLength / 2);
        
        for (let i = 0; i < output.length; i++) {
            output[i] = view.getInt16(i * 2, true) / 0x7FFF;
        }
        
        return output;
    }
    
    applySmoothTransition(sample, sampleIndex) {
        // 在帧边界应用淡入淡出，避免爆音
        if (this.outputIndex < this.fadeInSamples) {
            const fadeRatio = this.outputIndex / this.fadeInSamples;
            sample = sample * fadeRatio + this.lastSample * (1 - fadeRatio);
        }
        
        return sample;
    }
    
    startPlayback() {
        this.isPlaying = true;
        this.playbackStarted = true;
        
        this.port.postMessage({
            type: 'playbackStarted',
            data: {
                bufferSize: this.jitterBuffer.length,
                timestamp: currentTime * 1000
            }
        });
    }
    
    pausePlayback() {
        this.isPlaying = false;
        
        this.port.postMessage({
            type: 'playbackPaused',
            data: {
                timestamp: currentTime * 1000
            }
        });
    }
    
    stopPlayback() {
        this.isPlaying = false;
        this.playbackStarted = false;
        this.flushBuffer();
        
        this.port.postMessage({
            type: 'playbackStopped',
            data: {
                timestamp: currentTime * 1000
            }
        });
    }
    
    flushBuffer() {
        this.jitterBuffer = [];
        this.outputBuffer.fill(0);
        this.outputIndex = 0;
        this.currentRate = this.normalRate;
    }
    
    updateConfig(config) {
        if (config.bufferSize !== undefined) {
            this.bufferSize = config.bufferSize;
        }
        if (config.targetSize !== undefined) {
            this.targetSize = config.targetSize;
        }
        if (config.adaptiveMode !== undefined) {
            this.adaptiveMode = config.adaptiveMode;
        }
    }
    
    reportStatus() {
        const bufferHealth = {
            size: this.jitterBuffer.length,
            target: this.targetSize,
            underruns: this.underrunCount,
            overruns: this.overrunCount,
            droppedFrames: this.droppedFrames,
            lateFrames: this.lateFrames,
            currentRate: this.currentRate,
            isPlaying: this.isPlaying
        };
        
        this.port.postMessage({
            type: 'bufferStatus',
            data: bufferHealth
        });
        
        // 重置计数器
        this.underrunCount = 0;
        this.overrunCount = 0;
    }
}

registerProcessor('player-processor', PlayerProcessor);
