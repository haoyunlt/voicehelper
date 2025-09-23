/**
 * 麦克风音频处理器 - 优化版AudioWorklet
 * 功能: VAD前置 + 16k/mono/PCM16 分帧(20ms) + 降噪处理
 */

class MicProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        
        // 音频配置
        this.sampleRate = 16000;  // 16kHz采样率
        this.frameSize = 320;     // 20ms @ 16kHz = 320 samples
        this.channels = 1;        // 单声道
        
        // 缓冲区
        this.inputBuffer = new Float32Array(this.frameSize);
        this.bufferIndex = 0;
        
        // VAD (Voice Activity Detection)
        this.vadThreshold = 0.01;
        this.vadFrameCount = 0;
        this.vadSilenceCount = 0;
        this.vadMaxSilence = 10;  // 最多10帧静音
        this.vadMinSpeech = 3;    // 最少3帧语音
        this.vadState = 'silence';
        
        // 音频预处理
        this.noiseGate = 0.005;
        this.highPassAlpha = 0.95;
        this.prevSample = 0;
        this.prevOutput = 0;
        
        // 性能监控
        this.frameCount = 0;
        this.processTime = 0;
        this.audioLevel = 0;
        
        console.log('MicProcessor initialized: 16kHz, 20ms frames, VAD enabled');
    }
    
    process(inputs, outputs, parameters) {
        const startTime = performance.now();
        const input = inputs[0];
        
        if (input && input.length > 0) {
            const inputChannel = input[0];
            
            if (inputChannel && inputChannel.length > 0) {
                this.processAudioFrame(inputChannel);
            }
        }
        
        // 性能统计
        this.processTime = performance.now() - startTime;
        this.frameCount++;
        
        // 每100帧报告一次性能
        if (this.frameCount % 100 === 0) {
            this.port.postMessage({
                type: 'performance',
                data: {
                    frameCount: this.frameCount,
                    avgProcessTime: this.processTime,
                    audioLevel: this.audioLevel,
                    vadState: this.vadState
                }
            });
        }
        
        return true;
    }
    
    processAudioFrame(audioData) {
        // 重采样到16kHz (如果需要)
        const resampledData = this.resampleTo16k(audioData);
        
        // 音频预处理
        const processedData = this.preprocessAudio(resampledData);
        
        // 填充帧缓冲区
        for (let i = 0; i < processedData.length; i++) {
            this.inputBuffer[this.bufferIndex] = processedData[i];
            this.bufferIndex++;
            
            // 当缓冲区满时(20ms帧)，进行VAD检测和发送
            if (this.bufferIndex >= this.frameSize) {
                this.processCompleteFrame();
                this.bufferIndex = 0;
            }
        }
    }
    
    processCompleteFrame() {
        // 计算音频能量
        this.audioLevel = this.calculateAudioLevel(this.inputBuffer);
        
        // VAD检测
        const vadResult = this.performVAD(this.audioLevel);
        
        // 根据VAD结果决定是否发送音频帧
        if (vadResult.shouldSend) {
            // 转换为PCM16格式
            const pcm16Data = this.floatToPCM16(this.inputBuffer);
            
            this.port.postMessage({
                type: 'audioFrame',
                data: {
                    audio: pcm16Data,
                    timestamp: currentTime * 1000, // 转换为毫秒
                    frameSize: this.frameSize,
                    sampleRate: this.sampleRate,
                    channels: this.channels,
                    audioLevel: this.audioLevel,
                    vadState: this.vadState,
                    sequenceNum: this.frameCount
                }
            });
        }
        
        // 发送VAD状态变化事件
        if (vadResult.stateChanged) {
            this.port.postMessage({
                type: 'vadStateChange',
                data: {
                    state: this.vadState,
                    timestamp: currentTime * 1000,
                    audioLevel: this.audioLevel
                }
            });
        }
    }
    
    resampleTo16k(audioData) {
        // 简单的重采样实现 (实际项目中应使用更高质量的重采样算法)
        const currentSampleRate = sampleRate;
        if (currentSampleRate === this.sampleRate) {
            return audioData;
        }
        
        const ratio = currentSampleRate / this.sampleRate;
        const outputLength = Math.floor(audioData.length / ratio);
        const output = new Float32Array(outputLength);
        
        for (let i = 0; i < outputLength; i++) {
            const srcIndex = i * ratio;
            const srcIndexFloor = Math.floor(srcIndex);
            const srcIndexCeil = Math.min(srcIndexFloor + 1, audioData.length - 1);
            const fraction = srcIndex - srcIndexFloor;
            
            // 线性插值
            output[i] = audioData[srcIndexFloor] * (1 - fraction) + 
                       audioData[srcIndexCeil] * fraction;
        }
        
        return output;
    }
    
    preprocessAudio(audioData) {
        const processed = new Float32Array(audioData.length);
        
        for (let i = 0; i < audioData.length; i++) {
            let sample = audioData[i];
            
            // 1. 噪声门限
            if (Math.abs(sample) < this.noiseGate) {
                sample = 0;
            }
            
            // 2. 高通滤波器 (去除低频噪声)
            const output = this.highPassAlpha * (this.prevOutput + sample - this.prevSample);
            this.prevSample = sample;
            this.prevOutput = output;
            sample = output;
            
            // 3. 软限幅
            if (sample > 0.95) sample = 0.95;
            if (sample < -0.95) sample = -0.95;
            
            processed[i] = sample;
        }
        
        return processed;
    }
    
    calculateAudioLevel(audioData) {
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i] * audioData[i];
        }
        return Math.sqrt(sum / audioData.length);
    }
    
    performVAD(audioLevel) {
        const isSpeech = audioLevel > this.vadThreshold;
        let stateChanged = false;
        let shouldSend = false;
        
        if (isSpeech) {
            this.vadSilenceCount = 0;
            this.vadFrameCount++;
            
            if (this.vadState === 'silence' && this.vadFrameCount >= this.vadMinSpeech) {
                this.vadState = 'speech';
                stateChanged = true;
            }
            
            if (this.vadState === 'speech') {
                shouldSend = true;
            }
        } else {
            this.vadSilenceCount++;
            this.vadFrameCount = 0;
            
            if (this.vadState === 'speech' && this.vadSilenceCount >= this.vadMaxSilence) {
                this.vadState = 'silence';
                stateChanged = true;
            }
            
            // 在语音状态下，即使是静音帧也发送(用于端点检测)
            if (this.vadState === 'speech' && this.vadSilenceCount < this.vadMaxSilence) {
                shouldSend = true;
            }
        }
        
        return { shouldSend, stateChanged };
    }
    
    floatToPCM16(float32Array) {
        const buffer = new ArrayBuffer(float32Array.length * 2);
        const view = new DataView(buffer);
        
        for (let i = 0; i < float32Array.length; i++) {
            const sample = Math.max(-1, Math.min(1, float32Array[i]));
            view.setInt16(i * 2, sample * 0x7FFF, true); // little-endian
        }
        
        return buffer;
    }
}

registerProcessor('mic-processor', MicProcessor);
