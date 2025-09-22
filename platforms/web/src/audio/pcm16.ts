/**
 * V2架构PCM16音频处理器
 * 基于BaseAudioProcessor和BasePlayer实现PCM16格式的音频处理
 */

import { BaseAudioProcessor, BasePlayer, AudioConfig } from './base';

export class PCM16Processor extends BaseAudioProcessor {
  private mediaRecorder?: MediaRecorder;
  private audioContext?: AudioContext;
  private workletNode?: AudioWorkletNode;
  private stream?: MediaStream;
  private onDataCallback?: (data: ArrayBuffer) => void;
  
  async start(): Promise<void> {
    if (this.isRecording) {
      return;
    }
    
    try {
      // 请求麦克风权限
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: this.config.sampleRate,
          channelCount: this.config.channels,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        }
      });
      
      // 创建音频上下文
      this.audioContext = new AudioContext({
        sampleRate: this.config.sampleRate
      });
      
      // 加载AudioWorklet处理器
      await this.loadAudioWorklet();
      
      // 创建音频节点
      const source = this.audioContext.createMediaStreamSource(this.stream);
      this.workletNode = new AudioWorkletNode(this.audioContext, 'pcm16-processor', {
        processorOptions: {
          sampleRate: this.config.sampleRate,
          channels: this.config.channels,
          bitDepth: this.config.bitDepth
        }
      });
      
      // 设置消息处理
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'audioData' && this.onDataCallback) {
          this.onDataCallback(event.data.buffer);
        }
      };
      
      // 连接音频节点
      source.connect(this.workletNode);
      
      this.isRecording = true;
      console.log('PCM16音频录制开始');
      
    } catch (error) {
      console.error('启动音频录制失败:', error);
      throw error;
    }
  }
  
  async stop(): Promise<void> {
    if (!this.isRecording) {
      return;
    }
    
    this.isRecording = false;
    
    // 停止音频流
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = undefined;
    }
    
    // 断开音频节点
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = undefined;
    }
    
    // 关闭音频上下文
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = undefined;
    }
    
    console.log('PCM16音频录制停止');
  }
  
  onAudioData(callback: (data: ArrayBuffer) => void): void {
    this.onDataCallback = callback;
  }
  
  private async loadAudioWorklet(): Promise<void> {
    if (!this.audioContext) {
      throw new Error('AudioContext not initialized');
    }
    
    try {
      // 尝试加载AudioWorklet模块
      await this.audioContext.audioWorklet.addModule('/audio/pcm16-processor.js');
    } catch (error) {
      // 如果加载失败，创建内联处理器
      console.warn('Failed to load external worklet, using inline processor');
      await this.createInlineWorklet();
    }
  }
  
  private async createInlineWorklet(): Promise<void> {
    const processorCode = `
      class PCM16Processor extends AudioWorkletProcessor {
        constructor(options) {
          super();
          this.sampleRate = options.processorOptions.sampleRate || 16000;
          this.channels = options.processorOptions.channels || 1;
          this.bitDepth = options.processorOptions.bitDepth || 16;
          this.bufferSize = 4096;
          this.buffer = new Float32Array(this.bufferSize);
          this.bufferIndex = 0;
        }
        
        process(inputs, outputs, parameters) {
          const input = inputs[0];
          if (input && input.length > 0) {
            const inputChannel = input[0];
            
            for (let i = 0; i < inputChannel.length; i++) {
              this.buffer[this.bufferIndex++] = inputChannel[i];
              
              if (this.bufferIndex >= this.bufferSize) {
                this.sendBuffer();
                this.bufferIndex = 0;
              }
            }
          }
          
          return true;
        }
        
        sendBuffer() {
          // 转换为PCM16格式
          const pcm16Buffer = new Int16Array(this.bufferSize);
          for (let i = 0; i < this.bufferSize; i++) {
            pcm16Buffer[i] = Math.max(-32768, Math.min(32767, this.buffer[i] * 32767));
          }
          
          this.port.postMessage({
            type: 'audioData',
            buffer: pcm16Buffer.buffer
          });
        }
      }
      
      registerProcessor('pcm16-processor', PCM16Processor);
    `;
    
    const blob = new Blob([processorCode], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    
    await this.audioContext!.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
  }
}

export class PCMChunkPlayer extends BasePlayer {
  private audioContext?: AudioContext;
  private gainNode?: GainNode;
  private currentSource?: AudioBufferSourceNode;
  private isInitialized = false;
  
  constructor() {
    super();
    this.initializeAudioContext();
  }
  
  private async initializeAudioContext(): Promise<void> {
    try {
      this.audioContext = new AudioContext({ sampleRate: 16000 });
      this.gainNode = this.audioContext.createGain();
      this.gainNode.connect(this.audioContext.destination);
      this.gainNode.gain.value = this.volume;
      this.isInitialized = true;
    } catch (error) {
      console.error('初始化音频上下文失败:', error);
    }
  }
  
  async play(audioData: ArrayBuffer): Promise<void> {
    if (!this.isInitialized || !this.audioContext || !this.gainNode) {
      throw new Error('Audio player not initialized');
    }
    
    // 如果音频上下文被暂停，恢复它
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
    
    try {
      // 检测音频格式并解码
      let audioBuffer: AudioBuffer;
      
      if (this.isMP3Data(audioData)) {
        // MP3数据，直接解码
        audioBuffer = await this.audioContext.decodeAudioData(audioData);
      } else {
        // 假设是PCM16数据，手动创建AudioBuffer
        audioBuffer = await this.createAudioBufferFromPCM16(audioData);
      }
      
      // 创建音频源
      const source = this.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(this.gainNode);
      
      // 播放音频
      this.currentSource = source;
      source.start();
      this.isPlaying = true;
      
      // 设置播放结束回调
      source.onended = () => {
        this.isPlaying = false;
        this.playNext();
      };
      
    } catch (error) {
      console.error('播放音频失败:', error);
      throw error;
    }
  }
  
  stop(): void {
    if (this.currentSource) {
      this.currentSource.stop();
      this.currentSource = undefined;
    }
    this.clearQueue();
    this.isPlaying = false;
  }
  
  setVolume(volume: number): void {
    this.volume = Math.max(0, Math.min(1, volume));
    if (this.gainNode) {
      this.gainNode.gain.value = this.volume;
    }
  }
  
  private async playNext(): Promise<void> {
    const nextData = this.dequeue();
    if (nextData) {
      await this.play(nextData);
    }
  }
  
  private isMP3Data(data: ArrayBuffer): boolean {
    const view = new Uint8Array(data);
    // 检查MP3文件头
    return view.length > 3 && 
           ((view[0] === 0xFF && (view[1] & 0xE0) === 0xE0) || // MP3 frame header
            (view[0] === 0x49 && view[1] === 0x44 && view[2] === 0x33)); // ID3 tag
  }
  
  private async createAudioBufferFromPCM16(data: ArrayBuffer): Promise<AudioBuffer> {
    if (!this.audioContext) {
      throw new Error('AudioContext not available');
    }
    
    const pcm16Data = new Int16Array(data);
    const sampleRate = 16000;
    const channels = 1;
    
    // 创建AudioBuffer
    const audioBuffer = this.audioContext.createBuffer(channels, pcm16Data.length, sampleRate);
    const channelData = audioBuffer.getChannelData(0);
    
    // 转换PCM16到Float32
    for (let i = 0; i < pcm16Data.length; i++) {
      channelData[i] = pcm16Data[i] / 32768;
    }
    
    return audioBuffer;
  }
}
