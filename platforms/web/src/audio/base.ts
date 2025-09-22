/**
 * V2架构音频处理器基类
 * 定义音频采集和播放的抽象接口
 */

export interface AudioConfig {
  sampleRate: number;
  channels: number;
  bitDepth: number;
}

export abstract class BaseAudioProcessor {
  protected config: AudioConfig;
  protected isRecording: boolean = false;
  
  constructor(config: AudioConfig = { sampleRate: 16000, channels: 1, bitDepth: 16 }) {
    this.config = config;
  }
  
  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;
  abstract onAudioData(callback: (data: ArrayBuffer) => void): void;
  
  protected validateConfig(config: AudioConfig): boolean {
    return config.sampleRate > 0 && config.channels > 0 && config.bitDepth > 0;
  }
  
  getConfig(): AudioConfig {
    return { ...this.config };
  }
  
  isActive(): boolean {
    return this.isRecording;
  }
}

export abstract class BasePlayer {
  protected isPlaying: boolean = false;
  protected queue: ArrayBuffer[] = [];
  protected volume: number = 1.0;
  
  abstract play(audioData: ArrayBuffer): Promise<void>;
  abstract stop(): void;
  abstract setVolume(volume: number): void;
  
  protected enqueue(data: ArrayBuffer): void {
    this.queue.push(data);
  }
  
  protected dequeue(): ArrayBuffer | undefined {
    return this.queue.shift();
  }
  
  getQueueLength(): number {
    return this.queue.length;
  }
  
  clearQueue(): void {
    this.queue = [];
  }
  
  getVolume(): number {
    return this.volume;
  }
  
  isActive(): boolean {
    return this.isPlaying;
  }
}
