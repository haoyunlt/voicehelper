import { Platform } from 'react-native'
import BackgroundJob from 'react-native-background-job'
import { AudioRecorder, AudioUtils } from 'react-native-audio'
import AsyncStorage from '@react-native-async-storage/async-storage'
import NetInfo from '@react-native-community/netinfo'

export interface VoiceOptimizationConfig {
  enableBackgroundProcessing: boolean
  enableBatteryOptimization: boolean
  enableNetworkAdaptation: boolean
  audioQuality: 'low' | 'medium' | 'high'
  maxRecordingDuration: number
  compressionLevel: number
}

export interface NetworkQuality {
  type: string
  isConnected: boolean
  strength: number
  bandwidth: number
  latency: number
}

export interface BatteryOptimization {
  level: number
  isLowPowerMode: boolean
  chargingState: string
  estimatedTimeRemaining: number
}

class VoiceOptimizationService {
  private config: VoiceOptimizationConfig
  private networkQuality: NetworkQuality | null = null
  private batteryInfo: BatteryOptimization | null = null
  private backgroundJobId: string | null = null
  private audioProcessor: AudioProcessor
  private networkAdapter: NetworkAdapter
  private batteryOptimizer: BatteryOptimizer

  constructor(config: Partial<VoiceOptimizationConfig> = {}) {
    this.config = {
      enableBackgroundProcessing: true,
      enableBatteryOptimization: true,
      enableNetworkAdaptation: true,
      audioQuality: 'medium',
      maxRecordingDuration: 300, // 5 minutes
      compressionLevel: 0.7,
      ...config
    }

    this.audioProcessor = new AudioProcessor(this.config)
    this.networkAdapter = new NetworkAdapter()
    this.batteryOptimizer = new BatteryOptimizer()

    this.initialize()
  }

  private async initialize() {
    // 监听网络状态变化
    NetInfo.addEventListener(this.handleNetworkStateChange.bind(this))

    // 监听电池状态变化
    if (Platform.OS === 'ios') {
      // iOS 电池监听
      this.batteryOptimizer.startBatteryMonitoring()
    } else {
      // Android 电池监听
      this.batteryOptimizer.startBatteryMonitoring()
    }

    // 加载保存的配置
    await this.loadSavedConfig()
  }

  private async loadSavedConfig() {
    try {
      const savedConfig = await AsyncStorage.getItem('voice_optimization_config')
      if (savedConfig) {
        this.config = { ...this.config, ...JSON.parse(savedConfig) }
      }
    } catch (error) {
      console.error('Failed to load saved config:', error)
    }
  }

  private async saveConfig() {
    try {
      await AsyncStorage.setItem('voice_optimization_config', JSON.stringify(this.config))
    } catch (error) {
      console.error('Failed to save config:', error)
    }
  }

  private handleNetworkStateChange(state: any) {
    this.networkQuality = {
      type: state.type,
      isConnected: state.isConnected,
      strength: state.details?.strength || 0,
      bandwidth: state.details?.bandwidth || 0,
      latency: state.details?.latency || 0
    }

    if (this.config.enableNetworkAdaptation) {
      this.adaptToNetworkConditions()
    }
  }

  private adaptToNetworkConditions() {
    if (!this.networkQuality) return

    const { type, strength, bandwidth } = this.networkQuality

    // 根据网络条件调整音频质量
    if (type === 'wifi' && strength > 0.8) {
      this.config.audioQuality = 'high'
      this.config.compressionLevel = 0.3
    } else if (type === 'cellular' && bandwidth > 1000000) { // > 1Mbps
      this.config.audioQuality = 'medium'
      this.config.compressionLevel = 0.5
    } else {
      this.config.audioQuality = 'low'
      this.config.compressionLevel = 0.8
    }

    console.log(`Network adapted: ${type}, quality: ${this.config.audioQuality}`)
  }

  public async startBackgroundProcessing(): Promise<boolean> {
    if (!this.config.enableBackgroundProcessing) return false

    try {
      const jobConfig = {
        jobKey: 'voiceProcessing',
        period: 15000, // 15 seconds
        requiredNetworkType: 'any',
        requiresCharging: false,
        requiresDeviceIdle: false
      }

      this.backgroundJobId = await BackgroundJob.start(jobConfig)
      console.log('Background processing started:', this.backgroundJobId)
      return true
    } catch (error) {
      console.error('Failed to start background processing:', error)
      return false
    }
  }

  public async stopBackgroundProcessing(): Promise<void> {
    if (this.backgroundJobId) {
      await BackgroundJob.stop(this.backgroundJobId)
      this.backgroundJobId = null
      console.log('Background processing stopped')
    }
  }

  public async optimizeAudioRecording(options: any = {}): Promise<any> {
    const optimizedOptions = {
      ...options,
      SampleRate: this.getOptimalSampleRate(),
      Channels: 1, // Mono for voice
      AudioQuality: this.config.audioQuality,
      AudioEncoding: this.getOptimalEncoding(),
      OutputFormat: this.getOptimalFormat(),
      MeteringEnabled: true,
      MeasurementMode: true
    }

    // 电池优化
    if (this.config.enableBatteryOptimization && this.batteryInfo?.isLowPowerMode) {
      optimizedOptions.SampleRate = Math.min(optimizedOptions.SampleRate, 16000)
      optimizedOptions.AudioQuality = 'low'
    }

    return optimizedOptions
  }

  private getOptimalSampleRate(): number {
    switch (this.config.audioQuality) {
      case 'high': return 44100
      case 'medium': return 22050
      case 'low': return 16000
      default: return 22050
    }
  }

  private getOptimalEncoding(): string {
    if (Platform.OS === 'ios') {
      return 'lpcm' // Linear PCM for iOS
    } else {
      return 'aac' // AAC for Android
    }
  }

  private getOptimalFormat(): string {
    if (Platform.OS === 'ios') {
      return 'caf' // Core Audio Format for iOS
    } else {
      return 'm4a' // M4A for Android
    }
  }

  public async processAudioForUpload(audioPath: string): Promise<string> {
    try {
      // 音频压缩
      const compressedPath = await this.audioProcessor.compressAudio(
        audioPath,
        this.config.compressionLevel
      )

      // 网络适应性处理
      if (this.networkQuality?.bandwidth && this.networkQuality.bandwidth < 500000) { // < 500kbps
        // 进一步压缩
        return await this.audioProcessor.compressAudio(compressedPath, 0.9)
      }

      return compressedPath
    } catch (error) {
      console.error('Audio processing failed:', error)
      return audioPath // 返回原始文件
    }
  }

  public getNetworkQuality(): NetworkQuality | null {
    return this.networkQuality
  }

  public getBatteryInfo(): BatteryOptimization | null {
    return this.batteryInfo
  }

  public updateConfig(newConfig: Partial<VoiceOptimizationConfig>) {
    this.config = { ...this.config, ...newConfig }
    this.saveConfig()
  }

  public getConfig(): VoiceOptimizationConfig {
    return { ...this.config }
  }

  public async getOptimizationRecommendations(): Promise<string[]> {
    const recommendations: string[] = []

    // 网络建议
    if (this.networkQuality) {
      if (this.networkQuality.type === 'cellular' && this.networkQuality.strength < 0.5) {
        recommendations.push('网络信号较弱，建议移动到信号更好的位置')
      }
      if (this.networkQuality.bandwidth < 100000) { // < 100kbps
        recommendations.push('网络带宽较低，已自动降低音频质量')
      }
    }

    // 电池建议
    if (this.batteryInfo) {
      if (this.batteryInfo.level < 0.2) {
        recommendations.push('电量较低，建议连接充电器')
      }
      if (this.batteryInfo.isLowPowerMode) {
        recommendations.push('低电量模式已启用，音频质量已优化')
      }
    }

    // 存储建议
    const availableStorage = await this.getAvailableStorage()
    if (availableStorage < 100 * 1024 * 1024) { // < 100MB
      recommendations.push('存储空间不足，建议清理设备存储')
    }

    return recommendations
  }

  private async getAvailableStorage(): Promise<number> {
    try {
      // 这里应该实现获取可用存储空间的逻辑
      // 返回字节数
      return 1024 * 1024 * 1024 // 1GB (示例)
    } catch (error) {
      console.error('Failed to get available storage:', error)
      return 0
    }
  }

  public destroy() {
    this.stopBackgroundProcessing()
    this.batteryOptimizer.stopBatteryMonitoring()
    // 清理其他资源
  }
}

class AudioProcessor {
  private config: VoiceOptimizationConfig

  constructor(config: VoiceOptimizationConfig) {
    this.config = config
  }

  async compressAudio(inputPath: string, compressionLevel: number): Promise<string> {
    try {
      // 这里应该实现音频压缩逻辑
      // 可以使用 FFmpeg 或其他音频处理库
      
      const outputPath = inputPath.replace(/\.[^/.]+$/, '_compressed.m4a')
      
      // 模拟压缩过程
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      console.log(`Audio compressed: ${inputPath} -> ${outputPath}`)
      return outputPath
    } catch (error) {
      console.error('Audio compression failed:', error)
      throw error
    }
  }

  async enhanceAudio(inputPath: string): Promise<string> {
    try {
      // 音频增强：降噪、音量归一化等
      const outputPath = inputPath.replace(/\.[^/.]+$/, '_enhanced.m4a')
      
      // 模拟增强过程
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      console.log(`Audio enhanced: ${inputPath} -> ${outputPath}`)
      return outputPath
    } catch (error) {
      console.error('Audio enhancement failed:', error)
      throw error
    }
  }
}

class NetworkAdapter {
  private currentQuality: string = 'medium'

  adaptQuality(networkQuality: NetworkQuality): string {
    const { type, strength, bandwidth } = networkQuality

    if (type === 'wifi' && strength > 0.8) {
      this.currentQuality = 'high'
    } else if (type === 'cellular') {
      if (bandwidth > 2000000) { // > 2Mbps
        this.currentQuality = 'medium'
      } else if (bandwidth > 500000) { // > 500kbps
        this.currentQuality = 'low'
      } else {
        this.currentQuality = 'very_low'
      }
    } else {
      this.currentQuality = 'low'
    }

    return this.currentQuality
  }

  getCurrentQuality(): string {
    return this.currentQuality
  }
}

class BatteryOptimizer {
  private batteryLevel: number = 1.0
  private isLowPowerMode: boolean = false
  private chargingState: string = 'unknown'

  startBatteryMonitoring() {
    // 这里应该实现电池状态监听
    // 不同平台有不同的实现方式
    
    if (Platform.OS === 'ios') {
      // iOS 电池监听实现
      this.startIOSBatteryMonitoring()
    } else {
      // Android 电池监听实现
      this.startAndroidBatteryMonitoring()
    }
  }

  private startIOSBatteryMonitoring() {
    // iOS 特定的电池监听逻辑
    console.log('Starting iOS battery monitoring')
  }

  private startAndroidBatteryMonitoring() {
    // Android 特定的电池监听逻辑
    console.log('Starting Android battery monitoring')
  }

  stopBatteryMonitoring() {
    console.log('Battery monitoring stopped')
  }

  getBatteryLevel(): number {
    return this.batteryLevel
  }

  isInLowPowerMode(): boolean {
    return this.isLowPowerMode
  }

  getChargingState(): string {
    return this.chargingState
  }

  shouldOptimizeForBattery(): boolean {
    return this.batteryLevel < 0.3 || this.isLowPowerMode
  }
}

export default VoiceOptimizationService
