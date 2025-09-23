/**
 * Barge-in检测Hook
 * 监听本地语音能量并触发TTS取消
 */
import { useState, useEffect, useCallback, useRef } from 'react';

interface BargeInConfig {
  energyThreshold: number;
  minSpeechDuration: number;
  debounceTime: number;
  enabled: boolean;
}

interface BargeInMetrics {
  currentEnergy: number;
  isDetecting: boolean;
  triggerCount: number;
  lastTriggerTime: number | null;
  averageEnergy: number;
  adaptiveThreshold: number;
}

interface UseBargeInOptions {
  sessionId: string;
  config?: Partial<BargeInConfig>;
  onBargeIn?: (metrics: BargeInMetrics) => void;
  onEnergyUpdate?: (energy: number) => void;
  onError?: (error: Error) => void;
}

const DEFAULT_CONFIG: BargeInConfig = {
  energyThreshold: 0.01,
  minSpeechDuration: 100, // ms
  debounceTime: 200, // ms
  enabled: true
};

export function useBargeIn(options: UseBargeInOptions) {
  const [config, setConfig] = useState<BargeInConfig>({
    ...DEFAULT_CONFIG,
    ...options.config
  });
  
  const [metrics, setMetrics] = useState<BargeInMetrics>({
    currentEnergy: 0,
    isDetecting: false,
    triggerCount: 0,
    lastTriggerTime: null,
    averageEnergy: 0,
    adaptiveThreshold: config.energyThreshold
  });

  const [isActive, setIsActive] = useState(false);

  // Refs for audio processing
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const processingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const lastTriggerTimeRef = useRef<number>(0);
  const energyHistoryRef = useRef<number[]>([]);
  const speechStartTimeRef = useRef<number | null>(null);

  // 初始化音频上下文和分析器
  const initializeAudio = useCallback(async () => {
    try {
      // 创建音频上下文
      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      // 获取麦克风权限
      mediaStreamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: false, // 关闭回声消除以获得原始音频
          noiseSuppression: false,
          autoGainControl: false,
          sampleRate: 16000
        }
      });

      // 创建分析器节点
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 2048;
      analyserRef.current.smoothingTimeConstant = 0.3;

      // 连接音频流到分析器
      const source = audioContextRef.current.createMediaStreamSource(mediaStreamRef.current);
      source.connect(analyserRef.current);

      console.log('Barge-in audio initialized');
      return true;

    } catch (error) {
      console.error('Failed to initialize barge-in audio:', error);
      options.onError?.(error as Error);
      return false;
    }
  }, [options]);

  // 计算音频能量
  const calculateEnergy = useCallback((): number => {
    if (!analyserRef.current) return 0;

    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyserRef.current.getByteTimeDomainData(dataArray);

    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      const sample = ((dataArray[i] || 0) - 128) / 128; // 转换到 -1 到 1 范围
      sum += sample * sample;
    }

    const rms = Math.sqrt(sum / bufferLength);
    return rms;
  }, []);

  // 更新自适应阈值
  const updateAdaptiveThreshold = useCallback((energy: number) => {
    energyHistoryRef.current.push(energy);
    
    // 保持最近100个样本
    if (energyHistoryRef.current.length > 100) {
      energyHistoryRef.current.shift();
    }

    if (energyHistoryRef.current.length >= 10) {
      // 计算背景噪声水平（25th percentile）
      const sortedEnergies = [...energyHistoryRef.current].sort((a, b) => a - b);
      const noiseFloor = sortedEnergies[Math.floor(sortedEnergies.length * 0.25)];
      
      // 自适应阈值 = 噪声底线 + 固定偏移
      const adaptiveThreshold = Math.max(
        (noiseFloor || 0) * 3, // 至少是噪声的3倍
        config.energyThreshold // 不低于配置的最小阈值
      );

      setMetrics(prev => ({
        ...prev,
        adaptiveThreshold,
        averageEnergy: sortedEnergies[Math.floor(sortedEnergies.length * 0.5)] || 0 // 中位数
      }));
    }
  }, [config.energyThreshold]);

  // 触发barge-in
  const triggerBargeIn = useCallback(async () => {
    const now = Date.now();
    
    // 防抖检查
    if (now - lastTriggerTimeRef.current < config.debounceTime) {
      return;
    }

    lastTriggerTimeRef.current = now;

    try {
      // 发送取消TTS请求
      const response = await fetch('/api/realtime/cancel-tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          session_id: options.sessionId,
          reason: 'barge_in_detected'
        })
      });

      if (!response.ok) {
        throw new Error(`Failed to cancel TTS: ${response.statusText}`);
      }

      // 更新指标
      setMetrics(prev => {
        const newMetrics = {
          ...prev,
          triggerCount: prev.triggerCount + 1,
          lastTriggerTime: now
        };
        
        options.onBargeIn?.(newMetrics);
        return newMetrics;
      });

      console.log('Barge-in triggered successfully');

    } catch (error) {
      console.error('Failed to trigger barge-in:', error);
      options.onError?.(error as Error);
    }
  }, [config.debounceTime, options]);

  // 音频处理循环
  const processAudio = useCallback(() => {
    if (!config.enabled || !analyserRef.current) return;

    const energy = calculateEnergy();
    const now = Date.now();

    // 更新当前能量
    setMetrics(prev => ({
      ...prev,
      currentEnergy: energy
    }));

    // 更新自适应阈值
    updateAdaptiveThreshold(energy);

    // 检测语音活动
    const currentThreshold = metrics.adaptiveThreshold || config.energyThreshold;
    const isVoiceActive = energy > currentThreshold;

    if (isVoiceActive) {
      if (speechStartTimeRef.current === null) {
        // 开始检测到语音
        speechStartTimeRef.current = now;
        setMetrics(prev => ({ ...prev, isDetecting: true }));
      } else if (now - speechStartTimeRef.current >= config.minSpeechDuration) {
        // 语音持续时间足够，触发barge-in
        if (!metrics.isDetecting) {
          triggerBargeIn();
        }
      }
    } else {
      // 没有检测到语音，重置状态
      speechStartTimeRef.current = null;
      setMetrics(prev => ({ ...prev, isDetecting: false }));
    }

    // 通知能量更新
    options.onEnergyUpdate?.(energy);

  }, [config, metrics.adaptiveThreshold, calculateEnergy, updateAdaptiveThreshold, triggerBargeIn, options]);

  // 开始监听
  const start = useCallback(async () => {
    if (isActive) return;

    const initialized = await initializeAudio();
    if (!initialized) return;

    // 开始音频处理循环
    processingIntervalRef.current = setInterval(processAudio, 50); // 每50ms检查一次

    setIsActive(true);
    console.log('Barge-in detection started');
  }, [isActive, initializeAudio, processAudio]);

  // 停止监听
  const stop = useCallback(() => {
    if (!isActive) return;

    // 清理定时器
    if (processingIntervalRef.current) {
      clearInterval(processingIntervalRef.current);
      processingIntervalRef.current = null;
    }

    // 停止媒体流
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    // 关闭音频上下文
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // 重置状态
    speechStartTimeRef.current = null;
    energyHistoryRef.current = [];
    
    setIsActive(false);
    setMetrics(prev => ({
      ...prev,
      isDetecting: false,
      currentEnergy: 0
    }));

    console.log('Barge-in detection stopped');
  }, [isActive]);

  // 更新配置
  const updateConfig = useCallback((newConfig: Partial<BargeInConfig>) => {
    setConfig(prev => ({ ...prev, ...newConfig }));
  }, []);

  // 手动触发barge-in（用于测试）
  const manualTrigger = useCallback(async () => {
    await triggerBargeIn();
  }, [triggerBargeIn]);

  // 重置指标
  const resetMetrics = useCallback(() => {
    setMetrics(prev => ({
      ...prev,
      triggerCount: 0,
      lastTriggerTime: null
    }));
    energyHistoryRef.current = [];
  }, []);

  // 清理资源
  useEffect(() => {
    return () => {
      stop();
    };
  }, [stop]);

  // 配置变化时重新启动
  useEffect(() => {
    if (isActive && config.enabled) {
      // 配置变化，重新启动
      stop();
      setTimeout(start, 100);
    } else if (isActive && !config.enabled) {
      // 禁用时停止
      stop();
    }
  }, [config.enabled, config.energyThreshold, config.minSpeechDuration]);

  return {
    // 状态
    isActive,
    config,
    metrics,

    // 方法
    start,
    stop,
    updateConfig,
    manualTrigger,
    resetMetrics,

    // 计算属性
    isDetecting: metrics.isDetecting,
    currentEnergy: metrics.currentEnergy,
    triggerCount: metrics.triggerCount,
    adaptiveThreshold: metrics.adaptiveThreshold
  };
}

// 可视化组件Hook
export function useBargeInVisualization(bargeInHook: ReturnType<typeof useBargeIn>) {
  const [energyHistory, setEnergyHistory] = useState<number[]>([]);
  const [maxHistoryLength] = useState(100);

  useEffect(() => {
    if (bargeInHook.currentEnergy !== undefined) {
      setEnergyHistory(prev => {
        const newHistory = [...prev, bargeInHook.currentEnergy];
        return newHistory.slice(-maxHistoryLength);
      });
    }
  }, [bargeInHook.currentEnergy, maxHistoryLength]);

  const getVisualizationData = useCallback(() => {
    return {
      energyHistory,
      currentEnergy: bargeInHook.currentEnergy,
      threshold: bargeInHook.adaptiveThreshold,
      isDetecting: bargeInHook.isDetecting,
      triggerCount: bargeInHook.triggerCount
    };
  }, [energyHistory, bargeInHook]);

  return {
    energyHistory,
    getVisualizationData
  };
}
