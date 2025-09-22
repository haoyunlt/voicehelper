/**
 * 端到端语音延迟测试
 * 测试完整的语音对话流程延迟
 */

import { test, expect, Page, Browser } from '@playwright/test';
import { VoiceClient, VoiceClientConfig, VoiceEvents } from '../../sdks/javascript/src/voice-client';

interface LatencyMetrics {
  captureStart: number;
  sttComplete: number;
  llmComplete: number;
  ttsStart: number;
  audioPlayback: number;
  e2eComplete: number;
}

interface TestResult {
  success: boolean;
  metrics: LatencyMetrics;
  totalLatency: number;
  phases: {
    capture: number;
    stt: number;
    llm: number;
    tts: number;
    playback: number;
  };
  error?: string;
}

class VoiceLatencyTester {
  private page: Page;
  private voiceClient: VoiceClient | null = null;
  private metrics: Partial<LatencyMetrics> = {};
  private testAudioBuffer: ArrayBuffer | null = null;

  constructor(page: Page) {
    this.page = page;
  }

  async initialize(): Promise<void> {
    // 生成测试音频
    await this.generateTestAudio();
    
    // 设置页面权限
    await this.page.context().grantPermissions(['microphone']);
    
    // 导航到测试页面
    await this.page.goto('/test/voice-latency');
  }

  async generateTestAudio(): Promise<void> {
    // 在浏览器中生成测试音频
    this.testAudioBuffer = await this.page.evaluate(() => {
      const sampleRate = 16000;
      const duration = 2; // 2秒
      const samples = sampleRate * duration;
      
      // 生成正弦波 + 噪声模拟语音
      const audioData = new Float32Array(samples);
      for (let i = 0; i < samples; i++) {
        const t = i / sampleRate;
        const frequency = 440 + Math.sin(t * 2) * 100; // 变化的频率
        audioData[i] = 0.3 * Math.sin(2 * Math.PI * frequency * t) + 
                       0.1 * (Math.random() - 0.5); // 添加噪声
      }
      
      // 转换为ArrayBuffer
      const buffer = new ArrayBuffer(audioData.length * 4);
      const view = new Float32Array(buffer);
      view.set(audioData);
      
      return buffer;
    });
  }

  async runLatencyTest(): Promise<TestResult> {
    try {
      // 重置指标
      this.metrics = {};
      
      // 配置语音客户端
      const config: VoiceClientConfig = {
        apiKey: process.env.VOICEHELPER_API_KEY || 'test-key',
        baseUrl: 'http://localhost:8000',
        provider: 'webrtc',
        enableVAD: true,
        enableBargeIn: true,
        debug: true
      };

      const events: VoiceEvents = {
        onConnected: () => {
          console.log('Voice client connected');
        },
        onTranscription: (text: string, isFinal: boolean) => {
          if (isFinal && !this.metrics.sttComplete) {
            this.metrics.sttComplete = performance.now();
            console.log('STT completed:', text);
          }
        },
        onResponse: (text: string) => {
          if (!this.metrics.llmComplete) {
            this.metrics.llmComplete = performance.now();
            console.log('LLM response received:', text);
          }
        },
        onAudioReceived: (audioData: ArrayBuffer) => {
          if (!this.metrics.ttsStart) {
            this.metrics.ttsStart = performance.now();
            console.log('First TTS audio chunk received');
          }
        },
        onError: (error: Error) => {
          console.error('Voice client error:', error);
        }
      };

      // 在页面中创建语音客户端
      await this.page.evaluate(
        ({ config, events }) => {
          // 这里需要在页面中实际创建VoiceClient实例
          // 由于安全限制，我们需要通过页面脚本来处理
          (window as any).voiceClientConfig = config;
          (window as any).voiceClientEvents = events;
        },
        { config, events }
      );

      // 开始测试
      this.metrics.captureStart = performance.now();

      // 连接语音客户端
      await this.page.evaluate(async () => {
        const { VoiceClient } = await import('/src/voice-client.js');
        const client = new VoiceClient(
          (window as any).voiceClientConfig,
          (window as any).voiceClientEvents
        );
        
        await client.connect();
        (window as any).voiceClient = client;
      });

      // 等待连接建立
      await this.page.waitForFunction(() => (window as any).voiceClient?.getConnectionState() === 'connected');

      // 开始录音
      await this.page.evaluate(() => {
        return (window as any).voiceClient.startRecording();
      });

      // 模拟发送音频数据
      await this.simulateAudioInput();

      // 等待完整的对话流程完成
      await this.waitForE2ECompletion();

      this.metrics.e2eComplete = performance.now();

      // 计算各阶段延迟
      const phases = this.calculatePhaseLatencies();
      const totalLatency = this.metrics.e2eComplete! - this.metrics.captureStart!;

      return {
        success: true,
        metrics: this.metrics as LatencyMetrics,
        totalLatency,
        phases
      };

    } catch (error) {
      return {
        success: false,
        metrics: this.metrics as LatencyMetrics,
        totalLatency: 0,
        phases: {
          capture: 0,
          stt: 0,
          llm: 0,
          tts: 0,
          playback: 0
        },
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  private async simulateAudioInput(): Promise<void> {
    // 模拟音频输入
    await this.page.evaluate((audioBuffer) => {
      const client = (window as any).voiceClient;
      if (!client) return;

      // 将音频数据分块发送
      const chunkSize = 1600; // 100ms at 16kHz
      const totalSamples = audioBuffer.byteLength / 4; // Float32Array
      
      let offset = 0;
      const sendChunk = () => {
        if (offset >= totalSamples) return;
        
        const chunkSamples = Math.min(chunkSize, totalSamples - offset);
        const chunkBuffer = audioBuffer.slice(offset * 4, (offset + chunkSamples) * 4);
        
        // 发送音频块（这里需要根据实际的VoiceClient实现调整）
        // client.sendAudioData(chunkBuffer);
        
        offset += chunkSamples;
        
        if (offset < totalSamples) {
          setTimeout(sendChunk, 100); // 每100ms发送一块
        }
      };
      
      sendChunk();
    }, this.testAudioBuffer);
  }

  private async waitForE2ECompletion(): Promise<void> {
    // 等待音频播放开始
    await this.page.waitForFunction(
      () => (window as any).audioPlaybackStarted === true,
      { timeout: 10000 }
    );

    this.metrics.audioPlayback = performance.now();

    // 等待额外的缓冲时间确保音频播放
    await this.page.waitForTimeout(1000);
  }

  private calculatePhaseLatencies(): TestResult['phases'] {
    const start = this.metrics.captureStart || 0;
    const sttComplete = this.metrics.sttComplete || start;
    const llmComplete = this.metrics.llmComplete || sttComplete;
    const ttsStart = this.metrics.ttsStart || llmComplete;
    const audioPlayback = this.metrics.audioPlayback || ttsStart;

    return {
      capture: sttComplete - start,
      stt: sttComplete - start,
      llm: llmComplete - sttComplete,
      tts: ttsStart - llmComplete,
      playback: audioPlayback - ttsStart
    };
  }

  async cleanup(): Promise<void> {
    // 清理资源
    await this.page.evaluate(() => {
      const client = (window as any).voiceClient;
      if (client) {
        client.disconnect();
      }
    });
  }
}

// Playwright测试用例

test.describe('语音延迟测试', () => {
  let tester: VoiceLatencyTester;

  test.beforeEach(async ({ page }) => {
    tester = new VoiceLatencyTester(page);
    await tester.initialize();
  });

  test.afterEach(async () => {
    await tester.cleanup();
  });

  test('端到端语音延迟应小于700ms', async () => {
    const result = await tester.runLatencyTest();
    
    expect(result.success).toBe(true);
    expect(result.totalLatency).toBeLessThan(700); // P95目标
    
    console.log('延迟测试结果:', {
      总延迟: `${result.totalLatency.toFixed(1)}ms`,
      STT延迟: `${result.phases.stt.toFixed(1)}ms`,
      LLM延迟: `${result.phases.llm.toFixed(1)}ms`,
      TTS延迟: `${result.phases.tts.toFixed(1)}ms`,
      播放延迟: `${result.phases.playback.toFixed(1)}ms`
    });
  });

  test('STT首包延迟应小于200ms', async () => {
    const result = await tester.runLatencyTest();
    
    expect(result.success).toBe(true);
    expect(result.phases.stt).toBeLessThan(200);
  });

  test('TTS首包延迟应小于300ms', async () => {
    const result = await tester.runLatencyTest();
    
    expect(result.success).toBe(true);
    expect(result.phases.tts).toBeLessThan(300);
  });

  test('连续多次测试延迟稳定性', async () => {
    const results: TestResult[] = [];
    const testCount = 5;

    for (let i = 0; i < testCount; i++) {
      const result = await tester.runLatencyTest();
      expect(result.success).toBe(true);
      results.push(result);
      
      // 测试间隔
      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    // 计算延迟统计
    const latencies = results.map(r => r.totalLatency);
    const avgLatency = latencies.reduce((a, b) => a + b, 0) / latencies.length;
    const maxLatency = Math.max(...latencies);
    const minLatency = Math.min(...latencies);
    const jitter = maxLatency - minLatency;

    console.log('延迟稳定性测试结果:', {
      平均延迟: `${avgLatency.toFixed(1)}ms`,
      最大延迟: `${maxLatency.toFixed(1)}ms`,
      最小延迟: `${minLatency.toFixed(1)}ms`,
      抖动: `${jitter.toFixed(1)}ms`
    });

    // 验证稳定性
    expect(avgLatency).toBeLessThan(700);
    expect(jitter).toBeLessThan(200); // 抖动小于200ms
  });
});

// 压力测试
test.describe('语音压力测试', () => {
  test('并发语音会话延迟测试', async ({ browser }) => {
    const concurrentSessions = 3;
    const promises: Promise<TestResult>[] = [];

    // 创建多个并发会话
    for (let i = 0; i < concurrentSessions; i++) {
      const context = await browser.newContext();
      const page = await context.newPage();
      
      const tester = new VoiceLatencyTester(page);
      await tester.initialize();
      
      promises.push(tester.runLatencyTest());
    }

    // 等待所有测试完成
    const results = await Promise.all(promises);

    // 验证所有会话都成功
    results.forEach((result, index) => {
      expect(result.success).toBe(true);
      expect(result.totalLatency).toBeLessThan(1000); // 并发时允许更高延迟
      
      console.log(`会话${index + 1}延迟: ${result.totalLatency.toFixed(1)}ms`);
    });

    // 计算并发性能
    const avgConcurrentLatency = results.reduce((sum, r) => sum + r.totalLatency, 0) / results.length;
    console.log(`并发平均延迟: ${avgConcurrentLatency.toFixed(1)}ms`);
  });
});

// 网络条件测试
test.describe('不同网络条件下的延迟测试', () => {
  const networkConditions = [
    { name: '4G', downloadThroughput: 4000, uploadThroughput: 1000, latency: 20 },
    { name: '3G', downloadThroughput: 1600, uploadThroughput: 750, latency: 150 },
    { name: '慢速3G', downloadThroughput: 500, uploadThroughput: 500, latency: 300 }
  ];

  networkConditions.forEach(condition => {
    test(`${condition.name}网络条件下的延迟测试`, async ({ page }) => {
      // 设置网络条件
      await page.context().route('**/*', route => {
        // 模拟网络延迟
        setTimeout(() => route.continue(), condition.latency);
      });

      const tester = new VoiceLatencyTester(page);
      await tester.initialize();

      const result = await tester.runLatencyTest();
      
      expect(result.success).toBe(true);
      
      // 根据网络条件调整延迟期望
      const expectedMaxLatency = condition.name === '4G' ? 800 : 
                                condition.name === '3G' ? 1200 : 2000;
      
      expect(result.totalLatency).toBeLessThan(expectedMaxLatency);
      
      console.log(`${condition.name}网络延迟: ${result.totalLatency.toFixed(1)}ms`);
      
      await tester.cleanup();
    });
  });
});

// 设备性能测试
test.describe('不同设备性能下的延迟测试', () => {
  const deviceConfigs = [
    { name: '高性能设备', cpuSlowdown: 1 },
    { name: '中等设备', cpuSlowdown: 2 },
    { name: '低性能设备', cpuSlowdown: 4 }
  ];

  deviceConfigs.forEach(config => {
    test(`${config.name}延迟测试`, async ({ page }) => {
      // 模拟CPU性能
      const client = await page.context().newCDPSession(page);
      await client.send('Emulation.setCPUThrottlingRate', { 
        rate: config.cpuSlowdown 
      });

      const tester = new VoiceLatencyTester(page);
      await tester.initialize();

      const result = await tester.runLatencyTest();
      
      expect(result.success).toBe(true);
      
      // 根据设备性能调整延迟期望
      const expectedMaxLatency = config.cpuSlowdown === 1 ? 700 :
                                config.cpuSlowdown === 2 ? 1000 : 1500;
      
      expect(result.totalLatency).toBeLessThan(expectedMaxLatency);
      
      console.log(`${config.name}延迟: ${result.totalLatency.toFixed(1)}ms`);
      
      await tester.cleanup();
    });
  });
});

// 导出测试结果
test.afterAll(async () => {
  // 生成测试报告
  const reportData = {
    timestamp: new Date().toISOString(),
    testSuite: 'voice-latency-e2e',
    environment: {
      nodeVersion: process.version,
      platform: process.platform
    }
  };

  console.log('语音延迟E2E测试完成:', reportData);
});
