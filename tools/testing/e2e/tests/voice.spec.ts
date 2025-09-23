import { test, expect, Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

/**
 * 语音交互功能端到端测试
 * 测试语音输入、语音合成、实时语音对话等功能
 */

test.describe('语音交互功能测试', () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    
    // 导航到聊天页面
    await page.goto('/chat');
    
    // 等待页面加载完成
    await page.waitForLoadState('networkidle');
    
    // 授予麦克风权限（模拟）
    await page.context().grantPermissions(['microphone']);
  });

  test('语音输入按钮可见性和状态', async () => {
    // 验证语音输入按钮存在
    await expect(page.locator('[data-testid="voice-input-button"]')).toBeVisible();
    
    // 验证初始状态
    const voiceButton = page.locator('[data-testid="voice-input-button"]');
    await expect(voiceButton).not.toHaveClass(/recording/);
    
    // 验证按钮可点击
    await expect(voiceButton).toBeEnabled();
  });

  test('语音录制状态切换', async () => {
    const voiceButton = page.locator('[data-testid="voice-input-button"]');
    
    // 开始录制
    await voiceButton.click();
    
    // 验证录制状态
    await expect(voiceButton).toHaveClass(/recording/);
    await expect(page.locator('[data-testid="recording-indicator"]')).toBeVisible();
    
    // 停止录制
    await voiceButton.click();
    
    // 验证停止状态
    await expect(voiceButton).not.toHaveClass(/recording/);
    await expect(page.locator('[data-testid="recording-indicator"]')).not.toBeVisible();
  });

  test('语音权限处理', async () => {
    // 拒绝麦克风权限
    await page.context().clearPermissions();
    
    const voiceButton = page.locator('[data-testid="voice-input-button"]');
    await voiceButton.click();
    
    // 验证权限提示
    await expect(page.locator('[data-testid="permission-dialog"]')).toBeVisible();
    await expect(page.locator('[data-testid="permission-message"]')).toContainText('麦克风权限');
    
    // 点击授权按钮
    await page.click('[data-testid="grant-permission"]');
    
    // 重新授予权限
    await page.context().grantPermissions(['microphone']);
  });

  test('语音转文字功能', async () => {
    // 模拟语音输入
    await simulateVoiceInput(page, '你好，这是一个语音测试');
    
    // 验证转录结果显示
    await expect(page.locator('[data-testid="voice-transcript"]')).toBeVisible();
    await expect(page.locator('[data-testid="voice-transcript"]')).toContainText('你好');
    
    // 验证转录文本自动填入输入框
    const inputValue = await page.inputValue('[data-testid="message-input"]');
    expect(inputValue).toContain('你好');
  });

  test('语音消息发送', async () => {
    // 模拟语音输入并发送
    await simulateVoiceInput(page, '请介绍一下人工智能');
    
    // 点击发送按钮
    await page.click('[data-testid="send-button"]');
    
    // 验证语音消息显示
    await expect(page.locator('[data-testid="user-message"]').last()).toBeVisible();
    await expect(page.locator('[data-testid="voice-message-indicator"]')).toBeVisible();
    
    // 验证AI响应
    await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout: 30000 });
  });

  test('语音合成播放', async () => {
    // 发送文本消息
    await page.fill('[data-testid="message-input"]', '请用语音回复我');
    await page.click('[data-testid="send-button"]');
    
    // 等待AI响应
    await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout: 30000 });
    
    // 查找语音播放按钮
    const playButton = page.locator('[data-testid="play-voice"]').last();
    
    if (await playButton.isVisible()) {
      // 点击播放按钮
      await playButton.click();
      
      // 验证播放状态
      await expect(page.locator('[data-testid="audio-playing"]')).toBeVisible();
      
      // 验证暂停功能
      await page.click('[data-testid="pause-voice"]');
      await expect(page.locator('[data-testid="audio-playing"]')).not.toBeVisible();
    }
  });

  test('实时语音对话', async () => {
    // 启用实时语音模式
    if (await page.locator('[data-testid="realtime-voice-toggle"]').isVisible()) {
      await page.click('[data-testid="realtime-voice-toggle"]');
      
      // 验证实时模式激活
      await expect(page.locator('[data-testid="realtime-indicator"]')).toBeVisible();
      
      // 模拟连续语音输入
      await simulateVoiceInput(page, '开始实时对话');
      
      // 验证实时响应
      await expect(page.locator('[data-testid="realtime-response"]')).toBeVisible({ timeout: 10000 });
      
      // 关闭实时模式
      await page.click('[data-testid="realtime-voice-toggle"]');
      await expect(page.locator('[data-testid="realtime-indicator"]')).not.toBeVisible();
    }
  });

  test('语音设置和配置', async () => {
    // 打开语音设置
    if (await page.locator('[data-testid="voice-settings"]').isVisible()) {
      await page.click('[data-testid="voice-settings"]');
      
      // 验证设置面板
      await expect(page.locator('[data-testid="voice-settings-panel"]')).toBeVisible();
      
      // 测试语音识别语言设置
      if (await page.locator('[data-testid="voice-language-select"]').isVisible()) {
        await page.selectOption('[data-testid="voice-language-select"]', 'zh-CN');
        
        // 验证设置保存
        const selectedValue = await page.inputValue('[data-testid="voice-language-select"]');
        expect(selectedValue).toBe('zh-CN');
      }
      
      // 测试语音合成设置
      if (await page.locator('[data-testid="tts-voice-select"]').isVisible()) {
        await page.selectOption('[data-testid="tts-voice-select"]', 'female');
        
        // 测试语音预览
        await page.click('[data-testid="voice-preview"]');
        await expect(page.locator('[data-testid="audio-playing"]')).toBeVisible();
      }
      
      // 关闭设置面板
      await page.click('[data-testid="close-settings"]');
      await expect(page.locator('[data-testid="voice-settings-panel"]')).not.toBeVisible();
    }
  });

  test('语音质量和延迟测试', async () => {
    // 记录开始时间
    const startTime = Date.now();
    
    // 模拟语音输入
    await simulateVoiceInput(page, '测试语音响应延迟');
    
    // 等待转录完成
    await expect(page.locator('[data-testid="voice-transcript"]')).toBeVisible();
    
    // 计算转录延迟
    const transcriptionTime = Date.now() - startTime;
    expect(transcriptionTime).toBeLessThan(5000); // 转录应在5秒内完成
    
    // 发送消息并测试TTS延迟
    await page.click('[data-testid="send-button"]');
    
    const responseStartTime = Date.now();
    await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout: 30000 });
    
    // 如果有语音播放，测试TTS延迟
    if (await page.locator('[data-testid="play-voice"]').last().isVisible()) {
      await page.click('[data-testid="play-voice"]');
      await expect(page.locator('[data-testid="audio-playing"]')).toBeVisible();
      
      const ttsTime = Date.now() - responseStartTime;
      expect(ttsTime).toBeLessThan(10000); // TTS应在10秒内开始播放
    }
  });

  test('语音错误处理', async () => {
    // 测试网络错误情况
    await page.route('**/api/voice/**', route => route.abort());
    
    const voiceButton = page.locator('[data-testid="voice-input-button"]');
    await voiceButton.click();
    
    // 模拟语音输入
    await simulateVoiceInput(page, '测试网络错误');
    
    // 验证错误提示
    await expect(page.locator('[data-testid="voice-error"]')).toBeVisible();
    await expect(page.locator('[data-testid="voice-error"]')).toContainText('网络错误');
    
    // 恢复网络
    await page.unroute('**/api/voice/**');
    
    // 测试重试功能
    if (await page.locator('[data-testid="retry-voice"]').isVisible()) {
      await page.click('[data-testid="retry-voice"]');
      await expect(page.locator('[data-testid="voice-error"]')).not.toBeVisible();
    }
  });

  test('语音消息历史', async () => {
    // 发送多条语音消息
    const messages = ['第一条语音消息', '第二条语音消息', '第三条语音消息'];
    
    for (const message of messages) {
      await simulateVoiceInput(page, message);
      await page.click('[data-testid="send-button"]');
      await expect(page.locator('[data-testid="user-message"]').last()).toBeVisible();
    }
    
    // 验证语音消息历史
    const voiceMessages = await page.locator('[data-testid="voice-message-indicator"]').count();
    expect(voiceMessages).toBe(3);
    
    // 测试语音消息回放
    const firstVoiceMessage = page.locator('[data-testid="voice-message-indicator"]').first();
    await firstVoiceMessage.click();
    
    // 验证回放功能
    await expect(page.locator('[data-testid="audio-playing"]')).toBeVisible();
  });

  test('多语言语音支持', async () => {
    // 测试中文语音
    await simulateVoiceInput(page, '你好，我是中文语音测试');
    await expect(page.locator('[data-testid="voice-transcript"]')).toContainText('你好');
    
    // 切换到英文（如果支持）
    if (await page.locator('[data-testid="voice-language-select"]').isVisible()) {
      await page.selectOption('[data-testid="voice-language-select"]', 'en-US');
      
      // 测试英文语音
      await simulateVoiceInput(page, 'Hello, this is English voice test');
      await expect(page.locator('[data-testid="voice-transcript"]')).toContainText('Hello');
    }
  });

  test('语音输入取消功能', async () => {
    const voiceButton = page.locator('[data-testid="voice-input-button"]');
    
    // 开始录制
    await voiceButton.click();
    await expect(page.locator('[data-testid="recording-indicator"]')).toBeVisible();
    
    // 测试取消录制
    if (await page.locator('[data-testid="cancel-recording"]').isVisible()) {
      await page.click('[data-testid="cancel-recording"]');
      
      // 验证录制已取消
      await expect(page.locator('[data-testid="recording-indicator"]')).not.toBeVisible();
      await expect(page.locator('[data-testid="voice-transcript"]')).not.toBeVisible();
    }
  });
});

/**
 * 模拟语音输入的辅助函数
 */
async function simulateVoiceInput(page: Page, text: string) {
  // 开始录制
  await page.click('[data-testid="voice-input-button"]');
  
  // 等待录制状态
  await expect(page.locator('[data-testid="recording-indicator"]')).toBeVisible();
  
  // 模拟录制时间
  await page.waitForTimeout(2000);
  
  // 模拟语音识别结果
  await page.evaluate((transcriptText) => {
    // 模拟WebSocket消息或API响应
    const event = new CustomEvent('voiceTranscript', {
      detail: { transcript: transcriptText, confidence: 0.95 }
    });
    window.dispatchEvent(event);
  }, text);
  
  // 停止录制
  await page.click('[data-testid="voice-input-button"]');
  
  // 等待录制停止
  await expect(page.locator('[data-testid="recording-indicator"]')).not.toBeVisible();
}

/**
 * 创建测试音频文件的辅助函数
 */
function createTestAudioFile(): string {
  // 创建一个简单的WAV文件头（静音）
  const sampleRate = 44100;
  const duration = 2; // 2秒
  const numSamples = sampleRate * duration;
  const numChannels = 1;
  const bytesPerSample = 2;
  
  const buffer = Buffer.alloc(44 + numSamples * bytesPerSample);
  
  // WAV文件头
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + numSamples * bytesPerSample, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(numChannels, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * numChannels * bytesPerSample, 28);
  buffer.writeUInt16LE(numChannels * bytesPerSample, 32);
  buffer.writeUInt16LE(bytesPerSample * 8, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(numSamples * bytesPerSample, 40);
  
  // 写入静音数据
  for (let i = 0; i < numSamples; i++) {
    buffer.writeInt16LE(0, 44 + i * bytesPerSample);
  }
  
  const testAudioPath = path.join(__dirname, '../fixtures/test-audio.wav');
  fs.writeFileSync(testAudioPath, buffer);
  
  return testAudioPath;
}
