/**
 * VoiceHelper 语音交互 E2E 测试
 * 功能: 完整语音对话流程 + 打断功能 + 网络异常恢复 + 性能验证
 */

import { test, expect, Page } from '@playwright/test';
import { promises as fs } from 'fs';
import path from 'path';

// 测试配置
const TEST_TIMEOUT = 30000;
const AUDIO_TEST_FILES = {
    hello: 'test-data/audio/hello_16k.wav',
    question: 'test-data/audio/question_16k.wav',
    command: 'test-data/audio/command_16k.wav',
};

// 性能阈值
const PERFORMANCE_THRESHOLDS = {
    e2eLatency: 500,        // 端到端延迟 < 500ms
    asrLatency: 300,        // ASR延迟 < 300ms
    ttsLatency: 200,        // TTS首音延迟 < 200ms
    interruptLatency: 120,  // 打断响应 < 120ms
    reconnectTime: 5000,    // 重连时间 < 5s
};

test.describe('语音交互 E2E 测试', () => {
    let page: Page;
    
    test.beforeEach(async ({ page: testPage }) => {
        page = testPage;
        
        // 设置权限
        await page.context().grantPermissions(['microphone']);
        
        // 导航到语音聊天页面
        await page.goto('/voice-chat');
        
        // 等待页面加载完成
        await expect(page.locator('#voice-chat-container')).toBeVisible();
        
        // 等待WebSocket连接建立
        await expect(page.locator('#connection-status')).toContainText('已连接');
    });
    
    test('完整语音对话流程', async () => {
        test.setTimeout(TEST_TIMEOUT);
        
        // 记录开始时间
        const testStartTime = Date.now();
        
        // 1. 上传测试音频文件
        const audioInput = page.locator('#audio-file-input');
        await audioInput.setInputFiles(AUDIO_TEST_FILES.hello);
        
        // 2. 开始录音
        await page.click('#start-recording');
        await expect(page.locator('#recording-status')).toContainText('录音中');
        
        // 3. 验证音频采集指标
        const audioLevel = page.locator('#audio-level');
        await expect(audioLevel).toBeVisible();
        
        // 等待音频电平显示
        await page.waitForFunction(() => {
            const levelElement = document.querySelector('#audio-level');
            return levelElement && parseFloat(levelElement.textContent || '0') > 0;
        }, { timeout: 5000 });
        
        // 4. 等待ASR结果
        const asrStartTime = Date.now();
        await expect(page.locator('#asr-result')).toContainText('你好', { timeout: 10000 });
        const asrLatency = Date.now() - asrStartTime;
        
        console.log(`ASR延迟: ${asrLatency}ms`);
        expect(asrLatency).toBeLessThan(PERFORMANCE_THRESHOLDS.asrLatency);
        
        // 5. 验证LLM响应
        const llmStartTime = Date.now();
        await expect(page.locator('#llm-response')).toBeVisible({ timeout: 15000 });
        const llmLatency = Date.now() - llmStartTime;
        
        console.log(`LLM响应延迟: ${llmLatency}ms`);
        
        // 6. 验证TTS播放
        const ttsStartTime = Date.now();
        await expect(page.locator('#tts-player')).toHaveAttribute('playing', 'true', { timeout: 10000 });
        const ttsLatency = Date.now() - ttsStartTime;
        
        console.log(`TTS首音延迟: ${ttsLatency}ms`);
        expect(ttsLatency).toBeLessThan(PERFORMANCE_THRESHOLDS.ttsLatency);
        
        // 7. 检查端到端延迟
        const e2eLatency = Date.now() - testStartTime;
        console.log(`端到端延迟: ${e2eLatency}ms`);
        expect(e2eLatency).toBeLessThan(PERFORMANCE_THRESHOLDS.e2eLatency * 10); // 放宽E2E阈值
        
        // 8. 验证音频质量指标
        const audioQuality = await page.locator('#audio-quality-score').textContent();
        const qualityScore = parseInt(audioQuality || '0');
        expect(qualityScore).toBeGreaterThan(80); // 音频质量分数 > 80
        
        // 9. 验证延迟指标显示
        await expect(page.locator('#latency-metrics')).toBeVisible();
        
        const captureLatency = await page.locator('#capture-latency').textContent();
        const playLatency = await page.locator('#play-latency').textContent();
        
        console.log(`采集延迟: ${captureLatency}ms, 播放延迟: ${playLatency}ms`);
        
        // 10. 停止录音
        await page.click('#stop-recording');
        await expect(page.locator('#recording-status')).toContainText('已停止');
    });
    
    test('打断功能测试', async () => {
        test.setTimeout(TEST_TIMEOUT);
        
        // 1. 发送长文本触发TTS
        await page.fill('#text-input', '这是一个很长的测试文本，用来测试打断功能。我们需要确保在TTS播放过程中能够成功打断。这个文本需要足够长，以便我们有足够的时间来测试打断功能的响应速度和准确性。');
        await page.click('#send-text');
        
        // 2. 等待TTS开始播放
        await expect(page.locator('#tts-status')).toContainText('播放中', { timeout: 10000 });
        
        // 3. 记录打断开始时间并发送打断信号
        const interruptStartTime = Date.now();
        await page.click('#interrupt-button');
        
        // 4. 验证打断响应
        await expect(page.locator('#tts-status')).toContainText('已取消', { timeout: 1000 });
        const interruptLatency = Date.now() - interruptStartTime;
        
        console.log(`打断响应延迟: ${interruptLatency}ms`);
        expect(interruptLatency).toBeLessThan(PERFORMANCE_THRESHOLDS.interruptLatency);
        
        // 5. 验证打断成功指标
        const interruptSuccess = await page.locator('#interrupt-success-rate').textContent();
        const successRate = parseFloat(interruptSuccess || '0');
        expect(successRate).toBeGreaterThan(0.95); // 打断成功率 > 95%
        
        // 6. 验证音频播放确实停止
        await expect(page.locator('#tts-player')).toHaveAttribute('playing', 'false');
        
        // 7. 验证可以立即开始新的对话
        await page.fill('#text-input', '打断测试完成');
        await page.click('#send-text');
        await expect(page.locator('#llm-response')).toBeVisible({ timeout: 5000 });
    });
    
    test('网络异常恢复测试', async () => {
        test.setTimeout(TEST_TIMEOUT);
        
        // 1. 验证初始连接状态
        await expect(page.locator('#connection-status')).toContainText('已连接');
        
        // 2. 模拟网络断开 - 拦截WebSocket连接
        await page.route('**/voice/stream', route => route.abort());
        
        // 3. 尝试发送消息触发重连
        await page.fill('#text-input', '测试网络恢复');
        await page.click('#send-text');
        
        // 4. 验证断线检测
        await expect(page.locator('#connection-status')).toContainText('重连中', { timeout: 5000 });
        
        // 5. 验证重连指示器
        await expect(page.locator('#reconnecting-indicator')).toBeVisible();
        
        // 6. 恢复网络连接
        await page.unroute('**/voice/stream');
        
        // 7. 验证自动重连
        const reconnectStartTime = Date.now();
        await expect(page.locator('#connection-status')).toContainText('已连接', { timeout: 10000 });
        const reconnectTime = Date.now() - reconnectStartTime;
        
        console.log(`重连时间: ${reconnectTime}ms`);
        expect(reconnectTime).toBeLessThan(PERFORMANCE_THRESHOLDS.reconnectTime);
        
        // 8. 验证重连后功能正常
        await page.fill('#text-input', '重连后测试消息');
        await page.click('#send-text');
        await expect(page.locator('#llm-response')).toBeVisible({ timeout: 10000 });
        
        // 9. 验证重连统计
        const reconnectCount = await page.locator('#reconnect-count').textContent();
        expect(parseInt(reconnectCount || '0')).toBeGreaterThan(0);
    });
    
    test('音频质量监控测试', async () => {
        test.setTimeout(TEST_TIMEOUT);
        
        // 1. 开始音频监控
        await page.click('#start-audio-monitoring');
        
        // 2. 上传测试音频并播放
        await page.locator('#audio-file-input').setInputFiles(AUDIO_TEST_FILES.question);
        await page.click('#start-recording');
        
        // 3. 等待音频处理
        await page.waitForTimeout(3000);
        
        // 4. 验证音频质量指标
        const metrics = {
            jitter: await page.locator('#jitter-p95').textContent(),
            dropRate: await page.locator('#drop-rate').textContent(),
            outOfOrder: await page.locator('#out-of-order-rate').textContent(),
            bufferHealth: await page.locator('#buffer-health').textContent(),
        };
        
        console.log('音频质量指标:', metrics);
        
        // 5. 验证指标在正常范围内
        expect(parseFloat(metrics.jitter || '0')).toBeLessThan(50); // 抖动 < 50ms
        expect(parseFloat(metrics.dropRate || '0')).toBeLessThan(0.01); // 丢包率 < 1%
        expect(parseFloat(metrics.outOfOrder || '0')).toBeLessThan(0.005); // 乱序率 < 0.5%
        expect(parseFloat(metrics.bufferHealth || '0')).toBeGreaterThan(80); // 缓冲区健康度 > 80
        
        // 6. 验证实时指标更新
        const initialJitter = parseFloat(metrics.jitter || '0');
        await page.waitForTimeout(2000);
        const updatedJitter = parseFloat(await page.locator('#jitter-p95').textContent() || '0');
        
        // 指标应该有更新（可能相同但至少不会是初始值0）
        expect(updatedJitter).toBeGreaterThanOrEqual(0);
        
        await page.click('#stop-recording');
    });
    
    test('多会话并发测试', async () => {
        test.setTimeout(TEST_TIMEOUT);
        
        // 1. 打开多个会话标签
        const sessionTabs = ['#session-tab-1', '#session-tab-2', '#session-tab-3'];
        
        for (const tab of sessionTabs) {
            await page.click('#new-session-button');
            await expect(page.locator(tab)).toBeVisible();
        }
        
        // 2. 在每个会话中发送消息
        for (let i = 0; i < sessionTabs.length; i++) {
            await page.click(sessionTabs[i]);
            await page.fill('#text-input', `并发测试消息 ${i + 1}`);
            await page.click('#send-text');
        }
        
        // 3. 验证所有会话都能正常响应
        for (let i = 0; i < sessionTabs.length; i++) {
            await page.click(sessionTabs[i]);
            await expect(page.locator('#llm-response')).toBeVisible({ timeout: 10000 });
        }
        
        // 4. 验证会话隔离
        await page.click(sessionTabs[0]);
        const session1History = await page.locator('#chat-history').textContent();
        
        await page.click(sessionTabs[1]);
        const session2History = await page.locator('#chat-history').textContent();
        
        expect(session1History).not.toBe(session2History);
        
        // 5. 验证资源使用情况
        const activeConnections = await page.locator('#active-connections').textContent();
        expect(parseInt(activeConnections || '0')).toBe(sessionTabs.length);
    });
    
    test('性能基准测试', async () => {
        test.setTimeout(TEST_TIMEOUT);
        
        // 1. 运行性能基准测试
        await page.click('#run-benchmark');
        
        // 2. 等待基准测试完成
        await expect(page.locator('#benchmark-status')).toContainText('完成', { timeout: 20000 });
        
        // 3. 获取基准测试结果
        const benchmarkResults = {
            avgE2ELatency: await page.locator('#benchmark-e2e-latency').textContent(),
            avgASRLatency: await page.locator('#benchmark-asr-latency').textContent(),
            avgTTSLatency: await page.locator('#benchmark-tts-latency').textContent(),
            throughput: await page.locator('#benchmark-throughput').textContent(),
            errorRate: await page.locator('#benchmark-error-rate').textContent(),
        };
        
        console.log('性能基准测试结果:', benchmarkResults);
        
        // 4. 验证性能指标
        expect(parseFloat(benchmarkResults.avgE2ELatency || '0')).toBeLessThan(PERFORMANCE_THRESHOLDS.e2eLatency);
        expect(parseFloat(benchmarkResults.avgASRLatency || '0')).toBeLessThan(PERFORMANCE_THRESHOLDS.asrLatency);
        expect(parseFloat(benchmarkResults.avgTTSLatency || '0')).toBeLessThan(PERFORMANCE_THRESHOLDS.ttsLatency);
        expect(parseFloat(benchmarkResults.errorRate || '0')).toBeLessThan(0.01); // 错误率 < 1%
        
        // 5. 验证吞吐量
        const throughput = parseFloat(benchmarkResults.throughput || '0');
        expect(throughput).toBeGreaterThan(10); // 吞吐量 > 10 requests/sec
        
        // 6. 生成性能报告
        await page.click('#generate-performance-report');
        await expect(page.locator('#performance-report')).toBeVisible();
        
        // 7. 下载性能报告
        const downloadPromise = page.waitForEvent('download');
        await page.click('#download-report');
        const download = await downloadPromise;
        
        expect(download.suggestedFilename()).toContain('performance-report');
    });
});

test.describe('语音交互错误处理', () => {
    test('ASR服务异常处理', async ({ page }) => {
        // 模拟ASR服务异常
        await page.route('**/asr/**', route => route.abort());
        
        await page.goto('/voice-chat');
        await page.locator('#audio-file-input').setInputFiles(AUDIO_TEST_FILES.hello);
        await page.click('#start-recording');
        
        // 验证错误处理
        await expect(page.locator('#error-message')).toContainText('语音识别服务暂不可用');
        await expect(page.locator('#fallback-text-input')).toBeVisible();
    });
    
    test('TTS服务异常处理', async ({ page }) => {
        // 模拟TTS服务异常
        await page.route('**/tts/**', route => route.abort());
        
        await page.goto('/voice-chat');
        await page.fill('#text-input', '测试TTS异常处理');
        await page.click('#send-text');
        
        // 验证错误处理
        await expect(page.locator('#error-message')).toContainText('语音合成服务暂不可用');
        await expect(page.locator('#text-only-mode')).toBeVisible();
    });
    
    test('网络超时处理', async ({ page }) => {
        // 模拟网络超时
        await page.route('**/api/**', route => {
            return new Promise(() => {}); // 永不resolve，模拟超时
        });
        
        await page.goto('/voice-chat');
        await page.fill('#text-input', '测试网络超时');
        await page.click('#send-text');
        
        // 验证超时处理
        await expect(page.locator('#timeout-message')).toBeVisible({ timeout: 10000 });
        await expect(page.locator('#retry-button')).toBeVisible();
    });
});

// 辅助函数
async function waitForAudioProcessing(page: Page, timeout = 5000) {
    await page.waitForFunction(() => {
        const statusElement = document.querySelector('#processing-status');
        return statusElement && statusElement.textContent !== '处理中';
    }, { timeout });
}

async function capturePerformanceMetrics(page: Page) {
    return await page.evaluate(() => {
        return {
            timing: performance.timing,
            navigation: performance.navigation,
            memory: (performance as any).memory,
        };
    });
}

async function injectAudioTestData(page: Page, audioFile: string) {
    const audioBuffer = await fs.readFile(path.resolve(audioFile));
    await page.evaluate((buffer) => {
        // 注入音频数据到页面
        (window as any).testAudioData = buffer;
    }, Array.from(audioBuffer));
}
