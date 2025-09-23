import { Page, expect } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

/**
 * 测试辅助工具函数
 * 提供常用的测试操作和断言
 */

/**
 * 等待元素加载并验证可见性
 */
export async function waitForElement(page: Page, selector: string, timeout: number = 10000) {
  await expect(page.locator(selector)).toBeVisible({ timeout });
}

/**
 * 等待多个元素加载
 */
export async function waitForElements(page: Page, selectors: string[], timeout: number = 10000) {
  for (const selector of selectors) {
    await waitForElement(page, selector, timeout);
  }
}

/**
 * 安全地填写输入框
 */
export async function safeFill(page: Page, selector: string, text: string) {
  await page.locator(selector).clear();
  await page.locator(selector).fill(text);
  
  // 验证输入是否成功
  const value = await page.inputValue(selector);
  expect(value).toBe(text);
}

/**
 * 安全地点击按钮
 */
export async function safeClick(page: Page, selector: string) {
  await expect(page.locator(selector)).toBeEnabled();
  await page.click(selector);
}

/**
 * 等待网络请求完成
 */
export async function waitForNetworkIdle(page: Page, timeout: number = 5000) {
  await page.waitForLoadState('networkidle', { timeout });
}

/**
 * 模拟语音输入
 */
export async function simulateVoiceInput(page: Page, transcript: string) {
  // 开始录制
  await safeClick(page, '[data-testid="voice-input-button"]');
  
  // 等待录制状态
  await waitForElement(page, '[data-testid="recording-indicator"]');
  
  // 模拟录制时间
  await page.waitForTimeout(1000);
  
  // 模拟语音识别结果
  await page.evaluate((text) => {
    const event = new CustomEvent('voiceTranscript', {
      detail: { transcript: text, confidence: 0.95 }
    });
    window.dispatchEvent(event);
  }, transcript);
  
  // 停止录制
  await safeClick(page, '[data-testid="voice-input-button"]');
  
  // 等待录制停止
  await expect(page.locator('[data-testid="recording-indicator"]')).not.toBeVisible();
}

/**
 * 发送聊天消息
 */
export async function sendChatMessage(page: Page, message: string) {
  await safeFill(page, '[data-testid="message-input"]', message);
  await safeClick(page, '[data-testid="send-button"]');
  
  // 验证消息已发送
  await expect(page.locator('[data-testid="user-message"]').last()).toContainText(message);
}

/**
 * 等待AI响应
 */
export async function waitForAIResponse(page: Page, timeout: number = 30000) {
  await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout });
  
  // 等待响应完成（内容不再变化）
  let previousText = '';
  let stableCount = 0;
  
  while (stableCount < 3) {
    await page.waitForTimeout(1000);
    const currentText = await page.locator('[data-testid="assistant-message"]').last().textContent() || '';
    
    if (currentText === previousText) {
      stableCount++;
    } else {
      stableCount = 0;
      previousText = currentText;
    }
  }
  
  return previousText;
}

/**
 * 创建测试文件
 */
export function createTestFile(filename: string, content: string): string {
  const testDir = path.join(__dirname, '../fixtures');
  if (!fs.existsSync(testDir)) {
    fs.mkdirSync(testDir, { recursive: true });
  }
  
  const filePath = path.join(testDir, filename);
  fs.writeFileSync(filePath, content, 'utf8');
  
  return filePath;
}

/**
 * 清理测试文件
 */
export function cleanupTestFile(filePath: string) {
  if (fs.existsSync(filePath)) {
    fs.unlinkSync(filePath);
  }
}

/**
 * 上传文件到数据集
 */
export async function uploadFileToDataset(page: Page, filePath: string, datasetName: string) {
  // 点击上传按钮
  await safeClick(page, '[data-testid="upload-button"]');
  
  // 等待上传对话框
  await waitForElement(page, '[data-testid="upload-dialog"]');
  
  // 选择文件
  const fileInput = page.locator('input[type="file"]');
  await fileInput.setInputFiles(filePath);
  
  // 设置数据集名称
  await safeFill(page, '[data-testid="dataset-name"]', datasetName);
  
  // 开始上传
  await safeClick(page, '[data-testid="start-upload"]');
  
  // 等待上传完成
  await expect(page.locator('[data-testid="upload-success"]')).toBeVisible({ timeout: 30000 });
}

/**
 * 验证响应时间
 */
export async function measureResponseTime(action: () => Promise<void>): Promise<number> {
  const startTime = Date.now();
  await action();
  return Date.now() - startTime;
}

/**
 * 验证内存使用
 */
export async function checkMemoryUsage(page: Page): Promise<{ used: number; total: number }> {
  return await page.evaluate(() => {
    const memory = (performance as any).memory;
    if (memory) {
      return {
        used: memory.usedJSHeapSize,
        total: memory.totalJSHeapSize
      };
    }
    return { used: 0, total: 0 };
  });
}

/**
 * 监听控制台错误
 */
export function setupConsoleErrorListener(page: Page): string[] {
  const errors: string[] = [];
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });
  
  page.on('pageerror', error => {
    errors.push(error.message);
  });
  
  return errors;
}

/**
 * 监听网络请求
 */
export function setupNetworkListener(page: Page) {
  const requests: any[] = [];
  const responses: any[] = [];
  
  page.on('request', request => {
    requests.push({
      url: request.url(),
      method: request.method(),
      resourceType: request.resourceType(),
      timestamp: Date.now()
    });
  });
  
  page.on('response', response => {
    responses.push({
      url: response.url(),
      status: response.status(),
      statusText: response.statusText(),
      timestamp: Date.now()
    });
  });
  
  return { requests, responses };
}

/**
 * 验证页面性能指标
 */
export async function getPerformanceMetrics(page: Page) {
  return await page.evaluate(() => {
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    const paint = performance.getEntriesByType('paint');
    
    return {
      // 页面加载时间
      loadTime: navigation.loadEventEnd - navigation.loadEventStart,
      // DOM内容加载时间
      domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
      // 首次内容绘制
      firstContentfulPaint: paint.find(p => p.name === 'first-contentful-paint')?.startTime || 0,
      // 最大内容绘制
      largestContentfulPaint: paint.find(p => p.name === 'largest-contentful-paint')?.startTime || 0
    };
  });
}

/**
 * 等待元素文本包含指定内容
 */
export async function waitForTextContent(page: Page, selector: string, expectedText: string, timeout: number = 10000) {
  await expect(page.locator(selector)).toContainText(expectedText, { timeout });
}

/**
 * 滚动到元素位置
 */
export async function scrollToElement(page: Page, selector: string) {
  await page.locator(selector).scrollIntoViewIfNeeded();
}

/**
 * 验证元素属性
 */
export async function verifyElementAttribute(page: Page, selector: string, attribute: string, expectedValue: string) {
  await expect(page.locator(selector)).toHaveAttribute(attribute, expectedValue);
}

/**
 * 验证元素CSS类
 */
export async function verifyElementClass(page: Page, selector: string, className: string) {
  await expect(page.locator(selector)).toHaveClass(new RegExp(className));
}

/**
 * 模拟键盘快捷键
 */
export async function pressShortcut(page: Page, shortcut: string) {
  await page.keyboard.press(shortcut);
}

/**
 * 验证下载文件
 */
export async function verifyDownload(page: Page, triggerAction: () => Promise<void>): Promise<string> {
  const downloadPromise = page.waitForEvent('download');
  await triggerAction();
  const download = await downloadPromise;
  
  // 保存下载文件
  const downloadPath = path.join(__dirname, '../downloads', download.suggestedFilename());
  await download.saveAs(downloadPath);
  
  return downloadPath;
}

/**
 * 模拟拖拽操作
 */
export async function dragAndDrop(page: Page, sourceSelector: string, targetSelector: string) {
  await page.dragAndDrop(sourceSelector, targetSelector);
}

/**
 * 验证响应式布局
 */
export async function testResponsiveLayout(page: Page, viewports: { width: number; height: number }[]) {
  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    await page.waitForTimeout(500); // 等待布局调整
    
    // 验证页面仍然可用
    await expect(page.locator('body')).toBeVisible();
  }
}

/**
 * 清理测试数据
 */
export async function cleanupTestData(page: Page) {
  // 清理可能创建的测试数据
  try {
    // 删除测试数据集
    await page.request.delete('/api/v1/datasets/playwright-test-dataset');
    
    // 删除测试用户
    await page.request.delete('/api/v1/users/playwright-test-user');
  } catch (error) {
    // 忽略清理错误
    console.log('清理测试数据时出错:', error);
  }
}

/**
 * 生成随机测试数据
 */
export function generateTestData() {
  const timestamp = Date.now();
  return {
    username: `testuser-${timestamp}`,
    email: `test-${timestamp}@example.com`,
    datasetName: `测试数据集-${timestamp}`,
    conversationId: `conv-${timestamp}`,
    requestId: `req-${timestamp}`
  };
}

/**
 * 验证API响应
 */
export async function verifyAPIResponse(page: Page, url: string, expectedStatus: number = 200) {
  const response = await page.request.get(url);
  expect(response.status()).toBe(expectedStatus);
  return response;
}

/**
 * 模拟网络条件
 */
export async function simulateNetworkConditions(page: Page, conditions: 'slow3g' | 'fast3g' | 'offline') {
  const context = page.context();
  
  switch (conditions) {
    case 'slow3g':
      await context.route('**/*', route => {
        setTimeout(() => route.continue(), 1000); // 1秒延迟
      });
      break;
    case 'fast3g':
      await context.route('**/*', route => {
        setTimeout(() => route.continue(), 100); // 100ms延迟
      });
      break;
    case 'offline':
      await context.route('**/*', route => route.abort());
      break;
  }
}

/**
 * 恢复网络条件
 */
export async function restoreNetworkConditions(page: Page) {
  await page.context().unroute('**/*');
}
