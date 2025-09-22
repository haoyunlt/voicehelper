import { test, expect, Page } from '@playwright/test';

/**
 * 冒烟测试 - 快速验证核心功能
 * 确保基本功能正常工作
 */

test.describe('冒烟测试', () => {
  test('首页加载正常 @smoke', async ({ page }) => {
    await page.goto('/');
    
    // 验证页面标题
    await expect(page).toHaveTitle(/智能聊天机器人/);
    
    // 验证主要导航链接
    await expect(page.locator('a[href="/chat"]')).toBeVisible();
    await expect(page.locator('a[href="/datasets"]')).toBeVisible();
    await expect(page.locator('a[href="/analytics"]')).toBeVisible();
  });

  test('聊天页面基本功能 @smoke', async ({ page }) => {
    await page.goto('/chat');
    
    // 等待页面加载
    await page.waitForLoadState('networkidle');
    
    // 验证基本元素存在
    await expect(page.locator('[data-testid="chat-container"]')).toBeVisible();
    await expect(page.locator('[data-testid="message-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="send-button"]')).toBeVisible();
    
    // 测试输入功能
    const testMessage = '测试消息';
    await page.fill('[data-testid="message-input"]', testMessage);
    
    // 验证输入值
    const inputValue = await page.inputValue('[data-testid="message-input"]');
    expect(inputValue).toBe(testMessage);
    
    // 验证发送按钮状态
    await expect(page.locator('[data-testid="send-button"]')).toBeEnabled();
  });

  test('API健康检查 @smoke', async ({ page }) => {
    // 检查后端API
    const backendResponse = await page.request.get('http://localhost:8080/health');
    expect(backendResponse.ok()).toBeTruthy();
    
    const backendData = await backendResponse.json();
    expect(backendData.status).toBe('ok');
    
    // 检查算法服务API
    const algoResponse = await page.request.get('http://localhost:8000/health');
    expect(algoResponse.ok()).toBeTruthy();
    
    const algoData = await algoResponse.json();
    expect(algoData.status).toBe('healthy');
  });

  test('数据集页面加载 @smoke', async ({ page }) => {
    await page.goto('/datasets');
    
    // 等待页面加载
    await page.waitForLoadState('networkidle');
    
    // 验证页面基本元素（使用first()避免多个匹配）
    await expect(page.locator('h1').first()).toBeVisible();
  });

  test('分析页面加载 @smoke', async ({ page }) => {
    await page.goto('/analytics');
    
    // 等待页面加载
    await page.waitForLoadState('networkidle');
    
    // 验证页面基本元素（使用first()避免多个匹配）
    await expect(page.locator('h1').first()).toBeVisible();
  });

  test('响应式设计基本验证 @smoke', async ({ page }) => {
    await page.goto('/chat');
    
    // 桌面视图
    await page.setViewportSize({ width: 1200, height: 800 });
    await expect(page.locator('[data-testid="chat-container"]')).toBeVisible();
    
    // 移动视图
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('[data-testid="chat-container"]')).toBeVisible();
    await expect(page.locator('[data-testid="message-input"]')).toBeVisible();
  });
});
