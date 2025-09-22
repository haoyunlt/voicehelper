import { test, expect, Page } from '@playwright/test';

/**
 * 聊天功能端到端测试
 * 测试文本聊天、流式响应、对话历史等核心功能
 */

test.describe('聊天功能测试', () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    
    // 导航到聊天页面
    await page.goto('/chat');
    
    // 等待页面加载完成
    await page.waitForLoadState('networkidle');
    
    // 验证页面基本元素
    await expect(page.locator('[data-testid="chat-container"]')).toBeVisible();
    await expect(page.locator('[data-testid="message-input"]')).toBeVisible();
    await expect(page.locator('[data-testid="send-button"]')).toBeVisible();
  });

  test('基本文本聊天功能 @smoke', async () => {
    // 输入测试消息
    const testMessage = '你好，这是一个测试消息';
    await page.fill('[data-testid="message-input"]', testMessage);
    
    // 发送消息
    await page.click('[data-testid="send-button"]');
    
    // 验证用户消息显示
    await expect(page.locator('[data-testid="user-message"]').last()).toContainText(testMessage);
    
    // 等待AI响应（增加超时时间）
    await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout: 60000 });
    
    // 验证响应不为空
    const assistantMessage = await page.locator('[data-testid="assistant-message"]').last().textContent();
    expect(assistantMessage).toBeTruthy();
    expect(assistantMessage!.length).toBeGreaterThan(0);
  });

  test('流式响应功能', async () => {
    const testMessage = '请详细介绍一下人工智能的发展历史';
    await page.fill('[data-testid="message-input"]', testMessage);
    
    // 监听流式响应
    let responseChunks: string[] = [];
    
    // 监听网络请求
    page.on('response', async (response) => {
      if (response.url().includes('/api/chat/stream')) {
        // 验证是流式响应
        expect(response.headers()['content-type']).toContain('text/stream');
      }
    });
    
    await page.click('[data-testid="send-button"]');
    
    // 等待响应开始
    await page.waitForSelector('[data-testid="assistant-message"]:last-child', { timeout: 10000 });
    
    // 监听响应内容变化
    const assistantMessage = page.locator('[data-testid="assistant-message"]').last();
    
    // 等待响应完成（内容不再变化）
    let previousText = '';
    let stableCount = 0;
    
    while (stableCount < 3) {
      await page.waitForTimeout(1000);
      const currentText = await assistantMessage.textContent() || '';
      
      if (currentText === previousText) {
        stableCount++;
      } else {
        stableCount = 0;
        previousText = currentText;
      }
    }
    
    // 验证最终响应
    const finalResponse = await assistantMessage.textContent();
    expect(finalResponse).toBeTruthy();
    expect(finalResponse!.length).toBeGreaterThan(50); // 详细回答应该比较长
  });

  test('对话历史功能', async () => {
    // 发送第一条消息
    const firstMessage = '我的名字是张三';
    await page.fill('[data-testid="message-input"]', firstMessage);
    await page.click('[data-testid="send-button"]');
    
    // 等待第一个响应
    await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout: 30000 });
    
    // 发送第二条消息（测试上下文记忆）
    const secondMessage = '你还记得我的名字吗？';
    await page.fill('[data-testid="message-input"]', secondMessage);
    await page.click('[data-testid="send-button"]');
    
    // 等待第二个响应
    await expect(page.locator('[data-testid="assistant-message"]').nth(1)).toBeVisible({ timeout: 30000 });
    
    // 验证AI记住了用户名字
    const secondResponse = await page.locator('[data-testid="assistant-message"]').nth(1).textContent();
    expect(secondResponse).toContain('张三');
    
    // 验证消息历史显示正确
    const userMessages = await page.locator('[data-testid="user-message"]').count();
    const assistantMessages = await page.locator('[data-testid="assistant-message"]').count();
    
    expect(userMessages).toBe(2);
    expect(assistantMessages).toBe(2);
  });

  test('输入验证和错误处理', async () => {
    // 测试空消息
    await page.click('[data-testid="send-button"]');
    
    // 验证空消息不会发送
    const messageCount = await page.locator('[data-testid="user-message"]').count();
    expect(messageCount).toBe(0);
    
    // 测试超长消息
    const longMessage = 'a'.repeat(10000);
    await page.fill('[data-testid="message-input"]', longMessage);
    
    // 验证输入限制或警告
    const inputValue = await page.inputValue('[data-testid="message-input"]');
    expect(inputValue.length).toBeLessThanOrEqual(5000); // 假设有长度限制
  });

  test('键盘快捷键', async () => {
    const testMessage = '测试键盘快捷键';
    await page.fill('[data-testid="message-input"]', testMessage);
    
    // 使用Enter键发送消息
    await page.press('[data-testid="message-input"]', 'Enter');
    
    // 验证消息已发送
    await expect(page.locator('[data-testid="user-message"]').last()).toContainText(testMessage);
    
    // 测试Shift+Enter换行（如果支持）
    await page.fill('[data-testid="message-input"]', '第一行');
    await page.press('[data-testid="message-input"]', 'Shift+Enter');
    await page.type('[data-testid="message-input"]', '第二行');
    
    const inputValue = await page.inputValue('[data-testid="message-input"]');
    expect(inputValue).toContain('\n'); // 验证换行符
  });

  test('响应中的引用和来源', async () => {
    const testMessage = '请告诉我关于测试文档的信息';
    await page.fill('[data-testid="message-input"]', testMessage);
    await page.click('[data-testid="send-button"]');
    
    // 等待响应
    await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout: 30000 });
    
    // 检查是否有引用信息
    const referencesExist = await page.locator('[data-testid="message-references"]').count() > 0;
    
    if (referencesExist) {
      // 验证引用格式
      await expect(page.locator('[data-testid="message-references"]').first()).toBeVisible();
      
      // 点击引用查看详情
      await page.click('[data-testid="reference-item"]');
      
      // 验证引用详情显示
      await expect(page.locator('[data-testid="reference-detail"]')).toBeVisible();
    }
  });

  test('消息操作功能', async () => {
    const testMessage = '这是一条测试消息';
    await page.fill('[data-testid="message-input"]', testMessage);
    await page.click('[data-testid="send-button"]');
    
    // 等待响应
    await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout: 30000 });
    
    // 测试复制消息功能
    await page.hover('[data-testid="assistant-message"]');
    
    if (await page.locator('[data-testid="copy-message"]').isVisible()) {
      await page.click('[data-testid="copy-message"]');
      
      // 验证复制成功提示
      await expect(page.locator('[data-testid="copy-success"]')).toBeVisible();
    }
    
    // 测试重新生成功能
    if (await page.locator('[data-testid="regenerate-message"]').isVisible()) {
      await page.click('[data-testid="regenerate-message"]');
      
      // 验证重新生成
      await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout: 30000 });
    }
  });

  test('聊天会话管理', async () => {
    // 发送一条消息建立会话
    await page.fill('[data-testid="message-input"]', '开始新会话');
    await page.click('[data-testid="send-button"]');
    
    // 等待响应
    await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout: 30000 });
    
    // 测试新建会话
    if (await page.locator('[data-testid="new-chat"]').isVisible()) {
      await page.click('[data-testid="new-chat"]');
      
      // 验证消息历史被清空
      const messageCount = await page.locator('[data-testid="user-message"]').count();
      expect(messageCount).toBe(0);
    }
    
    // 测试会话历史（如果有侧边栏）
    if (await page.locator('[data-testid="chat-history"]').isVisible()) {
      await expect(page.locator('[data-testid="chat-history-item"]')).toHaveCount(1);
    }
  });

  test('响应式设计测试', async () => {
    // 测试桌面视图
    await page.setViewportSize({ width: 1200, height: 800 });
    await expect(page.locator('[data-testid="chat-container"]')).toBeVisible();
    
    // 测试平板视图
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('[data-testid="chat-container"]')).toBeVisible();
    
    // 测试手机视图
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('[data-testid="chat-container"]')).toBeVisible();
    
    // 在手机视图下测试聊天功能
    const mobileMessage = '手机端测试消息';
    await page.fill('[data-testid="message-input"]', mobileMessage);
    await page.click('[data-testid="send-button"]');
    
    await expect(page.locator('[data-testid="user-message"]').last()).toContainText(mobileMessage);
  });
});
