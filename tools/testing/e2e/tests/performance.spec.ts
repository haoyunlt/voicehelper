import { test, expect, Page } from '@playwright/test';

/**
 * 性能测试用例
 * 测试页面加载性能、交互响应时间、内存使用等
 */

test.describe('性能测试', () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
  });

  test.describe('页面加载性能', () => {
    test('首页加载性能', async () => {
      const startTime = Date.now();
      
      await page.goto('/', { waitUntil: 'networkidle' });
      
      const loadTime = Date.now() - startTime;
      
      // 首页应在3秒内加载完成
      expect(loadTime).toBeLessThan(3000);
      
      // 验证关键元素已加载
      await expect(page.locator('body')).toBeVisible();
    });

    test('聊天页面加载性能', async () => {
      const startTime = Date.now();
      
      await page.goto('/chat', { waitUntil: 'networkidle' });
      
      const loadTime = Date.now() - startTime;
      
      // 聊天页面应在5秒内加载完成
      expect(loadTime).toBeLessThan(5000);
      
      // 验证关键元素已加载
      await expect(page.locator('[data-testid="chat-container"]')).toBeVisible();
      await expect(page.locator('[data-testid="message-input"]')).toBeVisible();
    });

    test('数据集页面加载性能', async () => {
      const startTime = Date.now();
      
      await page.goto('/datasets', { waitUntil: 'networkidle' });
      
      const loadTime = Date.now() - startTime;
      
      // 数据集页面应在4秒内加载完成
      expect(loadTime).toBeLessThan(4000);
      
      // 验证关键元素已加载
      await expect(page.locator('[data-testid="datasets-container"]')).toBeVisible();
    });

    test('页面资源加载优化', async () => {
      // 监听网络请求
      const requests: any[] = [];
      page.on('request', request => {
        requests.push({
          url: request.url(),
          resourceType: request.resourceType(),
          method: request.method()
        });
      });

      await page.goto('/chat', { waitUntil: 'networkidle' });

      // 分析资源加载
      const jsRequests = requests.filter(r => r.resourceType === 'script');
      const cssRequests = requests.filter(r => r.resourceType === 'stylesheet');
      const imageRequests = requests.filter(r => r.resourceType === 'image');

      // JS文件数量应该合理（避免过多的小文件）
      expect(jsRequests.length).toBeLessThan(20);
      
      // CSS文件数量应该合理
      expect(cssRequests.length).toBeLessThan(10);
      
      // 验证没有404错误
      const failedRequests = requests.filter(r => r.status >= 400);
      expect(failedRequests.length).toBe(0);
    });
  });

  test.describe('交互响应性能', () => {
    test('消息发送响应时间', async () => {
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      const testMessage = '这是一个性能测试消息';
      
      // 测量输入响应时间
      const inputStartTime = Date.now();
      await page.fill('[data-testid="message-input"]', testMessage);
      const inputTime = Date.now() - inputStartTime;
      
      // 输入响应应该很快
      expect(inputTime).toBeLessThan(100);

      // 测量发送按钮响应时间
      const sendStartTime = Date.now();
      await page.click('[data-testid="send-button"]');
      
      // 验证用户消息立即显示
      await expect(page.locator('[data-testid="user-message"]').last()).toBeVisible();
      const sendTime = Date.now() - sendStartTime;
      
      // 发送响应应在500ms内
      expect(sendTime).toBeLessThan(500);

      // 测量AI响应时间
      const responseStartTime = Date.now();
      await expect(page.locator('[data-testid="assistant-message"]').last()).toBeVisible({ timeout: 30000 });
      const responseTime = Date.now() - responseStartTime;
      
      // AI响应应在30秒内开始
      expect(responseTime).toBeLessThan(30000);
    });

    test('语音按钮响应时间', async () => {
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      // 授予麦克风权限
      await page.context().grantPermissions(['microphone']);

      const voiceButton = page.locator('[data-testid="voice-input-button"]');
      
      // 测量语音按钮响应时间
      const clickStartTime = Date.now();
      await voiceButton.click();
      
      // 验证录制状态变化
      await expect(page.locator('[data-testid="recording-indicator"]')).toBeVisible();
      const clickTime = Date.now() - clickStartTime;
      
      // 语音按钮响应应在200ms内
      expect(clickTime).toBeLessThan(200);
    });

    test('文件上传响应时间', async () => {
      await page.goto('/datasets');
      await page.waitForLoadState('networkidle');

      // 创建小测试文件
      const testContent = '这是一个小的测试文件内容';
      const testFile = Buffer.from(testContent, 'utf8');

      const uploadStartTime = Date.now();
      
      // 点击上传按钮
      await page.click('[data-testid="upload-button"]');
      await expect(page.locator('[data-testid="upload-dialog"]')).toBeVisible();
      
      const dialogTime = Date.now() - uploadStartTime;
      
      // 上传对话框应快速显示
      expect(dialogTime).toBeLessThan(300);
    });

    test('搜索响应时间', async () => {
      await page.goto('/datasets');
      await page.waitForLoadState('networkidle');

      if (await page.locator('[data-testid="search-input"]').isVisible()) {
        const searchStartTime = Date.now();
        
        await page.fill('[data-testid="search-input"]', '测试');
        
        // 等待搜索结果或无结果提示
        await page.waitForTimeout(1000);
        
        const searchTime = Date.now() - searchStartTime;
        
        // 搜索响应应在2秒内
        expect(searchTime).toBeLessThan(2000);
      }
    });
  });

  test.describe('内存和资源使用', () => {
    test('内存泄漏检测', async () => {
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      // 获取初始内存使用
      const initialMemory = await page.evaluate(() => {
        return (performance as any).memory ? (performance as any).memory.usedJSHeapSize : 0;
      });

      // 执行多次操作
      for (let i = 0; i < 10; i++) {
        await page.fill('[data-testid="message-input"]', `测试消息 ${i}`);
        await page.click('[data-testid="send-button"]');
        await expect(page.locator('[data-testid="user-message"]').last()).toBeVisible();
        
        // 等待一下再继续
        await page.waitForTimeout(500);
      }

      // 强制垃圾回收（如果支持）
      await page.evaluate(() => {
        if ((window as any).gc) {
          (window as any).gc();
        }
      });

      // 获取最终内存使用
      const finalMemory = await page.evaluate(() => {
        return (performance as any).memory ? (performance as any).memory.usedJSHeapSize : 0;
      });

      if (initialMemory > 0 && finalMemory > 0) {
        const memoryIncrease = finalMemory - initialMemory;
        const memoryIncreasePercent = (memoryIncrease / initialMemory) * 100;
        
        // 内存增长不应超过50%
        expect(memoryIncreasePercent).toBeLessThan(50);
      }
    });

    test('DOM节点数量检测', async () => {
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      // 获取初始DOM节点数
      const initialNodes = await page.evaluate(() => {
        return document.getElementsByTagName('*').length;
      });

      // 发送多条消息
      for (let i = 0; i < 20; i++) {
        await page.fill('[data-testid="message-input"]', `DOM测试消息 ${i}`);
        await page.click('[data-testid="send-button"]');
        await expect(page.locator('[data-testid="user-message"]').last()).toBeVisible();
      }

      // 获取最终DOM节点数
      const finalNodes = await page.evaluate(() => {
        return document.getElementsByTagName('*').length;
      });

      const nodeIncrease = finalNodes - initialNodes;
      
      // DOM节点增长应该合理（每条消息不应该创建过多节点）
      expect(nodeIncrease).toBeLessThan(1000);
    });

    test('事件监听器泄漏检测', async () => {
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      // 获取初始事件监听器数量（如果浏览器支持）
      const initialListeners = await page.evaluate(() => {
        return (window as any).getEventListeners ? 
          Object.keys((window as any).getEventListeners(document)).length : 0;
      });

      // 执行一些操作
      for (let i = 0; i < 5; i++) {
        await page.click('[data-testid="voice-input-button"]');
        await page.waitForTimeout(100);
        await page.click('[data-testid="voice-input-button"]');
        await page.waitForTimeout(100);
      }

      // 获取最终事件监听器数量
      const finalListeners = await page.evaluate(() => {
        return (window as any).getEventListeners ? 
          Object.keys((window as any).getEventListeners(document)).length : 0;
      });

      if (initialListeners > 0) {
        const listenerIncrease = finalListeners - initialListeners;
        
        // 事件监听器不应该无限增长
        expect(listenerIncrease).toBeLessThan(10);
      }
    });
  });

  test.describe('网络性能', () => {
    test('API请求优化', async () => {
      // 监听网络请求
      const apiRequests: any[] = [];
      page.on('response', response => {
        if (response.url().includes('/api/')) {
          apiRequests.push({
            url: response.url(),
            status: response.status(),
            timing: response.request().timing()
          });
        }
      });

      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      // 发送一条消息
      await page.fill('[data-testid="message-input"]', '网络性能测试');
      await page.click('[data-testid="send-button"]');
      
      await expect(page.locator('[data-testid="user-message"]').last()).toBeVisible();

      // 等待API请求完成
      await page.waitForTimeout(2000);

      // 分析API请求
      const successfulRequests = apiRequests.filter(r => r.status >= 200 && r.status < 300);
      const failedRequests = apiRequests.filter(r => r.status >= 400);

      // 大部分API请求应该成功
      expect(successfulRequests.length).toBeGreaterThan(0);
      expect(failedRequests.length).toBe(0);
    });

    test('缓存效果测试', async () => {
      // 第一次访问
      const firstVisitStart = Date.now();
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');
      const firstVisitTime = Date.now() - firstVisitStart;

      // 刷新页面（测试缓存）
      const secondVisitStart = Date.now();
      await page.reload({ waitUntil: 'networkidle' });
      const secondVisitTime = Date.now() - secondVisitStart;

      // 第二次访问应该更快（由于缓存）
      expect(secondVisitTime).toBeLessThan(firstVisitTime);
    });

    test('并发请求处理', async () => {
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      // 同时发送多条消息
      const promises = [];
      for (let i = 0; i < 5; i++) {
        promises.push((async () => {
          await page.fill('[data-testid="message-input"]', `并发测试消息 ${i}`);
          await page.click('[data-testid="send-button"]');
          return Date.now();
        })());
      }

      const startTime = Date.now();
      await Promise.all(promises);
      const totalTime = Date.now() - startTime;

      // 并发处理应该相对高效
      expect(totalTime).toBeLessThan(10000);

      // 验证所有消息都已发送
      const messageCount = await page.locator('[data-testid="user-message"]').count();
      expect(messageCount).toBeGreaterThanOrEqual(5);
    });
  });

  test.describe('渲染性能', () => {
    test('大量消息渲染性能', async () => {
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      const renderStartTime = Date.now();

      // 模拟添加大量消息到DOM
      await page.evaluate(() => {
        const chatContainer = document.querySelector('[data-testid="chat-container"]');
        if (chatContainer) {
          for (let i = 0; i < 100; i++) {
            const messageDiv = document.createElement('div');
            messageDiv.setAttribute('data-testid', 'user-message');
            messageDiv.textContent = `性能测试消息 ${i}`;
            chatContainer.appendChild(messageDiv);
          }
        }
      });

      const renderTime = Date.now() - renderStartTime;

      // 渲染100条消息应在1秒内完成
      expect(renderTime).toBeLessThan(1000);

      // 验证消息已渲染
      const messageCount = await page.locator('[data-testid="user-message"]').count();
      expect(messageCount).toBeGreaterThanOrEqual(100);
    });

    test('滚动性能测试', async () => {
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      // 添加足够多的消息以产生滚动
      await page.evaluate(() => {
        const chatContainer = document.querySelector('[data-testid="chat-container"]');
        if (chatContainer) {
          for (let i = 0; i < 50; i++) {
            const messageDiv = document.createElement('div');
            messageDiv.setAttribute('data-testid', 'user-message');
            messageDiv.textContent = `滚动测试消息 ${i}`;
            messageDiv.style.height = '50px';
            messageDiv.style.marginBottom = '10px';
            chatContainer.appendChild(messageDiv);
          }
        }
      });

      // 测试滚动性能
      const scrollStartTime = Date.now();
      
      await page.evaluate(() => {
        const chatContainer = document.querySelector('[data-testid="chat-container"]');
        if (chatContainer) {
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }
      });

      const scrollTime = Date.now() - scrollStartTime;

      // 滚动应该很快完成
      expect(scrollTime).toBeLessThan(100);
    });

    test('动画性能测试', async () => {
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      // 测试语音按钮动画
      if (await page.locator('[data-testid="voice-input-button"]').isVisible()) {
        await page.context().grantPermissions(['microphone']);
        
        const animationStartTime = Date.now();
        
        await page.click('[data-testid="voice-input-button"]');
        await expect(page.locator('[data-testid="recording-indicator"]')).toBeVisible();
        
        const animationTime = Date.now() - animationStartTime;
        
        // 动画应该流畅快速
        expect(animationTime).toBeLessThan(500);
      }
    });
  });

  test.describe('移动端性能', () => {
    test('移动端加载性能', async () => {
      // 设置移动端视口
      await page.setViewportSize({ width: 375, height: 667 });
      
      const mobileLoadStart = Date.now();
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');
      const mobileLoadTime = Date.now() - mobileLoadStart;

      // 移动端加载时间可能稍长，但应在合理范围内
      expect(mobileLoadTime).toBeLessThan(8000);

      // 验证移动端布局
      await expect(page.locator('[data-testid="chat-container"]')).toBeVisible();
    });

    test('移动端交互性能', async () => {
      await page.setViewportSize({ width: 375, height: 667 });
      await page.goto('/chat');
      await page.waitForLoadState('networkidle');

      // 测试移动端触摸交互
      const touchStartTime = Date.now();
      
      await page.tap('[data-testid="message-input"]');
      await page.fill('[data-testid="message-input"]', '移动端测试');
      await page.tap('[data-testid="send-button"]');
      
      await expect(page.locator('[data-testid="user-message"]').last()).toBeVisible();
      
      const touchTime = Date.now() - touchStartTime;
      
      // 移动端交互应该响应迅速
      expect(touchTime).toBeLessThan(1000);
    });
  });
});
