import { defineConfig, devices } from '@playwright/test';

/**
 * VoiceHelper Playwright 配置
 * 端到端测试配置文件
 */
export default defineConfig({
  // 测试目录
  testDir: './tests',
  
  // 全局设置
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  
  // 报告配置
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results.json' }],
    ['junit', { outputFile: 'test-results.xml' }]
  ],
  
  // 全局测试配置
  use: {
    // 基础URL
    baseURL: 'http://localhost:3000',
    
    // 浏览器配置
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    
    // 网络配置
    ignoreHTTPSErrors: true,
    
    // 等待配置
    actionTimeout: 30000,
    navigationTimeout: 30000,
    
    // 额外的HTTP头
    extraHTTPHeaders: {
      'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
    }
  },

  // 项目配置 - 多浏览器测试
  projects: [
    // 桌面浏览器
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
    
    // 移动设备
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
    },
    
    // 平板设备
    {
      name: 'iPad',
      use: { ...devices['iPad Pro'] },
    }
  ],

  // 测试服务器配置
  webServer: [
    {
      command: 'npm run dev',
      port: 3000,
      cwd: '../frontend',
      reuseExistingServer: !process.env.CI,
      timeout: 120000,
    },
    {
      command: 'go run cmd/server/main.go',
      port: 8080,
      cwd: '../backend',
      reuseExistingServer: !process.env.CI,
      timeout: 60000,
    },
    {
      command: 'python -m uvicorn app.main:app --host 0.0.0.0 --port 8000',
      port: 8000,
      cwd: '../algo',
      reuseExistingServer: !process.env.CI,
      timeout: 60000,
    }
  ],

  // 全局设置和清理
  globalSetup: require.resolve('./global-setup'),
  globalTeardown: require.resolve('./global-teardown'),
  
  // 测试超时
  timeout: 60000,
  expect: {
    timeout: 10000
  }
});
