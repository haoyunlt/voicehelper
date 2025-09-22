import { chromium, FullConfig } from '@playwright/test';

/**
 * Playwright 全局设置
 * 在所有测试开始前执行的初始化操作
 */
async function globalSetup(config: FullConfig) {
  console.log('🚀 开始全局测试设置...');
  
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // 1. 等待服务启动
    console.log('⏳ 等待服务启动...');
    await waitForServices(page);
    
    // 2. 初始化测试数据
    console.log('📊 初始化测试数据...');
    await initializeTestData(page);
    
    // 3. 创建测试用户
    console.log('👤 创建测试用户...');
    await createTestUser(page);
    
    // 4. 验证系统健康状态
    console.log('🏥 验证系统健康状态...');
    await verifySystemHealth(page);
    
    console.log('✅ 全局设置完成');
    
  } catch (error) {
    console.error('❌ 全局设置失败:', error);
    throw error;
  } finally {
    await browser.close();
  }
}

/**
 * 等待所有服务启动
 */
async function waitForServices(page: any) {
  const services = [
    { name: '前端服务', url: 'http://localhost:3000', timeout: 120000 },
    { name: '后端服务', url: 'http://localhost:8080/health', timeout: 60000 },
    { name: '算法服务', url: 'http://localhost:8000/health', timeout: 60000 }
  ];
  
  for (const service of services) {
    console.log(`  等待 ${service.name} 启动...`);
    await waitForService(page, service.url, service.timeout);
    console.log(`  ✅ ${service.name} 已启动`);
  }
}

/**
 * 等待单个服务启动
 */
async function waitForService(page: any, url: string, timeout: number) {
  const startTime = Date.now();
  
  while (Date.now() - startTime < timeout) {
    try {
      const response = await page.request.get(url);
      if (response.ok()) {
        return;
      }
    } catch (error) {
      // 服务还未启动，继续等待
    }
    
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
  
  throw new Error(`服务 ${url} 在 ${timeout}ms 内未启动`);
}

/**
 * 初始化测试数据
 */
async function initializeTestData(page: any) {
  try {
    // 创建测试数据集
    const testDataset = {
      name: 'playwright-test-dataset',
      description: 'Playwright自动化测试数据集',
      documents: [
        {
          title: '测试文档1',
          content: '这是一个用于Playwright测试的示例文档。它包含了基本的问答内容。'
        },
        {
          title: '测试文档2', 
          content: '这是另一个测试文档，用于验证RAG检索功能。'
        }
      ]
    };
    
    // 通过API创建测试数据
    const response = await page.request.post('http://localhost:8080/api/v1/datasets', {
      data: testDataset,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok()) {
      console.warn('创建测试数据集失败，可能已存在');
    }
    
  } catch (error) {
    console.warn('初始化测试数据时出错:', error);
  }
}

/**
 * 创建测试用户
 */
async function createTestUser(page: any) {
  try {
    const testUser = {
      username: 'playwright-test-user',
      email: 'test@playwright.com',
      password: 'test123456',
      role: 'user'
    };
    
    const response = await page.request.post('http://localhost:8080/api/v1/auth/register', {
      data: testUser,
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok()) {
      console.warn('创建测试用户失败，可能已存在');
    }
    
    // 保存测试用户信息到环境变量
    process.env.TEST_USER_EMAIL = testUser.email;
    process.env.TEST_USER_PASSWORD = testUser.password;
    
  } catch (error) {
    console.warn('创建测试用户时出错:', error);
  }
}

/**
 * 验证系统健康状态
 */
async function verifySystemHealth(page: any) {
  const healthChecks = [
    { name: '后端健康检查', url: 'http://localhost:8080/health' },
    { name: '算法服务健康检查', url: 'http://localhost:8000/health' },
    { name: 'API ping测试', url: 'http://localhost:8080/api/v1/ping' }
  ];
  
  for (const check of healthChecks) {
    try {
      const response = await page.request.get(check.url);
      if (!response.ok()) {
        throw new Error(`${check.name} 失败: ${response.status()}`);
      }
      console.log(`  ✅ ${check.name} 通过`);
    } catch (error) {
      console.error(`  ❌ ${check.name} 失败:`, error);
      throw error;
    }
  }
}

export default globalSetup;
