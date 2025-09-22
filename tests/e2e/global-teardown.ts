import { chromium, FullConfig } from '@playwright/test';

/**
 * Playwright 全局清理
 * 在所有测试结束后执行的清理操作
 */
async function globalTeardown(config: FullConfig) {
  console.log('🧹 开始全局测试清理...');
  
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // 1. 清理测试数据
    console.log('🗑️ 清理测试数据...');
    await cleanupTestData(page);
    
    // 2. 清理测试用户
    console.log('👤 清理测试用户...');
    await cleanupTestUser(page);
    
    // 3. 清理临时文件
    console.log('📁 清理临时文件...');
    await cleanupTempFiles();
    
    // 4. 生成测试报告摘要
    console.log('📊 生成测试报告摘要...');
    await generateTestSummary();
    
    console.log('✅ 全局清理完成');
    
  } catch (error) {
    console.error('❌ 全局清理失败:', error);
    // 不抛出错误，避免影响测试结果
  } finally {
    await browser.close();
  }
}

/**
 * 清理测试数据
 */
async function cleanupTestData(page: any) {
  try {
    // 删除测试数据集
    const response = await page.request.delete('http://localhost:8080/api/v1/datasets/playwright-test-dataset', {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (response.ok()) {
      console.log('  ✅ 测试数据集已删除');
    } else {
      console.warn('  ⚠️ 删除测试数据集失败或不存在');
    }
    
  } catch (error) {
    console.warn('清理测试数据时出错:', error);
  }
}

/**
 * 清理测试用户
 */
async function cleanupTestUser(page: any) {
  try {
    // 删除测试用户（如果API支持）
    const response = await page.request.delete('http://localhost:8080/api/v1/users/playwright-test-user', {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (response.ok()) {
      console.log('  ✅ 测试用户已删除');
    } else {
      console.warn('  ⚠️ 删除测试用户失败或不存在');
    }
    
  } catch (error) {
    console.warn('清理测试用户时出错:', error);
  }
}

/**
 * 清理临时文件
 */
async function cleanupTempFiles() {
  try {
    const fs = require('fs').promises;
    const path = require('path');
    
    // 清理测试生成的临时文件
    const tempDirs = [
      './test-results',
      './playwright-report/temp',
      './downloads'
    ];
    
    for (const dir of tempDirs) {
      try {
        const fullPath = path.resolve(dir);
        await fs.rmdir(fullPath, { recursive: true });
        console.log(`  ✅ 已清理临时目录: ${dir}`);
      } catch (error) {
        // 目录可能不存在，忽略错误
      }
    }
    
  } catch (error) {
    console.warn('清理临时文件时出错:', error);
  }
}

/**
 * 生成测试报告摘要
 */
async function generateTestSummary() {
  try {
    const fs = require('fs').promises;
    const path = require('path');
    
    // 读取测试结果
    let testResults = null;
    try {
      const resultsPath = path.resolve('./test-results.json');
      const resultsContent = await fs.readFile(resultsPath, 'utf8');
      testResults = JSON.parse(resultsContent);
    } catch (error) {
      console.warn('无法读取测试结果文件');
      return;
    }
    
    // 生成摘要
    const summary = {
      timestamp: new Date().toISOString(),
      totalTests: testResults.stats?.total || 0,
      passed: testResults.stats?.passed || 0,
      failed: testResults.stats?.failed || 0,
      skipped: testResults.stats?.skipped || 0,
      duration: testResults.stats?.duration || 0,
      environment: {
        node: process.version,
        platform: process.platform,
        arch: process.arch
      },
      coverage: {
        // 如果有覆盖率数据，在这里添加
      }
    };
    
    // 保存摘要
    const summaryPath = path.resolve('./test-summary.json');
    await fs.writeFile(summaryPath, JSON.stringify(summary, null, 2));
    
    console.log('  ✅ 测试报告摘要已生成');
    console.log(`     总测试数: ${summary.totalTests}`);
    console.log(`     通过: ${summary.passed}`);
    console.log(`     失败: ${summary.failed}`);
    console.log(`     跳过: ${summary.skipped}`);
    console.log(`     耗时: ${(summary.duration / 1000).toFixed(2)}s`);
    
  } catch (error) {
    console.warn('生成测试报告摘要时出错:', error);
  }
}

export default globalTeardown;
