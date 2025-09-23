import { test, expect, Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

/**
 * 数据集管理功能端到端测试
 * 测试文件上传、数据集创建、文档管理等功能
 */

test.describe('数据集管理功能测试', () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    
    // 导航到数据集页面
    await page.goto('/datasets');
    
    // 等待页面加载完成
    await page.waitForLoadState('networkidle');
    
    // 验证页面基本元素
    await expect(page.locator('[data-testid="datasets-container"]')).toBeVisible();
  });

  test('数据集页面基本元素', async () => {
    // 验证页面标题
    await expect(page.locator('h1')).toContainText('数据集管理');
    
    // 验证上传按钮
    await expect(page.locator('[data-testid="upload-button"]')).toBeVisible();
    
    // 验证数据集列表
    await expect(page.locator('[data-testid="datasets-list"]')).toBeVisible();
    
    // 验证搜索框
    if (await page.locator('[data-testid="search-input"]').isVisible()) {
      await expect(page.locator('[data-testid="search-input"]')).toBeEnabled();
    }
  });

  test('文件上传功能', async () => {
    // 创建测试文件
    const testFilePath = createTestDocument();
    
    try {
      // 点击上传按钮
      await page.click('[data-testid="upload-button"]');
      
      // 验证上传对话框
      await expect(page.locator('[data-testid="upload-dialog"]')).toBeVisible();
      
      // 选择文件
      const fileInput = page.locator('input[type="file"]');
      await fileInput.setInputFiles(testFilePath);
      
      // 验证文件选择
      await expect(page.locator('[data-testid="selected-files"]')).toContainText('test-document.txt');
      
      // 设置数据集名称
      await page.fill('[data-testid="dataset-name"]', 'Playwright测试数据集');
      
      // 开始上传
      await page.click('[data-testid="start-upload"]');
      
      // 验证上传进度
      await expect(page.locator('[data-testid="upload-progress"]')).toBeVisible();
      
      // 等待上传完成
      await expect(page.locator('[data-testid="upload-success"]')).toBeVisible({ timeout: 30000 });
      
      // 验证数据集出现在列表中
      await expect(page.locator('[data-testid="dataset-item"]')).toContainText('Playwright测试数据集');
      
    } finally {
      // 清理测试文件
      if (fs.existsSync(testFilePath)) {
        fs.unlinkSync(testFilePath);
      }
    }
  });

  test('批量文件上传', async () => {
    // 创建多个测试文件
    const testFiles = [
      createTestDocument('文档1', '这是第一个测试文档的内容'),
      createTestDocument('文档2', '这是第二个测试文档的内容'),
      createTestDocument('文档3', '这是第三个测试文档的内容')
    ];
    
    try {
      await page.click('[data-testid="upload-button"]');
      await expect(page.locator('[data-testid="upload-dialog"]')).toBeVisible();
      
      // 选择多个文件
      const fileInput = page.locator('input[type="file"]');
      await fileInput.setInputFiles(testFiles);
      
      // 验证多文件选择
      await expect(page.locator('[data-testid="selected-files"]')).toContainText('3 个文件');
      
      // 设置数据集信息
      await page.fill('[data-testid="dataset-name"]', '批量上传测试');
      await page.fill('[data-testid="dataset-description"]', '这是批量上传的测试数据集');
      
      // 开始上传
      await page.click('[data-testid="start-upload"]');
      
      // 验证批量上传进度
      await expect(page.locator('[data-testid="batch-upload-progress"]')).toBeVisible();
      
      // 等待所有文件上传完成
      await expect(page.locator('[data-testid="upload-success"]')).toBeVisible({ timeout: 60000 });
      
      // 验证文档数量
      await expect(page.locator('[data-testid="document-count"]')).toContainText('3');
      
    } finally {
      // 清理测试文件
      testFiles.forEach(filePath => {
        if (fs.existsSync(filePath)) {
          fs.unlinkSync(filePath);
        }
      });
    }
  });

  test('支持的文件格式验证', async () => {
    // 测试支持的格式
    const supportedFormats = [
      { name: 'test.txt', content: '文本文件内容' },
      { name: 'test.md', content: '# Markdown文件\n这是markdown内容' },
      { name: 'test.pdf', content: 'PDF文件内容' } // 简化处理
    ];
    
    for (const format of supportedFormats) {
      const testFile = createTestFile(format.name, format.content);
      
      try {
        await page.click('[data-testid="upload-button"]');
        await expect(page.locator('[data-testid="upload-dialog"]')).toBeVisible();
        
        const fileInput = page.locator('input[type="file"]');
        await fileInput.setInputFiles(testFile);
        
        // 验证文件格式被接受
        await expect(page.locator('[data-testid="file-format-valid"]')).toBeVisible();
        
        // 关闭对话框
        await page.click('[data-testid="cancel-upload"]');
        
      } finally {
        if (fs.existsSync(testFile)) {
          fs.unlinkSync(testFile);
        }
      }
    }
  });

  test('不支持的文件格式处理', async () => {
    // 创建不支持的文件格式
    const unsupportedFile = createTestFile('test.exe', '不支持的文件内容');
    
    try {
      await page.click('[data-testid="upload-button"]');
      await expect(page.locator('[data-testid="upload-dialog"]')).toBeVisible();
      
      const fileInput = page.locator('input[type="file"]');
      await fileInput.setInputFiles(unsupportedFile);
      
      // 验证错误提示
      await expect(page.locator('[data-testid="file-format-error"]')).toBeVisible();
      await expect(page.locator('[data-testid="file-format-error"]')).toContainText('不支持的文件格式');
      
    } finally {
      if (fs.existsSync(unsupportedFile)) {
        fs.unlinkSync(unsupportedFile);
      }
    }
  });

  test('数据集详情查看', async () => {
    // 假设已有数据集，点击查看详情
    if (await page.locator('[data-testid="dataset-item"]').first().isVisible()) {
      await page.click('[data-testid="dataset-item"]');
      
      // 验证详情页面
      await expect(page.locator('[data-testid="dataset-detail"]')).toBeVisible();
      
      // 验证数据集信息
      await expect(page.locator('[data-testid="dataset-name"]')).toBeVisible();
      await expect(page.locator('[data-testid="dataset-description"]')).toBeVisible();
      await expect(page.locator('[data-testid="document-list"]')).toBeVisible();
      
      // 验证文档列表
      const documentCount = await page.locator('[data-testid="document-item"]').count();
      expect(documentCount).toBeGreaterThan(0);
    }
  });

  test('文档搜索功能', async () => {
    // 在搜索框中输入关键词
    if (await page.locator('[data-testid="search-input"]').isVisible()) {
      await page.fill('[data-testid="search-input"]', '测试');
      
      // 等待搜索结果
      await page.waitForTimeout(1000);
      
      // 验证搜索结果
      const searchResults = await page.locator('[data-testid="search-result"]').count();
      
      if (searchResults > 0) {
        // 验证搜索结果包含关键词
        await expect(page.locator('[data-testid="search-result"]').first()).toContainText('测试');
      }
      
      // 清空搜索
      await page.fill('[data-testid="search-input"]', '');
      await page.waitForTimeout(500);
    }
  });

  test('数据集删除功能', async () => {
    // 创建一个测试数据集用于删除
    const testFile = createTestDocument('删除测试', '这是用于删除测试的文档');
    
    try {
      // 上传测试数据集
      await page.click('[data-testid="upload-button"]');
      await expect(page.locator('[data-testid="upload-dialog"]')).toBeVisible();
      
      const fileInput = page.locator('input[type="file"]');
      await fileInput.setInputFiles(testFile);
      
      await page.fill('[data-testid="dataset-name"]', '待删除数据集');
      await page.click('[data-testid="start-upload"]');
      
      await expect(page.locator('[data-testid="upload-success"]')).toBeVisible({ timeout: 30000 });
      
      // 找到刚创建的数据集并删除
      const datasetItem = page.locator('[data-testid="dataset-item"]').filter({ hasText: '待删除数据集' });
      
      if (await datasetItem.isVisible()) {
        // 点击删除按钮
        await datasetItem.locator('[data-testid="delete-dataset"]').click();
        
        // 确认删除
        await expect(page.locator('[data-testid="delete-confirm-dialog"]')).toBeVisible();
        await page.click('[data-testid="confirm-delete"]');
        
        // 验证删除成功
        await expect(page.locator('[data-testid="delete-success"]')).toBeVisible();
        
        // 验证数据集从列表中消失
        await expect(datasetItem).not.toBeVisible();
      }
      
    } finally {
      if (fs.existsSync(testFile)) {
        fs.unlinkSync(testFile);
      }
    }
  });

  test('数据集编辑功能', async () => {
    // 假设已有数据集，测试编辑功能
    if (await page.locator('[data-testid="dataset-item"]').first().isVisible()) {
      const firstDataset = page.locator('[data-testid="dataset-item"]').first();
      
      // 点击编辑按钮
      await firstDataset.locator('[data-testid="edit-dataset"]').click();
      
      // 验证编辑对话框
      await expect(page.locator('[data-testid="edit-dataset-dialog"]')).toBeVisible();
      
      // 修改数据集信息
      const newName = '编辑后的数据集名称';
      const newDescription = '这是编辑后的描述';
      
      await page.fill('[data-testid="edit-dataset-name"]', newName);
      await page.fill('[data-testid="edit-dataset-description"]', newDescription);
      
      // 保存修改
      await page.click('[data-testid="save-dataset"]');
      
      // 验证修改成功
      await expect(page.locator('[data-testid="edit-success"]')).toBeVisible();
      
      // 验证数据集信息已更新
      await expect(page.locator('[data-testid="dataset-item"]').first()).toContainText(newName);
    }
  });

  test('文档预览功能', async () => {
    // 进入数据集详情
    if (await page.locator('[data-testid="dataset-item"]').first().isVisible()) {
      await page.click('[data-testid="dataset-item"]');
      await expect(page.locator('[data-testid="dataset-detail"]')).toBeVisible();
      
      // 点击文档预览
      if (await page.locator('[data-testid="document-item"]').first().isVisible()) {
        await page.click('[data-testid="document-item"]');
        
        // 验证预览对话框
        await expect(page.locator('[data-testid="document-preview"]')).toBeVisible();
        
        // 验证文档内容
        await expect(page.locator('[data-testid="document-content"]')).toBeVisible();
        
        // 关闭预览
        await page.click('[data-testid="close-preview"]');
        await expect(page.locator('[data-testid="document-preview"]')).not.toBeVisible();
      }
    }
  });

  test('数据集统计信息', async () => {
    // 验证统计信息显示
    if (await page.locator('[data-testid="dataset-stats"]').isVisible()) {
      // 验证总数据集数量
      await expect(page.locator('[data-testid="total-datasets"]')).toBeVisible();
      
      // 验证总文档数量
      await expect(page.locator('[data-testid="total-documents"]')).toBeVisible();
      
      // 验证存储使用情况
      if (await page.locator('[data-testid="storage-usage"]').isVisible()) {
        await expect(page.locator('[data-testid="storage-usage"]')).toBeVisible();
      }
    }
  });

  test('数据集导出功能', async () => {
    if (await page.locator('[data-testid="dataset-item"]').first().isVisible()) {
      const firstDataset = page.locator('[data-testid="dataset-item"]').first();
      
      // 点击导出按钮
      if (await firstDataset.locator('[data-testid="export-dataset"]').isVisible()) {
        await firstDataset.locator('[data-testid="export-dataset"]').click();
        
        // 验证导出选项
        await expect(page.locator('[data-testid="export-options"]')).toBeVisible();
        
        // 选择导出格式
        await page.selectOption('[data-testid="export-format"]', 'json');
        
        // 开始导出
        await page.click('[data-testid="start-export"]');
        
        // 验证导出进度
        await expect(page.locator('[data-testid="export-progress"]')).toBeVisible();
        
        // 等待导出完成
        await expect(page.locator('[data-testid="export-complete"]')).toBeVisible({ timeout: 30000 });
      }
    }
  });

  test('响应式设计测试', async () => {
    // 测试桌面视图
    await page.setViewportSize({ width: 1200, height: 800 });
    await expect(page.locator('[data-testid="datasets-container"]')).toBeVisible();
    
    // 测试平板视图
    await page.setViewportSize({ width: 768, height: 1024 });
    await expect(page.locator('[data-testid="datasets-container"]')).toBeVisible();
    
    // 在平板视图下测试上传功能
    await page.click('[data-testid="upload-button"]');
    await expect(page.locator('[data-testid="upload-dialog"]')).toBeVisible();
    
    // 测试手机视图
    await page.setViewportSize({ width: 375, height: 667 });
    await expect(page.locator('[data-testid="datasets-container"]')).toBeVisible();
    
    // 验证移动端布局
    if (await page.locator('[data-testid="mobile-menu"]').isVisible()) {
      await page.click('[data-testid="mobile-menu"]');
      await expect(page.locator('[data-testid="mobile-nav"]')).toBeVisible();
    }
  });
});

/**
 * 创建测试文档的辅助函数
 */
function createTestDocument(title: string = 'test-document', content: string = '这是一个测试文档的内容。它包含了一些示例文本用于测试文档上传和处理功能。'): string {
  const testDir = path.join(__dirname, '../fixtures');
  if (!fs.existsSync(testDir)) {
    fs.mkdirSync(testDir, { recursive: true });
  }
  
  const filePath = path.join(testDir, `${title}.txt`);
  fs.writeFileSync(filePath, content, 'utf8');
  
  return filePath;
}

/**
 * 创建指定格式测试文件的辅助函数
 */
function createTestFile(filename: string, content: string): string {
  const testDir = path.join(__dirname, '../fixtures');
  if (!fs.existsSync(testDir)) {
    fs.mkdirSync(testDir, { recursive: true });
  }
  
  const filePath = path.join(testDir, filename);
  fs.writeFileSync(filePath, content, 'utf8');
  
  return filePath;
}
