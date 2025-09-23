# VoiceHelper E2E 测试套件

基于 Playwright 的端到端测试套件，用于测试 VoiceHelper 系统的完整功能。

## 🚀 快速开始

### 环境准备

1. 安装依赖：
```bash
cd tests/e2e
npm install
```

2. 安装浏览器：
```bash
npm run install-browsers
```

3. 安装系统依赖（Linux）：
```bash
npm run install-deps
```

### 启动服务

在运行测试前，确保所有服务都已启动：

```bash
# 启动基础设施服务
docker-compose -f deploy/docker-compose.local.yml up -d

# 启动前端服务
cd frontend && npm run dev

# 启动后端服务
cd backend && go run cmd/server/main.go

# 启动算法服务
cd algo && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 🧪 运行测试

### 基本命令

```bash
# 运行所有测试
npm test

# 运行特定测试文件
npm run test:chat      # 聊天功能测试
npm run test:voice     # 语音交互测试
npm run test:datasets  # 数据集管理测试
npm run test:api       # API集成测试
npm run test:performance # 性能测试

# 带界面运行测试
npm run test:headed

# 调试模式
npm run test:debug

# 交互式UI模式
npm run test:ui
```

### 测试标签

```bash
# 冒烟测试（快速验证核心功能）
npm run test:smoke

# 回归测试（完整功能验证）
npm run test:regression
```

### 指定浏览器

```bash
# 只在Chrome中运行
npx playwright test --project=chromium

# 只在Firefox中运行
npx playwright test --project=firefox

# 只在移动端运行
npx playwright test --project="Mobile Chrome"
```

## 📊 测试报告

### 查看报告

```bash
# 查看HTML报告
npm run report

# 生成JUnit报告
npx playwright test --reporter=junit

# 生成JSON报告
npx playwright test --reporter=json
```

### 报告文件

- `playwright-report/` - HTML报告
- `test-results.json` - JSON格式结果
- `test-results.xml` - JUnit格式结果
- `test-summary.json` - 测试摘要

## 🏗️ 测试结构

```
tests/e2e/
├── playwright.config.ts     # Playwright配置
├── global-setup.ts         # 全局设置
├── global-teardown.ts      # 全局清理
├── tests/                  # 测试用例
│   ├── chat.spec.ts        # 聊天功能测试
│   ├── voice.spec.ts       # 语音交互测试
│   ├── datasets.spec.ts    # 数据集管理测试
│   ├── api.spec.ts         # API集成测试
│   └── performance.spec.ts # 性能测试
├── fixtures/               # 测试数据
└── utils/                  # 测试工具
```

## 🧪 测试覆盖范围

### 功能测试

#### 聊天功能 (`chat.spec.ts`)
- ✅ 基本文本聊天
- ✅ 流式响应
- ✅ 对话历史
- ✅ 消息操作（复制、重新生成）
- ✅ 键盘快捷键
- ✅ 响应式设计

#### 语音交互 (`voice.spec.ts`)
- ✅ 语音输入按钮状态
- ✅ 语音录制和转文字
- ✅ 语音合成播放
- ✅ 实时语音对话
- ✅ 语音设置配置
- ✅ 多语言支持
- ✅ 错误处理

#### 数据集管理 (`datasets.spec.ts`)
- ✅ 文件上传（单个/批量）
- ✅ 支持格式验证
- ✅ 数据集CRUD操作
- ✅ 文档搜索和预览
- ✅ 数据集导出
- ✅ 响应式设计

### API测试 (`api.spec.ts`)
- ✅ 健康检查和基础API
- ✅ 聊天API（流式/非流式）
- ✅ 语音API（ASR/TTS）
- ✅ 数据集管理API
- ✅ 集成服务API
- ✅ 认证和授权
- ✅ 错误处理和边界测试
- ✅ 安全测试

### 性能测试 (`performance.spec.ts`)
- ✅ 页面加载性能
- ✅ 交互响应时间
- ✅ 内存使用监控
- ✅ 网络性能优化
- ✅ 渲染性能
- ✅ 移动端性能

## 🔧 配置说明

### 浏览器支持

- **桌面浏览器**: Chrome, Firefox, Safari
- **移动设备**: Mobile Chrome, Mobile Safari
- **平板设备**: iPad Pro

### 环境配置

测试环境通过以下方式配置：

1. **服务地址**: 
   - 前端: `http://localhost:3000`
   - 后端: `http://localhost:8080`
   - 算法: `http://localhost:8000`

2. **超时设置**:
   - 操作超时: 30秒
   - 导航超时: 30秒
   - 测试超时: 60秒

3. **重试策略**:
   - CI环境: 2次重试
   - 本地环境: 0次重试

### 测试数据

测试使用以下数据：

- **测试用户**: `playwright-test-user`
- **测试数据集**: `playwright-test-dataset`
- **测试文档**: 动态生成的文本文件

## 🐛 调试指南

### 常见问题

1. **服务未启动**
   ```bash
   # 检查服务状态
   curl http://localhost:3000  # 前端
   curl http://localhost:8080/health  # 后端
   curl http://localhost:8000/health  # 算法
   ```

2. **权限问题**
   ```bash
   # 授予麦克风权限（在测试中自动处理）
   await page.context().grantPermissions(['microphone']);
   ```

3. **元素未找到**
   ```bash
   # 使用调试模式查看页面状态
   npm run test:debug
   ```

### 调试技巧

1. **截图调试**:
   ```typescript
   await page.screenshot({ path: 'debug.png' });
   ```

2. **控制台日志**:
   ```typescript
   page.on('console', msg => console.log(msg.text()));
   ```

3. **网络监控**:
   ```typescript
   page.on('response', response => {
     console.log(response.url(), response.status());
   });
   ```

## 📈 持续集成

### GitHub Actions

```yaml
name: E2E Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
      - name: Install dependencies
        run: npm ci
      - name: Install Playwright
        run: npx playwright install --with-deps
      - name: Run tests
        run: npm test
      - name: Upload report
        uses: actions/upload-artifact@v3
        with:
          name: playwright-report
          path: playwright-report/
```

### Docker集成

```dockerfile
FROM mcr.microsoft.com/playwright:v1.40.0-focal
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
CMD ["npm", "test"]
```

## 🤝 贡献指南

### 添加新测试

1. 在相应的 `.spec.ts` 文件中添加测试用例
2. 使用描述性的测试名称
3. 添加适当的测试标签 (`@smoke`, `@regression`)
4. 确保测试的独立性和可重复性

### 测试最佳实践

1. **使用数据测试ID**: `data-testid="element-name"`
2. **等待元素状态**: 使用 `expect().toBeVisible()` 而不是 `waitForTimeout()`
3. **清理测试数据**: 在测试后清理创建的数据
4. **错误处理**: 测试正常流程和异常情况
5. **性能考虑**: 避免不必要的等待和操作

### 代码规范

- 使用 TypeScript
- 遵循 ESLint 规则
- 添加适当的注释
- 保持测试简洁明了

## 📞 支持

如有问题或建议，请：

1. 查看 [Playwright 官方文档](https://playwright.dev/)
2. 检查现有的 [Issues](https://github.com/your-repo/issues)
3. 创建新的 Issue 或 Pull Request

---

**Happy Testing! 🎭**
