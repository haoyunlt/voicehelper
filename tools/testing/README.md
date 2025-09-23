# VoiceHelper 测试套件

本目录包含了 VoiceHelper 项目的完整测试套件，涵盖单元测试、集成测试、性能测试和端到端测试。

## 📁 目录结构

```
tools/testing/
├── unit/                           # 单元测试
│   ├── security/                   # 安全测试
│   │   └── test_security.py       # 认证、授权、输入验证、XSS防护
│   ├── error_handling/             # 异常处理测试
│   │   └── test_error_handling.py # 网络异常、服务不可用、数据错误
│   ├── voice/                      # 语音功能测试
│   │   └── test_voice_processing.py # ASR、TTS、实时处理、情感分析
│   └── multimodal/                 # 多模态测试
│       └── test_multimodal_fusion.py # 文本+图像+语音融合
├── integration/                    # 集成测试
│   └── test_service_integration.py # 服务间集成、数据库、缓存、消息队列
├── performance/                    # 性能测试
│   └── test_comprehensive_performance.py # 并发、内存、响应时间、负载
├── e2e/                           # 端到端测试
│   └── test_business_workflows.py # 完整业务流程测试
├── datasets/                      # 测试数据集
│   ├── chat/                      # 对话测试数据
│   ├── voice/                     # 语音测试数据
│   ├── security/                  # 安全测试数据
│   └── performance/               # 性能测试数据
├── test_runner_comprehensive.py   # 综合测试运行器
└── README.md                      # 本文件
```

## 🚀 快速开始

### 1. 安装测试依赖

```bash
# 安装测试依赖包
pip install -r requirements-test.txt

# 或者使用项目根目录的安装脚本
./setup_all_venvs.sh
```

### 2. 运行所有测试

```bash
# 运行完整测试套件
python tools/testing/test_runner_comprehensive.py

# 只运行关键测试
python tools/testing/test_runner_comprehensive.py --critical-only

# 运行指定测试套件
python tools/testing/test_runner_comprehensive.py --suites security voice_processing
```

### 3. 运行单个测试套件

```bash
# 安全测试
pytest tools/testing/unit/security/test_security.py -v

# 语音处理测试
pytest tools/testing/unit/voice/test_voice_processing.py -v

# 性能测试
pytest tools/testing/performance/test_comprehensive_performance.py -v

# 端到端测试
pytest tools/testing/e2e/test_business_workflows.py -v
```

## 📊 测试覆盖范围

### 单元测试 (Unit Tests)

#### 安全测试 (`test_security.py`)
- **认证安全**: JWT令牌验证、过期检测、篡改检测
- **输入验证**: SQL注入防护、XSS攻击防护、文件上传验证
- **数据隐私**: 敏感数据脱敏、数据加密、安全日志
- **API安全**: 安全HTTP头、CORS配置、请求频率限制

#### 异常处理测试 (`test_error_handling.py`)
- **网络异常**: 连接超时、连接拒绝、HTTP错误状态
- **数据验证**: JSON解析错误、数据类型验证、文件处理错误
- **资源耗尽**: 内存限制、并发限制、磁盘空间
- **服务依赖**: 数据库连接、外部API熔断器、优雅降级

#### 语音处理测试 (`test_voice_processing.py`)
- **ASR处理**: 音频格式验证、转录准确性、噪声处理、实时流式处理
- **TTS处理**: 文本预处理、语音配置、音频质量评估
- **情感分析**: 多模态情感分析、情感趋势分析

#### 多模态融合测试 (`test_multimodal_fusion.py`)
- **数据预处理**: 文本、图像、音频预处理
- **跨模态注意力**: 文本-图像、音频-文本、多模态自注意力
- **融合策略**: 早期融合、晚期融合、注意力融合、门控融合、分层融合
- **性能优化**: 并行处理、缓存优化、自适应融合

### 集成测试 (Integration Tests)

#### 服务集成测试 (`test_service_integration.py`)
- **后端-算法服务**: 服务发现、健康检查、请求路由、负载均衡、熔断器
- **数据库集成**: 连接池、事务管理、数据库迁移
- **缓存服务**: 缓存层集成、分布式缓存一致性
- **消息队列**: 异步任务处理、事件驱动架构

### 性能测试 (Performance Tests)

#### 综合性能测试 (`test_comprehensive_performance.py`)
- **并发性能**: 异步并发请求、线程池性能、连接池性能
- **内存性能**: 内存使用监控、内存泄漏检测、垃圾回收性能
- **响应时间**: API响应时间分布、数据库查询性能
- **压力测试**: 渐进式负载增加、资源耗尽场景

### 端到端测试 (E2E Tests)

#### 业务流程测试 (`test_business_workflows.py`)
- **用户入门流程**: 注册、会话创建、引导完成、偏好设置
- **对话流程**: 文本对话、语音对话、多模态交互、历史管理
- **文档管理**: 上传、处理、搜索、删除完整生命周期
- **跨服务集成**: 端到端用户旅程测试

## 🎯 测试策略

### 测试金字塔

```
        E2E Tests (少量)
       ┌─────────────────┐
       │  业务流程测试    │
       └─────────────────┘
      ┌───────────────────────┐
      │   Integration Tests   │  
      │     集成测试          │
      └───────────────────────┘
    ┌─────────────────────────────┐
    │        Unit Tests           │
    │        单元测试             │
    └─────────────────────────────┘
```

### 测试分类

| 测试类型 | 数量占比 | 执行频率 | 主要目的 |
|---------|---------|----------|----------|
| **单元测试** | 70% | 每次提交 | 验证单个组件功能 |
| **集成测试** | 20% | 每日构建 | 验证服务间交互 |
| **端到端测试** | 10% | 发布前 | 验证完整业务流程 |

### 性能基准

| 指标 | 目标值 | 测试覆盖 |
|------|--------|----------|
| **API响应时间** | P95 < 200ms | ✅ |
| **语音处理时间** | < 150ms | ✅ |
| **文档处理吞吐量** | > 10 docs/min | ✅ |
| **并发用户数** | > 100 | ✅ |
| **内存使用** | < 2GB | ✅ |
| **错误率** | < 0.1% | ✅ |

## 🔧 配置说明

### pytest 配置 (`pytest.ini`)

```ini
[tool:pytest]
testpaths = tools/testing
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: 单元测试
    integration: 集成测试
    e2e: 端到端测试
    performance: 性能测试
    slow: 慢速测试
```

### 测试环境变量

```bash
# 测试数据库
TEST_DATABASE_URL=postgresql://test:test@localhost:5432/voicehelper_test

# 测试Redis
TEST_REDIS_URL=redis://localhost:6379/1

# 测试服务端点
TEST_BACKEND_URL=http://localhost:8080
TEST_ALGO_URL=http://localhost:8000

# 性能测试配置
PERFORMANCE_TEST_DURATION=60
PERFORMANCE_TEST_USERS=100
```

## 📈 测试报告

### 生成测试报告

```bash
# 生成HTML和JSON报告
python tools/testing/test_runner_comprehensive.py

# 只生成JSON报告
python tools/testing/test_runner_comprehensive.py --no-html

# 自定义输出目录
python tools/testing/test_runner_comprehensive.py --output-dir custom/path
```

### 报告内容

- **执行总结**: 总体状态、执行时间、成功率
- **套件详情**: 每个测试套件的详细结果
- **性能指标**: 响应时间、吞吐量、资源使用
- **改进建议**: 基于测试结果的优化建议

## 🚨 故障排除

### 常见问题

1. **测试依赖缺失**
   ```bash
   pip install -r requirements-test.txt
   ```

2. **服务连接失败**
   ```bash
   # 检查服务状态
   docker-compose ps
   
   # 重启服务
   docker-compose restart
   ```

3. **权限问题**
   ```bash
   # 确保测试目录有写权限
   chmod -R 755 tools/testing/
   ```

4. **内存不足**
   ```bash
   # 跳过内存密集型测试
   pytest -m "not slow"
   ```

### 调试技巧

```bash
# 详细输出
pytest -v -s

# 只运行失败的测试
pytest --lf

# 在第一个失败时停止
pytest -x

# 并行运行测试
pytest -n auto
```

## 🔄 持续集成

### GitHub Actions 配置

```yaml
name: Test Suite
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements-test.txt
      - name: Run tests
        run: python tools/testing/test_runner_comprehensive.py --critical-only
```

## 📚 最佳实践

### 编写测试用例

1. **命名规范**: `test_功能描述_场景`
2. **结构清晰**: Arrange-Act-Assert 模式
3. **独立性**: 测试间不应相互依赖
4. **可重复**: 每次运行结果一致
5. **有意义**: 测试真实的业务场景

### 性能测试

1. **基准设定**: 建立性能基线
2. **渐进测试**: 逐步增加负载
3. **资源监控**: 监控CPU、内存、网络
4. **回归检测**: 防止性能退化

### 测试数据

1. **数据隔离**: 测试数据独立管理
2. **数据清理**: 测试后清理数据
3. **数据生成**: 使用工厂模式生成测试数据
4. **敏感数据**: 避免使用真实敏感数据

## 🤝 贡献指南

1. **添加新测试**: 遵循现有的目录结构和命名规范
2. **更新文档**: 同步更新相关文档
3. **性能影响**: 考虑新测试对整体执行时间的影响
4. **代码审查**: 提交前进行代码审查

---

如有问题或建议，请提交 Issue 或 Pull Request。
