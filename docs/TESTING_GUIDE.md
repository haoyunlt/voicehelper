# VoiceHelper 测试指南

## 📋 测试体系概览

本指南整合了所有测试相关文档，提供从单元测试到性能测试的完整测试解决方案。

## 🎯 测试覆盖范围

### 1. 单元测试 (Unit Tests)
- **后端处理器测试** (`tests/unit/backend/test_handlers.py`)
  - API处理器功能验证
  - 错误处理机制测试
  - 数据验证逻辑测试

- **算法服务测试** (`tests/unit/algo/test_services.py`)
  - RAG检索服务测试
  - 语音处理服务测试
  - 文档入库服务测试

### 2. 集成测试 (Integration Tests)
- **API端点集成** (`tests/integration/test_api_endpoints.py`)
  - 后端与算法服务集成
  - 服务间通信验证
  - 数据流完整性测试

### 3. 端到端测试 (End-to-End Tests)
- **完整工作流测试** (`tests/e2e/test_complete_workflows.py`)
  - 文本对话完整流程
  - 文档入库与查询流程
  - 语音交互完整流程

### 4. 性能测试 (Performance Tests)
- **统一性能测试** (`tests/unified_performance_test.py`)
  - 系统资源监控
  - API响应时间测试
  - 并发处理能力测试
  - 内存使用测试

- **统一基准测试** (`tests/unified_benchmark_test.py`)
  - 建立性能基线
  - 版本间性能对比
  - 回归测试验证

## 🛠️ 测试工具集

### 快速开始工具
1. **Makefile** - 统一的命令入口
   ```bash
   make help           # 查看所有命令
   make test           # 运行所有测试
   make test-unit      # 运行单元测试
   make test-integration # 运行集成测试
   make test-e2e       # 运行端到端测试
   make test-performance # 运行性能测试
   ```

2. **统一测试运行器** (`tests/run_all_tests.py`)
   ```bash
   # 运行所有测试类型
   python tests/run_all_tests.py
   
   # 运行指定测试类型
   python tests/run_all_tests.py --tests unit integration performance
   
   # 详细输出模式
   python tests/run_all_tests.py --verbose
   ```

### 专项测试工具
1. **性能测试套件**
   ```bash
   # 快速性能测试
   python tests/unified_performance_test.py --test-type quick
   
   # 综合性能测试
   python tests/unified_performance_test.py --test-type comprehensive
   
   # 基准测试
   python tests/unified_benchmark_test.py
   ```

2. **模块测试** (`tests/module_test_runner.py`)
   ```bash
   # 运行模块测试
   python tests/module_test_runner.py
   ```

## 🏗️ 测试架构

### 测试金字塔
```
测试金字塔
    ┌─────────────────┐
    │   E2E Tests     │  ← 端到端测试 (少量)
    │   (UI/Workflow) │
    ├─────────────────┤
    │ Integration     │  ← 集成测试 (适量)
    │ Tests (API)     │
    ├─────────────────┤
    │   Unit Tests    │  ← 单元测试 (大量)
    │ (Components)    │
    └─────────────────┘
         ┌─────────────────┐
         │ Performance     │  ← 性能测试 (专项)
         │ Tests           │
         └─────────────────┘
```

### 测试覆盖范围

| 测试层级 | 覆盖范围 | 测试工具 | 执行频率 |
|---------|---------|----------|----------|
| **单元测试** | 函数、类、模块 | pytest | 每次提交 |
| **集成测试** | API接口、服务间调用 | pytest + requests | 每日构建 |
| **端到端测试** | 完整业务流程 | pytest + selenium | 发布前 |
| **性能测试** | 负载、压力、基准 | locust + 自定义脚本 | 定期执行 |

## 🛠️ 测试环境准备

### 1. 安装测试依赖

```bash
# 创建虚拟环境
python3 -m venv test_venv
source test_venv/bin/activate

# 安装测试依赖
pip install -r requirements-test.txt

# 安装额外依赖
pip install pytest pytest-asyncio pytest-cov
pip install requests aiohttp psutil
pip install locust playwright
```

### 2. 环境配置

```bash
# 测试环境变量
export TEST_ENV=testing
export BACKEND_URL=http://localhost:8080
export ALGO_URL=http://localhost:8000
export FRONTEND_URL=http://localhost:3000
```

### 3. 服务启动

```bash
# 启动测试服务
docker-compose up -d

# 等待服务就绪
./scripts/wait-for-services.sh

# 验证服务状态
curl http://localhost:8080/health
curl http://localhost:8000/health
```

## 🧪 测试执行指南

### 1. 单元测试

```bash
# 运行所有单元测试
pytest tests/unit/ -v

# 运行特定模块测试
pytest tests/unit/backend/ -v
pytest tests/unit/algo/ -v

# 生成覆盖率报告
pytest tests/unit/ --cov=backend --cov=algo --cov-report=html
```

### 2. 集成测试

```bash
# 运行集成测试
pytest tests/integration/ -v

# 运行API端点测试
pytest tests/integration/test_api_endpoints.py -v
```

### 3. 端到端测试

```bash
# 安装Playwright浏览器
npx playwright install

# 运行E2E测试
npx playwright test

# 运行特定测试
npx playwright test tests/e2e/tests/smoke.spec.ts
```

### 4. 性能测试

```bash
# 快速性能测试
python tests/unified_performance_test.py --test-type quick

# 综合性能测试
python tests/unified_performance_test.py --test-type comprehensive

# 基准测试
python tests/unified_benchmark_test.py

# 优化验证测试
python tests/performance/optimization_validator.py
```

## 📊 测试报告分析

### 1. 测试结果解读

#### 单元测试报告
```bash
# 查看测试结果
pytest tests/unit/ --html=reports/unit_test_report.html --self-contained-html

# 查看覆盖率报告
pytest tests/unit/ --cov=backend --cov=algo --cov-report=html
open htmlcov/index.html
```

#### 性能测试报告
```bash
# 查看性能测试结果
python tests/unified_performance_test.py --output reports/performance_report.json

# 查看基准测试结果
python tests/unified_benchmark_test.py --output reports/benchmark_report.json
```

### 2. 测试指标解读

#### 性能指标
- **响应时间**: <100ms (优秀), <200ms (良好), <500ms (可接受)
- **并发处理**: >100 req/s (优秀), >50 req/s (良好), >20 req/s (可接受)
- **内存使用**: <2GB (优秀), <4GB (良好), <8GB (可接受)
- **CPU使用**: <50% (优秀), <70% (良好), <90% (可接受)

#### 覆盖率指标
- **代码覆盖率**: >90% (优秀), >80% (良好), >70% (可接受)
- **分支覆盖率**: >85% (优秀), >75% (良好), >65% (可接受)
- **函数覆盖率**: >95% (优秀), >90% (良好), >85% (可接受)

## 🔧 持续集成配置

### 1. GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements-test.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run unit tests
      run: pytest tests/unit/ --cov=backend --cov=algo --cov-report=xml
    
    - name: Run integration tests
      run: pytest tests/integration/
    
    - name: Run performance tests
      run: python tests/unified_performance_test.py --test-type quick
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. 测试环境配置

```yaml
# docker-compose.test.yml
version: '3.8'
services:
  test-db:
    image: postgres:15
    environment:
      POSTGRES_DB: chatbot_test
      POSTGRES_USER: chatbot
      POSTGRES_PASSWORD: chatbot123
    ports:
      - "5433:5432"
  
  test-redis:
    image: redis:7
    ports:
      - "6380:6379"
  
    ports:
      - "19531:19530"
```

## 🎯 最佳实践

### 1. 测试编写原则

#### 单元测试原则
- **单一职责**: 每个测试只验证一个功能
- **独立性**: 测试之间不相互依赖
- **可重复**: 测试结果可重复
- **快速执行**: 单元测试应该快速完成

#### 集成测试原则
- **真实环境**: 使用真实的服务环境
- **数据隔离**: 使用独立的测试数据
- **清理机制**: 测试后清理数据
- **错误处理**: 验证错误处理机制

#### 性能测试原则
- **基准建立**: 建立性能基准
- **回归检测**: 检测性能回归
- **负载模拟**: 模拟真实负载
- **资源监控**: 监控系统资源

### 2. 测试数据管理

```python
# 测试数据工厂
class TestDataFactory:
    @staticmethod
    def create_user():
        return {
            "id": "test_user_123",
            "name": "Test User",
            "email": "test@example.com"
        }
    
    @staticmethod
    def create_message():
        return {
            "id": "test_message_123",
            "content": "Test message",
            "user_id": "test_user_123",
            "created_at": "2025-01-01T00:00:00Z"
        }
```

### 3. 测试环境隔离

```python
# 测试环境隔离
import pytest
import os

@pytest.fixture(scope="session")
def test_environment():
    # 设置测试环境变量
    os.environ["TEST_ENV"] = "testing"
    os.environ["DATABASE_URL"] = "postgres://test:test@localhost:5433/test_db"
    os.environ["REDIS_URL"] = "redis://localhost:6380"
    
    yield
    
    # 清理测试环境
    os.environ.pop("TEST_ENV", None)
```

## 🚨 故障排除

### 1. 常见测试问题

#### 测试环境问题
```bash
# 检查服务状态
docker-compose ps

# 检查端口占用
netstat -tulpn | grep :8080

# 检查日志
docker-compose logs backend
```

#### 测试执行问题
```bash
# 清理测试缓存
pytest --cache-clear

# 重新安装依赖
pip install -r requirements-test.txt --force-reinstall

# 检查Python环境
python --version
pip list
```

### 2. 性能测试问题

#### 性能测试失败
```bash
# 检查系统资源
top
free -h
df -h

# 检查服务性能
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/health

# 运行性能诊断
python tests/performance/optimization_validator.py
```

## 📚 相关文档

- [统一部署指南](UNIFIED_DEPLOYMENT_GUIDE.md)
- [环境配置指南](UNIFIED_ENVIRONMENT_GUIDE.md)
- [故障排除指南](TROUBLESHOOTING_GUIDE.md)
- [性能优化指南](BEST_PRACTICES.md#性能优化最佳实践)

---

**测试指南完成！** 🎉

如有问题，请参考 [故障排除指南](TROUBLESHOOTING_GUIDE.md) 或提交 Issue。
