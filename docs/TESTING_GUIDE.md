# VoiceHelper 测试指南

本文档详细介绍VoiceHelper项目的测试策略、测试用例设计和测试执行方法。

## 📋 目录

- [测试架构概览](#测试架构概览)
- [测试环境准备](#测试环境准备)
- [测试类型说明](#测试类型说明)
- [测试执行指南](#测试执行指南)
- [性能测试详解](#性能测试详解)
- [测试报告分析](#测试报告分析)
- [持续集成配置](#持续集成配置)
- [最佳实践](#最佳实践)

## 🏗️ 测试架构概览

VoiceHelper采用多层次测试策略，确保系统的可靠性和性能：

```text
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
```text

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
# 安装测试依赖
pip install -r requirements-test.txt

# 或者使用脚本自动检查和安装
./scripts/run_tests.sh --check
```text

### 2. 环境变量配置

创建测试环境配置文件 `.env.test`：

```bash
# 服务地址配置
TEST_BACKEND_URL=http://localhost:8080
TEST_ALGO_URL=http://localhost:8000
TEST_FRONTEND_URL=http://localhost:3000
TEST_WS_URL=ws://localhost:8080

# 测试超时配置
TEST_TIMEOUT=30
TEST_API_RESPONSE_TIME=2.0
TEST_WS_CONNECT_TIME=5.0
TEST_SUCCESS_RATE=95.0

# 测试数据目录
TEST_DATA_DIR=tests/data

# 数据库配置（测试用）
TEST_DATABASE_URL=postgresql://test:test@localhost:5432/voicehelper_test
TEST_REDIS_URL=redis://localhost:6379/1

# 日志级别
LOG_LEVEL=INFO
TESTING=true
```text

### 3. 启动测试服务

```bash
# 启动后端服务
cd backend && go run cmd/server/main.go &

# 启动算法服务
cd algo && python app/main.py &

# 启动前端服务（可选，用于E2E测试）
cd frontend && npm run dev &
```text

## 📝 测试类型说明

### 1. 单元测试 (Unit Tests)

**目标**：测试单个函数、类或模块的功能正确性

**位置**：`tests/unit/`

**特点**：
- 快速执行（< 1秒）
- 无外部依赖
- 高覆盖率（目标 > 80%）

**示例**：

```python
# tests/unit/backend/test_handlers.py
def test_health_check_success():
    """测试健康检查成功"""
    services = Mock()
    handlers = Handlers(services)
    
    mock_context = Mock()
    handlers.HealthCheck(mock_context)
    
    mock_context.JSON.assert_called_once()
    call_args = mock_context.JSON.call_args
    assert call_args[0][0] == 200
    assert call_args[0][1]["status"] == "ok"
```text

### 2. 集成测试 (Integration Tests)

**目标**：测试服务间的接口和数据流

**位置**：`tests/integration/`

**特点**：
- 需要真实服务运行
- 测试API接口
- 验证数据一致性

**示例**：

```python
# tests/integration/test_api_endpoints.py
def test_chat_completion_api(self, api_base_url, test_auth_token):
    """测试聊天完成API"""
    headers = {"Authorization": f"Bearer {test_auth_token}"}
    chat_data = {
        "messages": [{"role": "user", "content": "测试消息"}],
        "temperature": 0.7
    }
    
    response = requests.post(
        f"{api_base_url}/api/v1/chat/completions",
        headers=headers,
        json=chat_data,
        timeout=30
    )
    
    assert response.status_code == 200
```text

### 3. 端到端测试 (E2E Tests)

**目标**：测试完整的用户业务流程

**位置**：`tests/e2e/`

**特点**：
- 模拟真实用户操作
- 跨多个服务
- 包含UI交互

**示例**：

```python
# tests/e2e/test_complete_workflows.py
def test_complete_user_journey(self, service_config):
    """测试完整用户旅程"""
    # 1. 用户注册登录
    token = self.test_user_registration_and_login_flow(service_config, test_user_data)
    
    # 2. 文档上传
    self.test_document_management_workflow(service_config)
    
    # 3. 智能问答
    qa_results = self.test_intelligent_qa_workflow(service_config)
    
    # 验证整体流程
    assert len(qa_results) > 0
    assert all(r["has_references"] for r in qa_results)
```text

## 🚀 测试执行指南

### 快速开始（推荐新用户）

```bash
# 方式一：使用Makefile（最简单）
make help                    # 查看所有可用命令
make test-quick             # 快速测试验证
make test                   # 运行所有测试
make test-unit              # 只运行单元测试

# 方式二：快速测试脚本
./scripts/quick_test.sh     # 快速验证测试框架和核心功能

# 方式三：测试演示
python scripts/demo_tests.py  # 交互式测试演示和环境检查
```text

### 使用完整测试脚本

```bash
# 显示帮助信息
./scripts/run_tests.sh --help

# 检查环境和依赖
./scripts/run_tests.sh --check

# 运行所有测试
./scripts/run_tests.sh --all

# 运行特定类型测试
./scripts/run_tests.sh --unit              # 单元测试
./scripts/run_tests.sh --integration       # 集成测试
./scripts/run_tests.sh --e2e               # 端到端测试

# 运行性能测试
./scripts/run_tests.sh --performance benchmark  # 基准测试
./scripts/run_tests.sh --performance load       # 负载测试
./scripts/run_tests.sh --performance stress     # 压力测试

# 生成测试报告
./scripts/run_tests.sh --report
```text

### 使用pytest直接执行

```bash
# 运行所有测试
pytest

# 运行特定目录的测试
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/

# 运行特定测试文件
pytest tests/unit/backend/test_handlers.py

# 运行特定测试函数
pytest tests/unit/backend/test_handlers.py::TestAuthMiddleware::test_valid_jwt_token

# 使用标记过滤测试
pytest -m "unit"           # 只运行单元测试
pytest -m "not slow"       # 跳过慢速测试
pytest -m "api and not external"  # 运行API测试但跳过外部依赖

# 并行执行测试
pytest -n auto            # 自动检测CPU核心数
pytest -n 4               # 使用4个进程

# 生成覆盖率报告
pytest --cov=backend --cov=algo --cov-report=html

# 详细输出
pytest -v -s              # 详细输出 + 不捕获stdout

# 只运行失败的测试
pytest --lf               # last failed
pytest --ff               # failed first
```text

## ⚡ 性能测试详解

### 1. 基准测试 (Benchmark Tests)

**目的**：建立性能基线，用于回归检测

```bash
# 运行基准测试
python scripts/performance/benchmark_test.py

# 选择测试模式
# 1. 快速基准测试 (推荐)
# 2. 完整基准测试套件  
# 3. 自定义测试
```text

**基准指标**：
- 健康检查：< 5ms
- 聊天完成：< 2s
- 文档查询：< 1s
- 并发处理：> 100 QPS

### 2. 负载测试 (Load Tests)

**目的**：验证系统在预期负载下的性能

```bash
# 使用Locust进行负载测试
locust -f scripts/performance/load_test.py \
       --host http://localhost:8080 \
       --users 50 \
       --spawn-rate 5 \
       --run-time 5m \
       --headless

# 或使用脚本
./scripts/run_tests.sh --performance load
```text

**负载配置**：
- 并发用户：50
- 启动速率：5用户/秒
- 持续时间：5分钟
- 成功率：> 95%

### 3. 压力测试 (Stress Tests)

**目的**：找出系统的性能极限和崩溃点

```bash
# 运行压力测试
python scripts/performance/stress_test.py

# 测试模式选择：
# 1. 后端压力测试
# 2. 算法服务压力测试  
# 3. 递增压力测试
```text

**压力测试场景**：
- 递增负载：10 → 1000用户
- 极限并发：找出最大QPS
- 资源监控：CPU、内存、网络

### 性能测试报告

测试完成后会生成以下报告：

```text
reports/
├── benchmark_20241221_143022.json     # 基准测试结果
├── load_test_report_20241221_143500.csv   # 负载测试详细数据
├── stress_test_summary_20241221_144000.json  # 压力测试摘要
└── performance_comparison.html        # 性能对比报告
```text

## 📊 测试报告分析

### 覆盖率报告

```bash
# 生成覆盖率报告
pytest --cov=backend --cov=algo --cov-report=html --cov-report=term

# 查看HTML报告
open reports/coverage_html/index.html
```text

**覆盖率目标**：
- 整体覆盖率：> 80%
- 核心模块：> 90%
- 关键路径：100%

### 性能报告分析

**关键指标**：

| 指标 | 目标值 | 警告阈值 | 说明 |
|------|--------|----------|------|
| 平均响应时间 | < 2s | > 3s | 用户体验关键指标 |
| P95响应时间 | < 5s | > 8s | 95%请求的响应时间 |
| 错误率 | < 1% | > 5% | 系统稳定性指标 |
| 吞吐量 | > 100 QPS | < 50 QPS | 系统处理能力 |
| CPU使用率 | < 70% | > 90% | 资源使用效率 |
| 内存使用率 | < 80% | > 95% | 内存泄漏检测 |

### 测试趋势分析

```python
# 性能趋势分析脚本示例
import pandas as pd
import matplotlib.pyplot as plt

# 读取历史测试数据
df = pd.read_csv('reports/performance_history.csv')

# 绘制响应时间趋势
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['avg_response_time'], label='平均响应时间')
plt.plot(df['date'], df['p95_response_time'], label='P95响应时间')
plt.xlabel('日期')
plt.ylabel('响应时间 (秒)')
plt.title('VoiceHelper 性能趋势')
plt.legend()
plt.savefig('reports/performance_trend.png')
```text

## 🔄 持续集成配置

### GitHub Actions 配置

```yaml
# .github/workflows/test.yml
name: VoiceHelper Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
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
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ --cov=backend --cov=algo --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Start services
      run: |
        # 启动后端和算法服务
        docker-compose -f docker-compose.test.yml up -d
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v

  performance-tests:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run benchmark tests
      run: |
        python scripts/performance/benchmark_test.py
    
    - name: Archive performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-reports
        path: reports/
```text

### 测试质量门禁

```yaml
# 质量门禁配置
quality_gates:
  unit_tests:
    coverage_threshold: 80
    max_duration: 300  # 5分钟
    
  integration_tests:
    success_rate: 95
    max_duration: 1800  # 30分钟
    
  performance_tests:
    avg_response_time: 2.0  # 2秒
    error_rate: 1.0  # 1%
    throughput: 100  # 100 QPS
```text

## 🎯 最佳实践

### 1. 测试设计原则

**FIRST原则**：
- **Fast**：测试应该快速执行
- **Independent**：测试之间相互独立
- **Repeatable**：测试结果可重复
- **Self-Validating**：测试有明确的通过/失败结果
- **Timely**：测试应该及时编写

### 2. 测试命名规范

```python
# 好的测试命名
def test_user_login_with_valid_credentials_returns_token():
    """测试：使用有效凭据登录应返回令牌"""
    pass

def test_chat_completion_with_empty_message_raises_validation_error():
    """测试：空消息的聊天完成应抛出验证错误"""
    pass

# 避免的命名
def test_login():  # 太模糊
def test_case_1():  # 无意义
```text

### 3. 测试数据管理

```python
# 使用fixture管理测试数据
@pytest.fixture
def sample_user():
    return {
        "username": "test_user",
        "email": "test@example.com",
        "password": "secure_password"
    }

# 使用工厂模式生成测试数据
class UserFactory:
    @staticmethod
    def create_user(**kwargs):
        defaults = {
            "username": f"user_{random.randint(1000, 9999)}",
            "email": f"test_{random.randint(1000, 9999)}@example.com",
            "password": "test_password"
        }
        defaults.update(kwargs)
        return defaults
```text

### 4. 异步测试最佳实践

```python
# 正确的异步测试写法
@pytest.mark.asyncio
async def test_async_chat_completion():
    """测试异步聊天完成"""
    async with aiohttp.ClientSession() as session:
        response = await session.post("/api/chat", json={"message": "test"})
        assert response.status == 200

# 使用超时控制
@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_long_running_operation():
    """测试长时间运行的操作"""
    result = await long_running_function()
    assert result is not None
```text

### 5. 错误处理测试

```python
def test_api_handles_invalid_input_gracefully():
    """测试API优雅处理无效输入"""
    invalid_data = {"invalid": "data"}
    
    response = requests.post("/api/chat", json=invalid_data)
    
    assert response.status_code == 400
    assert "error" in response.json()
    assert "validation" in response.json()["error"].lower()
```text

### 6. 性能测试最佳实践

```python
def test_api_response_time_under_threshold():
    """测试API响应时间在阈值内"""
    start_time = time.time()
    
    response = requests.get("/api/health")
    
    response_time = time.time() - start_time
    assert response_time < 0.1  # 100ms阈值
    assert response.status_code == 200
```text

## 🔧 故障排除

### 常见问题及解决方案

#### 1. 测试环境连接失败

```bash
# 检查服务状态
curl http://localhost:8080/health
curl http://localhost:8000/health

# 检查端口占用
netstat -tulpn | grep :8080
netstat -tulpn | grep :8000
```text

#### 2. 测试依赖安装失败

```bash
# 升级pip
pip install --upgrade pip

# 清理缓存重新安装
pip cache purge
pip install -r requirements-test.txt --no-cache-dir
```text

#### 3. 异步测试超时

```python
# 增加超时时间
@pytest.mark.timeout(60)
async def test_slow_operation():
    pass

# 或在pytest.ini中配置
# timeout = 60
```text

#### 4. 覆盖率统计不准确

```bash
# 清理之前的覆盖率数据
coverage erase

# 重新运行测试
pytest --cov=backend --cov=algo --cov-report=html
```text

## 🛠️ 测试工具说明

### 新增测试工具

#### 1. 测试框架验证

```bash
# 验证pytest和相关工具是否正常工作
python -m pytest tests/test_framework_validation.py -v
```text

#### 2. 快速测试脚本

```bash
# 快速验证测试环境和核心功能
./scripts/quick_test.sh
```text

#### 3. 测试演示脚本

```bash
# 交互式测试演示，展示各种测试类型
python scripts/demo_tests.py
```text

#### 4. Makefile命令

```bash
# 查看所有可用命令
make help

# 开发环境设置
make dev-setup

# 代码质量检查
make lint format coverage

# 服务管理
make start-backend start-algo start-frontend
make stop-services
```text

### 测试文件结构

```text
tests/
├── conftest.py                    # pytest配置和fixtures
├── test_framework_validation.py   # 测试框架验证
├── unit/                          # 单元测试
│   ├── backend/
│   │   └── test_handlers.py
│   └── algo/
│       └── test_services.py
├── integration/                   # 集成测试
│   └── test_api_endpoints.py
└── e2e/                          # 端到端测试
    └── test_complete_workflows.py

scripts/
├── run_tests.sh                  # 完整测试脚本
├── quick_test.sh                 # 快速测试脚本
├── demo_tests.py                 # 测试演示脚本
└── performance/                  # 性能测试脚本
    ├── load_test.py
    ├── stress_test.py
    └── benchmark_test.py
```text

## 📚 参考资源

- [pytest官方文档](https://docs.pytest.org/)
- [Locust性能测试指南](https://docs.locust.io/)
- [Selenium WebDriver文档](https://selenium-python.readthedocs.io/)
- [aiohttp测试指南](https://docs.aiohttp.org/en/stable/testing.html)
- [测试驱动开发最佳实践](https://testdriven.io/)
- [Make命令教程](https://www.gnu.org/software/make/manual/)

---

**最后更新**: 2024-12-21  
**维护者**: VoiceHelper Team  
**版本**: v2.0.0
