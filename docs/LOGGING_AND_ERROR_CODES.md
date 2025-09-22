# VoiceHelper 日志系统与错误码指南

## 📋 概述

VoiceHelper 采用统一的日志系统和错误码体系，提供结构化的日志记录和标准化的错误处理机制。本文档详细介绍了如何使用日志系统和错误码。

## 🎯 设计原则

### 日志系统设计原则
- **结构化日志**: 使用JSON格式，便于解析和分析
- **统一格式**: Go和Python服务使用相同的日志格式
- **丰富上下文**: 包含网络信息、性能指标、业务上下文
- **分级记录**: 支持多种日志级别和类型
- **可观测性**: 便于监控、告警和故障排查

### 错误码设计原则
- **统一编码**: 采用6位数字编码，便于识别和管理
- **分层设计**: 按服务、模块、错误类型分层
- **国际化**: 支持中英文错误信息
- **HTTP映射**: 错误码自动映射到HTTP状态码

## 🔢 错误码体系

### 错误码格式

错误码采用6位数字格式：`XYZABC`

- **X**: 服务类型
  - 1: Gateway (网关)
  - 2: Auth (认证)
  - 3: Chat (聊天)
  - 4: Voice (语音)
  - 5: RAG (检索增强生成)
  - 6: Storage (存储)
  - 7: Integration (集成)
  - 8: Monitor (监控)
  - 9: Common (通用)

- **Y**: 模块类型
  - 0: 通用
  - 1: API
  - 2: Service
  - 3: Database
  - 4: Cache
  - 5: Network
  - 6: File
  - 7: Config
  - 8: Security
  - 9: Performance

- **Z**: 错误类型
  - 0: 成功
  - 1: 客户端错误
  - 2: 服务端错误
  - 3: 网络错误
  - 4: 数据错误
  - 5: 权限错误
  - 6: 配置错误
  - 7: 性能错误
  - 8: 安全错误
  - 9: 未知错误

- **ABC**: 具体错误序号 (001-999)

### 常用错误码

#### 成功码
- `000000`: 操作成功

#### Gateway错误 (1xxxxx)
- `102001`: Gateway内部错误
- `111001`: 无效请求
- `111002`: 缺少参数
- `111005`: 请求频率超限
- `133001`: 网络错误

#### 认证错误 (2xxxxx)
- `211001`: 无效凭证
- `211002`: Token过期
- `211004`: 权限不足
- `281001`: 安全违规

#### 聊天错误 (3xxxxx)
- `311001`: 无效消息
- `311003`: 会话不存在
- `371001`: 响应超时

#### 语音错误 (4xxxxx)
- `411001`: 音频格式无效
- `411004`: 语音识别失败
- `411005`: 语音合成失败

#### RAG错误 (5xxxxx)
- `511001`: 无效查询
- `511003`: 索引失败
- `511004`: 检索失败
- `533001`: 向量数据库错误

#### 系统错误 (9xxxxx)
- `902001`: 系统内部错误
- `933001`: 网络超时
- `961002`: 配置无效

## 📝 日志系统

### 日志级别

- **DEBUG**: 调试信息，开发环境使用
- **INFO**: 一般信息，正常业务流程
- **WARNING**: 警告信息，需要关注但不影响功能
- **ERROR**: 错误信息，功能异常但服务可继续
- **CRITICAL**: 严重错误，服务不可用

### 日志类型

- **startup**: 启动日志
- **request**: 请求日志
- **response**: 响应日志
- **error**: 错误日志
- **debug**: 调试日志
- **performance**: 性能日志
- **security**: 安全日志
- **business**: 业务日志
- **system**: 系统日志

### 日志格式

```json
{
  "timestamp": "2024-12-21T10:30:45.123Z",
  "level": "info",
  "type": "request",
  "service": "voicehelper-backend",
  "module": "chat",
  "message": "GET /api/v1/chat/completions",
  "error_code": 511004,
  "network": {
    "local_ip": "192.168.1.100",
    "local_port": "8080",
    "remote_ip": "192.168.1.50",
    "remote_port": "54321",
    "url": "http://192.168.1.100:8080/api/v1/chat/completions",
    "method": "GET",
    "user_agent": "Mozilla/5.0...",
    "request_id": "req_123456789"
  },
  "context": {
    "user_id": "user_123",
    "session_id": "sess_456",
    "custom_field": "custom_value"
  },
  "duration_ms": 150.5,
  "request_size": 1024,
  "response_size": 2048,
  "status_code": 200,
  "stack_trace": "..."
}
```

## 🛠️ 使用指南

### Go服务使用

#### 1. 初始化日志器

```go
import (
    "github.com/voicehelper/common/logger"
    "github.com/voicehelper/common/errors"
)

func main() {
    // 初始化默认日志器
    logger.InitDefaultLogger("voicehelper-backend")
    log := logger.GetDefaultLogger()
    
    // 或创建带模块名的日志器
    log := logger.NewLogger("voicehelper-backend").WithModule("auth")
}
```

#### 2. 记录不同类型的日志

```go
// 启动日志
logger.Startup("服务启动", 
    logger.F("host", "0.0.0.0"),
    logger.F("port", 8080),
    logger.F("local_ip", "192.168.1.100"),
)

// 请求日志 (通过中间件自动记录)
router.Use(logger.GinLoggerMiddleware(log))

// 业务日志
logger.Business("用户登录成功", 
    logger.F("user_id", "user_123"),
    logger.F("login_method", "password"),
)

// 错误日志
logger.ErrorWithCode(errors.AuthTokenExpired, "Token验证失败",
    logger.F("token", "xxx..."),
    logger.F("user_id", "user_123"),
)

// 性能日志
start := time.Now()
// ... 执行操作 ...
logger.Performance("数据库查询", time.Since(start),
    logger.F("query", "SELECT * FROM users"),
    logger.F("rows", 100),
)

// 安全日志
logger.Security("可疑登录尝试",
    logger.F("ip", "192.168.1.200"),
    logger.F("attempts", 5),
)
```

#### 3. 错误处理

```go
// 抛出自定义错误
if err != nil {
    errorInfo := errors.GetErrorInfo(errors.AuthTokenExpired)
    c.JSON(errorInfo.HTTPStatus, gin.H{
        "error": errorInfo,
    })
    return
}

// 记录错误日志
logger.ErrorWithCode(errors.GatewayInternalError, "处理请求失败",
    logger.F("error", err.Error()),
    logger.F("request_id", requestID),
)
```

### Python服务使用

#### 1. 初始化日志器

```python
from common.logger import init_logger, get_logger
from common.errors import ErrorCode, VoiceHelperError

# 初始化日志器
init_logger("voicehelper-algo")
logger = get_logger("main")

# 或获取带模块名的日志器
logger = get_logger("voice_processing")
```

#### 2. 记录不同类型的日志

```python
# 启动日志
logger.startup("算法服务启动", context={
    "host": "0.0.0.0",
    "port": 8000,
    "local_ip": "192.168.1.100",
})

# 业务日志
logger.business("文档入库完成", context={
    "task_id": "task_123",
    "files_count": 5,
    "duration_ms": 1500,
})

# 错误日志
logger.error_with_code(ErrorCode.RAG_RETRIEVAL_FAILED, "检索失败", context={
    "query": "用户查询内容",
    "collection": "documents",
})

# 性能日志
import time
start_time = time.time()
# ... 执行操作 ...
duration_ms = (time.time() - start_time) * 1000
logger.performance("向量检索", duration_ms, context={
    "top_k": 10,
    "collection": "documents",
})

# 异常日志
try:
    # ... 可能出错的操作 ...
    pass
except Exception as e:
    logger.exception("处理请求异常", e, context={
        "request_id": "req_123",
        "user_id": "user_456",
    })
```

#### 3. FastAPI集成

```python
from fastapi import FastAPI, Request
from common.logger import LoggingMiddleware
from common.errors import VoiceHelperError, ErrorCode

app = FastAPI()

# 添加日志中间件
app.middleware("http")(LoggingMiddleware())

# 异常处理器
@app.exception_handler(VoiceHelperError)
async def voicehelper_exception_handler(request: Request, exc: VoiceHelperError):
    logger.error_with_code(exc.code, f"VoiceHelper错误: {exc.message}", context={
        "method": request.method,
        "url": str(request.url),
        "details": exc.details,
    })
    
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict()
    )

# API端点
@app.post("/query")
async def query_documents(request: QueryRequest, http_request: Request):
    try:
        # 验证请求
        if not request.messages:
            raise VoiceHelperError(ErrorCode.RAG_INVALID_QUERY, "没有提供查询消息")
        
        # 处理请求...
        
    except VoiceHelperError:
        raise
    except Exception as e:
        logger.exception("查询处理失败", e)
        raise VoiceHelperError(ErrorCode.RAG_RETRIEVAL_FAILED, f"查询失败: {str(e)}")
```

## 🔧 配置说明

### 环境变量

```bash
# 日志级别
LOG_LEVEL=info

# 服务名称
SERVICE_NAME=voicehelper-backend

# 网络配置
HOST=0.0.0.0
PORT=8080

# 日志文件配置 (可选)
LOG_FILE_PATH=/var/log/voicehelper/app.log
LOG_MAX_SIZE=100MB
LOG_MAX_FILES=10
LOG_MAX_AGE=30
```

### 日志级别配置

- **开发环境**: `LOG_LEVEL=debug`
- **测试环境**: `LOG_LEVEL=info`
- **生产环境**: `LOG_LEVEL=warning`

## 📊 监控与分析

### 日志聚合

推荐使用ELK Stack (Elasticsearch, Logstash, Kibana) 进行日志聚合和分析：

```yaml
# docker-compose.yml
version: '3'
services:
  elasticsearch:
    image: elasticsearch:7.14.0
    environment:
      - discovery.type=single-node
    ports:
      - "9200:9200"
  
  logstash:
    image: logstash:7.14.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
  
  kibana:
    image: kibana:7.14.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
```

### 告警规则

基于错误码和日志级别设置告警：

```yaml
# Prometheus告警规则
groups:
- name: voicehelper_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(voicehelper_errors_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "VoiceHelper错误率过高"
      description: "错误率: {{ $value }}"
  
  - alert: ServiceDown
    expr: up{job="voicehelper"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "VoiceHelper服务下线"
```

## 🎯 最佳实践

### 1. 日志记录原则

- **关键路径必记**: 启动、关闭、请求处理、错误处理
- **性能指标必记**: 响应时间、吞吐量、资源使用
- **安全事件必记**: 认证失败、权限检查、可疑行为
- **业务事件必记**: 用户操作、数据变更、状态转换

### 2. 错误处理原则

- **统一错误码**: 使用预定义的错误码，不要自定义
- **详细上下文**: 提供足够的上下文信息便于排查
- **用户友好**: 对外暴露的错误信息要用户友好
- **安全考虑**: 不要在错误信息中泄露敏感信息

### 3. 性能考虑

- **异步记录**: 使用异步方式记录日志，避免阻塞主流程
- **批量处理**: 对于高频日志，考虑批量写入
- **采样记录**: 对于调试日志，可以采用采样方式
- **存储优化**: 定期清理过期日志，控制存储成本

### 4. 开发建议

- **本地开发**: 使用DEBUG级别，便于调试
- **测试环境**: 使用INFO级别，记录关键流程
- **生产环境**: 使用WARNING级别，只记录重要信息
- **监控告警**: 基于ERROR和CRITICAL级别设置告警

## 🔍 故障排查

### 常见问题

1. **日志不输出**
   - 检查LOG_LEVEL环境变量
   - 确认日志器初始化正确
   - 验证日志格式配置

2. **错误码不正确**
   - 检查错误码定义
   - 确认错误码映射关系
   - 验证HTTP状态码

3. **性能影响**
   - 调整日志级别
   - 使用异步记录
   - 优化日志内容

### 调试技巧

```bash
# 查看实时日志
tail -f /var/log/voicehelper/app.log | jq .

# 过滤错误日志
grep '"level":"error"' /var/log/voicehelper/app.log | jq .

# 统计错误码
grep '"error_code"' /var/log/voicehelper/app.log | jq -r .error_code | sort | uniq -c

# 分析响应时间
grep '"type":"response"' /var/log/voicehelper/app.log | jq -r .duration_ms | awk '{sum+=$1; count++} END {print "平均响应时间:", sum/count, "ms"}'
```

## 📚 参考资源

- [Structured Logging Best Practices](https://www.elastic.co/guide/en/ecs/current/ecs-logging.html)
- [HTTP Status Codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)
- [JSON Logging Format](https://www.elastic.co/guide/en/ecs/current/ecs-field-reference.html)
- [Prometheus Monitoring](https://prometheus.io/docs/practices/naming/)

---

**最后更新**: 2024-12-21  
**维护者**: VoiceHelper Team  
**版本**: v1.0.0
