# 服务发现功能指南

## 概述

VoiceHelper 现已实现完整的服务发现功能，支持自动发现、注册、健康检查和负载均衡。该功能使系统能够动态发现和管理第三方服务，提高系统的可扩展性和可靠性。

## 🚀 核心功能

### 1. 自动服务发现
- **OpenAPI/Swagger 规范解析**: 自动解析服务的 API 文档
- **健康检查端点发现**: 自动发现服务的健康检查接口
- **服务信息提取**: 提取服务名称、版本、描述等元信息
- **端点映射**: 自动映射服务的所有可用端点

### 2. 服务注册表
- **动态服务注册**: 支持手动和自动服务注册
- **服务分类管理**: 按类别组织服务（API、数据库、消息队列等）
- **服务状态跟踪**: 实时跟踪服务状态和健康状况
- **元数据管理**: 存储和管理服务的详细元数据

### 3. 健康检查系统
- **自动健康监控**: 定期检查服务健康状态
- **多种检查方式**: 支持 HTTP、TCP 等多种健康检查方式
- **健康状态分类**: 健康、不健康、未知、超时等状态
- **故障检测**: 快速检测服务故障并更新状态

### 4. 负载均衡器
- **多种策略**: 轮询、加权随机、最小延迟、健康优先、优先级等
- **熔断器模式**: 防止级联故障，提高系统稳定性
- **服务选择**: 智能选择最佳服务实例
- **故障转移**: 自动故障转移到健康的服务实例

## 📋 API 接口

### 服务发现 API

#### 1. 发现服务能力
```http
POST /api/v1/integrations/discover
Content-Type: application/json

{
  "url": "https://api.example.com",
  "headers": {
    "Authorization": "Bearer token"
  },
  "timeout": 30
}
```

**响应示例:**
```json
{
  "success": true,
  "url": "https://api.example.com",
  "result": {
    "service_url": "https://api.example.com",
    "service_name": "Example API",
    "version": "1.0.0",
    "description": "Example service description",
    "capabilities": ["openapi", "health"],
    "endpoints": [
      {
        "path": "/users",
        "method": "GET",
        "description": "Get users",
        "tags": ["users"]
      }
    ],
    "health_check": {
      "path": "/health",
      "method": "GET",
      "interval": "30s",
      "timeout": "10s"
    },
    "status": "discovered"
  }
}
```

#### 2. 发现并注册服务
```http
POST /api/v1/integrations/discover-and-register
Content-Type: application/json

{
  "url": "https://api.example.com",
  "service_id": "example-api"
}
```

#### 3. 获取发现缓存
```http
GET /api/v1/integrations/discovery/cache
```

#### 4. 清理发现缓存
```http
DELETE /api/v1/integrations/discovery/cache
```

### 服务注册表 API

#### 1. 列出注册的服务
```http
GET /api/v1/integrations/registry/services?category=api
```

#### 2. 列出健康的服务
```http
GET /api/v1/integrations/registry/services/healthy
```

#### 3. 检查服务健康状态
```http
GET /api/v1/integrations/registry/services/{service_id}/health
```

**响应示例:**
```json
{
  "service_id": "example-api",
  "health": "healthy",
  "checked_at": "2023-09-22T13:27:37Z"
}
```

#### 4. 获取服务统计信息
```http
GET /api/v1/integrations/registry/services/{service_id}/stats
```

**响应示例:**
```json
{
  "service_id": "example-api",
  "stats": {
    "request_count": 1250,
    "error_count": 15,
    "last_request_time": "2023-09-22T13:25:00Z",
    "average_latency": "150ms",
    "health_status": "healthy",
    "last_health_check": "2023-09-22T13:27:00Z",
    "uptime": "24h30m",
    "error_rate": 0.012
  }
}
```

#### 5. 记录服务请求
```http
POST /api/v1/integrations/registry/services/{service_id}/record
Content-Type: application/json

{
  "latency": "150ms",
  "success": true
}
```

## 🔧 配置选项

### 服务注册表配置
```go
config := &discovery.RegistryConfig{
    HealthCheckInterval: 30 * time.Second,  // 健康检查间隔
    HealthCheckTimeout:  10 * time.Second,  // 健康检查超时
    HealthCheckEnabled:  true,              // 启用健康检查
    AutoDiscovery:       true,              // 启用自动发现
    DiscoveryMethods:    []string{"openapi", "swagger", "health", "info"},
}
```

### 负载均衡器配置
```go
config := &discovery.LoadBalancerConfig{
    Strategy:            discovery.StrategyHealthyFirst,  // 负载均衡策略
    FailureThreshold:    5,                              // 失败阈值
    RecoveryTimeout:     30 * time.Second,               // 恢复超时
    EnableCircuitBreaker: true,                          // 启用熔断器
}
```

## 💡 使用示例

### 1. 基本服务发现
```go
package main

import (
    "context"
    "time"
    "chatbot/pkg/discovery"
)

func main() {
    // 创建服务发现实例
    serviceDiscovery := discovery.NewServiceDiscovery()
    
    // 发现服务
    request := &discovery.DiscoveryRequest{
        URL:              "https://api.example.com",
        Timeout:          30 * time.Second,
        FollowRedirects:  true,
        DiscoveryMethods: []string{"openapi", "health", "info"},
    }
    
    result, err := serviceDiscovery.DiscoverService(context.Background(), request)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("发现服务: %s v%s\n", result.ServiceName, result.Version)
    fmt.Printf("端点数量: %d\n", len(result.Endpoints))
}
```

### 2. 服务注册和管理
```go
package main

import (
    "context"
    "chatbot/pkg/discovery"
)

func main() {
    // 创建服务注册表
    registry := discovery.NewServiceRegistry(nil)
    
    // 注册服务
    service := &discovery.RegisteredService{
        ID:       "my-api",
        Name:     "My API Service",
        URL:      "https://my-api.example.com",
        Category: "api",
        Status:   discovery.StatusActive,
        Weight:   100,
        Priority: 1,
    }
    
    err := registry.RegisterService(context.Background(), service)
    if err != nil {
        panic(err)
    }
    
    // 列出所有服务
    services := registry.ListServices()
    fmt.Printf("注册的服务数量: %d\n", len(services))
    
    // 列出健康的服务
    healthyServices := registry.ListHealthyServices()
    fmt.Printf("健康的服务数量: %d\n", len(healthyServices))
}
```

### 3. 负载均衡
```go
package main

import (
    "chatbot/pkg/discovery"
)

func main() {
    // 创建服务注册表和负载均衡器
    registry := discovery.NewServiceRegistry(nil)
    loadBalancer := discovery.NewLoadBalancer(registry, nil)
    
    // 选择服务
    selection, err := loadBalancer.SelectService("api", nil)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("选择的服务: %s (%s)\n", selection.Service.Name, selection.Reason)
    
    // 选择多个服务（用于冗余）
    selections, err := loadBalancer.SelectMultipleServices("api", 3, nil)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("选择了 %d 个服务用于冗余\n", len(selections))
}
```

### 4. 熔断器使用
```go
package main

import (
    "chatbot/pkg/discovery"
)

func main() {
    registry := discovery.NewServiceRegistry(nil)
    loadBalancer := discovery.NewLoadBalancer(registry, nil)
    
    // 模拟服务调用
    serviceID := "my-api"
    
    // 记录成功请求
    loadBalancer.RecordSuccess(serviceID)
    
    // 记录失败请求
    loadBalancer.RecordFailure(serviceID)
    
    // 获取熔断器状态
    status := loadBalancer.GetCircuitBreakerStatus()
    fmt.Printf("熔断器状态: %v\n", status)
    
    // 重置熔断器
    loadBalancer.ResetCircuitBreaker(serviceID)
}
```

## 🎯 负载均衡策略

### 1. 健康优先 (HealthyFirst)
优先选择健康的服务实例，在健康服务中使用加权随机选择。

### 2. 轮询 (RoundRobin)
按顺序轮流选择服务实例。

### 3. 加权随机 (WeightedRandom)
根据服务权重进行随机选择，权重越高被选中的概率越大。

### 4. 最小延迟 (LeastLatency)
选择平均延迟最低的服务实例。

### 5. 优先级 (Priority)
根据服务优先级选择，优先级相同的服务中使用加权随机。

## 🔍 服务发现方法

### 1. OpenAPI 发现
- 路径: `/openapi.json`, `/openapi.yaml`, `/v3/api-docs`
- 提取: 服务信息、端点列表、参数定义

### 2. Swagger 发现
- 路径: `/swagger.json`, `/swagger.yaml`, `/v2/api-docs`
- 提取: 服务信息、端点列表

### 3. 健康检查发现
- 路径: `/health`, `/healthz`, `/health/check`
- 提取: 健康检查配置

### 4. 服务信息发现
- 路径: `/info`, `/version`, `/actuator/info`
- 提取: 服务名称、版本、描述

### 5. 根路径发现
- 路径: `/`
- 提取: 基本服务信息

## 📊 监控和统计

### 服务统计指标
- **请求计数**: 总请求数和错误请求数
- **延迟统计**: 平均延迟和延迟分布
- **健康状态**: 当前健康状态和最后检查时间
- **运行时间**: 服务注册后的运行时间
- **错误率**: 错误请求占总请求的比例

### 负载均衡统计
- **策略信息**: 当前使用的负载均衡策略
- **熔断器状态**: 各服务的熔断器状态
- **选择统计**: 各服务被选择的次数和原因

## 🛠️ 故障排除

### 常见问题

#### 1. 服务发现失败
- **检查网络连接**: 确保能够访问目标服务
- **检查认证**: 验证 API 密钥或认证信息
- **检查超时设置**: 增加超时时间
- **检查发现方法**: 尝试不同的发现方法

#### 2. 健康检查失败
- **检查健康检查端点**: 确认服务提供健康检查接口
- **检查健康检查路径**: 验证健康检查路径是否正确
- **检查响应格式**: 确认健康检查返回正确的状态码

#### 3. 负载均衡问题
- **检查服务权重**: 确认服务权重配置正确
- **检查服务状态**: 确认服务状态为活跃
- **检查熔断器**: 检查熔断器是否意外打开

### 调试模式
```go
import "github.com/sirupsen/logrus"

// 启用调试日志
logrus.SetLevel(logrus.DebugLevel)

// 创建服务发现实例
serviceDiscovery := discovery.NewServiceDiscovery()
```

## 🚀 最佳实践

### 1. 服务注册
- 使用有意义的服务 ID 和名称
- 设置合适的权重和优先级
- 提供详细的服务元数据
- 定期更新服务信息

### 2. 健康检查
- 实现专门的健康检查端点
- 返回详细的健康状态信息
- 设置合适的检查间隔
- 处理健康检查的依赖关系

### 3. 负载均衡
- 根据业务需求选择合适的策略
- 设置合理的熔断器阈值
- 监控服务性能指标
- 定期评估和调整配置

### 4. 监控和告警
- 监控服务发现成功率
- 监控健康检查状态
- 设置服务不可用告警
- 监控负载均衡效果

## 📈 性能优化

### 1. 缓存优化
- 启用发现结果缓存
- 设置合适的缓存 TTL
- 定期清理过期缓存

### 2. 并发优化
- 并行执行健康检查
- 使用连接池
- 设置合适的超时时间

### 3. 网络优化
- 使用 HTTP/2
- 启用连接复用
- 设置合适的重试策略

## 🔮 未来扩展

### 计划功能
- **服务网格集成**: 与 Istio、Linkerd 等服务网格集成
- **更多发现方法**: 支持 gRPC 反射、GraphQL 内省等
- **智能路由**: 基于请求内容的智能路由
- **A/B 测试**: 支持流量分割和 A/B 测试

### 扩展接口
- **自定义发现器**: 支持自定义服务发现逻辑
- **自定义健康检查**: 支持自定义健康检查方法
- **自定义负载均衡**: 支持自定义负载均衡策略

## 📄 总结

VoiceHelper 的服务发现功能提供了完整的服务管理解决方案，包括自动发现、注册、健康检查和负载均衡。该功能大大提高了系统的可扩展性、可靠性和可维护性，为构建分布式系统提供了强有力的支持。

通过合理配置和使用这些功能，可以实现：
- 🔄 **动态服务管理**: 自动发现和注册服务
- 🏥 **健康监控**: 实时监控服务健康状态  
- ⚖️ **智能负载均衡**: 根据多种策略分配请求
- 🛡️ **故障保护**: 熔断器防止级联故障
- 📊 **性能监控**: 详细的性能统计和监控

这些功能共同构成了一个强大而灵活的服务发现和管理平台。
