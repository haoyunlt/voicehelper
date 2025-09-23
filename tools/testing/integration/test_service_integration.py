"""
服务间集成测试用例
测试覆盖：后端-算法服务集成、数据库连接、消息队列、缓存服务、外部API集成
"""

import pytest
import asyncio
import json
import time
import aiohttp
import aioredis
from unittest.mock import Mock, patch, AsyncMock
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
from concurrent.futures import ThreadPoolExecutor


class TestBackendAlgoServiceIntegration:
    """后端-算法服务集成测试"""
    
    @pytest.fixture
    def mock_algo_service(self):
        """模拟算法服务"""
        class MockAlgoService:
            def __init__(self):
                self.base_url = "http://localhost:8000"
                self.is_healthy = True
                self.response_delay = 0.1
            
            async def health_check(self):
                """健康检查"""
                if not self.is_healthy:
                    raise aiohttp.ClientConnectorError(None, OSError("Service unavailable"))
                
                await asyncio.sleep(self.response_delay)
                return {"status": "healthy", "version": "1.0.0", "timestamp": time.time()}
            
            async def chat_completion(self, messages, stream=False):
                """对话完成"""
                await asyncio.sleep(self.response_delay)
                
                if stream:
                    async def stream_response():
                        chunks = [
                            '{"type": "refs", "refs": [{"source": "test.pdf", "content": "测试内容"}]}',
                            '{"type": "message", "content": "这是"}',
                            '{"type": "message", "content": "测试回答"}',
                            '{"type": "end"}'
                        ]
                        for chunk in chunks:
                            yield chunk
                            await asyncio.sleep(0.05)
                    
                    return stream_response()
                else:
                    return {
                        "response": "这是测试回答",
                        "references": [{"source": "test.pdf", "content": "测试内容"}],
                        "model": "test-model",
                        "usage": {"prompt_tokens": 10, "completion_tokens": 20}
                    }
            
            async def voice_process(self, audio_data, language="zh-CN"):
                """语音处理"""
                await asyncio.sleep(self.response_delay * 2)  # 语音处理较慢
                
                return {
                    "transcript": "这是语音转文字结果",
                    "confidence": 0.95,
                    "emotion": {
                        "primary": "neutral",
                        "confidence": 0.8,
                        "all_emotions": {"neutral": 0.8, "happy": 0.2}
                    },
                    "audio_features": {
                        "duration": 2.5,
                        "sample_rate": 16000,
                        "quality_score": 85
                    }
                }
            
            async def document_ingest(self, file_data, metadata=None):
                """文档入库"""
                await asyncio.sleep(self.response_delay * 3)  # 文档处理较慢
                
                return {
                    "document_id": "doc_123",
                    "chunks_created": 15,
                    "vectors_indexed": 15,
                    "processing_time": 2.5,
                    "status": "completed"
                }
        
        return MockAlgoService()
    
    @pytest.mark.asyncio
    async def test_service_discovery_and_health_check(self, mock_algo_service):
        """测试服务发现和健康检查"""
        class ServiceRegistry:
            def __init__(self):
                self.services = {}
            
            def register_service(self, name, url, health_endpoint="/health"):
                """注册服务"""
                self.services[name] = {
                    "url": url,
                    "health_endpoint": health_endpoint,
                    "status": "unknown",
                    "last_check": None
                }
            
            async def check_service_health(self, name):
                """检查服务健康状态"""
                if name not in self.services:
                    return {"healthy": False, "error": "Service not registered"}
                
                service = self.services[name]
                try:
                    # 模拟健康检查调用
                    if name == "algo_service":
                        health_data = await mock_algo_service.health_check()
                        service["status"] = "healthy"
                        service["last_check"] = time.time()
                        return {"healthy": True, "data": health_data}
                    else:
                        raise Exception("Unknown service")
                
                except Exception as e:
                    service["status"] = "unhealthy"
                    service["last_check"] = time.time()
                    return {"healthy": False, "error": str(e)}
            
            async def get_healthy_services(self):
                """获取健康的服务列表"""
                healthy_services = []
                for name, service in self.services.items():
                    health_result = await self.check_service_health(name)
                    if health_result["healthy"]:
                        healthy_services.append({
                            "name": name,
                            "url": service["url"],
                            "status": service["status"],
                            "last_check": service["last_check"]
                        })
                return healthy_services
        
        # 测试服务注册和健康检查
        registry = ServiceRegistry()
        registry.register_service("algo_service", "http://localhost:8000")
        
        # 检查健康服务
        health_result = await registry.check_service_health("algo_service")
        assert health_result["healthy"]
        assert "version" in health_result["data"]
        
        # 获取健康服务列表
        healthy_services = await registry.get_healthy_services()
        assert len(healthy_services) == 1
        assert healthy_services[0]["name"] == "algo_service"
        
        # 测试服务不可用情况
        mock_algo_service.is_healthy = False
        health_result = await registry.check_service_health("algo_service")
        assert not health_result["healthy"]
        assert "error" in health_result
    
    @pytest.mark.asyncio
    async def test_request_routing_and_load_balancing(self, mock_algo_service):
        """测试请求路由和负载均衡"""
        class LoadBalancer:
            def __init__(self):
                self.services = []
                self.current_index = 0
                self.request_counts = {}
            
            def add_service(self, service_instance):
                """添加服务实例"""
                self.services.append(service_instance)
                service_id = id(service_instance)
                self.request_counts[service_id] = 0
            
            def get_next_service(self, strategy="round_robin"):
                """获取下一个服务实例"""
                if not self.services:
                    return None
                
                if strategy == "round_robin":
                    service = self.services[self.current_index]
                    self.current_index = (self.current_index + 1) % len(self.services)
                    return service
                elif strategy == "least_connections":
                    # 选择请求数最少的服务
                    service = min(self.services, key=lambda s: self.request_counts[id(s)])
                    return service
                else:
                    return self.services[0]
            
            async def route_request(self, request_type, *args, **kwargs):
                """路由请求"""
                service = self.get_next_service()
                if not service:
                    return {"success": False, "error": "No available services"}
                
                service_id = id(service)
                self.request_counts[service_id] += 1
                
                try:
                    if request_type == "chat":
                        result = await service.chat_completion(*args, **kwargs)
                    elif request_type == "voice":
                        result = await service.voice_process(*args, **kwargs)
                    elif request_type == "ingest":
                        result = await service.document_ingest(*args, **kwargs)
                    else:
                        raise ValueError(f"Unknown request type: {request_type}")
                    
                    return {"success": True, "data": result, "service_id": service_id}
                
                except Exception as e:
                    return {"success": False, "error": str(e), "service_id": service_id}
                finally:
                    self.request_counts[service_id] -= 1
        
        # 创建负载均衡器并添加服务实例
        load_balancer = LoadBalancer()
        
        # 创建多个服务实例
        service1 = mock_algo_service
        service2 = mock_algo_service  # 在实际应用中这会是不同的实例
        
        load_balancer.add_service(service1)
        load_balancer.add_service(service2)
        
        # 测试轮询负载均衡
        results = []
        for i in range(4):
            result = await load_balancer.route_request(
                "chat", 
                messages=[{"role": "user", "content": f"测试消息{i}"}]
            )
            results.append(result)
        
        # 验证请求被分发到不同服务
        assert all(result["success"] for result in results)
        service_ids = [result["service_id"] for result in results]
        assert len(set(service_ids)) > 1  # 至少使用了不同的服务实例
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, mock_algo_service):
        """测试熔断器集成"""
        class CircuitBreakerProxy:
            def __init__(self, service, failure_threshold=3, timeout=5):
                self.service = service
                self.failure_threshold = failure_threshold
                self.timeout = timeout
                self.failure_count = 0
                self.last_failure_time = None
                self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
            
            async def call_service(self, method_name, *args, **kwargs):
                """通过熔断器调用服务"""
                current_time = time.time()
                
                # 检查熔断器状态
                if self.state == "OPEN":
                    if current_time - self.last_failure_time < self.timeout:
                        return {
                            "success": False,
                            "error": "Circuit breaker is OPEN",
                            "circuit_state": self.state
                        }
                    else:
                        self.state = "HALF_OPEN"
                
                try:
                    # 调用服务方法
                    method = getattr(self.service, method_name)
                    result = await method(*args, **kwargs)
                    
                    # 成功调用，重置失败计数
                    if self.state == "HALF_OPEN":
                        self.state = "CLOSED"
                        self.failure_count = 0
                    
                    return {"success": True, "data": result, "circuit_state": self.state}
                
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = current_time
                    
                    # 检查是否需要打开熔断器
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                    
                    return {
                        "success": False,
                        "error": str(e),
                        "circuit_state": self.state,
                        "failure_count": self.failure_count
                    }
        
        # 创建熔断器代理
        circuit_breaker = CircuitBreakerProxy(mock_algo_service, failure_threshold=2, timeout=1)
        
        # 测试正常调用
        result = await circuit_breaker.call_service("health_check")
        assert result["success"]
        assert result["circuit_state"] == "CLOSED"
        
        # 模拟服务故障
        mock_algo_service.is_healthy = False
        
        # 连续失败调用触发熔断
        for i in range(3):
            result = await circuit_breaker.call_service("health_check")
            assert not result["success"]
        
        # 验证熔断器打开
        assert circuit_breaker.state == "OPEN"
        
        # 测试熔断器拒绝请求
        result = await circuit_breaker.call_service("health_check")
        assert not result["success"]
        assert "Circuit breaker is OPEN" in result["error"]
        
        # 等待超时后测试半开状态
        await asyncio.sleep(1.1)
        mock_algo_service.is_healthy = True
        
        result = await circuit_breaker.call_service("health_check")
        assert result["success"]
        assert circuit_breaker.state == "CLOSED"


class TestDatabaseIntegration:
    """数据库集成测试"""
    
    @pytest.fixture
    def mock_db_connection(self):
        """模拟数据库连接"""
        class MockDBConnection:
            def __init__(self):
                self.is_connected = True
                self.data_store = {}
                self.transaction_active = False
            
            async def execute(self, query, params=None):
                """执行SQL查询"""
                if not self.is_connected:
                    raise psycopg2.OperationalError("Connection lost")
                
                # 模拟查询执行
                await asyncio.sleep(0.01)  # 模拟数据库延迟
                
                if query.strip().upper().startswith("SELECT"):
                    # 模拟查询结果
                    if "users" in query.lower():
                        return [
                            {"id": 1, "username": "test_user", "email": "test@example.com"},
                            {"id": 2, "username": "admin", "email": "admin@example.com"}
                        ]
                    elif "conversations" in query.lower():
                        return [
                            {"id": 1, "user_id": 1, "title": "Test Conversation", "created_at": time.time()}
                        ]
                    else:
                        return []
                
                elif query.strip().upper().startswith("INSERT"):
                    # 模拟插入操作
                    return {"rowcount": 1, "lastrowid": 123}
                
                elif query.strip().upper().startswith("UPDATE"):
                    # 模拟更新操作
                    return {"rowcount": 1}
                
                elif query.strip().upper().startswith("DELETE"):
                    # 模拟删除操作
                    return {"rowcount": 1}
                
                else:
                    return {"rowcount": 0}
            
            async def begin_transaction(self):
                """开始事务"""
                if self.transaction_active:
                    raise Exception("Transaction already active")
                self.transaction_active = True
            
            async def commit_transaction(self):
                """提交事务"""
                if not self.transaction_active:
                    raise Exception("No active transaction")
                self.transaction_active = False
            
            async def rollback_transaction(self):
                """回滚事务"""
                if not self.transaction_active:
                    raise Exception("No active transaction")
                self.transaction_active = False
            
            async def close(self):
                """关闭连接"""
                self.is_connected = False
        
        return MockDBConnection()
    
    @pytest.mark.asyncio
    async def test_database_connection_pool(self, mock_db_connection):
        """测试数据库连接池"""
        class ConnectionPool:
            def __init__(self, max_connections=10):
                self.max_connections = max_connections
                self.available_connections = []
                self.active_connections = set()
                self.connection_count = 0
            
            async def get_connection(self):
                """获取数据库连接"""
                if self.available_connections:
                    conn = self.available_connections.pop()
                    self.active_connections.add(conn)
                    return conn
                
                if self.connection_count < self.max_connections:
                    # 创建新连接（在实际应用中这里会创建真实的数据库连接）
                    conn = mock_db_connection
                    self.connection_count += 1
                    self.active_connections.add(conn)
                    return conn
                
                # 连接池已满，等待可用连接
                raise Exception("Connection pool exhausted")
            
            async def return_connection(self, conn):
                """归还连接到池中"""
                if conn in self.active_connections:
                    self.active_connections.remove(conn)
                    if conn.is_connected:
                        self.available_connections.append(conn)
                    else:
                        self.connection_count -= 1
            
            async def close_all(self):
                """关闭所有连接"""
                for conn in list(self.active_connections) + self.available_connections:
                    await conn.close()
                self.active_connections.clear()
                self.available_connections.clear()
                self.connection_count = 0
        
        # 测试连接池
        pool = ConnectionPool(max_connections=3)
        
        # 获取连接
        conn1 = await pool.get_connection()
        assert conn1 is not None
        assert len(pool.active_connections) == 1
        
        # 执行查询
        result = await conn1.execute("SELECT * FROM users")
        assert len(result) == 2
        assert result[0]["username"] == "test_user"
        
        # 归还连接
        await pool.return_connection(conn1)
        assert len(pool.available_connections) == 1
        assert len(pool.active_connections) == 0
        
        # 重用连接
        conn2 = await pool.get_connection()
        assert conn2 is conn1  # 应该是同一个连接
        
        await pool.close_all()
    
    @pytest.mark.asyncio
    async def test_database_transaction_management(self, mock_db_connection):
        """测试数据库事务管理"""
        class TransactionManager:
            def __init__(self, connection):
                self.connection = connection
            
            async def execute_in_transaction(self, operations):
                """在事务中执行多个操作"""
                try:
                    await self.connection.begin_transaction()
                    
                    results = []
                    for operation in operations:
                        query = operation["query"]
                        params = operation.get("params")
                        result = await self.connection.execute(query, params)
                        results.append(result)
                    
                    await self.connection.commit_transaction()
                    return {"success": True, "results": results}
                
                except Exception as e:
                    await self.connection.rollback_transaction()
                    return {"success": False, "error": str(e)}
        
        # 测试事务管理
        tx_manager = TransactionManager(mock_db_connection)
        
        # 成功的事务
        operations = [
            {"query": "INSERT INTO users (username, email) VALUES (%s, %s)", 
             "params": ("new_user", "new@example.com")},
            {"query": "UPDATE users SET last_login = NOW() WHERE username = %s", 
             "params": ("new_user",)}
        ]
        
        result = await tx_manager.execute_in_transaction(operations)
        assert result["success"]
        assert len(result["results"]) == 2
        
        # 模拟事务失败
        mock_db_connection.is_connected = False
        
        result = await tx_manager.execute_in_transaction(operations)
        assert not result["success"]
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_database_migration_and_schema_validation(self):
        """测试数据库迁移和模式验证"""
        class MigrationManager:
            def __init__(self, connection):
                self.connection = connection
                self.migrations = []
            
            def add_migration(self, version, up_sql, down_sql):
                """添加迁移"""
                self.migrations.append({
                    "version": version,
                    "up_sql": up_sql,
                    "down_sql": down_sql
                })
            
            async def get_current_version(self):
                """获取当前数据库版本"""
                try:
                    result = await self.connection.execute(
                        "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
                    )
                    return result[0]["version"] if result else 0
                except:
                    return 0
            
            async def apply_migrations(self, target_version=None):
                """应用迁移"""
                current_version = await self.get_current_version()
                
                if target_version is None:
                    target_version = max(m["version"] for m in self.migrations) if self.migrations else 0
                
                applied_migrations = []
                
                for migration in sorted(self.migrations, key=lambda m: m["version"]):
                    if migration["version"] > current_version and migration["version"] <= target_version:
                        try:
                            await self.connection.begin_transaction()
                            
                            # 执行迁移SQL
                            await self.connection.execute(migration["up_sql"])
                            
                            # 记录迁移版本
                            await self.connection.execute(
                                "INSERT INTO schema_migrations (version, applied_at) VALUES (%s, %s)",
                                (migration["version"], time.time())
                            )
                            
                            await self.connection.commit_transaction()
                            applied_migrations.append(migration["version"])
                            
                        except Exception as e:
                            await self.connection.rollback_transaction()
                            return {
                                "success": False,
                                "error": f"Migration {migration['version']} failed: {str(e)}",
                                "applied_migrations": applied_migrations
                            }
                
                return {
                    "success": True,
                    "applied_migrations": applied_migrations,
                    "current_version": target_version
                }
        
        # 测试迁移管理
        migration_manager = MigrationManager(mock_db_connection)
        
        # 添加测试迁移
        migration_manager.add_migration(
            version=1,
            up_sql="CREATE TABLE test_table (id SERIAL PRIMARY KEY, name VARCHAR(100))",
            down_sql="DROP TABLE test_table"
        )
        
        migration_manager.add_migration(
            version=2,
            up_sql="ALTER TABLE test_table ADD COLUMN email VARCHAR(255)",
            down_sql="ALTER TABLE test_table DROP COLUMN email"
        )
        
        # 应用迁移
        result = await migration_manager.apply_migrations()
        assert result["success"]
        assert len(result["applied_migrations"]) == 2
        assert result["current_version"] == 2


class TestCacheIntegration:
    """缓存服务集成测试"""
    
    @pytest.fixture
    def mock_redis_client(self):
        """模拟Redis客户端"""
        class MockRedisClient:
            def __init__(self):
                self.data = {}
                self.expiry = {}
                self.is_connected = True
            
            async def get(self, key):
                """获取值"""
                if not self.is_connected:
                    raise aioredis.ConnectionError("Redis connection failed")
                
                # 检查过期
                if key in self.expiry and time.time() > self.expiry[key]:
                    del self.data[key]
                    del self.expiry[key]
                    return None
                
                return self.data.get(key)
            
            async def set(self, key, value, ex=None):
                """设置值"""
                if not self.is_connected:
                    raise aioredis.ConnectionError("Redis connection failed")
                
                self.data[key] = value
                if ex:
                    self.expiry[key] = time.time() + ex
                
                return True
            
            async def delete(self, key):
                """删除键"""
                if key in self.data:
                    del self.data[key]
                    if key in self.expiry:
                        del self.expiry[key]
                    return 1
                return 0
            
            async def exists(self, key):
                """检查键是否存在"""
                return key in self.data and (
                    key not in self.expiry or time.time() <= self.expiry[key]
                )
            
            async def flushall(self):
                """清空所有数据"""
                self.data.clear()
                self.expiry.clear()
                return True
        
        return MockRedisClient()
    
    @pytest.mark.asyncio
    async def test_cache_layer_integration(self, mock_redis_client):
        """测试缓存层集成"""
        class CacheService:
            def __init__(self, redis_client, default_ttl=3600):
                self.redis = redis_client
                self.default_ttl = default_ttl
            
            async def get_or_set(self, key, fetch_func, ttl=None):
                """获取缓存或设置新值"""
                if ttl is None:
                    ttl = self.default_ttl
                
                try:
                    # 尝试从缓存获取
                    cached_value = await self.redis.get(key)
                    if cached_value is not None:
                        return {
                            "success": True,
                            "data": json.loads(cached_value),
                            "from_cache": True
                        }
                    
                    # 缓存未命中，调用获取函数
                    fresh_data = await fetch_func()
                    
                    # 存储到缓存
                    await self.redis.set(key, json.dumps(fresh_data), ex=ttl)
                    
                    return {
                        "success": True,
                        "data": fresh_data,
                        "from_cache": False
                    }
                
                except Exception as e:
                    # 缓存失败，直接返回数据
                    try:
                        fresh_data = await fetch_func()
                        return {
                            "success": True,
                            "data": fresh_data,
                            "from_cache": False,
                            "cache_error": str(e)
                        }
                    except Exception as fetch_error:
                        return {
                            "success": False,
                            "error": str(fetch_error),
                            "cache_error": str(e)
                        }
            
            async def invalidate_pattern(self, pattern):
                """按模式失效缓存"""
                # 简化实现，实际应用中需要使用Redis的SCAN命令
                keys_to_delete = []
                for key in mock_redis_client.data.keys():
                    if pattern in key:
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    await self.redis.delete(key)
                
                return len(keys_to_delete)
        
        # 测试缓存服务
        cache_service = CacheService(mock_redis_client, default_ttl=60)
        
        # 模拟数据获取函数
        call_count = 0
        async def fetch_user_data():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # 模拟数据库查询延迟
            return {"id": 123, "name": "Test User", "email": "test@example.com"}
        
        # 第一次调用 - 应该从数据源获取
        result1 = await cache_service.get_or_set("user:123", fetch_user_data)
        assert result1["success"]
        assert not result1["from_cache"]
        assert call_count == 1
        
        # 第二次调用 - 应该从缓存获取
        result2 = await cache_service.get_or_set("user:123", fetch_user_data)
        assert result2["success"]
        assert result2["from_cache"]
        assert call_count == 1  # 没有再次调用fetch函数
        
        # 测试缓存失效
        invalidated_count = await cache_service.invalidate_pattern("user:")
        assert invalidated_count == 1
        
        # 失效后再次调用 - 应该重新获取
        result3 = await cache_service.get_or_set("user:123", fetch_user_data)
        assert result3["success"]
        assert not result3["from_cache"]
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_distributed_cache_consistency(self, mock_redis_client):
        """测试分布式缓存一致性"""
        class DistributedCacheManager:
            def __init__(self, redis_client):
                self.redis = redis_client
                self.local_cache = {}
                self.cache_version = {}
            
            async def set_with_version(self, key, value, ttl=3600):
                """设置带版本的缓存"""
                version = int(time.time() * 1000)  # 毫秒时间戳作为版本
                
                cache_data = {
                    "value": value,
                    "version": version,
                    "timestamp": time.time()
                }
                
                # 存储到Redis
                await self.redis.set(key, json.dumps(cache_data), ex=ttl)
                
                # 更新本地缓存
                self.local_cache[key] = cache_data
                self.cache_version[key] = version
                
                return version
            
            async def get_with_consistency_check(self, key):
                """获取缓存并检查一致性"""
                # 先检查本地缓存
                local_data = self.local_cache.get(key)
                
                # 从Redis获取最新版本
                redis_data_str = await self.redis.get(key)
                if not redis_data_str:
                    # Redis中没有数据，清理本地缓存
                    if key in self.local_cache:
                        del self.local_cache[key]
                        del self.cache_version[key]
                    return None
                
                redis_data = json.loads(redis_data_str)
                
                # 检查版本一致性
                if local_data and local_data["version"] == redis_data["version"]:
                    # 版本一致，返回本地缓存
                    return {
                        "value": local_data["value"],
                        "source": "local_cache",
                        "version": local_data["version"]
                    }
                else:
                    # 版本不一致或本地无缓存，更新本地缓存
                    self.local_cache[key] = redis_data
                    self.cache_version[key] = redis_data["version"]
                    
                    return {
                        "value": redis_data["value"],
                        "source": "redis",
                        "version": redis_data["version"]
                    }
            
            async def invalidate_everywhere(self, key):
                """在所有地方失效缓存"""
                # 从Redis删除
                await self.redis.delete(key)
                
                # 从本地缓存删除
                if key in self.local_cache:
                    del self.local_cache[key]
                    del self.cache_version[key]
                
                return True
        
        # 测试分布式缓存一致性
        cache_manager = DistributedCacheManager(mock_redis_client)
        
        # 设置缓存
        version1 = await cache_manager.set_with_version("config:app", {"theme": "dark", "lang": "zh"})
        
        # 第一次获取 - 应该从Redis获取
        result1 = await cache_manager.get_with_consistency_check("config:app")
        assert result1["source"] == "redis"
        assert result1["version"] == version1
        
        # 第二次获取 - 应该从本地缓存获取
        result2 = await cache_manager.get_with_consistency_check("config:app")
        assert result2["source"] == "local_cache"
        assert result2["version"] == version1
        
        # 模拟另一个实例更新了缓存
        await asyncio.sleep(0.001)  # 确保时间戳不同
        version2 = await cache_manager.set_with_version("config:app", {"theme": "light", "lang": "en"})
        
        # 再次获取 - 应该检测到版本不一致并更新
        result3 = await cache_manager.get_with_consistency_check("config:app")
        assert result3["source"] == "redis"
        assert result3["version"] == version2
        assert result3["value"]["theme"] == "light"


class TestMessageQueueIntegration:
    """消息队列集成测试"""
    
    @pytest.fixture
    def mock_message_queue(self):
        """模拟消息队列"""
        class MockMessageQueue:
            def __init__(self):
                self.queues = {}
                self.subscribers = {}
                self.message_id_counter = 0
            
            async def publish(self, topic, message, headers=None):
                """发布消息"""
                if topic not in self.queues:
                    self.queues[topic] = []
                
                self.message_id_counter += 1
                message_data = {
                    "id": self.message_id_counter,
                    "topic": topic,
                    "payload": message,
                    "headers": headers or {},
                    "timestamp": time.time(),
                    "processed": False
                }
                
                self.queues[topic].append(message_data)
                
                # 通知订阅者
                if topic in self.subscribers:
                    for callback in self.subscribers[topic]:
                        try:
                            await callback(message_data)
                        except Exception as e:
                            print(f"Subscriber error: {e}")
                
                return message_data["id"]
            
            async def subscribe(self, topic, callback):
                """订阅主题"""
                if topic not in self.subscribers:
                    self.subscribers[topic] = []
                
                self.subscribers[topic].append(callback)
            
            async def get_messages(self, topic, limit=10):
                """获取消息"""
                if topic not in self.queues:
                    return []
                
                messages = self.queues[topic][-limit:]
                return messages
            
            async def ack_message(self, topic, message_id):
                """确认消息处理"""
                if topic in self.queues:
                    for msg in self.queues[topic]:
                        if msg["id"] == message_id:
                            msg["processed"] = True
                            return True
                return False
        
        return MockMessageQueue()
    
    @pytest.mark.asyncio
    async def test_async_task_processing(self, mock_message_queue):
        """测试异步任务处理"""
        class TaskProcessor:
            def __init__(self, message_queue):
                self.mq = message_queue
                self.processed_tasks = []
                self.failed_tasks = []
            
            async def submit_task(self, task_type, task_data, priority="normal"):
                """提交任务"""
                task_message = {
                    "task_type": task_type,
                    "task_data": task_data,
                    "priority": priority,
                    "submitted_at": time.time(),
                    "retry_count": 0
                }
                
                topic = f"tasks.{priority}"
                message_id = await self.mq.publish(topic, task_message)
                
                return {"task_id": message_id, "status": "submitted"}
            
            async def process_task(self, message):
                """处理任务"""
                task = message["payload"]
                task_type = task["task_type"]
                task_data = task["task_data"]
                
                try:
                    if task_type == "document_processing":
                        # 模拟文档处理
                        await asyncio.sleep(0.1)
                        result = {
                            "processed_chunks": len(task_data.get("content", "")),
                            "processing_time": 0.1
                        }
                    elif task_type == "voice_synthesis":
                        # 模拟语音合成
                        await asyncio.sleep(0.05)
                        result = {
                            "audio_duration": len(task_data.get("text", "")) * 0.1,
                            "synthesis_time": 0.05
                        }
                    else:
                        raise ValueError(f"Unknown task type: {task_type}")
                    
                    # 记录成功处理的任务
                    self.processed_tasks.append({
                        "message_id": message["id"],
                        "task_type": task_type,
                        "result": result,
                        "processed_at": time.time()
                    })
                    
                    # 确认消息
                    await self.mq.ack_message(message["topic"], message["id"])
                    
                    return {"success": True, "result": result}
                
                except Exception as e:
                    # 记录失败的任务
                    self.failed_tasks.append({
                        "message_id": message["id"],
                        "task_type": task_type,
                        "error": str(e),
                        "failed_at": time.time()
                    })
                    
                    return {"success": False, "error": str(e)}
            
            async def start_worker(self, topics=None):
                """启动工作进程"""
                if topics is None:
                    topics = ["tasks.high", "tasks.normal", "tasks.low"]
                
                for topic in topics:
                    await self.mq.subscribe(topic, self.process_task)
        
        # 测试任务处理
        processor = TaskProcessor(mock_message_queue)
        
        # 启动工作进程
        await processor.start_worker()
        
        # 提交不同类型的任务
        task1 = await processor.submit_task(
            "document_processing",
            {"content": "这是一个测试文档内容"},
            "high"
        )
        
        task2 = await processor.submit_task(
            "voice_synthesis",
            {"text": "这是要合成的文本"},
            "normal"
        )
        
        # 等待任务处理
        await asyncio.sleep(0.2)
        
        # 验证任务处理结果
        assert len(processor.processed_tasks) == 2
        assert task1["status"] == "submitted"
        assert task2["status"] == "submitted"
        
        # 验证处理结果
        doc_task = next(t for t in processor.processed_tasks if t["task_type"] == "document_processing")
        voice_task = next(t for t in processor.processed_tasks if t["task_type"] == "voice_synthesis")
        
        assert doc_task["result"]["processed_chunks"] > 0
        assert voice_task["result"]["audio_duration"] > 0
    
    @pytest.mark.asyncio
    async def test_event_driven_architecture(self, mock_message_queue):
        """测试事件驱动架构"""
        class EventBus:
            def __init__(self, message_queue):
                self.mq = message_queue
                self.event_handlers = {}
            
            async def emit_event(self, event_type, event_data, source="system"):
                """发出事件"""
                event = {
                    "event_type": event_type,
                    "event_data": event_data,
                    "source": source,
                    "event_id": f"evt_{int(time.time() * 1000)}",
                    "timestamp": time.time()
                }
                
                topic = f"events.{event_type}"
                await self.mq.publish(topic, event)
                
                return event["event_id"]
            
            async def register_handler(self, event_type, handler_func):
                """注册事件处理器"""
                topic = f"events.{event_type}"
                
                async def wrapper(message):
                    event = message["payload"]
                    try:
                        await handler_func(event)
                    except Exception as e:
                        print(f"Event handler error: {e}")
                
                await self.mq.subscribe(topic, wrapper)
                
                if event_type not in self.event_handlers:
                    self.event_handlers[event_type] = []
                self.event_handlers[event_type].append(handler_func)
        
        # 创建事件总线
        event_bus = EventBus(mock_message_queue)
        
        # 事件处理结果记录
        handled_events = []
        
        # 注册事件处理器
        async def user_registered_handler(event):
            user_data = event["event_data"]
            # 模拟发送欢迎邮件
            await asyncio.sleep(0.01)
            handled_events.append({
                "handler": "welcome_email",
                "user_id": user_data["user_id"],
                "processed_at": time.time()
            })
        
        async def user_activity_handler(event):
            activity_data = event["event_data"]
            # 模拟记录用户活动
            await asyncio.sleep(0.01)
            handled_events.append({
                "handler": "activity_log",
                "user_id": activity_data["user_id"],
                "action": activity_data["action"],
                "processed_at": time.time()
            })
        
        # 注册处理器
        await event_bus.register_handler("user_registered", user_registered_handler)
        await event_bus.register_handler("user_activity", user_activity_handler)
        
        # 发出事件
        await event_bus.emit_event("user_registered", {
            "user_id": 123,
            "username": "new_user",
            "email": "new_user@example.com"
        })
        
        await event_bus.emit_event("user_activity", {
            "user_id": 123,
            "action": "login",
            "ip_address": "192.168.1.100"
        })
        
        # 等待事件处理
        await asyncio.sleep(0.1)
        
        # 验证事件处理
        assert len(handled_events) == 2
        
        welcome_event = next(e for e in handled_events if e["handler"] == "welcome_email")
        activity_event = next(e for e in handled_events if e["handler"] == "activity_log")
        
        assert welcome_event["user_id"] == 123
        assert activity_event["action"] == "login"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
