"""
负载测试脚本
使用Locust框架进行分布式负载测试
测试目标：验证系统在正常负载下的性能表现
"""

import json
import time
import random
import base64
from locust import HttpUser, task, between, events
from locust.runners import MasterRunner
import logging
import csv
import os
from datetime import datetime
from typing import Dict, List, Any


class VoiceHelperLoadTest(HttpUser):
    """VoiceHelper负载测试用户类"""
    
    wait_time = between(1, 3)  # 用户操作间隔1-3秒
    
    def on_start(self):
        """用户开始时的初始化"""
        self.auth_token = self.get_auth_token()
        self.conversation_id = f"load_test_{self.environment.runner.user_count}_{int(time.time())}"
        self.message_count = 0
        
        # 性能指标记录
        self.response_times = []
        self.error_count = 0
        self.success_count = 0
    
    def get_auth_token(self) -> str:
        """获取认证令牌"""
        try:
            login_data = {
                "username": f"load_test_user_{random.randint(1, 1000)}",
                "password": "load_test_password"
            }
            
            response = self.client.post(
                "/api/v1/auth/login",
                json=login_data,
                timeout=10,
                catch_response=True
            )
            
            if response.status_code == 200:
                return response.json().get("token", "load_test_token")
            else:
                return "load_test_token"
                
        except Exception as e:
            logging.warning(f"获取认证令牌失败: {e}")
            return "load_test_token"
    
    @task(5)
    def test_health_check(self):
        """健康检查测试 - 权重5"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="health_check"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    response.success()
                else:
                    response.failure(f"健康检查失败: {data}")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(3)
    def test_ping_endpoint(self):
        """Ping接口测试 - 权重3"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        with self.client.get(
            "/api/v1/ping",
            headers=headers,
            catch_response=True,
            name="ping"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("message") == "pong":
                    response.success()
                else:
                    response.failure("Ping响应格式错误")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(10)
    def test_chat_completion(self):
        """聊天完成测试 - 权重10（主要功能）"""
        self.message_count += 1
        
        # 随机选择测试消息
        test_messages = [
            "你好，我想了解VoiceHelper的功能",
            "请介绍一下系统的技术架构",
            "如何使用语音功能？",
            "系统支持哪些平台？",
            "性能指标如何？",
            "有什么安全保障措施？",
            "如何进行文档管理？",
            "支持多语言吗？",
            "可以集成第三方服务吗？",
            "如何进行系统监控？"
        ]
        
        message = random.choice(test_messages)
        
        chat_data = {
            "conversation_id": self.conversation_id,
            "messages": [
                {
                    "role": "user",
                    "content": f"{message} (消息#{self.message_count})"
                }
            ],
            "top_k": 5,
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": True
        }
        
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        with self.client.post(
            "/api/v1/chat/completions",
            headers=headers,
            json=chat_data,
            catch_response=True,
            name="chat_completion",
            stream=True
        ) as response:
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            if response.status_code == 200:
                # 处理流式响应
                content_received = False
                chunk_count = 0
                
                try:
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                chunk_count += 1
                                data_str = line_str[6:]
                                if data_str != '[DONE]':
                                    try:
                                        data = json.loads(data_str)
                                        if 'content' in data or 'message' in data:
                                            content_received = True
                                    except json.JSONDecodeError:
                                        pass
                    
                    if content_received or chunk_count > 0:
                        response.success()
                        self.success_count += 1
                    else:
                        response.failure("未收到有效的流式响应")
                        self.error_count += 1
                        
                except Exception as e:
                    response.failure(f"处理流式响应失败: {e}")
                    self.error_count += 1
            else:
                response.failure(f"HTTP {response.status_code}")
                self.error_count += 1
    
    @task(2)
    def test_integration_services(self):
        """集成服务测试 - 权重2"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        with self.client.get(
            "/api/v1/integrations/services",
            headers=headers,
            catch_response=True,
            name="integration_services"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "services" in data:
                    response.success()
                else:
                    response.failure("服务列表格式错误")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def test_integration_health(self):
        """集成服务健康检查 - 权重1"""
        headers = {"Authorization": f"Bearer {self.auth_token}"}
        
        with self.client.get(
            "/api/v1/integrations/health",
            headers=headers,
            catch_response=True,
            name="integration_health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")


class AlgoServiceLoadTest(HttpUser):
    """算法服务负载测试"""
    
    host = "http://localhost:8000"  # 算法服务地址
    wait_time = between(2, 5)  # 算法服务响应较慢，增加等待时间
    
    def on_start(self):
        """初始化"""
        self.test_documents_ingested = False
        self.query_count = 0
    
    @task(1)
    def test_algo_health(self):
        """算法服务健康检查"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="algo_health"
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("算法服务状态异常")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def test_document_ingest(self):
        """文档入库测试"""
        if self.test_documents_ingested:
            return  # 避免重复入库
        
        test_doc = {
            "files": [
                {
                    "name": f"load_test_doc_{int(time.time())}.txt",
                    "content": f"""
                    负载测试文档 {random.randint(1, 1000)}
                    
                    这是一个用于负载测试的文档。
                    包含了关于VoiceHelper系统的信息。
                    
                    功能特性：
                    - 智能对话
                    - 语音识别
                    - 多模态交互
                    - 企业级安全
                    
                    技术架构：
                    - 微服务设计
                    - 分布式部署
                    - 高可用保障
                    - 性能优化
                    
                    创建时间: {datetime.now().isoformat()}
                    """,
                    "metadata": {
                        "category": "load_test",
                        "test_id": f"load_test_{int(time.time())}",
                        "created_by": "load_test_user"
                    }
                }
            ],
            "collection_name": "load_test_collection"
        }
        
        with self.client.post(
            "/ingest",
            json=test_doc,
            catch_response=True,
            name="document_ingest",
            timeout=60
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "task_id" in data:
                    response.success()
                    self.test_documents_ingested = True
                else:
                    response.failure("入库响应格式错误")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(8)
    def test_document_query(self):
        """文档查询测试"""
        self.query_count += 1
        
        # 测试查询列表
        test_queries = [
            "VoiceHelper有什么功能？",
            "系统架构是怎样的？",
            "如何保证系统安全？",
            "性能优化措施有哪些？",
            "支持哪些技术栈？",
            "如何进行部署？",
            "监控方案如何？",
            "扩展性如何？"
        ]
        
        query = random.choice(test_queries)
        
        query_data = {
            "messages": [
                {
                    "role": "user",
                    "content": f"{query} (查询#{self.query_count})"
                }
            ],
            "top_k": 5,
            "temperature": 0.3,
            "max_tokens": 300
        }
        
        with self.client.post(
            "/query",
            json=query_data,
            catch_response=True,
            name="document_query",
            timeout=30
        ) as response:
            if response.status_code == 200:
                # 处理NDJSON流式响应
                response_lines = response.text.strip().split('\n')
                valid_responses = 0
                
                for line in response_lines:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if data.get("type") in ["refs", "message", "end"]:
                                valid_responses += 1
                        except json.JSONDecodeError:
                            pass
                
                if valid_responses > 0:
                    response.success()
                else:
                    response.failure("未收到有效的查询响应")
            else:
                response.failure(f"HTTP {response.status_code}")


# 性能监控和报告
class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {
            "requests": [],
            "errors": [],
            "response_times": [],
            "throughput": []
        }
        self.start_time = time.time()
    
    def record_request(self, request_type: str, response_time: float, success: bool):
        """记录请求"""
        timestamp = time.time()
        
        self.metrics["requests"].append({
            "timestamp": timestamp,
            "type": request_type,
            "response_time": response_time,
            "success": success
        })
        
        self.metrics["response_times"].append(response_time)
        
        if not success:
            self.metrics["errors"].append({
                "timestamp": timestamp,
                "type": request_type
            })
    
    def calculate_throughput(self, window_seconds: int = 60):
        """计算吞吐量"""
        current_time = time.time()
        window_start = current_time - window_seconds
        
        recent_requests = [
            r for r in self.metrics["requests"]
            if r["timestamp"] >= window_start
        ]
        
        return len(recent_requests) / window_seconds
    
    def get_statistics(self):
        """获取统计信息"""
        if not self.metrics["response_times"]:
            return {}
        
        response_times = self.metrics["response_times"]
        total_requests = len(self.metrics["requests"])
        total_errors = len(self.metrics["errors"])
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)],
            "p99_response_time": sorted(response_times)[int(len(response_times) * 0.99)],
            "throughput_1min": self.calculate_throughput(60),
            "throughput_5min": self.calculate_throughput(300),
            "test_duration": time.time() - self.start_time
        }
    
    def export_csv_report(self, filename: str):
        """导出CSV报告"""
        stats = self.get_statistics()
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # 写入统计信息
            writer.writerow(["Metric", "Value"])
            for key, value in stats.items():
                writer.writerow([key, value])
            
            # 写入详细请求数据
            writer.writerow([])
            writer.writerow(["Timestamp", "Request Type", "Response Time", "Success"])
            
            for request in self.metrics["requests"]:
                writer.writerow([
                    datetime.fromtimestamp(request["timestamp"]).isoformat(),
                    request["type"],
                    request["response_time"],
                    request["success"]
                ])


# Locust事件处理
performance_monitor = PerformanceMonitor()

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """请求事件处理"""
    success = exception is None
    performance_monitor.record_request(name, response_time / 1000.0, success)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """测试结束事件处理"""
    stats = performance_monitor.get_statistics()
    
    print("\n" + "="*50)
    print("负载测试结果统计")
    print("="*50)
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # 导出详细报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"load_test_report_{timestamp}.csv"
    performance_monitor.export_csv_report(report_filename)
    
    print(f"\n详细报告已导出: {report_filename}")
    
    # 性能阈值检查
    if stats.get("avg_response_time", 0) > 2.0:
        print("⚠️  警告: 平均响应时间超过2秒")
    
    if stats.get("error_rate", 0) > 5.0:
        print("⚠️  警告: 错误率超过5%")
    
    if stats.get("p95_response_time", 0) > 5.0:
        print("⚠️  警告: P95响应时间超过5秒")


# 负载测试配置
class LoadTestConfig:
    """负载测试配置"""
    
    # 基础配置
    BACKEND_HOST = "http://localhost:8080"
    ALGO_HOST = "http://localhost:8000"
    
    # 负载配置
    USERS = 50  # 并发用户数
    SPAWN_RATE = 5  # 每秒启动用户数
    RUN_TIME = "5m"  # 运行时间
    
    # 性能阈值
    MAX_AVG_RESPONSE_TIME = 2.0  # 最大平均响应时间(秒)
    MAX_ERROR_RATE = 5.0  # 最大错误率(%)
    MAX_P95_RESPONSE_TIME = 5.0  # 最大P95响应时间(秒)
    MIN_THROUGHPUT = 100  # 最小吞吐量(请求/分钟)


if __name__ == "__main__":
    # 运行负载测试的示例命令
    print("负载测试脚本")
    print("使用Locust运行:")
    print(f"locust -f {__file__} --host {LoadTestConfig.BACKEND_HOST}")
    print(f"或者: locust -f {__file__} --host {LoadTestConfig.BACKEND_HOST} --users {LoadTestConfig.USERS} --spawn-rate {LoadTestConfig.SPAWN_RATE} --run-time {LoadTestConfig.RUN_TIME} --headless")
