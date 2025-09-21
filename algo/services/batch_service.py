"""
批量化服务

为算法服务提供批量化处理能力
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import json
import time
from ..core.integrated_batch_system import IntegratedBatchSystem, BatchingConfig
from ..core.llm_client import LLMClient  # 假设存在LLM客户端

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """批量请求"""
    messages: List[Dict[str, str]]
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1000
    stream: bool = False
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None


@dataclass
class BatchResponse:
    """批量响应"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    request_id: str
    processing_time: float


class LLMBatchService:
    """LLM批量化服务"""
    
    def __init__(
        self,
        llm_client: Optional['LLMClient'] = None,
        config: Optional[BatchingConfig] = None
    ):
        self.llm_client = llm_client or self._create_default_client()
        self.config = config or self._create_default_config()
        
        # 初始化批处理系统
        self.batch_system = IntegratedBatchSystem(
            processing_func=self._process_llm_batch,
            config=self.config
        )
        
        self._started = False
    
    def _create_default_client(self) -> 'LLMClient':
        """创建默认LLM客户端"""
        # 这里应该返回实际的LLM客户端实例
        # 暂时返回模拟客户端
        return MockLLMClient()
    
    def _create_default_config(self) -> BatchingConfig:
        """创建默认配置"""
        return BatchingConfig(
            initial_batch_size=6,
            min_batch_size=1,
            max_batch_size=20,
            max_wait_time=0.08,  # 80ms
            enable_request_merging=True,
            similarity_threshold=0.88,
            merge_window=3.0,
            enable_dynamic_adjustment=True,
            monitor_interval=10.0
        )
    
    async def start(self):
        """启动服务"""
        if not self._started:
            await self.batch_system.start()
            self._started = True
            logger.info("LLM batch service started")
    
    async def stop(self):
        """停止服务"""
        if self._started:
            await self.batch_system.stop()
            self._started = False
            logger.info("LLM batch service stopped")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        priority: int = 0,
        timeout: float = 30.0
    ) -> BatchResponse:
        """
        聊天补全 (批量化处理)
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数
            user_id: 用户ID
            conversation_id: 会话ID
            priority: 优先级
            timeout: 超时时间
        
        Returns:
            批量响应
        """
        if not self._started:
            await self.start()
        
        # 构建请求内容
        content = self._build_request_content(messages, model, temperature, max_tokens)
        
        # 构建参数
        parameters = {
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'user_id': user_id,
            'conversation_id': conversation_id
        }
        
        # 提交到批处理系统
        result = await self.batch_system.process_request(
            content=content,
            model=model,
            parameters=parameters,
            priority=priority,
            timeout=timeout
        )
        
        return result
    
    def _build_request_content(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """构建请求内容用于相似性比较"""
        # 提取最后几条消息的内容
        recent_messages = messages[-3:] if len(messages) > 3 else messages
        
        # 构建内容字符串
        content_parts = []
        for msg in recent_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            content_parts.append(f"{role}: {content}")
        
        # 添加模型和关键参数
        content = " | ".join(content_parts)
        content += f" [model:{model}, temp:{temperature:.1f}, max:{max_tokens}]"
        
        return content
    
    async def _process_llm_batch(self, requests: List[Dict[str, Any]]) -> List[BatchResponse]:
        """处理LLM批次"""
        batch_start = time.time()
        
        try:
            # 按模型分组
            model_groups = {}
            for i, req in enumerate(requests):
                model = req.get('model', 'gpt-3.5-turbo')
                if model not in model_groups:
                    model_groups[model] = []
                model_groups[model].append((i, req))
            
            # 并行处理不同模型
            all_results = [None] * len(requests)
            
            async def process_model_group(model: str, group_requests: List[tuple]):
                """处理单个模型的请求组"""
                indices, reqs = zip(*group_requests)
                
                # 构建批量请求
                batch_requests = []
                for req in reqs:
                    batch_req = BatchRequest(
                        messages=req['parameters']['messages'],
                        model=model,
                        temperature=req['parameters'].get('temperature', 0.7),
                        max_tokens=req['parameters'].get('max_tokens', 1000),
                        user_id=req['parameters'].get('user_id'),
                        conversation_id=req['parameters'].get('conversation_id')
                    )
                    batch_requests.append(batch_req)
                
                # 调用LLM客户端
                responses = await self.llm_client.batch_chat_completion(batch_requests)
                
                # 将结果放回正确位置
                for idx, response in zip(indices, responses):
                    all_results[idx] = response
            
            # 并行处理所有模型组
            tasks = [
                process_model_group(model, group)
                for model, group in model_groups.items()
            ]
            
            await asyncio.gather(*tasks)
            
            # 设置处理时间
            processing_time = time.time() - batch_start
            for result in all_results:
                if result:
                    result.processing_time = processing_time
            
            return all_results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            
            # 创建错误响应
            error_responses = []
            for req in requests:
                error_response = BatchResponse(
                    content=f"Error: {str(e)}",
                    model=req.get('model', 'unknown'),
                    usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                    finish_reason='error',
                    request_id=req.get('id', 'unknown'),
                    processing_time=time.time() - batch_start
                )
                error_responses.append(error_response)
            
            return error_responses
    
    async def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = self.batch_system.get_stats()
        
        # 添加服务特定统计
        service_stats = {
            'service_name': 'LLMBatchService',
            'started': self._started,
            'config': {
                'initial_batch_size': self.config.initial_batch_size,
                'max_batch_size': self.config.max_batch_size,
                'max_wait_time': self.config.max_wait_time,
                'enable_request_merging': self.config.enable_request_merging,
                'similarity_threshold': self.config.similarity_threshold
            }
        }
        
        return {**stats, **service_stats}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 发送测试请求
            test_messages = [{"role": "user", "content": "Hello"}]
            
            start_time = time.time()
            response = await asyncio.wait_for(
                self.chat_completion(
                    messages=test_messages,
                    max_tokens=10,
                    priority=1  # 高优先级
                ),
                timeout=5.0
            )
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time': response_time,
                'batch_system_running': self._started,
                'test_response': response.content[:50] + '...' if len(response.content) > 50 else response.content
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'batch_system_running': self._started
            }


class MockLLMClient:
    """模拟LLM客户端"""
    
    async def batch_chat_completion(self, requests: List[BatchRequest]) -> List[BatchResponse]:
        """批量聊天补全"""
        # 模拟处理时间
        batch_size = len(requests)
        base_time = 0.3
        
        # 批次效率：批次越大，单个请求的平均时间越短
        efficiency = min(1.5, batch_size / 4.0)
        processing_time = base_time / efficiency
        
        await asyncio.sleep(processing_time)
        
        # 生成响应
        responses = []
        for i, req in enumerate(requests):
            # 模拟不同的响应长度
            if 'translate' in req.messages[-1].get('content', '').lower():
                content = f"Translation result for request {i}"
                tokens = 15
            elif 'summarize' in req.messages[-1].get('content', '').lower():
                content = f"Summary: This is a summarized response for request {i}"
                tokens = 25
            else:
                content = f"AI response to '{req.messages[-1].get('content', 'N/A')}' (request {i})"
                tokens = 30
            
            response = BatchResponse(
                content=content,
                model=req.model,
                usage={
                    'prompt_tokens': len(req.messages[-1].get('content', '')) // 4,
                    'completion_tokens': tokens,
                    'total_tokens': len(req.messages[-1].get('content', '')) // 4 + tokens
                },
                finish_reason='stop',
                request_id=f"req_{i}_{int(time.time() * 1000)}",
                processing_time=processing_time
            )
            responses.append(response)
        
        return responses


# FastAPI集成示例
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from pydantic import BaseModel
    
    class ChatRequest(BaseModel):
        messages: List[Dict[str, str]]
        model: str = "gpt-3.5-turbo"
        temperature: float = 0.7
        max_tokens: int = 1000
        stream: bool = False
        user_id: Optional[str] = None
        conversation_id: Optional[str] = None
    
    class ChatResponseModel(BaseModel):
        content: str
        model: str
        usage: Dict[str, int]
        finish_reason: str
        request_id: str
        processing_time: float
    
    # 全局服务实例
    batch_service: Optional[LLMBatchService] = None
    
    def create_batch_api(app: FastAPI):
        """创建批量化API"""
        global batch_service
        
        @app.on_event("startup")
        async def startup_event():
            global batch_service
            batch_service = LLMBatchService()
            await batch_service.start()
        
        @app.on_event("shutdown")
        async def shutdown_event():
            global batch_service
            if batch_service:
                await batch_service.stop()
        
        @app.post("/v1/chat/completions", response_model=ChatResponseModel)
        async def chat_completions(request: ChatRequest):
            """聊天补全接口"""
            if not batch_service:
                raise HTTPException(status_code=503, detail="Batch service not available")
            
            try:
                response = await batch_service.chat_completion(
                    messages=request.messages,
                    model=request.model,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    user_id=request.user_id,
                    conversation_id=request.conversation_id
                )
                
                return ChatResponseModel(
                    content=response.content,
                    model=response.model,
                    usage=response.usage,
                    finish_reason=response.finish_reason,
                    request_id=response.request_id,
                    processing_time=response.processing_time
                )
                
            except asyncio.TimeoutError:
                raise HTTPException(status_code=408, detail="Request timeout")
            except Exception as e:
                logger.error(f"Chat completion error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/v1/batch/stats")
        async def get_batch_stats():
            """获取批处理统计"""
            if not batch_service:
                raise HTTPException(status_code=503, detail="Batch service not available")
            
            return await batch_service.get_service_stats()
        
        @app.get("/v1/batch/health")
        async def batch_health_check():
            """批处理健康检查"""
            if not batch_service:
                raise HTTPException(status_code=503, detail="Batch service not available")
            
            health = await batch_service.health_check()
            
            if health['status'] != 'healthy':
                raise HTTPException(status_code=503, detail=health)
            
            return health

except ImportError:
    # FastAPI不可用时的占位符
    def create_batch_api(app):
        pass


# 使用示例
async def example_usage():
    """使用示例"""
    
    # 创建批量服务
    service = LLMBatchService()
    await service.start()
    
    try:
        # 模拟并发请求
        async def send_request(i: int):
            messages = [
                {"role": "user", "content": f"Hello, this is request {i}"}
            ]
            
            response = await service.chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                user_id=f"user_{i % 5}",  # 5个不同用户
                conversation_id=f"conv_{i % 10}"  # 10个不同会话
            )
            
            return response
        
        # 发送20个并发请求
        print("Sending 20 concurrent requests...")
        tasks = [send_request(i) for i in range(20)]
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        # 统计结果
        successful = sum(1 for r in responses if isinstance(r, BatchResponse))
        failed = len(responses) - successful
        
        print(f"\n=== 结果统计 ===")
        print(f"总请求: {len(responses)}")
        print(f"成功: {successful}")
        print(f"失败: {failed}")
        print(f"总耗时: {total_time:.3f}s")
        print(f"平均延迟: {total_time/len(responses):.3f}s")
        
        # 显示服务统计
        stats = await service.get_service_stats()
        print(f"\n=== 服务统计 ===")
        for key, value in stats.items():
            if isinstance(value, float):
                if 'rate' in key or 'efficiency' in key:
                    print(f"{key}: {value:.2%}")
                else:
                    print(f"{key}: {value:.3f}")
            elif isinstance(value, dict):
                print(f"{key}: {json.dumps(value, indent=2)}")
            else:
                print(f"{key}: {value}")
        
        # 健康检查
        health = await service.health_check()
        print(f"\n=== 健康检查 ===")
        print(json.dumps(health, indent=2))
        
    finally:
        await service.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())
