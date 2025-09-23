"""
VoiceHelper SDK Client - v1.9.0
主要的SDK客户端实现
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, BinaryIO
import aiohttp
import aiofiles
from pathlib import Path

from .types import *
from .exceptions import VoiceHelperError, APIError, NetworkError, AuthenticationError


class VoiceHelperSDK:
    """VoiceHelper SDK 主客户端类"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.voicehelper.com/v1",
                 timeout: int = 30,
                 retries: int = 3,
                 debug: bool = False):
        """
        初始化SDK客户端
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            timeout: 请求超时时间（秒）
            retries: 重试次数
            debug: 是否启用调试模式
        """
        self.config = VoiceHelperConfig(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            retries=retries,
            debug=debug
        )
        
        self.session: Optional[aiohttp.ClientSession] = None
        self._base_headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.config.api_key,
            "User-Agent": f"VoiceHelper-SDK-Python/{__version__}",
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()
    
    async def _ensure_session(self):
        """确保aiohttp会话存在"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._base_headers
            )
    
    async def close(self):
        """关闭客户端会话"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    # ==================== 聊天对话 API ====================
    
    async def create_chat_completion(self,
                                   messages: List[ChatMessage],
                                   model: ModelType = ModelType.GPT_4_TURBO,
                                   stream: bool = False,
                                   max_tokens: Optional[int] = None,
                                   temperature: Optional[float] = None,
                                   multimodal_config: Optional[MultimodalConfig] = None) -> ChatCompletionResponse:
        """
        创建聊天完成
        
        Args:
            messages: 消息列表
            model: 使用的模型
            stream: 是否流式响应
            max_tokens: 最大token数
            temperature: 温度参数
            multimodal_config: 多模态配置
            
        Returns:
            聊天完成响应
        """
        if stream:
            raise ValueError("流式响应请使用 create_chat_completion_stream 方法")
        
        request_data = {
            "messages": [msg.dict() for msg in messages],
            "model": model.value,
            "stream": False,
        }
        
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        if temperature is not None:
            request_data["temperature"] = temperature
        if multimodal_config is not None:
            request_data["multimodal_config"] = multimodal_config.dict()
        
        response_data = await self._make_request("POST", "/chat/completions", request_data)
        return ChatCompletionResponse(**response_data)
    
    async def create_chat_completion_stream(self,
                                          messages: List[ChatMessage],
                                          model: ModelType = ModelType.GPT_4_TURBO,
                                          max_tokens: Optional[int] = None,
                                          temperature: Optional[float] = None,
                                          multimodal_config: Optional[MultimodalConfig] = None) -> AsyncGenerator[ChatCompletionStreamResponse, None]:
        """
        创建流式聊天完成
        
        Args:
            messages: 消息列表
            model: 使用的模型
            max_tokens: 最大token数
            temperature: 温度参数
            multimodal_config: 多模态配置
            
        Yields:
            流式聊天完成响应
        """
        request_data = {
            "messages": [msg.dict() for msg in messages],
            "model": model.value,
            "stream": True,
        }
        
        if max_tokens is not None:
            request_data["max_tokens"] = max_tokens
        if temperature is not None:
            request_data["temperature"] = temperature
        if multimodal_config is not None:
            request_data["multimodal_config"] = multimodal_config.dict()
        
        await self._ensure_session()
        
        async with self.session.post(
            f"{self.config.base_url}/chat/completions",
            json=request_data,
            headers={"Accept": "text/event-stream"}
        ) as response:
            if response.status != 200:
                await self._handle_error_response(response)
            
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = line[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk_data = json.loads(data)
                        yield ChatCompletionStreamResponse(**chunk_data)
                    except json.JSONDecodeError:
                        continue
    
    async def get_conversations(self,
                              limit: int = 20,
                              offset: int = 0,
                              filter_type: ConversationFilter = ConversationFilter.ALL) -> ConversationList:
        """获取对话列表"""
        params = {
            "limit": limit,
            "offset": offset,
            "filter": filter_type.value
        }
        
        response_data = await self._make_request("GET", "/chat/conversations", params=params)
        return ConversationList(**response_data)
    
    async def get_conversation(self, conversation_id: str) -> Conversation:
        """获取对话详情"""
        response_data = await self._make_request("GET", f"/chat/conversations/{conversation_id}")
        return Conversation(**response_data)
    
    # ==================== 语音处理 API ====================
    
    async def synthesize_voice(self,
                             text: str,
                             voice: VoiceType = VoiceType.ALLOY,
                             emotion: EmotionType = EmotionType.NEUTRAL,
                             speed: float = 1.0,
                             language: LanguageCode = LanguageCode.ZH_CN,
                             streaming: bool = False) -> Union[bytes, VoiceSynthesisResponse]:
        """
        语音合成
        
        Args:
            text: 要合成的文本
            voice: 语音类型
            emotion: 情感类型
            speed: 语速
            language: 语言
            streaming: 是否流式合成
            
        Returns:
            音频数据或合成响应
        """
        request_data = {
            "text": text,
            "voice": voice.value,
            "emotion": emotion.value,
            "speed": speed,
            "language": language.value,
            "streaming": streaming
        }
        
        await self._ensure_session()
        
        async with self.session.post(
            f"{self.config.base_url}/voice/synthesize",
            json=request_data
        ) as response:
            if response.status != 200:
                await self._handle_error_response(response)
            
            content_type = response.headers.get('content-type', '')
            
            if 'audio/' in content_type:
                # 返回二进制音频数据
                return await response.read()
            else:
                # 返回JSON响应
                response_data = await response.json()
                return VoiceSynthesisResponse(**response_data)
    
    async def recognize_voice(self,
                            audio_data: Union[bytes, BinaryIO, str, Path],
                            language: LanguageCode = LanguageCode.ZH_CN,
                            enable_emotion_detection: bool = False,
                            enable_speaker_separation: bool = False,
                            noise_reduction: bool = True) -> VoiceRecognitionResponse:
        """
        语音识别
        
        Args:
            audio_data: 音频数据（字节、文件对象或文件路径）
            language: 识别语言
            enable_emotion_detection: 是否启用情感检测
            enable_speaker_separation: 是否启用说话人分离
            noise_reduction: 是否启用噪声抑制
            
        Returns:
            语音识别响应
        """
        # 处理音频数据
        if isinstance(audio_data, (str, Path)):
            async with aiofiles.open(audio_data, 'rb') as f:
                audio_bytes = await f.read()
        elif hasattr(audio_data, 'read'):
            if asyncio.iscoroutinefunction(audio_data.read):
                audio_bytes = await audio_data.read()
            else:
                audio_bytes = audio_data.read()
        else:
            audio_bytes = audio_data
        
        # 准备表单数据
        data = aiohttp.FormData()
        data.add_field('audio', audio_bytes, content_type='audio/wav')
        
        config = {
            "language": language.value,
            "enable_emotion_detection": enable_emotion_detection,
            "enable_speaker_separation": enable_speaker_separation,
            "noise_reduction": noise_reduction
        }
        data.add_field('config', json.dumps(config), content_type='application/json')
        
        await self._ensure_session()
        
        # 临时移除Content-Type头，让aiohttp自动设置
        headers = {k: v for k, v in self._base_headers.items() if k != 'Content-Type'}
        
        async with self.session.post(
            f"{self.config.base_url}/voice/recognize",
            data=data,
            headers=headers
        ) as response:
            if response.status != 200:
                await self._handle_error_response(response)
            
            response_data = await response.json()
            return VoiceRecognitionResponse(**response_data)
    
    # ==================== 图像理解 API ====================
    
    async def analyze_image(self,
                          image_data: Union[bytes, BinaryIO, str, Path],
                          tasks: Optional[List[VisionTask]] = None,
                          query: Optional[str] = None) -> VisionAnalysisResponse:
        """
        图像分析
        
        Args:
            image_data: 图像数据（字节、文件对象或文件路径）
            tasks: 分析任务列表
            query: 用户查询
            
        Returns:
            图像分析响应
        """
        # 处理图像数据
        if isinstance(image_data, (str, Path)):
            async with aiofiles.open(image_data, 'rb') as f:
                image_bytes = await f.read()
        elif hasattr(image_data, 'read'):
            if asyncio.iscoroutinefunction(image_data.read):
                image_bytes = await image_data.read()
            else:
                image_bytes = image_data.read()
        else:
            image_bytes = image_data
        
        # 准备表单数据
        data = aiohttp.FormData()
        data.add_field('image', image_bytes, content_type='image/jpeg')
        
        if tasks:
            for task in tasks:
                data.add_field('tasks', task.value)
        
        if query:
            data.add_field('query', query)
        
        await self._ensure_session()
        
        # 临时移除Content-Type头
        headers = {k: v for k, v in self._base_headers.items() if k != 'Content-Type'}
        
        async with self.session.post(
            f"{self.config.base_url}/vision/analyze",
            data=data,
            headers=headers
        ) as response:
            if response.status != 200:
                await self._handle_error_response(response)
            
            response_data = await response.json()
            return VisionAnalysisResponse(**response_data)
    
    # ==================== 多模态融合 API ====================
    
    async def fuse_modalities(self,
                            modality_inputs: List[ModalityInput],
                            fusion_strategy: FusionStrategy = FusionStrategy.HIERARCHICAL,
                            context: Optional[Dict[str, Any]] = None) -> MultimodalFusionResponse:
        """
        多模态融合
        
        Args:
            modality_inputs: 模态输入列表
            fusion_strategy: 融合策略
            context: 上下文信息
            
        Returns:
            多模态融合响应
        """
        request_data = {
            "modality_inputs": [inp.dict() for inp in modality_inputs],
            "fusion_strategy": fusion_strategy.value,
        }
        
        if context:
            request_data["context"] = context
        
        response_data = await self._make_request("POST", "/multimodal/fuse", request_data)
        return MultimodalFusionResponse(**response_data)
    
    # ==================== MCP服务集成 API ====================
    
    async def get_services(self,
                         category: Optional[ServiceCategory] = None,
                         status: ServiceStatus = ServiceStatus.ACTIVE) -> ServiceList:
        """获取可用服务列表"""
        params = {"status": status.value}
        if category:
            params["category"] = category.value
        
        response_data = await self._make_request("GET", "/mcp/services", params=params)
        return ServiceList(**response_data)
    
    async def call_mcp_service(self,
                             service_name: str,
                             operation: str,
                             params: Dict[str, Any]) -> MCPServiceCallResponse:
        """调用MCP服务"""
        request_data = {
            "operation": operation,
            "params": params
        }
        
        response_data = await self._make_request("POST", f"/mcp/services/{service_name}/call", request_data)
        return MCPServiceCallResponse(**response_data)
    
    # ==================== 数据集管理 API ====================
    
    async def get_datasets(self) -> DatasetList:
        """获取数据集列表"""
        response_data = await self._make_request("GET", "/datasets")
        return DatasetList(**response_data)
    
    async def create_dataset(self,
                           name: str,
                           description: Optional[str] = None,
                           config: Optional[Dict[str, Any]] = None) -> Dataset:
        """创建数据集"""
        request_data = {"name": name}
        if description:
            request_data["description"] = description
        if config:
            request_data["config"] = config
        
        response_data = await self._make_request("POST", "/datasets", request_data)
        return Dataset(**response_data)
    
    async def ingest_data(self,
                        dataset_id: str,
                        files: List[Union[str, Path, BinaryIO]],
                        metadata: Optional[Dict[str, Any]] = None) -> IngestResponse:
        """数据摄取"""
        data = aiohttp.FormData()
        
        # 添加文件
        for file_item in files:
            if isinstance(file_item, (str, Path)):
                async with aiofiles.open(file_item, 'rb') as f:
                    file_data = await f.read()
                    filename = Path(file_item).name
                    data.add_field('files', file_data, filename=filename)
            elif hasattr(file_item, 'read'):
                if asyncio.iscoroutinefunction(file_item.read):
                    file_data = await file_item.read()
                else:
                    file_data = file_item.read()
                filename = getattr(file_item, 'name', 'file')
                data.add_field('files', file_data, filename=filename)
        
        # 添加元数据
        if metadata:
            data.add_field('metadata', json.dumps(metadata), content_type='application/json')
        
        await self._ensure_session()
        
        headers = {k: v for k, v in self._base_headers.items() if k != 'Content-Type'}
        
        async with self.session.post(
            f"{self.config.base_url}/datasets/{dataset_id}/ingest",
            data=data,
            headers=headers
        ) as response:
            if response.status != 200:
                await self._handle_error_response(response)
            
            response_data = await response.json()
            return IngestResponse(**response_data)
    
    # ==================== 系统监控 API ====================
    
    async def get_system_health(self) -> SystemHealth:
        """系统健康检查"""
        response_data = await self._make_request("GET", "/system/health")
        return SystemHealth(**response_data)
    
    async def get_system_metrics(self) -> SystemMetrics:
        """系统指标"""
        response_data = await self._make_request("GET", "/system/metrics")
        return SystemMetrics(**response_data)
    
    # ==================== 工具方法 ====================
    
    async def _make_request(self,
                          method: str,
                          endpoint: str,
                          data: Optional[Dict[str, Any]] = None,
                          params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """通用请求方法"""
        await self._ensure_session()
        
        url = f"{self.config.base_url}{endpoint}"
        
        for attempt in range(self.config.retries):
            try:
                if self.config.debug:
                    print(f"[VoiceHelper SDK] {method} {url}")
                    if data:
                        print(f"[VoiceHelper SDK] Data: {data}")
                
                kwargs = {}
                if data:
                    kwargs['json'] = data
                if params:
                    kwargs['params'] = params
                
                async with self.session.request(method, url, **kwargs) as response:
                    if response.status >= 400:
                        await self._handle_error_response(response)
                    
                    response_data = await response.json()
                    
                    if self.config.debug:
                        print(f"[VoiceHelper SDK] Response: {response_data}")
                    
                    return response_data
                    
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.config.retries - 1:
                    raise NetworkError(f"网络请求失败: {str(e)}")
                
                # 指数退避
                delay = min(2 ** attempt, 10)
                await asyncio.sleep(delay)
        
        raise NetworkError("达到最大重试次数")
    
    async def _handle_error_response(self, response: aiohttp.ClientResponse):
        """处理错误响应"""
        try:
            error_data = await response.json()
            error = error_data.get('error', {})
            
            error_code = error.get('code', 'unknown_error')
            error_message = error.get('message', f'HTTP {response.status}')
            error_type = error.get('type', 'api_error')
            
            if response.status == 401:
                raise AuthenticationError(error_message, error_code)
            elif response.status == 400:
                raise APIError(error_message, error_code, error_type)
            else:
                raise VoiceHelperError(error_message, error_code, error_type)
                
        except (json.JSONDecodeError, KeyError):
            raise VoiceHelperError(f"HTTP {response.status} {response.reason}")
    
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        self.config.api_key = api_key
        self._base_headers["X-API-Key"] = api_key
    
    def set_base_url(self, base_url: str):
        """设置基础URL"""
        self.config.base_url = base_url
    
    def set_debug(self, debug: bool):
        """设置调试模式"""
        self.config.debug = debug
