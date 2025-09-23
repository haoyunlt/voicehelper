"""
VoiceHelper AI Python SDK
完整的Python客户端SDK

@version 1.9.0
@author VoiceHelper Team
@license MIT
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Union, AsyncGenerator, BinaryIO, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

import httpx
import websockets
from pydantic import BaseModel, Field, validator

# ==================== 配置和模型 ====================

@dataclass
class VoiceHelperConfig:
    """VoiceHelper SDK配置"""
    api_key: str
    base_url: str = "https://api.voicehelper.ai/v1"
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    debug: bool = False


class Message(BaseModel):
    """消息模型"""
    role: str = Field(..., description="消息角色")
    content: Union[str, List[Dict[str, Any]]] = Field(..., description="消息内容")
    name: Optional[str] = Field(None, description="消息发送者名称")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="工具调用列表")
    tool_call_id: Optional[str] = Field(None, description="工具调用ID")

    @validator('role')
    def validate_role(cls, v):
        if v not in ['system', 'user', 'assistant', 'tool']:
            raise ValueError('role must be one of: system, user, assistant, tool')
        return v


class ChatCompletionRequest(BaseModel):
    """对话完成请求"""
    messages: List[Message] = Field(..., description="对话消息列表")
    model: str = Field("gpt-4", description="使用的模型")
    temperature: Optional[float] = Field(0.7, ge=0, le=2, description="随机性控制")
    max_tokens: Optional[int] = Field(None, ge=1, description="最大生成token数")
    stream: bool = Field(False, description="是否流式返回")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="可用工具列表")
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(None, description="工具选择策略")


class ChatCompletionResponse(BaseModel):
    """对话完成响应"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None


class TranscriptionRequest(BaseModel):
    """语音转录请求"""
    model: str = Field("whisper-1", description="使用的模型")
    language: Optional[str] = Field(None, description="语言代码")
    response_format: str = Field("json", description="响应格式")


class TranscriptionResponse(BaseModel):
    """语音转录响应"""
    text: str
    language: Optional[str] = None
    duration: Optional[float] = None
    segments: Optional[List[Dict[str, Any]]] = None


class SynthesisRequest(BaseModel):
    """语音合成请求"""
    text: str = Field(..., max_length=4096, description="要合成的文本")
    voice: str = Field("alloy", description="语音类型")
    response_format: str = Field("mp3", description="音频格式")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="语速")
    emotion: str = Field("neutral", description="情感类型")

    @validator('voice')
    def validate_voice(cls, v):
        valid_voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer', 'zh-female-1', 'zh-male-1']
        if v not in valid_voices:
            raise ValueError(f'voice must be one of: {valid_voices}')
        return v

    @validator('emotion')
    def validate_emotion(cls, v):
        valid_emotions = ['neutral', 'happy', 'sad', 'angry', 'excited', 'calm']
        if v not in valid_emotions:
            raise ValueError(f'emotion must be one of: {valid_emotions}')
        return v


class VoiceHelperError(Exception):
    """VoiceHelper SDK异常"""
    
    def __init__(self, message: str, error_type: str = "client_error", code: str = "unknown_error", param: Optional[str] = None):
        self.message = message
        self.error_type = error_type
        self.code = code
        self.param = param
        super().__init__(self.message)

    def __str__(self):
        return f"VoiceHelperError({self.error_type}): {self.message} (code: {self.code})"


# ==================== 主要SDK类 ====================

class VoiceHelperSDK:
    """VoiceHelper AI Python SDK主类"""

    def __init__(self, config: VoiceHelperConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if self.config.debug:
            logging.basicConfig(level=logging.DEBUG)
        
        # 创建HTTP客户端
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "VoiceHelper-Python-SDK/1.9.0",
            }
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """关闭客户端"""
        await self.client.aclose()

    def _format_error(self, response: httpx.Response) -> VoiceHelperError:
        """格式化错误响应"""
        try:
            error_data = response.json().get("error", {})
            return VoiceHelperError(
                message=error_data.get("message", "Unknown error"),
                error_type=error_data.get("type", "api_error"),
                code=error_data.get("code", str(response.status_code)),
                param=error_data.get("param")
            )
        except Exception:
            return VoiceHelperError(
                message=f"HTTP {response.status_code}: {response.text}",
                error_type="http_error",
                code=str(response.status_code)
            )

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """带重试的请求"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if self.config.debug:
                    self.logger.debug(f"Request attempt {attempt + 1}: {method} {url}")
                
                response = await self.client.request(method, url, **kwargs)
                
                if response.is_success:
                    return response
                
                # 检查是否应该重试
                if response.status_code in [408, 429, 500, 502, 503, 504] and attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                
                raise self._format_error(response)
                
            except httpx.RequestError as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                break
        
        raise VoiceHelperError(
            message=f"Request failed after {self.config.max_retries + 1} attempts: {str(last_exception)}",
            error_type="network_error",
            code="max_retries_exceeded"
        )

    # ==================== 对话接口 ====================

    async def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """创建对话完成"""
        response = await self._request_with_retry(
            "POST",
            "/chat/completions",
            json=request.dict(exclude_none=True)
        )
        
        return ChatCompletionResponse(**response.json())

    async def create_chat_completion_stream(
        self,
        request: ChatCompletionRequest,
        chunk_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """创建流式对话完成"""
        stream_request = request.copy()
        stream_request.stream = True
        
        async with self.client.stream(
            "POST",
            "/chat/completions",
            json=stream_request.dict(exclude_none=True),
            headers={"Accept": "text/event-stream"}
        ) as response:
            if not response.is_success:
                raise self._format_error(response)
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # 移除 "data: " 前缀
                    
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data)
                        if chunk_callback:
                            chunk_callback(chunk)
                        yield chunk
                    except json.JSONDecodeError:
                        continue

    async def list_conversations(
        self,
        limit: int = 20,
        offset: int = 0,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取对话列表"""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        
        response = await self._request_with_retry("GET", "/chat/conversations", params=params)
        return response.json()

    async def create_conversation(
        self,
        title: Optional[str] = None,
        initial_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建新对话"""
        data = {}
        if title:
            data["title"] = title
        if initial_message:
            data["initial_message"] = initial_message
        
        response = await self._request_with_retry("POST", "/chat/conversations", json=data)
        return response.json()

    async def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话详情"""
        response = await self._request_with_retry("GET", f"/chat/conversations/{conversation_id}")
        return response.json()

    async def delete_conversation(self, conversation_id: str) -> None:
        """删除对话"""
        await self._request_with_retry("DELETE", f"/chat/conversations/{conversation_id}")

    # ==================== 语音接口 ====================

    async def transcribe_audio(
        self,
        audio_file: Union[BinaryIO, bytes, Path],
        request: Optional[TranscriptionRequest] = None
    ) -> TranscriptionResponse:
        """语音转文字"""
        if request is None:
            request = TranscriptionRequest()
        
        # 准备文件数据
        if isinstance(audio_file, Path):
            files = {"file": audio_file.open("rb")}
        elif isinstance(audio_file, bytes):
            files = {"file": ("audio.wav", audio_file, "audio/wav")}
        else:
            files = {"file": audio_file}
        
        # 准备表单数据
        data = request.dict(exclude_none=True)
        
        response = await self._request_with_retry(
            "POST",
            "/voice/transcribe",
            files=files,
            data=data
        )
        
        return TranscriptionResponse(**response.json())

    async def synthesize_text(self, request: SynthesisRequest) -> bytes:
        """文字转语音"""
        response = await self._request_with_retry(
            "POST",
            "/voice/synthesize",
            json=request.dict(exclude_none=True)
        )
        
        return response.content

    async def create_realtime_voice_connection(
        self,
        model: str = "gpt-4-voice",
        voice: str = "alloy"
    ) -> websockets.WebSocketServerProtocol:
        """创建实时语音连接"""
        ws_url = self.config.base_url.replace("http", "ws") + "/voice/realtime"
        headers = {"Authorization": f"Bearer {self.config.api_key}"}
        
        return await websockets.connect(
            f"{ws_url}?model={model}&voice={voice}",
            extra_headers=headers
        )

    # ==================== 知识库接口 ====================

    async def list_datasets(
        self,
        limit: int = 20,
        offset: int = 0
    ) -> Dict[str, Any]:
        """获取数据集列表"""
        params = {"limit": limit, "offset": offset}
        response = await self._request_with_retry("GET", "/knowledge/datasets", params=params)
        return response.json()

    async def create_dataset(
        self,
        name: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建数据集"""
        data = {"name": name}
        if description:
            data["description"] = description
        
        response = await self._request_with_retry("POST", "/knowledge/datasets", json=data)
        return response.json()

    async def upload_document(
        self,
        dataset_id: str,
        file: Union[BinaryIO, bytes, Path],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """上传文档"""
        # 准备文件数据
        if isinstance(file, Path):
            files = {"file": file.open("rb")}
        elif isinstance(file, bytes):
            files = {"file": ("document.txt", file, "text/plain")}
        else:
            files = {"file": file}
        
        # 准备表单数据
        data = {}
        if metadata:
            data["metadata"] = json.dumps(metadata)
        
        response = await self._request_with_retry(
            "POST",
            f"/knowledge/datasets/{dataset_id}/documents",
            files=files,
            data=data
        )
        
        return response.json()

    async def search_knowledge(
        self,
        query: str,
        datasets: Optional[List[str]] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """知识检索"""
        data = {"query": query, "limit": limit}
        if datasets:
            data["datasets"] = datasets
        
        response = await self._request_with_retry("POST", "/knowledge/search", json=data)
        return response.json()

    # ==================== 服务集成接口 ====================

    async def list_available_services(
        self,
        category: Optional[str] = None,
        status: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取可用服务列表"""
        params = {}
        if category:
            params["category"] = category
        if status:
            params["status"] = status
        
        response = await self._request_with_retry("GET", "/integrations/services", params=params)
        return response.json()

    async def list_connections(self) -> Dict[str, Any]:
        """获取已连接服务"""
        response = await self._request_with_retry("GET", "/integrations/connections")
        return response.json()

    async def create_connection(
        self,
        service_id: str,
        config: Dict[str, Any],
        name: Optional[str] = None
    ) -> Dict[str, Any]:
        """创建服务连接"""
        data = {"service_id": service_id, "config": config}
        if name:
            data["name"] = name
        
        response = await self._request_with_retry("POST", "/integrations/connections", json=data)
        return response.json()

    async def execute_service_operation(
        self,
        connection_id: str,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """执行服务操作"""
        data = {"operation": operation}
        if parameters:
            data["parameters"] = parameters
        
        response = await self._request_with_retry(
            "POST",
            f"/integrations/connections/{connection_id}/execute",
            json=data
        )
        
        return response.json()

    # ==================== 用户管理接口 ====================

    async def get_user_profile(self) -> Dict[str, Any]:
        """获取用户资料"""
        response = await self._request_with_retry("GET", "/users/profile")
        return response.json()

    async def update_user_profile(
        self,
        name: Optional[str] = None,
        avatar_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """更新用户资料"""
        data = {}
        if name:
            data["name"] = name
        if avatar_url:
            data["avatar_url"] = avatar_url
        
        response = await self._request_with_retry("PUT", "/users/profile", json=data)
        return response.json()

    async def get_user_usage(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取使用统计"""
        params = {}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        
        response = await self._request_with_retry("GET", "/users/usage", params=params)
        return response.json()

    # ==================== 工具方法 ====================

    def set_api_key(self, api_key: str) -> None:
        """设置API密钥"""
        self.config.api_key = api_key
        self.client.headers["Authorization"] = f"Bearer {api_key}"

    def set_base_url(self, base_url: str) -> None:
        """设置基础URL"""
        self.config.base_url = base_url
        self.client.base_url = base_url

    def get_config(self) -> VoiceHelperConfig:
        """获取当前配置"""
        return self.config

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        response = await self._request_with_retry("GET", "/health")
        return response.json()


# ==================== 同步版本SDK ====================

class VoiceHelperSyncSDK:
    """VoiceHelper AI Python SDK同步版本"""

    def __init__(self, config: VoiceHelperConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if self.config.debug:
            logging.basicConfig(level=logging.DEBUG)
        
        # 创建同步HTTP客户端
        self.client = httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "VoiceHelper-Python-SDK/1.9.0",
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """关闭客户端"""
        self.client.close()

    def _format_error(self, response: httpx.Response) -> VoiceHelperError:
        """格式化错误响应"""
        try:
            error_data = response.json().get("error", {})
            return VoiceHelperError(
                message=error_data.get("message", "Unknown error"),
                error_type=error_data.get("type", "api_error"),
                code=error_data.get("code", str(response.status_code)),
                param=error_data.get("param")
            )
        except Exception:
            return VoiceHelperError(
                message=f"HTTP {response.status_code}: {response.text}",
                error_type="http_error",
                code=str(response.status_code)
            )

    def _request_with_retry(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> httpx.Response:
        """带重试的请求"""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                if self.config.debug:
                    self.logger.debug(f"Request attempt {attempt + 1}: {method} {url}")
                
                response = self.client.request(method, url, **kwargs)
                
                if response.is_success:
                    return response
                
                # 检查是否应该重试
                if response.status_code in [408, 429, 500, 502, 503, 504] and attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                
                raise self._format_error(response)
                
            except httpx.RequestError as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                break
        
        raise VoiceHelperError(
            message=f"Request failed after {self.config.max_retries + 1} attempts: {str(last_exception)}",
            error_type="network_error",
            code="max_retries_exceeded"
        )

    # 同步版本的所有方法...
    def create_chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """创建对话完成"""
        response = self._request_with_retry(
            "POST",
            "/chat/completions",
            json=request.dict(exclude_none=True)
        )
        
        return ChatCompletionResponse(**response.json())

    def transcribe_audio(
        self,
        audio_file: Union[BinaryIO, bytes, Path],
        request: Optional[TranscriptionRequest] = None
    ) -> TranscriptionResponse:
        """语音转文字"""
        if request is None:
            request = TranscriptionRequest()
        
        # 准备文件数据
        if isinstance(audio_file, Path):
            files = {"file": audio_file.open("rb")}
        elif isinstance(audio_file, bytes):
            files = {"file": ("audio.wav", audio_file, "audio/wav")}
        else:
            files = {"file": audio_file}
        
        # 准备表单数据
        data = request.dict(exclude_none=True)
        
        response = self._request_with_retry(
            "POST",
            "/voice/transcribe",
            files=files,
            data=data
        )
        
        return TranscriptionResponse(**response.json())

    def synthesize_text(self, request: SynthesisRequest) -> bytes:
        """文字转语音"""
        response = self._request_with_retry(
            "POST",
            "/voice/synthesize",
            json=request.dict(exclude_none=True)
        )
        
        return response.content

    # 其他同步方法...


# ==================== 便捷函数 ====================

def create_voicehelper_client(config: VoiceHelperConfig) -> VoiceHelperSDK:
    """创建VoiceHelper SDK实例"""
    return VoiceHelperSDK(config)


def create_voicehelper_sync_client(config: VoiceHelperConfig) -> VoiceHelperSyncSDK:
    """创建VoiceHelper SDK同步实例"""
    return VoiceHelperSyncSDK(config)


async def chat(
    api_key: str,
    message: str,
    model: str = "gpt-4",
    temperature: float = 0.7,
    base_url: Optional[str] = None
) -> str:
    """简单的聊天函数"""
    config = VoiceHelperConfig(api_key=api_key, base_url=base_url or "https://api.voicehelper.ai/v1")
    
    async with VoiceHelperSDK(config) as client:
        request = ChatCompletionRequest(
            messages=[Message(role="user", content=message)],
            model=model,
            temperature=temperature
        )
        
        response = await client.create_chat_completion(request)
        return response.choices[0]["message"]["content"]


async def transcribe(
    api_key: str,
    audio_file: Union[BinaryIO, bytes, Path],
    model: str = "whisper-1",
    language: Optional[str] = None,
    base_url: Optional[str] = None
) -> str:
    """简单的语音转文字函数"""
    config = VoiceHelperConfig(api_key=api_key, base_url=base_url or "https://api.voicehelper.ai/v1")
    
    async with VoiceHelperSDK(config) as client:
        request = TranscriptionRequest(model=model, language=language)
        response = await client.transcribe_audio(audio_file, request)
        return response.text


async def synthesize(
    api_key: str,
    text: str,
    voice: str = "alloy",
    speed: float = 1.0,
    emotion: str = "neutral",
    base_url: Optional[str] = None
) -> bytes:
    """简单的文字转语音函数"""
    config = VoiceHelperConfig(api_key=api_key, base_url=base_url or "https://api.voicehelper.ai/v1")
    
    async with VoiceHelperSDK(config) as client:
        request = SynthesisRequest(text=text, voice=voice, speed=speed, emotion=emotion)
        return await client.synthesize_text(request)


# 同步版本的便捷函数
def chat_sync(
    api_key: str,
    message: str,
    model: str = "gpt-4",
    temperature: float = 0.7,
    base_url: Optional[str] = None
) -> str:
    """简单的聊天函数(同步版本)"""
    config = VoiceHelperConfig(api_key=api_key, base_url=base_url or "https://api.voicehelper.ai/v1")
    
    with VoiceHelperSyncSDK(config) as client:
        request = ChatCompletionRequest(
            messages=[Message(role="user", content=message)],
            model=model,
            temperature=temperature
        )
        
        response = client.create_chat_completion(request)
        return response.choices[0]["message"]["content"]


def transcribe_sync(
    api_key: str,
    audio_file: Union[BinaryIO, bytes, Path],
    model: str = "whisper-1",
    language: Optional[str] = None,
    base_url: Optional[str] = None
) -> str:
    """简单的语音转文字函数(同步版本)"""
    config = VoiceHelperConfig(api_key=api_key, base_url=base_url or "https://api.voicehelper.ai/v1")
    
    with VoiceHelperSyncSDK(config) as client:
        request = TranscriptionRequest(model=model, language=language)
        response = client.transcribe_audio(audio_file, request)
        return response.text


def synthesize_sync(
    api_key: str,
    text: str,
    voice: str = "alloy",
    speed: float = 1.0,
    emotion: str = "neutral",
    base_url: Optional[str] = None
) -> bytes:
    """简单的文字转语音函数(同步版本)"""
    config = VoiceHelperConfig(api_key=api_key, base_url=base_url or "https://api.voicehelper.ai/v1")
    
    with VoiceHelperSyncSDK(config) as client:
        request = SynthesisRequest(text=text, voice=voice, speed=speed, emotion=emotion)
        return client.synthesize_text(request)


# ==================== 导出 ====================

__all__ = [
    # 主要类
    "VoiceHelperSDK",
    "VoiceHelperSyncSDK",
    "VoiceHelperConfig",
    "VoiceHelperError",
    
    # 模型类
    "Message",
    "ChatCompletionRequest", 
    "ChatCompletionResponse",
    "TranscriptionRequest",
    "TranscriptionResponse", 
    "SynthesisRequest",
    
    # 便捷函数
    "create_voicehelper_client",
    "create_voicehelper_sync_client",
    "chat",
    "transcribe", 
    "synthesize",
    "chat_sync",
    "transcribe_sync",
    "synthesize_sync",
]
