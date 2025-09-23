"""
Chatbot Python SDK Client
"""

import asyncio
import json
from typing import Optional, Dict, Any, List, AsyncGenerator, Union
from pathlib import Path

import httpx
import websockets
from websockets.exceptions import WebSocketException

from .types import *
from .errors import *


class ChatbotClient:
    """
    Chatbot API客户端
    
    Args:
        api_key: API密钥或JWT令牌
        base_url: API基础URL，默认为 https://api.chatbot.ai/v1
        timeout: 请求超时时间(秒)，默认为30
        tenant_id: 租户ID，默认为 default
    
    Example:
        >>> client = ChatbotClient(api_key="your-api-key")
        >>> conversation = await client.conversations.create(title="新对话")
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.chatbot.ai/v1",
        timeout: float = 30.0,
        tenant_id: str = "default"
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.tenant_id = tenant_id
        
        # 创建HTTP客户端
        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Tenant-ID": self.tenant_id,
            }
        )
        
        # 初始化子模块
        self.auth = AuthAPI(self)
        self.conversations = ConversationsAPI(self)
        self.messages = MessagesAPI(self)
        self.voice = VoiceAPI(self)
        self.datasets = DatasetsAPI(self)
        self.search = SearchAPI(self)
        self.system = SystemAPI(self)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """关闭HTTP客户端"""
        await self._http_client.aclose()
    
    async def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> httpx.Response:
        """发送HTTP请求"""
        try:
            response = await self._http_client.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            error_data = {}
            try:
                error_data = e.response.json()
            except:
                pass
            
            raise APIError(
                message=error_data.get("error", str(e)),
                status_code=e.response.status_code,
                error_code=error_data.get("code"),
                details=error_data.get("details")
            )
        except httpx.RequestError as e:
            raise NetworkError(f"Network error: {str(e)}")


class AuthAPI:
    """认证相关API"""
    
    def __init__(self, client: ChatbotClient):
        self.client = client
    
    async def wechat_login(self, request: LoginRequest) -> LoginResponse:
        """微信小程序登录"""
        response = await self.client._request(
            "POST",
            "/auth/wechat/login",
            json=request.dict()
        )
        return LoginResponse(**response.json())
    
    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """刷新访问令牌"""
        response = await self.client._request(
            "POST", 
            "/auth/refresh",
            json={"refresh_token": refresh_token}
        )
        return TokenResponse(**response.json())


class ConversationsAPI:
    """对话相关API"""
    
    def __init__(self, client: ChatbotClient):
        self.client = client
    
    async def list(
        self,
        page: int = 1,
        limit: int = 20,
        status: Optional[str] = None
    ) -> PaginatedResponse[Conversation]:
        """获取对话列表"""
        params = {"page": page, "limit": limit}
        if status:
            params["status"] = status
        
        response = await self.client._request("GET", "/conversations", params=params)
        data = response.json()
        
        return PaginatedResponse(
            data=[Conversation(**item) for item in data["conversations"]],
            pagination=Pagination(**data["pagination"])
        )
    
    async def create(self, request: CreateConversationRequest) -> Conversation:
        """创建对话"""
        response = await self.client._request(
            "POST",
            "/conversations", 
            json=request.dict(exclude_none=True)
        )
        return Conversation(**response.json())
    
    async def get(self, conversation_id: str) -> Conversation:
        """获取对话详情"""
        response = await self.client._request("GET", f"/conversations/{conversation_id}")
        return Conversation(**response.json())
    
    async def update(
        self,
        conversation_id: str,
        request: UpdateConversationRequest
    ) -> Conversation:
        """更新对话"""
        response = await self.client._request(
            "PUT",
            f"/conversations/{conversation_id}",
            json=request.dict(exclude_none=True)
        )
        return Conversation(**response.json())
    
    async def delete(self, conversation_id: str) -> None:
        """删除对话"""
        await self.client._request("DELETE", f"/conversations/{conversation_id}")


class MessagesAPI:
    """消息相关API"""
    
    def __init__(self, client: ChatbotClient):
        self.client = client
    
    async def list(
        self,
        conversation_id: str,
        page: int = 1,
        limit: int = 20,
        before: Optional[str] = None
    ) -> PaginatedResponse[Message]:
        """获取消息列表"""
        params = {"page": page, "limit": limit}
        if before:
            params["before"] = before
        
        response = await self.client._request(
            "GET",
            f"/conversations/{conversation_id}/messages",
            params=params
        )
        data = response.json()
        
        return PaginatedResponse(
            data=[Message(**item) for item in data["messages"]],
            pagination=Pagination(**data["pagination"])
        )
    
    async def send(
        self,
        conversation_id: str,
        request: SendMessageRequest
    ) -> Message:
        """发送消息 (非流式)"""
        response = await self.client._request(
            "POST",
            f"/conversations/{conversation_id}/messages",
            json={**request.dict(), "stream": False}
        )
        return Message(**response.json())
    
    async def send_stream(
        self,
        conversation_id: str,
        request: SendMessageRequest
    ) -> AsyncGenerator[StreamEvent, None]:
        """发送消息 (流式)"""
        async with self.client._http_client.stream(
            "POST",
            f"/conversations/{conversation_id}/messages",
            json={**request.dict(), "stream": True}
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        yield StreamEvent(**data)
                    except json.JSONDecodeError:
                        continue


class VoiceAPI:
    """语音相关API"""
    
    def __init__(self, client: ChatbotClient):
        self.client = client
    
    async def transcribe(
        self,
        audio_file: Union[str, Path, bytes],
        language: str = "zh-CN"
    ) -> TranscribeResponse:
        """语音转文字"""
        if isinstance(audio_file, (str, Path)):
            with open(audio_file, "rb") as f:
                audio_data = f.read()
        else:
            audio_data = audio_file
        
        files = {"audio": audio_data}
        data = {"language": language}
        
        response = await self.client._request(
            "POST",
            "/voice/transcribe",
            files=files,
            data=data
        )
        return TranscribeResponse(**response.json())
    
    async def synthesize(
        self,
        text: str,
        voice: str = "female",
        speed: float = 1.0
    ) -> bytes:
        """文字转语音"""
        response = await self.client._request(
            "POST",
            "/voice/synthesize",
            json={
                "text": text,
                "voice": voice,
                "speed": speed
            }
        )
        return response.content


class DatasetsAPI:
    """数据集相关API"""
    
    def __init__(self, client: ChatbotClient):
        self.client = client
    
    async def list(
        self,
        page: int = 1,
        limit: int = 20
    ) -> PaginatedResponse[Dataset]:
        """获取数据集列表"""
        response = await self.client._request(
            "GET",
            "/datasets",
            params={"page": page, "limit": limit}
        )
        data = response.json()
        
        return PaginatedResponse(
            data=[Dataset(**item) for item in data["datasets"]],
            pagination=Pagination(**data["pagination"])
        )
    
    async def create(self, request: CreateDatasetRequest) -> Dataset:
        """创建数据集"""
        response = await self.client._request(
            "POST",
            "/datasets",
            json=request.dict()
        )
        return Dataset(**response.json())
    
    async def get(self, dataset_id: str) -> Dataset:
        """获取数据集详情"""
        response = await self.client._request("GET", f"/datasets/{dataset_id}")
        return Dataset(**response.json())
    
    async def upload_document(
        self,
        dataset_id: str,
        file_path: Union[str, Path],
        name: Optional[str] = None
    ) -> Document:
        """上传文档"""
        file_path = Path(file_path)
        
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            data = {}
            if name:
                data["name"] = name
            
            response = await self.client._request(
                "POST",
                f"/datasets/{dataset_id}/documents",
                files=files,
                data=data
            )
        
        return Document(**response.json())


class SearchAPI:
    """搜索相关API"""
    
    def __init__(self, client: ChatbotClient):
        self.client = client
    
    async def query(self, request: SearchRequest) -> SearchResponse:
        """知识搜索"""
        response = await self.client._request(
            "POST",
            "/search",
            json=request.dict(exclude_none=True)
        )
        data = response.json()
        
        return SearchResponse(
            results=[SearchResult(**item) for item in data["results"]],
            total=data["total"]
        )


class SystemAPI:
    """系统相关API"""
    
    def __init__(self, client: ChatbotClient):
        self.client = client
    
    async def health(self) -> HealthResponse:
        """健康检查"""
        response = await self.client._request("GET", "/health")
        return HealthResponse(**response.json())
    
    async def metrics(self) -> str:
        """获取系统指标"""
        response = await self.client._request(
            "GET",
            "/metrics",
            headers={"Accept": "text/plain"}
        )
        return response.text


class WebSocketClient:
    """WebSocket客户端 (用于实时通信)"""
    
    def __init__(
        self,
        client: ChatbotClient,
        conversation_id: str
    ):
        self.client = client
        self.conversation_id = conversation_id
        self.ws_url = (
            client.base_url.replace("http", "ws") + 
            f"/conversations/{conversation_id}/ws"
        )
        self.websocket = None
    
    async def connect(self):
        """连接WebSocket"""
        headers = {
            "Authorization": f"Bearer {self.client.api_key}",
            "X-Tenant-ID": self.client.tenant_id,
        }
        
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=headers
            )
        except Exception as e:
            raise NetworkError(f"WebSocket connection failed: {str(e)}")
    
    async def send_message(self, message: Dict[str, Any]):
        """发送WebSocket消息"""
        if not self.websocket:
            raise ChatbotError("WebSocket not connected")
        
        try:
            await self.websocket.send(json.dumps(message))
        except WebSocketException as e:
            raise NetworkError(f"Failed to send WebSocket message: {str(e)}")
    
    async def receive_messages(self) -> AsyncGenerator[Dict[str, Any], None]:
        """接收WebSocket消息"""
        if not self.websocket:
            raise ChatbotError("WebSocket not connected")
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    yield data
                except json.JSONDecodeError:
                    continue
        except WebSocketException as e:
            raise NetworkError(f"WebSocket receive error: {str(e)}")
    
    async def close(self):
        """关闭WebSocket连接"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
