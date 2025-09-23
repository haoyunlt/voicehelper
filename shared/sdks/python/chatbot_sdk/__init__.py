"""
Chatbot Python SDK

Official Python SDK for the Chatbot API.

Example:
    >>> from chatbot_sdk import ChatbotClient
    >>> 
    >>> client = ChatbotClient(
    ...     api_key="your-api-key",
    ...     base_url="https://api.chatbot.ai/v1"
    ... )
    >>> 
    >>> # 创建对话
    >>> conversation = await client.conversations.create(title="新对话")
    >>> 
    >>> # 发送消息
    >>> async for event in client.messages.send_stream(
    ...     conversation.id,
    ...     content="你好，世界！"
    ... ):
    ...     if event.type == "delta":
    ...         print(event.content, end="")
    ...     elif event.type == "done":
    ...         print(f"\n消息ID: {event.message_id}")
"""

from .client import ChatbotClient
from .types import *
from .errors import *

__version__ = "1.6.0"
__author__ = "Chatbot Team"
__email__ = "support@chatbot.ai"

__all__ = [
    "ChatbotClient",
    # Types
    "User",
    "Conversation", 
    "Message",
    "Reference",
    "Dataset",
    "Document",
    "SearchResult",
    "StreamEvent",
    # Requests
    "LoginRequest",
    "CreateConversationRequest",
    "SendMessageRequest",
    "CreateDatasetRequest",
    "SearchRequest",
    # Responses
    "LoginResponse",
    "SearchResponse",
    "HealthResponse",
    # Errors
    "ChatbotError",
    "APIError",
    "NetworkError",
    "ValidationError",
    "ConfigError",
    "StreamError",
    "FileError",
]
