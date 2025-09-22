"""
ASR/TTS适配器基类
定义ASR和TTS适配器的抽象接口
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Iterator, Callable, Optional, Dict, Any
from pydantic import BaseModel

from ..base.protocols import AsrStream, TtsStream
from ..base.mixins import RetryableMixin, ObservableMixin


class BaseAsrAdapter(BaseModel, AsrStream, RetryableMixin, ObservableMixin):
    """ASR适配器基类"""
    
    provider: str = "base"
    language: str = "zh-CN"
    sample_rate: int = 16000
    
    # 回调函数
    _partial_callback: Optional[Callable[[int, str], None]] = None
    _final_callback: Optional[Callable[[int, str, float], None]] = None
    
    # 会话状态
    _session_id: Optional[str] = None
    _is_active: bool = False
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def start(self, sr: int, codec: str, lang: str) -> str:
        """
        开始ASR会话
        
        Args:
            sr: 采样率
            codec: 音频编码
            lang: 语言代码
            
        Returns:
            会话ID
        """
        pass
    
    @abstractmethod
    def feed(self, seq: int, chunk: bytes) -> None:
        """
        输入音频数据
        
        Args:
            seq: 序列号
            chunk: 音频数据块
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """停止ASR会话"""
        pass
    
    def on_partial(self, cb: Callable[[int, str], None]) -> None:
        """
        设置部分识别结果回调
        
        Args:
            cb: 回调函数(序列号, 部分文本)
        """
        self._partial_callback = cb
    
    def on_final(self, cb: Callable[[int, str, float], None]) -> None:
        """
        设置最终识别结果回调
        
        Args:
            cb: 回调函数(序列号, 最终文本, 置信度)
        """
        self._final_callback = cb
    
    def _emit_partial(self, seq: int, text: str):
        """发送部分识别结果"""
        if self._partial_callback:
            self._partial_callback(seq, text)
    
    def _emit_final(self, seq: int, text: str, confidence: float):
        """发送最终识别结果"""
        if self._final_callback:
            self._final_callback(seq, text, confidence)


class BaseTtsAdapter(BaseModel, TtsStream, RetryableMixin, ObservableMixin):
    """TTS适配器基类"""
    
    provider: str = "base"
    voice: str = "default"
    speed: float = 1.0
    pitch: float = 1.0
    
    # 活跃请求跟踪
    _active_requests: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    def synthesize(self, text_iter: Iterator[str]) -> Iterator[bytes]:
        """
        流式文本转语音
        
        Args:
            text_iter: 文本迭代器
            
        Yields:
            音频数据块
        """
        pass
    
    @abstractmethod
    def cancel(self, request_id: str) -> None:
        """
        取消TTS请求
        
        Args:
            request_id: 请求ID
        """
        pass
    
    def synthesize_text(self, text: str) -> Iterator[bytes]:
        """
        单个文本转语音的便捷方法
        
        Args:
            text: 要转换的文本
            
        Yields:
            音频数据块
        """
        return self.synthesize(iter([text]))
    
    def _generate_request_id(self) -> str:
        """生成请求ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _track_request(self, request_id: str, request_data: Any):
        """跟踪请求"""
        self._active_requests[request_id] = request_data
    
    def _untrack_request(self, request_id: str):
        """取消跟踪请求"""
        self._active_requests.pop(request_id, None)
