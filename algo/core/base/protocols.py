"""
V2架构协议定义
定义流式回调、ASR、TTS等核心协议接口
"""

from typing import Iterable, Iterator, Protocol, Callable, Optional


class StreamCallback(Protocol):
    """流式回调协议"""
    
    def __call__(self, event: str, payload: dict) -> None:
        """
        发送流式事件
        
        Args:
            event: 事件类型
            payload: 事件数据
        """
        ...


class AsrStream(Protocol):
    """ASR 流式处理协议"""
    
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
        ...
    
    def feed(self, seq: int, chunk: bytes) -> None:
        """
        输入音频数据
        
        Args:
            seq: 序列号
            chunk: 音频数据块
        """
        ...
    
    def stop(self) -> None:
        """停止ASR会话"""
        ...
    
    def on_partial(self, cb: Callable[[int, str], None]) -> None:
        """
        设置部分识别结果回调
        
        Args:
            cb: 回调函数(序列号, 部分文本)
        """
        ...
    
    def on_final(self, cb: Callable[[int, str, float], None]) -> None:
        """
        设置最终识别结果回调
        
        Args:
            cb: 回调函数(序列号, 最终文本, 置信度)
        """
        ...


class TtsStream(Protocol):
    """TTS 流式合成协议"""
    
    def synthesize(self, text_iter: Iterable[str]) -> Iterator[bytes]:
        """
        流式文本转语音
        
        Args:
            text_iter: 文本迭代器
            
        Yields:
            音频数据块
        """
        ...
    
    def cancel(self, request_id: str) -> None:
        """
        取消TTS请求
        
        Args:
            request_id: 请求ID
        """
        ...
