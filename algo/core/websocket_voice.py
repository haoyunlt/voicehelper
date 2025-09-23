"""
WebSocket语音流处理器
实现算法服务端的WebSocket语音流接口，支持实时ASR、RAG查询和TTS流式响应
"""

import asyncio
import json
import base64
import time
from typing import Dict, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from datetime import datetime
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = None

try:
    from fastapi import WebSocket
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    WebSocket = None

from loguru import logger

from core.enhanced_voice_services import EnhancedVoiceService
from core.models import VoiceQueryRequest, VoiceQueryResponse
from common.errors import VoiceHelperError, ErrorCode


@dataclass
class WebSocketSession:
    """WebSocket会话"""
    session_id: str
    websocket: Union[WebSocketServerProtocol, WebSocket]
    created_at: datetime
    last_activity: datetime
    audio_buffer: bytes = b""
    transcript_buffer: str = ""
    is_processing: bool = False
    request_id: Optional[str] = None


class WebSocketVoiceHandler:
    """WebSocket语音处理器"""
    
    def __init__(self, voice_service: EnhancedVoiceService):
        self.voice_service = voice_service
        self.active_sessions: Dict[str, WebSocketSession] = {}
        self.cleanup_interval = 300  # 5分钟清理一次
        
        # 清理任务将在事件循环启动后创建
        self._cleanup_task = None
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """处理WebSocket连接（websockets库）"""
        session_id = self._generate_session_id()
        session = WebSocketSession(
            session_id=session_id,
            websocket=websocket,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        
        try:
            logger.info(f"WebSocket语音连接建立: {session_id}")
            
            # 发送连接确认
            await self._send_message(session, {
                "type": "connected",
                "session_id": session_id,
                "timestamp": int(time.time() * 1000)
            })
            
            # 处理消息循环
            async for message in websocket:
                await self._handle_message(session, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket连接关闭: {session_id}")
        except Exception as e:
            logger.error(f"WebSocket连接错误: {e}")
        finally:
            # 清理会话
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    async def handle_websocket_connection(self, websocket: WebSocket):
        """处理FastAPI WebSocket连接"""
        session_id = self._generate_session_id()
        session = WebSocketSession(
            session_id=session_id,
            websocket=websocket,
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        
        try:
            logger.info(f"FastAPI WebSocket语音连接建立: {session_id}")
            
            # 发送连接确认
            await self._send_message(session, {
                "type": "connected",
                "session_id": session_id,
                "timestamp": int(time.time() * 1000)
            })
            
            # 处理消息循环
            while True:
                try:
                    message = await websocket.receive()
                    if message["type"] == "websocket.receive":
                        if "bytes" in message:
                            await self._handle_message(session, message["bytes"])
                        elif "text" in message:
                            await self._handle_message(session, message["text"])
                    elif message["type"] == "websocket.disconnect":
                        break
                except Exception as e:
                    logger.error(f"接收WebSocket消息失败: {e}")
                    break
                
        except Exception as e:
            logger.error(f"FastAPI WebSocket连接错误: {e}")
        finally:
            # 清理会话
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
    
    async def _handle_message(self, session: WebSocketSession, message):
        """处理WebSocket消息"""
        try:
            session.last_activity = datetime.now()
            
            if isinstance(message, bytes):
                # 二进制消息 - 音频数据
                await self._handle_audio_data(session, message)
            else:
                # 文本消息 - 控制命令
                data = json.loads(message)
                await self._handle_control_message(session, data)
                
        except Exception as e:
            logger.error(f"处理WebSocket消息失败: {e}")
            await self._send_error(session, f"Message processing error: {str(e)}")
    
    async def _handle_audio_data(self, session: WebSocketSession, audio_data: bytes):
        """处理音频数据"""
        try:
            # 解析音频帧头部（前20字节）
            if len(audio_data) < 20:
                await self._send_error(session, "Invalid audio frame: too short")
                return
            
            header = self._parse_audio_header(audio_data[:20])
            audio_chunk = audio_data[20:]
            
            # 添加到音频缓冲区
            session.audio_buffer += audio_chunk
            
            # 检查是否有足够的音频数据进行处理
            min_chunk_size = header.get('sample_rate', 16000) * 0.5  # 0.5秒的音频
            
            if len(session.audio_buffer) >= min_chunk_size or header.get('is_final', False):
                await self._process_audio_chunk(session, header.get('is_final', False))
                
        except Exception as e:
            logger.error(f"处理音频数据失败: {e}")
            await self._send_error(session, f"Audio processing error: {str(e)}")
    
    async def _handle_control_message(self, session: WebSocketSession, data: Dict[str, Any]):
        """处理控制消息"""
        message_type = data.get('type')
        
        if message_type == 'start':
            await self._handle_start_command(session, data)
        elif message_type == 'stop':
            await self._handle_stop_command(session, data)
        elif message_type == 'cancel':
            await self._handle_cancel_command(session, data)
        elif message_type == 'ping':
            await self._handle_ping_command(session, data)
        else:
            logger.warning(f"未知控制消息类型: {message_type}")
    
    async def _handle_start_command(self, session: WebSocketSession, data: Dict[str, Any]):
        """处理开始命令"""
        session.request_id = data.get('request_id', self._generate_request_id())
        session.audio_buffer = b""
        session.transcript_buffer = ""
        session.is_processing = False
        
        await self._send_message(session, {
            "type": "started",
            "request_id": session.request_id,
            "session_id": session.session_id
        })
    
    async def _handle_stop_command(self, session: WebSocketSession, data: Dict[str, Any]):
        """处理停止命令"""
        if session.audio_buffer:
            # 处理剩余的音频数据
            await self._process_audio_chunk(session, is_final=True)
        
        await self._send_message(session, {
            "type": "stopped",
            "session_id": session.session_id
        })
    
    async def _handle_cancel_command(self, session: WebSocketSession, data: Dict[str, Any]):
        """处理取消命令"""
        session.is_processing = False
        session.audio_buffer = b""
        session.transcript_buffer = ""
        
        if session.request_id:
            await self.voice_service.cancel_request(session.request_id)
        
        await self._send_message(session, {
            "type": "cancelled",
            "session_id": session.session_id
        })
    
    async def _handle_ping_command(self, session: WebSocketSession, data: Dict[str, Any]):
        """处理心跳命令"""
        await self._send_message(session, {
            "type": "pong",
            "timestamp": int(time.time() * 1000)
        })
    
    async def _process_audio_chunk(self, session: WebSocketSession, is_final: bool = False):
        """处理音频块"""
        if session.is_processing:
            return
        
        session.is_processing = True
        
        try:
            # 创建语音查询请求
            request = VoiceQueryRequest(
                audio_data=base64.b64encode(session.audio_buffer).decode('utf-8'),
                session_id=session.session_id,
                lang="zh-CN",
                is_final=is_final
            )
            
            # 处理语音查询
            async for response in self.voice_service.process_voice_query(request):
                await self._send_voice_response(session, response)
            
            # 清空已处理的音频缓冲区
            if is_final:
                session.audio_buffer = b""
            else:
                # 保留一部分音频用于上下文
                keep_size = len(session.audio_buffer) // 4
                session.audio_buffer = session.audio_buffer[-keep_size:]
                
        except Exception as e:
            logger.error(f"处理音频块失败: {e}")
            await self._send_error(session, f"Audio chunk processing error: {str(e)}")
        finally:
            session.is_processing = False
    
    async def _send_voice_response(self, session: WebSocketSession, response: VoiceQueryResponse):
        """发送语音响应"""
        response_data = {
            "type": response.type,
            "session_id": session.session_id,
            "timestamp": int(time.time() * 1000)
        }
        
        if response.type == "asr_partial":
            response_data["text"] = response.text
            response_data["is_final"] = False
            session.transcript_buffer = response.text
            
        elif response.type == "asr_final":
            response_data["text"] = response.text
            response_data["is_final"] = True
            session.transcript_buffer = response.text
            
        elif response.type == "llm_delta":
            response_data["delta"] = response.text
            
        elif response.type == "refs":
            response_data["refs"] = [
                {
                    "id": ref.id,
                    "source": ref.source,
                    "title": ref.title,
                    "content": ref.content,
                    "score": ref.score
                }
                for ref in (response.refs or [])
            ]
            
        elif response.type == "tts_chunk":
            response_data["seq"] = response.seq
            response_data["pcm"] = response.pcm
            
        elif response.type == "done":
            response_data["final"] = True
            
        elif response.type == "error":
            response_data["error"] = response.error
        
        await self._send_message(session, response_data)
    
    async def _send_message(self, session: WebSocketSession, data: Dict[str, Any]):
        """发送消息到WebSocket"""
        try:
            message = json.dumps(data, ensure_ascii=False)
            
            # 检查WebSocket类型并使用相应的发送方法
            if FASTAPI_AVAILABLE and isinstance(session.websocket, WebSocket):
                # FastAPI WebSocket
                await session.websocket.send_text(message)
            elif WEBSOCKETS_AVAILABLE and isinstance(session.websocket, WebSocketServerProtocol):
                # websockets库
                await session.websocket.send(message)
            else:
                # 通用方法
                await session.websocket.send(message)
                
        except Exception as e:
            logger.error(f"发送WebSocket消息失败: {e}")
    
    async def _send_error(self, session: WebSocketSession, error_message: str):
        """发送错误消息"""
        await self._send_message(session, {
            "type": "error",
            "error": error_message,
            "session_id": session.session_id,
            "timestamp": int(time.time() * 1000)
        })
    
    def _parse_audio_header(self, header_data: bytes) -> Dict[str, Any]:
        """解析音频帧头部"""
        import struct
        
        try:
            # 解析头部格式: timestamp(8) + seq(4) + sample_rate(4) + channels(2) + frame_size(2)
            timestamp, seq, sample_rate, channels, frame_size = struct.unpack('<QIHHI', header_data)
            
            return {
                'timestamp': timestamp,
                'sequence': seq,
                'sample_rate': sample_rate,
                'channels': channels,
                'frame_size': frame_size,
                'is_final': False  # 可以通过其他方式传递
            }
        except struct.error as e:
            logger.error(f"解析音频头部失败: {e}")
            return {
                'timestamp': int(time.time() * 1000),
                'sequence': 0,
                'sample_rate': 16000,
                'channels': 1,
                'frame_size': 640,
                'is_final': False
            }
    
    def _generate_session_id(self) -> str:
        """生成会话ID"""
        import uuid
        return f"ws_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    def _generate_request_id(self) -> str:
        """生成请求ID"""
        import uuid
        return f"req_{uuid.uuid4().hex[:8]}"
    
    async def _cleanup_sessions(self):
        """清理过期会话"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = datetime.now()
                expired_sessions = []
                
                for session_id, session in self.active_sessions.items():
                    # 超过30分钟无活动的会话被认为过期
                    if (current_time - session.last_activity).total_seconds() > 1800:
                        expired_sessions.append(session_id)
                
                # 清理过期会话
                for session_id in expired_sessions:
                    session = self.active_sessions.pop(session_id, None)
                    if session:
                        try:
                            await session.websocket.close()
                        except:
                            pass
                        logger.info(f"清理过期WebSocket会话: {session_id}")
                
            except Exception as e:
                logger.error(f"清理会话任务失败: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        return {
            "total_sessions": len(self.active_sessions),
            "active_sessions": [
                {
                    "session_id": session.session_id,
                    "created_at": session.created_at.isoformat(),
                    "last_activity": session.last_activity.isoformat(),
                    "is_processing": session.is_processing,
                    "buffer_size": len(session.audio_buffer),
                    "transcript_length": len(session.transcript_buffer)
                }
                for session in self.active_sessions.values()
            ]
        }


# WebSocket服务器启动函数
async def start_websocket_server(voice_service: EnhancedVoiceService, host: str = "0.0.0.0", port: int = 8001):
    """启动WebSocket语音服务器"""
    handler = WebSocketVoiceHandler(voice_service)
    
    logger.info(f"启动WebSocket语音服务器: ws://{host}:{port}")
    
    async with websockets.serve(handler.handle_connection, host, port):
        logger.info(f"WebSocket语音服务器已启动: ws://{host}:{port}")
        await asyncio.Future()  # 保持服务器运行
