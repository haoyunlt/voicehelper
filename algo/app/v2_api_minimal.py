#!/usr/bin/env python3
"""
VoiceHelper 算法服务 V2 API (最小版本)
解决路径问题的简化实现
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path

# 设置Python路径
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from loguru import logger
from datetime import datetime, timedelta
import threading
import uuid

# 尝试导入核心模块，如果失败则使用模拟版本
try:
    from core.base import StreamCallback
except ImportError:
    logger.warning("无法导入 core.base.StreamCallback，使用模拟版本")
    class StreamCallback:
        def __init__(self, callback_fn=None):
            self.callback_fn = callback_fn
        
        def __call__(self, event: str, data: Dict[str, Any]):
            if self.callback_fn:
                self.callback_fn(event, data)

try:
    from core.graph.chat_voice import ChatVoiceAgentGraph
    REAL_AGENT = True
except ImportError:
    logger.warning("无法导入 ChatVoiceAgentGraph，使用模拟版本")
    REAL_AGENT = False
    class ChatVoiceAgentGraph:
        def __init__(self, retriever=None, **kwargs):
            self.retriever = retriever
            logger.info("使用模拟 ChatVoiceAgentGraph")
        
        def stream(self, query: str, **kwargs):
            # 模拟流式响应
            yield {"event": "agent_thinking", "data": {"content": "正在思考..."}}
            yield {"event": "agent_response", "data": {"content": f"模拟回复: {query}"}}

try:
    from common.logger import setup_logger
except ImportError:
    logger.warning("无法导入 common.logger，使用默认配置")
    def setup_logger():
        pass

# 请求/响应模型
class ChatRequest(BaseModel):
    query: str
    session_id: str
    context: Dict[str, Any] = {}
    max_tokens: int = 1000
    temperature: float = 0.7

class VoiceRequest(BaseModel):
    session_id: str
    audio_data: str  # base64编码的音频数据
    format: str = "wav"
    sample_rate: int = 16000

# 会话管理器
class ChatSessionManager:
    """聊天会话管理器"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.RLock()
    
    def create_session(self, session_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建新的聊天会话"""
        with self.session_lock:
            session_info = {
                "session_id": session_id,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "status": "active",
                "request_data": request_data,
                "cancel_event": asyncio.Event(),
            }
            self.active_sessions[session_id] = session_info
            logger.info(f"创建会话: {session_id}")
            return session_info
    
    def cancel_session(self, session_id: str) -> bool:
        """取消会话"""
        with self.session_lock:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                session["status"] = "cancelled"
                session["cancel_event"].set()
                logger.info(f"取消会话: {session_id}")
                return True
            return False
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        with self.session_lock:
            return self.active_sessions.get(session_id)
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        with self.session_lock:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if (current_time - session["last_activity"]).total_seconds() > 3600:  # 1小时超时
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                session = self.active_sessions.pop(session_id, {})
                if session and "cancel_event" in session:
                    session["cancel_event"].set()
                logger.info(f"清理过期会话: {session_id}")

# 全局服务实例
session_manager = ChatSessionManager()
agent_graph = None

def init_services():
    """初始化服务"""
    global agent_graph
    
    try:
        if REAL_AGENT:
            # 尝试初始化真实的Agent（需要retriever）
            logger.info("尝试初始化真实Agent服务...")
            # 这里需要真实的retriever，暂时跳过
            raise ImportError("需要完整依赖")
        else:
            # 使用模拟版本
            agent_graph = ChatVoiceAgentGraph(retriever=None)
            logger.info("✅ Agent服务初始化完成（模拟版本）")
    except Exception as e:
        logger.warning(f"使用模拟Agent服务: {e}")
        agent_graph = ChatVoiceAgentGraph(retriever=None)  # 使用模拟版本

# 创建FastAPI应用
app = FastAPI(
    title="VoiceHelper Algorithm Service V2",
    description="VoiceHelper 算法服务 V2 API (最小版本)",
    version="2.0.0-minimal"
)

@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    logger.info("🚀 VoiceHelper Algorithm Service V2 启动中...")
    init_services()
    logger.info("✅ 服务启动完成")

@app.get("/")
async def root():
    """根路径"""
    return {
        "service": "VoiceHelper Algorithm Service V2",
        "version": "2.0.0-minimal",
        "status": "running",
        "timestamp": time.time()
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "2.0.0-minimal",
        "services": {
            "agent_graph": agent_graph is not None,
            "session_manager": session_manager is not None,
        },
        "active_sessions": len(session_manager.active_sessions),
        "timestamp": time.time()
    }

@app.post("/api/v1/chat/stream")
async def stream_chat(request: ChatRequest):
    """流式聊天接口"""
    if not agent_graph:
        raise HTTPException(status_code=500, detail="Agent服务未初始化")
    
    # 创建会话
    session = session_manager.create_session(request.session_id, request.dict())
    
    async def event_generator():
        """SSE事件生成器"""
        try:
            # 执行Agent流式处理
            for result in agent_graph.stream(request.query):
                # 检查取消状态
                if session["cancel_event"].is_set():
                    yield {
                        "event": "cancelled",
                        "data": json.dumps({"message": "会话已取消"})
                    }
                    break
                
                event_type = result.get("event", "message")
                event_data = result.get("data", {})
                
                # 构建SSE事件
                yield {
                    "event": event_type,
                    "data": json.dumps({
                        "session_id": request.session_id,
                        "content": event_data.get("content", ""),
                        "timestamp": time.time(),
                        **event_data
                    })
                }
                
                # 更新会话活动时间
                session["last_activity"] = datetime.now()
                
                await asyncio.sleep(0.1)  # 防止过快发送
                
        except Exception as e:
            logger.error(f"流式聊天错误: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
        finally:
            # 清理会话
            session["status"] = "completed"
    
    return EventSourceResponse(event_generator())

@app.post("/api/v1/chat/cancel")
async def cancel_chat(request: Dict[str, str]):
    """取消聊天"""
    session_id = request.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少session_id")
    
    success = session_manager.cancel_session(session_id)
    
    if success:
        return {
            "status": "cancelled", 
            "session_id": session_id,
            "timestamp": time.time()
        }
    else:
        raise HTTPException(status_code=404, detail="会话不存在")

@app.get("/api/v1/chat/session/{session_id}")
async def get_session_status(session_id: str):
    """获取会话状态"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "created_at": session["created_at"].isoformat(),
        "last_activity": session["last_activity"].isoformat(),
    }

@app.get("/api/v1/chat/sessions")
async def list_sessions():
    """列出所有会话"""
    sessions = []
    for session_id, session in session_manager.active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "status": session["status"],
            "created_at": session["created_at"].isoformat(),
            "last_activity": session["last_activity"].isoformat(),
        })
    
    return {
        "sessions": sessions,
        "total": len(sessions),
        "timestamp": time.time()
    }

@app.websocket("/ws/voice")
async def websocket_voice_endpoint(websocket: WebSocket):
    """WebSocket语音处理端点"""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    logger.info(f"WebSocket语音连接建立: {session_id}")
    
    try:
        while True:
            # 接收消息
            message = await websocket.receive_json()
            message_type = message.get("type")
            
            if message_type == "audio_chunk":
                # 处理音频块
                await websocket.send_json({
                    "type": "asr_result",
                    "session_id": session_id,
                    "text": "模拟语音识别结果",
                    "is_final": False,
                    "timestamp": time.time()
                })
                
            elif message_type == "audio_end":
                # 音频结束
                await websocket.send_json({
                    "type": "asr_result",
                    "session_id": session_id,
                    "text": "完整的语音识别结果",
                    "is_final": True,
                    "timestamp": time.time()
                })
                
                # 生成回复
                await websocket.send_json({
                    "type": "agent_response",
                    "session_id": session_id,
                    "text": "这是对语音输入的回复",
                    "timestamp": time.time()
                })
                
            elif message_type == "ping":
                # 心跳
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": time.time()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket语音连接断开: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket语音处理错误: {e}")
        await websocket.close()

# 定期清理任务
async def cleanup_task():
    """定期清理过期会话"""
    while True:
        try:
            session_manager.cleanup_expired_sessions()
            await asyncio.sleep(300)  # 5分钟清理一次
        except Exception as e:
            logger.error(f"清理任务错误: {e}")
            await asyncio.sleep(60)

@app.on_event("startup")
async def start_cleanup_task():
    """启动清理任务"""
    asyncio.create_task(cleanup_task())

if __name__ == "__main__":
    import uvicorn
    
    # 设置日志
    setup_logger()
    
    # 获取配置
    host = os.getenv("ALGO_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("ALGO_SERVICE_PORT", "8070"))
    debug = os.getenv("ALGO_SERVICE_DEBUG", "true").lower() == "true"
    
    logger.info(f"启动服务: {host}:{port}")
    
    uvicorn.run(
        "v2_api_minimal:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )
