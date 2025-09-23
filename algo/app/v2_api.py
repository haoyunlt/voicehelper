"""
V2架构FastAPI接口
基于父类/子类设计模式的SSE和WebSocket接口实现
"""

import os
import json
import time
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from loguru import logger
from datetime import datetime, timedelta
import threading
import uuid

from core.base import StreamCallback
from core.rag.bge_faiss_retriever import BGEFaissRetriever
from core.graph.chat_voice import ChatVoiceAgentGraph
from core.tools import FetchTool, FsReadTool, GithubReadTool
from core.asr_tts.openai import OpenAIAsrAdapter, OpenAITtsAdapter


# 请求/响应模型
class ChatRequest(BaseModel):
    query: str
    session_id: str
    context: Optional[Dict[str, Any]] = None


class VoiceConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    language: str = "zh-CN"


class VoiceStartMessage(BaseModel):
    type: str = "start"
    session_id: str
    config: VoiceConfig


class VoiceAudioMessage(BaseModel):
    type: str = "audio"
    data: str  # base64编码的音频数据


class VoiceStopMessage(BaseModel):
    type: str = "stop"
    session_id: str


# 全局服务实例
retriever = None
agent_graph = None
asr_adapter = None
tts_adapter = None

# 会话管理器
class ChatSessionManager:
    """聊天会话管理器"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.RLock()
        self.cleanup_task = None
    
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
                "generator": None,
                "response_chunks": []
            }
            self.active_sessions[session_id] = session_info
            logger.info(f"创建聊天会话: {session_id}")
            return session_info
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        with self.session_lock:
            return self.active_sessions.get(session_id)
    
    def update_session_activity(self, session_id: str):
        """更新会话活动时间"""
        with self.session_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["last_activity"] = datetime.now()
    
    def cancel_session(self, session_id: str) -> bool:
        """取消会话"""
        with self.session_lock:
            session = self.active_sessions.get(session_id)
            if session:
                session["status"] = "cancelled"
                session["cancel_event"].set()
                logger.info(f"会话已取消: {session_id}")
                return True
            return False
    
    def complete_session(self, session_id: str):
        """完成会话"""
        with self.session_lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id]["status"] = "completed"
                logger.info(f"会话已完成: {session_id}")
    
    def remove_session(self, session_id: str):
        """移除会话"""
        with self.session_lock:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                logger.info(f"会话已移除: {session_id}")
    
    def cleanup_expired_sessions(self):
        """清理过期会话"""
        with self.session_lock:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                # 超过30分钟未活动的会话视为过期
                if (current_time - session["last_activity"]).total_seconds() > 1800:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                session = self.active_sessions.pop(session_id, {})
                if session:
                    if "cancel_event" in session:
                        session["cancel_event"].set()
                    logger.info(f"清理过期会话: {session_id}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        with self.session_lock:
            stats = {
                "total_sessions": len(self.active_sessions),
                "active_sessions": sum(1 for s in self.active_sessions.values() if s["status"] == "active"),
                "cancelled_sessions": sum(1 for s in self.active_sessions.values() if s["status"] == "cancelled"),
                "completed_sessions": sum(1 for s in self.active_sessions.values() if s["status"] == "completed"),
                "sessions": [
                    {
                        "session_id": session["session_id"],
                        "status": session["status"],
                        "created_at": session["created_at"].isoformat(),
                        "last_activity": session["last_activity"].isoformat(),
                        "duration": (datetime.now() - session["created_at"]).total_seconds()
                    }
                    for session in self.active_sessions.values()
                ]
            }
            return stats

# 全局会话管理器实例
session_manager = ChatSessionManager()


def init_v2_services():
    """初始化V2架构服务"""
    global retriever, agent_graph, asr_adapter, tts_adapter
    
    try:
        # 初始化BGE+FAISS检索器
        retriever = BGEFaissRetriever(
            model_name="BAAI/bge-large-zh-v1.5",
            tenant_id="default",
            dataset_id="knowledge_base"
        )
        
        # 初始化工具
        tools = [
            FetchTool(),
            FsReadTool(),
            GithubReadTool()
        ]
        
        # 初始化Agent图
        agent_graph = ChatVoiceAgentGraph(
            retriever=retriever,
            tools=tools
        )
        
        # 初始化ASR/TTS适配器
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            asr_adapter = OpenAIAsrAdapter(api_key=openai_api_key)
            tts_adapter = OpenAITtsAdapter(api_key=openai_api_key)
        
        # 启动会话清理任务
        start_session_cleanup_task()
        
        logger.info("V2架构服务初始化完成")
        
    except Exception as e:
        logger.error(f"V2架构服务初始化失败: {e}")
        raise


def start_session_cleanup_task():
    """启动会话清理任务"""
    async def cleanup_task():
        while True:
            try:
                session_manager.cleanup_expired_sessions()
                await asyncio.sleep(300)  # 每5分钟清理一次
            except Exception as e:
                logger.error(f"会话清理任务失败: {e}")
                await asyncio.sleep(60)  # 出错后1分钟后重试
    
    # 在后台启动清理任务
    asyncio.create_task(cleanup_task())


def create_v2_app() -> FastAPI:
    """创建V2架构FastAPI应用"""
    app = FastAPI(
        title="VoiceHelper V2 API",
        description="基于V2架构的语音增强聊天助手",
        version="2.0.0"
    )
    
    # 初始化服务
    init_v2_services()
    
    return app


app = create_v2_app()


@app.post("/api/v1/chat/stream")
async def stream_chat(request: ChatRequest):
    """
    流式聊天接口
    
    Args:
        request: 聊天请求
        
    Returns:
        SSE流式响应
    """
    if not agent_graph:
        raise HTTPException(status_code=500, detail="Agent服务未初始化")
    
    # 创建会话
    session_info = session_manager.create_session(
        request.session_id, 
        {"query": request.query, "context": request.context}
    )
    
    async def event_generator():
        """SSE事件生成器"""
        try:
            # 创建回调函数
            def callback(event: str, payload: dict):
                # 在异步上下文中发送事件
                pass
            
            # 执行Agent流式处理
            for result in agent_graph.stream(request.query, cb=callback):
                # 检查是否被取消
                if session_info["cancel_event"].is_set():
                    logger.info(f"聊天会话被取消: {request.session_id}")
                    yield {
                        "event": "cancelled",
                        "data": json.dumps({
                            "status": "cancelled",
                            "session_id": request.session_id,
                            "message": "聊天已被取消"
                        })
                    }
                    break
                
                # 更新会话活动时间
                session_manager.update_session_activity(request.session_id)
                
                event_type = result.get("event", "message")
                event_data = result.get("data", {})
                
                # 构建SSE事件
                yield {
                    "event": event_type,
                    "data": json.dumps({
                        "meta": {
                            "session_id": request.session_id,
                            "timestamp": time.time()
                        },
                        "data": event_data
                    }, ensure_ascii=False)
                }
            
            # 检查是否正常完成（未被取消）
            if not session_info["cancel_event"].is_set():
                # 发送完成事件
                yield {
                    "event": "done",
                    "data": json.dumps({"status": "completed"})
                }
                session_manager.complete_session(request.session_id)
            
        except Exception as e:
            logger.error(f"流式聊天处理失败: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
        finally:
            # 清理会话（延迟清理，给客户端时间接收最后的事件）
            asyncio.create_task(delayed_session_cleanup(request.session_id))
    
    return EventSourceResponse(event_generator())


async def delayed_session_cleanup(session_id: str):
    """延迟清理会话"""
    await asyncio.sleep(5)  # 等待5秒确保客户端接收到所有事件
    session_manager.remove_session(session_id)


@app.post("/api/v1/chat/cancel")
async def cancel_chat(request: Dict[str, str]):
    """
    取消聊天
    
    Args:
        request: 包含session_id的请求
        
    Returns:
        取消结果
    """
    session_id = request.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="缺少session_id")
    
    # 实现取消逻辑
    success = session_manager.cancel_session(session_id)
    
    if success:
        logger.info(f"成功取消聊天会话: {session_id}")
        return {
            "status": "cancelled", 
            "session_id": session_id,
            "message": "聊天会话已成功取消",
            "timestamp": time.time()
        }
    else:
        logger.warning(f"会话不存在或已结束: {session_id}")
        raise HTTPException(
            status_code=404, 
            detail=f"会话 {session_id} 不存在或已结束"
        )


@app.websocket("/api/v1/voice/stream")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket语音流接口 - 增强版
    
    Args:
        websocket: WebSocket连接
    """
    await websocket.accept()
    
    # 检查服务状态
    if not agent_graph:
        await websocket.send_json({
            "type": "error",
            "data": {"error": "语音服务未初始化"}
        })
        await websocket.close()
        return
    
    session_id = None
    voice_session = None
    
    try:
        logger.info("New voice WebSocket connection established")
        
        while True:
            # 接收WebSocket消息
            message = await websocket.receive_json()
            msg_type = message.get("type")
            
            if msg_type == "voice_start":
                # 开始语音会话
                session_id = message.get("session_id")
                config = message.get("config", {})
                
                logger.info(f"Starting voice session: {session_id}")
                
                # 创建语音会话
                voice_session = {
                    "session_id": session_id,
                    "config": config,
                    "start_time": time.time(),
                    "audio_buffer": b"",
                    "transcript_buffer": ""
                }
                
                await websocket.send_json({
                    "type": "session_started",
                    "session_id": session_id,
                    "status": "started"
                })
                
            elif msg_type == "voice_audio":
                # 处理音频数据
                if voice_session:
                    import base64
                    audio_data = base64.b64decode(message.get("data", ""))
                    
                    # 处理音频数据
                    try:
                        # 模拟ASR处理（简化版本）
                        audio_data = base64.b64decode(message.get("data", ""))
                        
                        # 模拟语音识别结果
                        if len(audio_data) > 1000:  # 假设有足够的音频数据
                            transcript = "这是模拟的语音识别结果"  # 实际应该调用ASR服务
                            
                            await websocket.send_json({
                                "type": "asr_partial",
                                "text": transcript,
                                "confidence": 0.8,
                                "session_id": session_id
                            })
                            
                            # 模拟最终结果
                            await websocket.send_json({
                                "type": "asr_final",
                                "text": transcript,
                                "confidence": 0.8,
                                "session_id": session_id
                            })
                            
                            # 触发Agent处理
                            if session_id:
                                await process_voice_query_enhanced(websocket, transcript, session_id)
                                
                    except Exception as e:
                        logger.error(f"Voice processing error: {e}")
                        await websocket.send_json({
                            "type": "error",
                            "error": str(e),
                            "session_id": session_id
                        })
                
            elif msg_type == "voice_stop":
                # 停止语音会话
                if voice_session:
                    logger.info(f"Stopping voice session: {session_id}")
                    voice_session = None
                
                await websocket.send_json({
                    "type": "session_stopped",
                    "session_id": session_id,
                    "status": "stopped"
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket处理失败: {e}")
        await websocket.send_json({
            "event": "error",
            "data": {"error": str(e)}
        })
    finally:
        # 清理资源
        if voice_session:
            logger.info(f"Cleaning up voice session: {session_id}")


async def process_voice_query_enhanced(websocket: WebSocket, query: str, session_id: str):
    """
    增强的语音查询处理
    """
    try:
        logger.info(f"Processing voice query: {query} for session: {session_id}")
        
        # 发送处理开始通知
        await websocket.send_json({
            "type": "agent_start",
            "session_id": session_id,
            "query": query
        })
        
        # 使用Agent处理查询
        if agent_graph:
            try:
                # 创建回调函数
                def callback(event: str, payload: dict):
                    # 在异步上下文中发送事件
                    pass
                
                # 流式处理Agent响应
                response_text = ""
                for result in agent_graph.stream(query, cb=callback):
                    event_type = result.get("event")
                    event_data = result.get("data", {})
                    
                    if event_type == "answer":
                        response_text = event_data.get("text", "")
                        # 发送Agent响应
                        await websocket.send_json({
                            "type": "agent_response",
                            "session_id": session_id,
                            "text": response_text,
                            "is_final": True
                        })
                        break
                
                # TTS处理（如果有TTS适配器）
                if response_text and tts_adapter:
                    try:
                        for audio_chunk in tts_adapter.synthesize_text(response_text):
                            import base64
                            audio_b64 = base64.b64encode(audio_chunk).decode()
                            await websocket.send_json({
                                "type": "tts_audio",
                                "session_id": session_id,
                                "audio_data": audio_b64,
                                "format": "wav"
                            })
                    except Exception as e:
                        logger.error(f"TTS processing error: {e}")
                        
            except Exception as e:
                logger.error(f"Agent processing error: {e}")
                await websocket.send_json({
                    "type": "error",
                    "session_id": session_id,
                    "error": str(e)
                })
        else:
            # 模拟响应
            response_text = f"收到您的语音输入：{query}"
            await websocket.send_json({
                "type": "agent_response",
                "session_id": session_id,
                "text": response_text,
                "is_final": True
            })
            
    except Exception as e:
        logger.error(f"Voice query processing error: {e}")
        await websocket.send_json({
            "type": "error",
            "session_id": session_id,
            "error": str(e)
        })


async def process_voice_query(websocket: WebSocket, query: str):
    """
    处理语音查询
    
    Args:
        websocket: WebSocket连接
        query: 查询文本
    """
    try:
        if not agent_graph:
            await websocket.send_json({
                "event": "error",
                "data": {"error": "Agent服务未初始化"}
            })
            return
        
        # 创建回调函数
        def callback(event: str, payload: dict):
            asyncio.create_task(websocket.send_json({
                "event": f"agent_{event}",
                "data": payload
            }))
        
        # 执行Agent处理
        for result in agent_graph.stream(query, cb=callback):
            event_type = result.get("event")
            event_data = result.get("data", {})
            
            if event_type == "answer":
                # 发送文本回答
                await websocket.send_json({
                    "event": "agent_response",
                    "data": event_data
                })
                
                # 生成TTS音频
                text = event_data.get("text", "")
                if text and tts_adapter:
                    try:
                        for audio_chunk in tts_adapter.synthesize_text(text):
                            import base64
                            audio_b64 = base64.b64encode(audio_chunk).decode()
                            await websocket.send_json({
                                "event": "tts_audio",
                                "data": audio_b64
                            })
                    except Exception as e:
                        logger.error(f"TTS处理失败: {e}")
            
            elif event_type == "error":
                await websocket.send_json({
                    "event": "agent_error",
                    "data": event_data
                })
                break
        
    except Exception as e:
        logger.error(f"语音查询处理失败: {e}")
        await websocket.send_json({
            "event": "error",
            "data": {"error": str(e)}
        })


@app.get("/api/v1/chat/session/{session_id}")
async def get_session_status(session_id: str):
    """
    获取会话状态
    
    Args:
        session_id: 会话ID
        
    Returns:
        会话状态信息
    """
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")
    
    return {
        "session_id": session["session_id"],
        "status": session["status"],
        "created_at": session["created_at"].isoformat(),
        "last_activity": session["last_activity"].isoformat(),
        "duration": (datetime.now() - session["created_at"]).total_seconds(),
        "request_data": session["request_data"]
    }


@app.get("/api/v1/chat/sessions")
async def list_active_sessions():
    """
    获取所有活跃会话列表
    
    Returns:
        活跃会话统计信息
    """
    return session_manager.get_session_stats()


@app.get("/api/v1/health")
async def health_check():
    """健康检查接口"""
    session_stats = session_manager.get_session_stats()
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": time.time(),
        "services": {
            "retriever": retriever is not None,
            "agent_graph": agent_graph is not None,
            "asr_adapter": asr_adapter is not None,
            "tts_adapter": tts_adapter is not None
        },
        "session_manager": {
            "total_sessions": session_stats["total_sessions"],
            "active_sessions": session_stats["active_sessions"],
            "cancelled_sessions": session_stats["cancelled_sessions"],
            "completed_sessions": session_stats["completed_sessions"]
        }
    }


if __name__ == "__main__":
    import uvicorn
    import os
    
    uvicorn.run(
        "v2_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8070)),
        reload=True
    )
