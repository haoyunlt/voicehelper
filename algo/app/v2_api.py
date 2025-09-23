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
        
        logger.info("V2架构服务初始化完成")
        
    except Exception as e:
        logger.error(f"V2架构服务初始化失败: {e}")
        raise


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
    
    async def event_generator():
        """SSE事件生成器"""
        try:
            # 创建回调函数
            def callback(event: str, payload: dict):
                # 在异步上下文中发送事件
                pass
            
            # 执行Agent流式处理
            for result in agent_graph.stream(request.query, cb=callback):
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
            
            # 发送完成事件
            yield {
                "event": "done",
                "data": json.dumps({"status": "completed"})
            }
            
        except Exception as e:
            logger.error(f"流式聊天处理失败: {e}")
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }
    
    return EventSourceResponse(event_generator())


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
    
    # TODO: 实现取消逻辑
    logger.info(f"取消聊天会话: {session_id}")
    
    return {"status": "cancelled", "session_id": session_id}


@app.websocket("/api/v1/voice/stream")
async def voice_websocket(websocket: WebSocket):
    """
    WebSocket语音流接口 - 增强版
    
    Args:
        websocket: WebSocket连接
    """
    await websocket.accept()
    
    # 检查服务状态
    if not enhanced_voice_service:
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
                    
                    # 使用增强语音服务处理
                    try:
                        # 模拟ASR处理
                        from core.voice_request import VoiceRequest
                        
                        voice_request = VoiceRequest(
                            conversation_id=session_id,
                            audio_chunk=message.get("data", ""),
                            is_final=False,
                            config=voice_session.get("config", {})
                        )
                        
                        # 异步处理语音
                        async for response in enhanced_voice_service.process_voice_request(voice_request):
                            if response.transcript:
                                await websocket.send_json({
                                    "type": "asr_partial",
                                    "text": response.transcript,
                                    "confidence": response.confidence or 0.8,
                                    "session_id": session_id
                                })
                            
                            if response.is_final and response.transcript:
                                await websocket.send_json({
                                    "type": "asr_final",
                                    "text": response.transcript,
                                    "confidence": response.confidence or 0.8,
                                    "session_id": session_id
                                })
                                
                                # 触发Agent处理
                                await process_voice_query_enhanced(websocket, response.transcript, session_id)
                                
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
        if agent_service:
            # 构建Agent请求
            agent_request = {
                "conversation_id": session_id,
                "message": query,
                "tools": ["web_search", "document_search"],
                "context": [],
                "max_tokens": 1000,
                "temperature": 0.7
            }
            
            # 流式处理Agent响应
            async for chunk in agent_service.stream_chat(agent_request):
                if chunk.get("type") == "agent_response":
                    # 发送Agent响应
                    await websocket.send_json({
                        "type": "agent_response",
                        "session_id": session_id,
                        "text": chunk.get("content", ""),
                        "is_final": chunk.get("is_final", False)
                    })
                    
                    # 如果是最终响应，进行TTS转换
                    if chunk.get("is_final") and enhanced_voice_service:
                        try:
                            # TTS处理
                            tts_response = await enhanced_voice_service.text_to_speech(
                                chunk.get("content", ""),
                                session_id
                            )
                            
                            if tts_response.get("audio_data"):
                                await websocket.send_json({
                                    "type": "tts_audio",
                                    "session_id": session_id,
                                    "audio_data": tts_response["audio_data"],
                                    "format": tts_response.get("format", "wav")
                                })
                        except Exception as e:
                            logger.error(f"TTS processing error: {e}")
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


@app.get("/api/v1/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "services": {
            "retriever": retriever is not None,
            "agent_graph": agent_graph is not None,
            "asr_adapter": asr_adapter is not None,
            "tts_adapter": tts_adapter is not None
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
