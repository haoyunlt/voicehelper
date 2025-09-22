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
    WebSocket语音流接口
    
    Args:
        websocket: WebSocket连接
    """
    await websocket.accept()
    
    if not asr_adapter or not tts_adapter:
        await websocket.send_json({
            "event": "error",
            "data": {"error": "ASR/TTS服务未初始化"}
        })
        await websocket.close()
        return
    
    session_id = None
    asr_session = None
    
    try:
        # 设置ASR回调
        def on_partial(seq: int, text: str):
            asyncio.create_task(websocket.send_json({
                "event": "asr_partial",
                "data": {"text": text, "confidence": 0.8}
            }))
        
        def on_final(seq: int, text: str, confidence: float):
            asyncio.create_task(websocket.send_json({
                "event": "asr_final", 
                "data": {"text": text, "confidence": confidence}
            }))
            
            # 触发Agent处理
            asyncio.create_task(process_voice_query(websocket, text))
        
        asr_adapter.on_partial(on_partial)
        asr_adapter.on_final(on_final)
        
        while True:
            # 接收WebSocket消息
            message = await websocket.receive_json()
            msg_type = message.get("type")
            
            if msg_type == "start":
                # 开始语音会话
                session_id = message.get("session_id")
                config = message.get("config", {})
                
                asr_session = asr_adapter.start(
                    sr=config.get("sample_rate", 16000),
                    codec="pcm",
                    lang=config.get("language", "zh-CN")
                )
                
                await websocket.send_json({
                    "event": "session_started",
                    "data": {"session_id": session_id, "asr_session": asr_session}
                })
                
            elif msg_type == "audio":
                # 处理音频数据
                if asr_session:
                    import base64
                    audio_data = base64.b64decode(message.get("data", ""))
                    asr_adapter.feed(0, audio_data)
                
            elif msg_type == "stop":
                # 停止语音会话
                if asr_session:
                    asr_adapter.stop()
                    asr_session = None
                
                await websocket.send_json({
                    "event": "session_stopped",
                    "data": {"session_id": session_id}
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
        if asr_session:
            try:
                asr_adapter.stop()
            except:
                pass


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
