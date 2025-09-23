"""
VoiceHelper V2 API - 增强版本
集成了OpenAI Whisper ASR、Edge-TTS、增强版FAISS RAG和Rasa对话管理
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 导入核心服务
from core.whisper_realtime_asr import WhisperRealtimeASR, ASRConfig, ASRResult
from core.edge_tts_service import EdgeTTSService, TTSConfig, TTSRequest, TTSResponse
from core.enhanced_faiss_rag import EnhancedFAISSRAG, Document, RetrievalResult
from core.rasa_dialogue import RasaDialogueManager, DialogueResponse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局服务实例
services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化服务
    logger.info("初始化VoiceHelper V2服务...")
    
    try:
        # 初始化ASR服务
        asr_config = ASRConfig(
            model_size="base",
            language="zh",
            vad_aggressiveness=2
        )
        services["asr"] = WhisperRealtimeASR(asr_config)
        await services["asr"].initialize()
        logger.info("ASR服务初始化完成")
        
        # 初始化TTS服务
        tts_config = TTSConfig(
            voice="zh-CN-XiaoxiaoNeural",
            cache_enabled=True,
            max_cache_size_mb=500
        )
        services["tts"] = EdgeTTSService(tts_config)
        await services["tts"].initialize()
        logger.info("TTS服务初始化完成")
        
        # 初始化RAG服务
        services["rag"] = EnhancedFAISSRAG(
            embedding_model="BAAI/bge-large-zh-v1.5",
            index_type="HNSW"
        )
        await services["rag"].initialize()
        logger.info("RAG服务初始化完成")
        
        # 初始化对话管理服务
        services["dialogue"] = RasaDialogueManager("http://localhost:5005")
        logger.info("对话管理服务初始化完成")
        
        logger.info("所有服务初始化完成")
        
    except Exception as e:
        logger.error(f"服务初始化失败: {e}")
        raise
    
    yield
    
    # 关闭时清理资源
    logger.info("清理服务资源...")
    if "dialogue" in services:
        await services["dialogue"].__aexit__(None, None, None)

# 创建FastAPI应用
app = FastAPI(
    title="VoiceHelper V2 API",
    description="基于OpenAI Whisper、Edge-TTS、FAISS和Rasa的智能语音助手API",
    version="2.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 数据模型 ====================

class AudioTranscribeRequest(BaseModel):
    audio_data: bytes = Field(..., description="音频数据")
    filename: str = Field(..., description="文件名")
    language: str = Field(default="zh", description="语言代码")
    model: str = Field(default="whisper-base", description="模型名称")
    user_id: str = Field(..., description="用户ID")
    tenant_id: str = Field(..., description="租户ID")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置参数")

class AudioTranscribeResponse(BaseModel):
    text: str = Field(..., description="识别文本")
    confidence: float = Field(..., description="置信度")
    language: str = Field(..., description="检测语言")
    duration: float = Field(..., description="音频时长")
    processing_time_ms: float = Field(..., description="处理时间(毫秒)")
    model_used: str = Field(..., description="使用的模型")

class TextSynthesizeRequest(BaseModel):
    text: str = Field(..., description="要合成的文本")
    voice: str = Field(default="zh-CN-XiaoxiaoNeural", description="语音")
    format: str = Field(default="mp3", description="音频格式")
    speed: float = Field(default=1.0, description="语速")
    emotion: Optional[str] = Field(None, description="情感")
    language: str = Field(default="zh-CN", description="语言")
    user_id: str = Field(..., description="用户ID")
    tenant_id: str = Field(..., description="租户ID")
    config: Dict[str, Any] = Field(default_factory=dict, description="配置参数")

class TextSynthesizeResponse(BaseModel):
    audio_data: str = Field(..., description="base64编码的音频数据")
    audio_url: Optional[str] = Field(None, description="音频URL")
    duration_ms: int = Field(..., description="音频时长(毫秒)")
    text_length: int = Field(..., description="文本长度")
    processing_time_ms: float = Field(..., description="处理时间(毫秒)")
    cached: bool = Field(..., description="是否来自缓存")
    voice_used: str = Field(..., description="使用的语音")
    text_hash: str = Field(..., description="文本哈希")

class DocumentSearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    top_k: int = Field(default=5, description="返回结果数量")
    score_threshold: float = Field(default=0.0, description="分数阈值")
    filters: Optional[Dict[str, Any]] = Field(None, description="过滤条件")
    user_id: str = Field(..., description="用户ID")
    tenant_id: str = Field(..., description="租户ID")

class DocumentSearchResponse(BaseModel):
    results: List[Dict[str, Any]] = Field(..., description="搜索结果")
    total: int = Field(..., description="结果总数")
    query: str = Field(..., description="查询文本")
    processing_time_ms: float = Field(..., description="处理时间(毫秒)")

class DialogueRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    user_id: str = Field(..., description="用户ID")
    session_id: str = Field(..., description="会话ID")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")

class DialogueResponse(BaseModel):
    text: str = Field(..., description="回复文本")
    intent: str = Field(..., description="识别意图")
    confidence: float = Field(..., description="置信度")
    entities: Dict[str, Any] = Field(..., description="实体")
    actions: List[str] = Field(..., description="动作列表")
    context_updates: Dict[str, Any] = Field(..., description="上下文更新")

# ==================== WebSocket连接管理 ====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, session_id: str, user_id: str, tenant_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        self.connection_metadata[session_id] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "connected_at": time.time(),
            "last_activity": time.time()
        }
        logger.info(f"WebSocket连接建立: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.connection_metadata:
            del self.connection_metadata[session_id]
        logger.info(f"WebSocket连接断开: {session_id}")

    async def send_message(self, session_id: str, message: Dict[str, Any]):
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_json(message)
                # 更新活动时间
                if session_id in self.connection_metadata:
                    self.connection_metadata[session_id]["last_activity"] = time.time()
            except Exception as e:
                logger.error(f"发送WebSocket消息失败: {e}")
                self.disconnect(session_id)

    def get_active_connections(self) -> Dict[str, Dict[str, Any]]:
        return self.connection_metadata.copy()

manager = ConnectionManager()

# ==================== API路由 ====================

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": time.time(),
        "services": {
            "asr": "asr" in services,
            "tts": "tts" in services,
            "rag": "rag" in services,
            "dialogue": "dialogue" in services
        }
    }

@app.post("/api/v2/voice/transcribe", response_model=AudioTranscribeResponse)
async def transcribe_audio(request: AudioTranscribeRequest):
    """语音转文字"""
    try:
        if "asr" not in services:
            raise HTTPException(status_code=503, detail="ASR服务不可用")
        
        start_time = time.time()
        asr_service = services["asr"]
        
        # 处理音频流
        results = []
        async for result in asr_service.process_audio_stream(request.audio_data):
            results.append(result)
            if result.is_final:
                break
        
        if not results:
            raise HTTPException(status_code=400, detail="音频处理失败")
        
        # 获取最终结果
        final_result = results[-1]
        processing_time = (time.time() - start_time) * 1000
        
        return AudioTranscribeResponse(
            text=final_result.text,
            confidence=final_result.confidence,
            language=final_result.language,
            duration=0.0,  # 需要从音频数据计算
            processing_time_ms=processing_time,
            model_used=request.model
        )
        
    except Exception as e:
        logger.error(f"语音转写失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音转写失败: {str(e)}")

@app.post("/api/v2/voice/synthesize", response_model=TextSynthesizeResponse)
async def synthesize_text(request: TextSynthesizeRequest):
    """文字转语音"""
    try:
        if "tts" not in services:
            raise HTTPException(status_code=503, detail="TTS服务不可用")
        
        tts_service = services["tts"]
        
        # 构建TTS请求
        tts_request = TTSRequest(
            text=request.text,
            voice=request.voice,
            rate=f"+{int((request.speed - 1) * 100)}%" if request.speed != 1.0 else "+0%",
            output_format=request.format,
            session_id=f"{request.user_id}:{request.tenant_id}"
        )
        
        # 执行语音合成
        tts_response = await tts_service.synthesize(tts_request)
        
        # 转换为base64
        import base64
        audio_data_b64 = base64.b64encode(tts_response.audio_data).decode('utf-8')
        
        return TextSynthesizeResponse(
            audio_data=audio_data_b64,
            audio_url=None,  # 可以实现文件存储和URL生成
            duration_ms=tts_response.duration_ms,
            text_length=tts_response.text_length,
            processing_time_ms=tts_response.processing_time_ms,
            cached=tts_response.cached,
            voice_used=tts_response.voice_used,
            text_hash=f"{hash(request.text):x}"
        )
        
    except Exception as e:
        logger.error(f"语音合成失败: {e}")
        raise HTTPException(status_code=500, detail=f"语音合成失败: {str(e)}")

@app.post("/api/v2/search/documents", response_model=DocumentSearchResponse)
async def search_documents(request: DocumentSearchRequest):
    """文档搜索"""
    try:
        if "rag" not in services:
            raise HTTPException(status_code=503, detail="RAG服务不可用")
        
        start_time = time.time()
        rag_service = services["rag"]
        
        # 执行搜索
        results = await rag_service.search(
            query=request.query,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            filters=request.filters
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # 转换结果格式
        search_results = []
        for result in results:
            search_results.append({
                "id": result.document.id,
                "content": result.document.content,
                "metadata": result.document.metadata,
                "score": result.score,
                "rank": result.rank
            })
        
        return DocumentSearchResponse(
            results=search_results,
            total=len(search_results),
            query=request.query,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"文档搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"文档搜索失败: {str(e)}")

@app.post("/api/v2/dialogue/process", response_model=DialogueResponse)
async def process_dialogue(request: DialogueRequest):
    """对话处理"""
    try:
        if "dialogue" not in services:
            raise HTTPException(status_code=503, detail="对话服务不可用")
        
        dialogue_service = services["dialogue"]
        
        # 处理对话
        async with dialogue_service:
            response = await dialogue_service.process_message(
                user_id=request.user_id,
                session_id=request.session_id,
                message=request.message,
                metadata=request.metadata
            )
        
        return DialogueResponse(
            text=response.text,
            intent=response.intent,
            confidence=response.confidence,
            entities=response.entities,
            actions=response.actions,
            context_updates=response.context_updates
        )
        
    except Exception as e:
        logger.error(f"对话处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")

@app.websocket("/api/v2/voice/stream/{session_id}")
async def voice_stream_websocket(websocket: WebSocket, session_id: str, user_id: str, tenant_id: str):
    """WebSocket语音流处理"""
    await manager.connect(websocket, session_id, user_id, tenant_id)
    
    try:
        # 发送连接确认
        await manager.send_message(session_id, {
            "type": "session_created",
            "session_id": session_id,
            "status": "ready",
            "timestamp": time.time()
        })
        
        while True:
            # 接收消息
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "start_recording":
                await manager.send_message(session_id, {
                    "type": "recording_started",
                    "session_id": session_id,
                    "status": "recording",
                    "timestamp": time.time()
                })
                
            elif message_type == "audio_chunk":
                # 处理音频块
                await process_audio_chunk(session_id, data)
                
            elif message_type == "stop_recording":
                await manager.send_message(session_id, {
                    "type": "recording_stopped",
                    "session_id": session_id,
                    "status": "stopped",
                    "timestamp": time.time()
                })
                
            elif message_type == "ping":
                await manager.send_message(session_id, {
                    "type": "pong",
                    "session_id": session_id,
                    "timestamp": time.time()
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket处理错误: {e}")
    finally:
        manager.disconnect(session_id)

async def process_audio_chunk(session_id: str, data: Dict[str, Any]):
    """处理音频块"""
    try:
        if "asr" not in services:
            await manager.send_message(session_id, {
                "type": "error",
                "message": "ASR服务不可用",
                "timestamp": time.time()
            })
            return
        
        # 解码音频数据
        import base64
        audio_data = base64.b64decode(data.get("data", ""))
        
        # 处理音频
        asr_service = services["asr"]
        async for result in asr_service.process_audio_stream(audio_data, session_id):
            await manager.send_message(session_id, {
                "type": "transcription",
                "session_id": session_id,
                "data": {
                    "text": result.text,
                    "confidence": result.confidence,
                    "is_final": result.is_final,
                    "language": result.language,
                    "processing_time_ms": result.processing_time_ms
                },
                "timestamp": time.time()
            })
            
            if result.is_final:
                break
                
    except Exception as e:
        logger.error(f"音频块处理失败: {e}")
        await manager.send_message(session_id, {
            "type": "error",
            "message": f"音频处理失败: {str(e)}",
            "timestamp": time.time()
        })

@app.get("/api/v2/stats/services")
async def get_service_stats():
    """获取服务统计信息"""
    stats = {
        "timestamp": time.time(),
        "services": {}
    }
    
    # ASR统计
    if "asr" in services:
        asr_stats = services["asr"].get_stats()
        stats["services"]["asr"] = asr_stats
    
    # TTS统计
    if "tts" in services:
        tts_stats = services["tts"].get_stats()
        stats["services"]["tts"] = tts_stats
    
    # RAG统计
    if "rag" in services:
        rag_stats = await services["rag"].get_stats()
        stats["services"]["rag"] = rag_stats
    
    # WebSocket连接统计
    connections = manager.get_active_connections()
    stats["websocket"] = {
        "active_connections": len(connections),
        "connections": connections
    }
    
    return stats

@app.get("/api/v2/voices/available")
async def get_available_voices():
    """获取可用语音列表"""
    try:
        if "tts" not in services:
            raise HTTPException(status_code=503, detail="TTS服务不可用")
        
        tts_service = services["tts"]
        voices = await tts_service.get_available_voices()
        
        return {
            "voices": voices,
            "total": len(voices),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"获取语音列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取语音列表失败: {str(e)}")

# ==================== 启动配置 ====================

if __name__ == "__main__":
    uvicorn.run(
        "v2_api_enhanced:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
