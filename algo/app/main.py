from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import os
import socket
import time
from dotenv import load_dotenv

from core.ingest import IngestService
from core.retrieve import RetrieveService
from core.voice import VoiceService
from core.enhanced_voice_services import EnhancedVoiceService
from core.websocket_voice import WebSocketVoiceHandler
from core.models import QueryRequest, IngestRequest, IngestResponse, VoiceQueryRequest, VoiceQueryResponse
from common.logger import init_logger, get_logger, LoggingMiddleware, get_uvicorn_log_config
from common.errors import ErrorCode, VoiceHelperError, get_error_info

# 加载环境变量
load_dotenv()

# 获取配置
SERVICE_NAME = os.getenv("SERVICE_NAME", os.getenv("ALGO_SERVICE_NAME", "voicehelper-algo"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", os.getenv("ALGO_PORT", 8000)))
ENV = os.getenv("ENV", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# 初始化日志器
init_logger(SERVICE_NAME)
logger = get_logger("main")

def get_local_ip() -> str:
    """获取本地IP地址"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"

# 获取本地IP
LOCAL_IP = get_local_ip()

app = FastAPI(
    title="VoiceHelper Algorithm Service",
    description="基于 RAG 的智能算法服务",
    version="1.9.0",
    docs_url="/docs" if ENV == "development" else None,
    redoc_url="/redoc" if ENV == "development" else None,
)

# 添加日志中间件
app.middleware("http")(LoggingMiddleware())

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 记录启动时间
start_time = time.time()

# 初始化服务
try:
    logger.startup("初始化算法服务组件", context={
        "service": SERVICE_NAME,
        "host": HOST,
        "port": PORT,
        "local_ip": LOCAL_IP,
        "environment": ENV,
        "log_level": LOG_LEVEL,
    })
    
    ingest_service = IngestService()
    retrieve_service = RetrieveService()
    voice_service = VoiceService(retrieve_service)
    # 创建默认语音配置
    from core.enhanced_voice_services import VoiceConfig, VoiceProvider
    voice_config = VoiceConfig()
    voice_config.primary_asr_provider = VoiceProvider.LOCAL
    voice_config.primary_tts_provider = VoiceProvider.EDGE_TTS
    enhanced_voice_service = EnhancedVoiceService(voice_config, retrieve_service)
    websocket_handler = WebSocketVoiceHandler(enhanced_voice_service)
    
    logger.startup("算法服务组件初始化完成", context={
        "components": ["IngestService", "RetrieveService", "VoiceService"],
        "status": "ready"
    })
    
except Exception as e:
    logger.exception("算法服务组件初始化失败", e)
    raise

# 异常处理器
@app.exception_handler(VoiceHelperError)
async def voicehelper_exception_handler(request: Request, exc: VoiceHelperError):
    """处理VoiceHelper自定义异常"""
    logger.error_with_code(exc.code, f"VoiceHelper错误: {exc.message}", context={
        "method": request.method,
        "url": str(request.url),
        "details": exc.details,
    })
    
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """处理HTTP异常"""
    logger.error(f"HTTP异常: {exc.detail}", context={
        "method": request.method,
        "url": str(request.url),
        "status_code": exc.status_code,
    })
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理通用异常"""
    logger.exception("未处理的异常", exc, context={
        "method": request.method,
        "url": str(request.url),
    })
    
    error_info = get_error_info(ErrorCode.SYSTEM_INTERNAL_ERROR)
    return JSONResponse(
        status_code=error_info.http_status,
        content=error_info.to_dict()
    )

@app.get("/")
async def root(request: Request):
    """根路径"""
    logger.debug("根路径访问", context={"client_ip": request.client.host if request.client else "unknown"})
    
    return {
        "message": "VoiceHelper Algorithm Service",
        "status": "running",
        "version": "1.9.0",
        "service": SERVICE_NAME,
        "local_ip": LOCAL_IP,
        "port": PORT,
        "endpoints": {
            "health": f"http://{LOCAL_IP}:{PORT}/health",
            "docs": f"http://{LOCAL_IP}:{PORT}/docs" if ENV == "development" else None,
            "ingest": f"http://{LOCAL_IP}:{PORT}/ingest",
            "query": f"http://{LOCAL_IP}:{PORT}/query",
            "voice_query": f"http://{LOCAL_IP}:{PORT}/voice/query",
        }
    }

@app.get("/health")
async def health_check(request: Request):
    """健康检查"""
    logger.debug("健康检查请求", context={
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", ""),
    })
    
    # 检查服务组件状态
    try:
        # 这里可以添加实际的健康检查逻辑
        # 例如检查数据库连接、外部服务等
        
        health_status = {
            "status": "healthy",
            "timestamp": int(time.time()),
            "service": SERVICE_NAME,
            "version": "1.9.0",
            "local_ip": LOCAL_IP,
            "port": PORT,
            "environment": ENV,
            "components": {
                "ingest_service": "healthy",
                "retrieve_service": "healthy", 
                "voice_service": "healthy",
            }
        }
        
        logger.business("健康检查完成", context=health_status)
        return health_status
        
    except Exception as e:
        logger.exception("健康检查失败", e)
        raise VoiceHelperError(ErrorCode.RAG_SERVICE_UNAVAILABLE, "服务健康检查失败")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """文档入库接口"""
    start_time = time.time()
    
    logger.business("文档入库请求", context={
        "files_count": len(request.files) if request.files else 0,
        "collection_name": getattr(request, 'collection_name', 'default'),
        "client_ip": http_request.client.host if http_request.client else "unknown",
    })
    
    try:
        # 验证请求
        if not request.files or len(request.files) == 0:
            raise VoiceHelperError(ErrorCode.RAG_INVALID_QUERY, "没有提供文档文件")
        
        # 生成任务ID
        task_id = ingest_service.generate_task_id()
        
        logger.info(f"生成入库任务ID: {task_id}", context={
            "task_id": task_id,
            "files_count": len(request.files),
        })
        
        # 后台处理入库任务
        background_tasks.add_task(
            ingest_service.process_ingest_task,
            task_id,
            request
        )
        
        duration_ms = (time.time() - start_time) * 1000
        logger.performance("文档入库任务创建", duration_ms, context={
            "task_id": task_id,
            "files_count": len(request.files),
        })
        
        return IngestResponse(task_id=task_id)
    
    except VoiceHelperError:
        raise
    except Exception as e:
        logger.exception("文档入库失败", e, context={
            "files_count": len(request.files) if request.files else 0,
        })
        raise VoiceHelperError(ErrorCode.RAG_INDEXING_FAILED, f"文档入库失败: {str(e)}")

@app.post("/query")
async def query_documents(request: QueryRequest, http_request: Request):
    """查询接口，返回流式响应"""
    start_time = time.time()
    
    logger.business("文档查询请求", context={
        "messages_count": len(request.messages) if request.messages else 0,
        "top_k": getattr(request, 'top_k', None),
        "client_ip": http_request.client.host if http_request.client else "unknown",
    })
    
    try:
        # 验证请求
        if not request.messages or len(request.messages) == 0:
            raise VoiceHelperError(ErrorCode.RAG_INVALID_QUERY, "没有提供查询消息")
        
        # 记录查询开始
        logger.info("开始处理查询请求", context={
            "messages_count": len(request.messages),
            "last_message": request.messages[-1].content[:100] if request.messages else "",
        })
        
        # 生成流式响应
        return StreamingResponse(
            retrieve_service.stream_query(request),
            media_type="application/x-ndjson"
        )
    
    except VoiceHelperError:
        raise
    except Exception as e:
        logger.exception("文档查询失败", e, context={
            "messages_count": len(request.messages) if request.messages else 0,
        })
        raise VoiceHelperError(ErrorCode.RAG_RETRIEVAL_FAILED, f"查询失败: {str(e)}")

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str, http_request: Request):
    """获取任务状态"""
    logger.debug(f"查询任务状态: {task_id}", context={
        "task_id": task_id,
        "client_ip": http_request.client.host if http_request.client else "unknown",
    })
    
    try:
        if not task_id or not task_id.strip():
            raise VoiceHelperError(ErrorCode.RAG_INVALID_QUERY, "任务ID不能为空")
        
        status = ingest_service.get_task_status(task_id)
        
        logger.business("任务状态查询", context={
            "task_id": task_id,
            "status": status.get("status", "unknown"),
            "progress": status.get("progress", 0),
        })
        
        return status
        
    except VoiceHelperError:
        raise
    except Exception as e:
        logger.exception("获取任务状态失败", e, context={"task_id": task_id})
        raise VoiceHelperError(ErrorCode.RAG_INTERNAL_ERROR, f"获取任务状态失败: {str(e)}")

@app.post("/voice/query")
async def voice_query(request: VoiceQueryRequest, http_request: Request):
    """语音查询接口"""
    start_time = time.time()
    
    logger.business("语音查询请求", context={
        "session_id": getattr(request, 'session_id', ''),
        "lang": getattr(request, 'lang', 'zh-CN'),
        "has_audio_data": bool(getattr(request, 'audio_data', None)),
        "client_ip": http_request.client.host if http_request.client else "unknown",
    })
    
    try:
        # 验证请求
        if not hasattr(request, 'audio_data') or not request.audio_data:
            raise VoiceHelperError(ErrorCode.VOICE_INVALID_FORMAT, "没有提供音频数据")
        
        logger.info("开始处理语音查询", context={
            "session_id": getattr(request, 'session_id', ''),
            "audio_data_length": len(request.audio_data) if request.audio_data else 0,
        })
        
        return StreamingResponse(
            voice_service.process_voice_query(request),
            media_type="application/x-ndjson"
        )
        
    except VoiceHelperError:
        raise
    except Exception as e:
        logger.exception("语音查询失败", e, context={
            "session_id": getattr(request, 'session_id', ''),
        })
        raise VoiceHelperError(ErrorCode.VOICE_PROCESSING_FAILED, f"语音查询失败: {str(e)}")

@app.websocket("/voice/stream")
async def websocket_voice_stream(websocket):
    """WebSocket语音流接口"""
    try:
        await websocket.accept()
        logger.info("WebSocket语音连接建立")
        
        # 委托给WebSocket处理器
        await websocket_handler.handle_websocket_connection(websocket)
        
    except Exception as e:
        logger.exception("WebSocket语音流处理失败", e)
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except:
            pass

@app.post("/cancel")
async def cancel_request(request: Request):
    """取消请求"""
    logger.debug("取消请求", context={
        "client_ip": request.client.host if request.client else "unknown",
        "headers": dict(request.headers),
    })
    
    try:
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            raise VoiceHelperError(ErrorCode.GATEWAY_MISSING_PARAMETER, "缺少请求ID")
        
        logger.info(f"取消请求: {request_id}", context={"request_id": request_id})
        
        await voice_service.cancel_request(request_id)
        
        logger.business("请求已取消", context={"request_id": request_id})
        return {"status": "cancelled", "request_id": request_id}
        
    except VoiceHelperError:
        raise
    except Exception as e:
        logger.exception("取消请求失败", e)
        raise VoiceHelperError(ErrorCode.SYSTEM_INTERNAL_ERROR, f"取消请求失败: {str(e)}")

@app.post("/reload")
async def reload_index(request: Request):
    """热重载索引"""
    try:
        dataset_id = request.query_params.get("dataset_id", "default")
        
        logger.info(f"开始热重载索引: {dataset_id}", context={
            "dataset_id": dataset_id,
            "client_ip": request.client.host if request.client else "unknown",
        })
        
        # 重载BGE+FAISS索引
        if hasattr(retrieve_service, 'rag_service') and retrieve_service.rag_service:
            result = await retrieve_service.rag_service.reload_index(dataset_id)
            
            logger.business("索引热重载完成", context={
                "dataset_id": dataset_id,
                "result": result
            })
            
            return {
                "status": "success",
                "message": f"索引 {dataset_id} 重载完成",
                "dataset_id": dataset_id,
                "timestamp": int(time.time() * 1000),
                **result
            }
        else:
            raise VoiceHelperError(ErrorCode.RAG_SERVICE_UNAVAILABLE, "RAG服务不可用")
            
    except VoiceHelperError:
        raise
    except Exception as e:
        logger.exception("索引热重载失败", e, context={"dataset_id": dataset_id})
        raise VoiceHelperError(ErrorCode.RAG_INTERNAL_ERROR, f"索引热重载失败: {str(e)}")

@app.get("/stats")
async def get_service_stats(request: Request):
    """获取服务统计信息"""
    try:
        logger.debug("获取服务统计信息", context={
            "client_ip": request.client.host if request.client else "unknown",
        })
        
        stats = {
            "service": SERVICE_NAME,
            "version": "1.9.0",
            "uptime": time.time() - start_time if 'start_time' in globals() else 0,
            "timestamp": int(time.time() * 1000),
            "components": {}
        }
        
        # 获取各组件统计
        if hasattr(retrieve_service, 'rag_service') and retrieve_service.rag_service:
            stats["components"]["rag"] = retrieve_service.rag_service.get_stats()
        
        if hasattr(enhanced_voice_service, 'get_stats'):
            stats["components"]["voice"] = enhanced_voice_service.get_stats()
        
        if hasattr(websocket_handler, 'get_session_stats'):
            stats["components"]["websocket"] = websocket_handler.get_session_stats()
        
        logger.business("服务统计信息查询", context=stats)
        return stats
        
    except Exception as e:
        logger.exception("获取服务统计失败", e)
        raise VoiceHelperError(ErrorCode.SYSTEM_INTERNAL_ERROR, f"获取统计信息失败: {str(e)}")

# 错误测试端点
@app.get("/error-test")
async def error_test(http_request: Request):
    """错误测试端点"""
    error_type = http_request.query_params.get("type", "internal")
    
    logger.debug(f"错误测试请求: {error_type}", context={
        "error_type": error_type,
        "client_ip": http_request.client.host if http_request.client else "unknown",
    })
    
    if error_type == "rag_error":
        raise VoiceHelperError(ErrorCode.RAG_RETRIEVAL_FAILED, "这是一个RAG错误测试")
    elif error_type == "voice_error":
        raise VoiceHelperError(ErrorCode.VOICE_ASR_FAILED, "这是一个语音错误测试")
    elif error_type == "auth_error":
        raise VoiceHelperError(ErrorCode.AUTH_TOKEN_EXPIRED, "这是一个认证错误测试")
    else:
        raise VoiceHelperError(ErrorCode.SYSTEM_INTERNAL_ERROR, "这是一个通用错误测试")

if __name__ == "__main__":
    # 启动服务器
    logger.startup("启动 VoiceHelper Algorithm Service", context={
        "service": SERVICE_NAME,
        "host": HOST,
        "port": PORT,
        "local_ip": LOCAL_IP,
        "environment": ENV,
        "log_level": LOG_LEVEL,
        "pid": os.getpid(),
        "service_url": f"http://{LOCAL_IP}:{PORT}",
        "health_check_url": f"http://{LOCAL_IP}:{PORT}/health",
        "docs_url": f"http://{LOCAL_IP}:{PORT}/docs" if ENV == "development" else None,
        "endpoints": [
            f"GET http://{LOCAL_IP}:{PORT}/",
            f"GET http://{LOCAL_IP}:{PORT}/health",
            f"POST http://{LOCAL_IP}:{PORT}/ingest",
            f"POST http://{LOCAL_IP}:{PORT}/query",
            f"POST http://{LOCAL_IP}:{PORT}/voice/query",
            f"GET http://{LOCAL_IP}:{PORT}/tasks/{{task_id}}",
            f"POST http://{LOCAL_IP}:{PORT}/cancel",
            f"GET http://{LOCAL_IP}:{PORT}/error-test",
        ]
    })
    
    try:
        uvicorn.run(
            "main:app",
            host=HOST,
            port=PORT,
            reload=True if ENV == "development" else False,
            log_config=get_uvicorn_log_config(),
            access_log=True,
        )
    except Exception as e:
        logger.exception("服务器启动失败", e, context={
            "host": HOST,
            "port": PORT,
            "service": SERVICE_NAME,
        })
        raise
