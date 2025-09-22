"""
VoiceHelper Backend - FastAPI Application
实时语音助手后端服务
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
import structlog

from app.config import settings
from app.routers import health, chat, realtime, tools_mcp, admin
from app.services.obs.tracing import setup_tracing
from app.services.obs.metrics import setup_metrics


# 配置结构化日志
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    logger.info("Starting VoiceHelper Backend", version="2.0.0")
    
    # 初始化追踪和指标
    setup_tracing()
    setup_metrics()
    
    yield
    
    # 关闭时清理
    logger.info("Shutting down VoiceHelper Backend")


# 创建FastAPI应用
app = FastAPI(
    title="VoiceHelper API",
    description="实时语音助手后端服务",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

if settings.TRUSTED_HOSTS:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.TRUSTED_HOSTS
    )


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """统一异常处理"""
    trace_id = trace.get_current_span().get_span_context().trace_id
    
    logger.error(
        "Unhandled exception",
        exc_info=exc,
        trace_id=f"{trace_id:032x}",
        path=request.url.path,
        method=request.method
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "Internal server error",
            "trace_id": f"{trace_id:032x}"
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理"""
    trace_id = trace.get_current_span().get_span_context().trace_id
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": f"HTTP_{exc.status_code}",
            "message": exc.detail,
            "trace_id": f"{trace_id:032x}"
        }
    )


# 路由注册
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(realtime.router, prefix="/realtime", tags=["realtime"])
app.include_router(tools_mcp.router, prefix="/tools", tags=["tools"])
app.include_router(admin.router, prefix="/admin", tags=["admin"])


# 根路径
@app.get("/")
async def root():
    return {
        "service": "VoiceHelper Backend",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs"
    }


# OpenTelemetry自动化仪表
FastAPIInstrumentor.instrument_app(app)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_config=None,  # 使用structlog
        access_log=False
    )
