"""
Dify-VoiceHelper 集成适配器
提供Dify与VoiceHelper之间的双向集成服务
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from adapters.dify_client import DifyClient
from adapters.voicehelper_client import VoiceHelperClient
from services.integration_service import IntegrationService
from services.workflow_service import WorkflowService
from services.data_sync_service import DataSyncService
from core.config import Settings
from core.database import DatabaseManager
from core.redis_client import RedisManager
from core.monitoring import setup_monitoring


# 配置设置
settings = Settings()

# 全局客户端实例
dify_client: Optional[DifyClient] = None
voicehelper_client: Optional[VoiceHelperClient] = None
integration_service: Optional[IntegrationService] = None
workflow_service: Optional[WorkflowService] = None
data_sync_service: Optional[DataSyncService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global dify_client, voicehelper_client, integration_service, workflow_service, data_sync_service
    
    logger.info("启动 Dify-VoiceHelper 集成适配器...")
    
    try:
        # 初始化数据库连接
        db_manager = DatabaseManager(settings.database_url)
        await db_manager.initialize()
        
        # 初始化Redis连接
        redis_manager = RedisManager(settings.redis_url)
        await redis_manager.initialize()
        
        # 初始化客户端
        dify_client = DifyClient(
            api_url=settings.dify_api_url,
            api_key=settings.dify_api_key
        )
        
        voicehelper_client = VoiceHelperClient(
            api_url=settings.voicehelper_api_url,
            algo_url=settings.voicehelper_algo_url
        )
        
        # 初始化服务
        integration_service = IntegrationService(
            dify_client=dify_client,
            voicehelper_client=voicehelper_client,
            db_manager=db_manager,
            redis_manager=redis_manager
        )
        
        workflow_service = WorkflowService(
            dify_client=dify_client,
            voicehelper_client=voicehelper_client
        )
        
        data_sync_service = DataSyncService(
            dify_client=dify_client,
            voicehelper_client=voicehelper_client,
            db_manager=db_manager
        )
        
        # 启动后台任务
        asyncio.create_task(data_sync_service.start_sync_tasks())
        
        logger.success("集成适配器启动成功")
        
        yield
        
    except Exception as e:
        logger.error(f"启动失败: {e}")
        raise
    finally:
        logger.info("关闭集成适配器...")
        
        # 清理资源
        if data_sync_service:
            await data_sync_service.stop_sync_tasks()
        
        if db_manager:
            await db_manager.close()
        
        if redis_manager:
            await redis_manager.close()


# 创建FastAPI应用
app = FastAPI(
    title="Dify-VoiceHelper Integration Adapter",
    description="提供Dify与VoiceHelper之间的双向集成服务",
    version="1.0.0",
    lifespan=lifespan
)

# 设置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置监控
setup_monitoring(app)


# ===========================================
# 数据模型
# ===========================================

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息")
    conversation_id: Optional[str] = Field(None, description="会话ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    app_id: str = Field(..., description="Dify应用ID")
    stream: bool = Field(False, description="是否流式响应")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class WorkflowRequest(BaseModel):
    """工作流请求模型"""
    workflow_id: str = Field(..., description="工作流ID")
    inputs: Dict[str, Any] = Field(..., description="输入参数")
    user_id: Optional[str] = Field(None, description="用户ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")


class SyncRequest(BaseModel):
    """数据同步请求模型"""
    sync_type: str = Field(..., description="同步类型: users|conversations|knowledge")
    direction: str = Field(..., description="同步方向: dify_to_vh|vh_to_dify|bidirectional")
    filters: Dict[str, Any] = Field(default_factory=dict, description="同步过滤条件")


# ===========================================
# 依赖注入
# ===========================================

def get_integration_service() -> IntegrationService:
    """获取集成服务实例"""
    if integration_service is None:
        raise HTTPException(status_code=500, detail="集成服务未初始化")
    return integration_service


def get_workflow_service() -> WorkflowService:
    """获取工作流服务实例"""
    if workflow_service is None:
        raise HTTPException(status_code=500, detail="工作流服务未初始化")
    return workflow_service


def get_data_sync_service() -> DataSyncService:
    """获取数据同步服务实例"""
    if data_sync_service is None:
        raise HTTPException(status_code=500, detail="数据同步服务未初始化")
    return data_sync_service


# ===========================================
# 健康检查和状态接口
# ===========================================

@app.get("/health")
async def health_check():
    """健康检查"""
    try:
        # 检查各服务状态
        dify_status = await dify_client.health_check() if dify_client else False
        vh_status = await voicehelper_client.health_check() if voicehelper_client else False
        
        return {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "services": {
                "dify": "healthy" if dify_status else "unhealthy",
                "voicehelper": "healthy" if vh_status else "unhealthy"
            }
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="健康检查失败")


@app.get("/status")
async def get_status(
    integration_svc: IntegrationService = Depends(get_integration_service)
):
    """获取详细状态信息"""
    try:
        status = await integration_svc.get_system_status()
        return JSONResponse(content=status)
    except Exception as e:
        logger.error(f"获取状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取状态失败")


# ===========================================
# 聊天集成接口
# ===========================================

@app.post("/api/v1/chat")
async def chat_with_dify(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    integration_svc: IntegrationService = Depends(get_integration_service)
):
    """通过Dify进行聊天"""
    try:
        if request.stream:
            # 流式响应
            return await integration_svc.stream_chat(
                message=request.message,
                app_id=request.app_id,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                metadata=request.metadata
            )
        else:
            # 普通响应
            response = await integration_svc.chat(
                message=request.message,
                app_id=request.app_id,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                metadata=request.metadata
            )
            
            # 后台记录聊天历史
            background_tasks.add_task(
                integration_svc.log_chat_history,
                request, response
            )
            
            return JSONResponse(content=response)
            
    except Exception as e:
        logger.error(f"聊天处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"聊天处理失败: {str(e)}")


@app.post("/api/v1/chat/voicehelper")
async def chat_with_voicehelper(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    integration_svc: IntegrationService = Depends(get_integration_service)
):
    """通过VoiceHelper进行聊天"""
    try:
        response = await integration_svc.chat_with_voicehelper(
            message=request.message,
            conversation_id=request.conversation_id,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        # 后台记录聊天历史
        background_tasks.add_task(
            integration_svc.log_chat_history,
            request, response
        )
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"VoiceHelper聊天处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"聊天处理失败: {str(e)}")


# ===========================================
# 工作流集成接口
# ===========================================

@app.post("/api/v1/workflow/run")
async def run_workflow(
    request: WorkflowRequest,
    workflow_svc: WorkflowService = Depends(get_workflow_service)
):
    """运行Dify工作流"""
    try:
        result = await workflow_svc.run_workflow(
            workflow_id=request.workflow_id,
            inputs=request.inputs,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"工作流运行失败: {e}")
        raise HTTPException(status_code=500, detail=f"工作流运行失败: {str(e)}")


@app.get("/api/v1/workflow/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    workflow_svc: WorkflowService = Depends(get_workflow_service)
):
    """获取工作流状态"""
    try:
        status = await workflow_svc.get_workflow_status(workflow_id)
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"获取工作流状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取工作流状态失败")


# ===========================================
# 数据同步接口
# ===========================================

@app.post("/api/v1/sync")
async def sync_data(
    request: SyncRequest,
    background_tasks: BackgroundTasks,
    sync_svc: DataSyncService = Depends(get_data_sync_service)
):
    """手动触发数据同步"""
    try:
        # 后台执行同步任务
        background_tasks.add_task(
            sync_svc.manual_sync,
            sync_type=request.sync_type,
            direction=request.direction,
            filters=request.filters
        )
        
        return JSONResponse(content={
            "message": "数据同步任务已启动",
            "sync_type": request.sync_type,
            "direction": request.direction
        })
        
    except Exception as e:
        logger.error(f"数据同步失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据同步失败: {str(e)}")


@app.get("/api/v1/sync/status")
async def get_sync_status(
    sync_svc: DataSyncService = Depends(get_data_sync_service)
):
    """获取数据同步状态"""
    try:
        status = await sync_svc.get_sync_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"获取同步状态失败: {e}")
        raise HTTPException(status_code=500, detail="获取同步状态失败")


# ===========================================
# 知识库集成接口
# ===========================================

@app.post("/api/v1/knowledge/sync")
async def sync_knowledge_base(
    background_tasks: BackgroundTasks,
    integration_svc: IntegrationService = Depends(get_integration_service)
):
    """同步知识库"""
    try:
        # 后台执行知识库同步
        background_tasks.add_task(integration_svc.sync_knowledge_base)
        
        return JSONResponse(content={
            "message": "知识库同步任务已启动"
        })
        
    except Exception as e:
        logger.error(f"知识库同步失败: {e}")
        raise HTTPException(status_code=500, detail=f"知识库同步失败: {str(e)}")


# ===========================================
# 配置管理接口
# ===========================================

@app.get("/api/v1/config")
async def get_config():
    """获取配置信息"""
    return {
        "dify_api_url": settings.dify_api_url,
        "voicehelper_api_url": settings.voicehelper_api_url,
        "adapter_version": "1.0.0",
        "features": {
            "chat_integration": True,
            "workflow_integration": True,
            "data_sync": True,
            "knowledge_sync": True
        }
    }


@app.post("/api/v1/config/reload")
async def reload_config():
    """重载配置"""
    try:
        # 重新加载配置
        global settings
        settings = Settings()
        
        return JSONResponse(content={
            "message": "配置重载成功"
        })
        
    except Exception as e:
        logger.error(f"配置重载失败: {e}")
        raise HTTPException(status_code=500, detail="配置重载失败")


# ===========================================
# 错误处理
# ===========================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "内部服务器错误"}
    )


# ===========================================
# 主程序入口
# ===========================================

if __name__ == "__main__":
    # 配置日志
    logger.add(
        "/app/logs/adapter.log",
        rotation="1 day",
        retention="30 days",
        level="INFO"
    )
    
    # 启动服务
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.adapter_port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )
