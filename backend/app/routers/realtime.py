"""
实时语音会话管理API
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
import structlog
import uuid
from datetime import datetime, timedelta

from app.deps import (
    get_redis, get_signaling_service, get_tts_service, 
    get_llm_router, require_auth
)
from app.models.schemas import (
    SessionCreateRequest, SessionCreateResponse, 
    CancelTTSRequest, BargeInRequest
)
from app.services.obs.tracing import span_ctx
from app.services.obs.metrics import record_session_event

logger = structlog.get_logger()
router = APIRouter()


@router.post("/session", response_model=SessionCreateResponse)
async def create_session(
    request: SessionCreateRequest,
    current_user: dict = Depends(require_auth),
    signaling_service = Depends(get_signaling_service),
    redis_client = Depends(get_redis)
):
    """
    创建实时语音会话
    
    返回WebRTC/WebSocket连接信息和服务路由策略
    """
    with span_ctx("session.create") as span:
        session_id = str(uuid.uuid4())
        user_id = current_user["user_id"]
        
        span.set_attribute("session_id", session_id)
        span.set_attribute("user_id", user_id)
        span.set_attribute("client_caps", str(request.client_capabilities))
        
        logger.info(
            "Creating voice session",
            session_id=session_id,
            user_id=user_id,
            capabilities=request.client_capabilities
        )
        
        # 生成会话令牌和策略
        token_data = await signaling_service.create_session_token(
            session_id=session_id,
            user_id=user_id,
            capabilities=request.client_capabilities
        )
        
        # 根据客户端能力和当前负载确定路由策略
        routing_policy = await _determine_routing_policy(
            request.client_capabilities,
            redis_client
        )
        
        # 存储会话信息到Redis
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(hours=2)).isoformat(),
            "routing_policy": routing_policy.dict(),
            "status": "active"
        }
        
        await redis_client.hset(
            f"session:{session_id}",
            mapping=session_data
        )
        await redis_client.expire(f"session:{session_id}", 7200)  # 2小时过期
        
        # 记录指标
        await record_session_event("session_created", session_id, user_id)
        
        return SessionCreateResponse(
            session_id=session_id,
            token=token_data["token"],
            expires_at=token_data["expires_at"],
            connection_info=token_data["connection_info"],
            routing_policy=routing_policy
        )


@router.post("/cancel-tts")
async def cancel_tts(
    request: CancelTTSRequest,
    current_user: dict = Depends(require_auth),
    tts_service = Depends(get_tts_service),
    redis_client = Depends(get_redis)
):
    """
    取消当前TTS播放（用于barge-in）
    """
    with span_ctx("tts.cancel") as span:
        session_id = request.session_id
        span.set_attribute("session_id", session_id)
        
        # 验证会话所有权
        session_data = await redis_client.hgetall(f"session:{session_id}")
        if not session_data or session_data.get("user_id") != current_user["user_id"]:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info("Cancelling TTS", session_id=session_id)
        
        # 取消TTS
        success = await tts_service.cancel(session_id)
        
        if success:
            # 更新会话状态
            await redis_client.hset(
                f"session:{session_id}",
                "last_tts_cancel",
                datetime.utcnow().isoformat()
            )
            
            # 记录指标
            await record_session_event("tts_cancelled", session_id, current_user["user_id"])
            
            return {"success": True, "message": "TTS cancelled"}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel TTS")


@router.post("/barge-in")
async def trigger_barge_in(
    request: BargeInRequest,
    current_user: dict = Depends(require_auth),
    tts_service = Depends(get_tts_service),
    redis_client = Depends(get_redis)
):
    """
    手动触发barge-in（调试用）
    """
    with span_ctx("barge_in.manual") as span:
        session_id = request.session_id
        span.set_attribute("session_id", session_id)
        
        # 验证会话
        session_data = await redis_client.hgetall(f"session:{session_id}")
        if not session_data or session_data.get("user_id") != current_user["user_id"]:
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(
            "Manual barge-in triggered",
            session_id=session_id,
            reason=request.reason
        )
        
        # 执行barge-in逻辑
        await tts_service.cancel(session_id)
        
        # 记录事件
        await redis_client.hset(
            f"session:{session_id}",
            "last_barge_in",
            datetime.utcnow().isoformat()
        )
        
        await record_session_event("barge_in_manual", session_id, current_user["user_id"])
        
        return {"success": True, "message": "Barge-in triggered"}


@router.get("/session/{session_id}/status")
async def get_session_status(
    session_id: str,
    current_user: dict = Depends(require_auth),
    redis_client = Depends(get_redis)
):
    """获取会话状态"""
    session_data = await redis_client.hgetall(f"session:{session_id}")
    if not session_data or session_data.get("user_id") != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "status": session_data.get("status"),
        "created_at": session_data.get("created_at"),
        "expires_at": session_data.get("expires_at"),
        "last_activity": session_data.get("last_activity")
    }


@router.delete("/session/{session_id}")
async def close_session(
    session_id: str,
    current_user: dict = Depends(require_auth),
    redis_client = Depends(get_redis),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """关闭会话"""
    session_data = await redis_client.hgetall(f"session:{session_id}")
    if not session_data or session_data.get("user_id") != current_user["user_id"]:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 标记会话为关闭
    await redis_client.hset(f"session:{session_id}", "status", "closed")
    
    # 后台清理资源
    background_tasks.add_task(_cleanup_session_resources, session_id)
    
    logger.info("Session closed", session_id=session_id)
    await record_session_event("session_closed", session_id, current_user["user_id"])
    
    return {"success": True, "message": "Session closed"}


async def _determine_routing_policy(client_capabilities: dict, redis_client) -> "RoutingPolicy":
    """根据客户端能力和系统负载确定路由策略"""
    from app.models.schemas import RoutingPolicy
    
    # 检查系统负载
    current_load = await redis_client.get("system:load") or "0"
    load_factor = float(current_load)
    
    # 根据客户端能力选择服务提供商
    stt_provider = "deepgram" if client_capabilities.get("webrtc", False) else "riva"
    tts_provider = "aura" if load_factor < 0.7 else "openai_rt"
    llm_model = "gpt-4o-mini" if load_factor < 0.8 else "gpt-3.5-turbo"
    
    return RoutingPolicy(
        stt_provider=stt_provider,
        tts_provider=tts_provider,
        llm_model=llm_model,
        max_latency_ms=700,
        enable_cache=True,
        enable_barge_in=True
    )


async def _cleanup_session_resources(session_id: str):
    """清理会话资源"""
    logger.info("Cleaning up session resources", session_id=session_id)
    # TODO: 清理WebRTC连接、取消未完成的TTS等
