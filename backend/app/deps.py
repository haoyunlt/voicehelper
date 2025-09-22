"""
依赖注入 - FastAPI Dependencies
"""
from typing import AsyncGenerator, Optional
from fastapi import Depends, HTTPException, Header
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
import structlog

from app.config import settings
from app.services.llm.router import LLMRouter
from app.services.stt.base import STTService
from app.services.tts.base import TTSService
from app.services.vad.semantic_vad import SemanticVAD
from app.services.webrtc.signaling import SignalingService

logger = structlog.get_logger()

# 数据库引擎
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_recycle=3600
)

AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Redis连接池
redis_pool = None


async def get_redis_pool():
    """获取Redis连接池"""
    global redis_pool
    if redis_pool is None:
        redis_pool = redis.ConnectionPool.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=20
        )
    return redis_pool


async def get_redis() -> AsyncGenerator[redis.Redis, None]:
    """获取Redis连接"""
    pool = await get_redis_pool()
    async with redis.Redis(connection_pool=pool) as r:
        yield r


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """获取数据库会话"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_current_user(
    authorization: Optional[str] = Header(None)
) -> Optional[dict]:
    """获取当前用户（简化版JWT验证）"""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    
    token = authorization.split(" ")[1]
    # TODO: 实现JWT验证逻辑
    return {"user_id": "test_user", "token": token}


def require_auth(
    current_user: Optional[dict] = Depends(get_current_user)
) -> dict:
    """需要认证的依赖"""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return current_user


# 服务依赖
_llm_router = None
_stt_service = None
_tts_service = None
_vad_service = None
_signaling_service = None


async def get_llm_router() -> LLMRouter:
    """获取LLM路由服务"""
    global _llm_router
    if _llm_router is None:
        _llm_router = LLMRouter()
    return _llm_router


async def get_stt_service() -> STTService:
    """获取STT服务"""
    global _stt_service
    if _stt_service is None:
        if settings.STT_PROVIDER == "deepgram":
            from app.services.stt.deepgram import DeepgramSTT
            _stt_service = DeepgramSTT()
        elif settings.STT_PROVIDER == "riva":
            from app.services.stt.riva import RivaSTT
            _stt_service = RivaSTT()
        else:
            raise ValueError(f"Unknown STT provider: {settings.STT_PROVIDER}")
    return _stt_service


async def get_tts_service() -> TTSService:
    """获取TTS服务"""
    global _tts_service
    if _tts_service is None:
        if settings.TTS_PROVIDER == "aura":
            from app.services.tts.aura import AuraTTS
            _tts_service = AuraTTS()
        elif settings.TTS_PROVIDER == "openai_rt":
            from app.services.tts.openai_rt import OpenAIRealtimeTTS
            _tts_service = OpenAIRealtimeTTS()
        elif settings.TTS_PROVIDER == "elevenlabs":
            from app.services.tts.elevenlabs import ElevenLabsTTS
            _tts_service = ElevenLabsTTS()
        else:
            raise ValueError(f"Unknown TTS provider: {settings.TTS_PROVIDER}")
    return _tts_service


async def get_vad_service() -> SemanticVAD:
    """获取VAD服务"""
    global _vad_service
    if _vad_service is None:
        _vad_service = SemanticVAD(
            min_speech_ms=settings.VAD_MIN_SPEECH_MS,
            min_silence_ms=settings.VAD_MIN_SILENCE_MS,
            energy_thresh=settings.VAD_ENERGY_THRESHOLD
        )
    return _vad_service


async def get_signaling_service() -> SignalingService:
    """获取信令服务"""
    global _signaling_service
    if _signaling_service is None:
        _signaling_service = SignalingService()
    return _signaling_service
