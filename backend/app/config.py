"""
配置中心 - 使用pydantic-settings
"""
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """应用配置"""
    
    # 基础配置
    DEBUG: bool = Field(default=False, env="DEBUG")
    PORT: int = Field(default=8000, env="PORT")
    HOST: str = Field(default="0.0.0.0", env="HOST")
    
    # CORS配置
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:3001"],
        env="ALLOWED_ORIGINS"
    )
    TRUSTED_HOSTS: Optional[List[str]] = Field(default=None, env="TRUSTED_HOSTS")
    
    # 数据库配置
    DATABASE_URL: str = Field(env="DATABASE_URL")
    REDIS_URL: str = Field(env="REDIS_URL")
    
    # 语音服务配置
    VOICE_MODE: str = Field(default="webrtc", env="VOICE_MODE")  # webrtc | websocket
    STT_PROVIDER: str = Field(default="deepgram", env="STT_PROVIDER")  # deepgram | riva
    TTS_PROVIDER: str = Field(default="aura", env="TTS_PROVIDER")  # aura | openai_rt | elevenlabs
    LLM_ROUTER: str = Field(default="default", env="LLM_ROUTER")  # default | lowcost-first | latency-first
    LATENCY_BUDGET_P95: int = Field(default=700, env="LATENCY_BUDGET_P95")
    
    # API密钥
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    DEEPGRAM_API_KEY: Optional[str] = Field(default=None, env="DEEPGRAM_API_KEY")
    ELEVENLABS_API_KEY: Optional[str] = Field(default=None, env="ELEVENLABS_API_KEY")
    
    # WebRTC配置
    LIVEKIT_KEY: Optional[str] = Field(default=None, env="LIVEKIT_KEY")
    LIVEKIT_SECRET: Optional[str] = Field(default=None, env="LIVEKIT_SECRET")
    LIVEKIT_URL: Optional[str] = Field(default=None, env="LIVEKIT_URL")
    
    # 本地化服务
    RIVA_SERVER: Optional[str] = Field(default=None, env="RIVA_SERVER")
    
    # 可观测性
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = Field(
        default=None, 
        env="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    PROMETHEUS_METRICS_PORT: int = Field(default=9090, env="PROMETHEUS_METRICS_PORT")
    
    # VAD配置
    VAD_MIN_SPEECH_MS: int = Field(default=120, env="VAD_MIN_SPEECH_MS")
    VAD_MIN_SILENCE_MS: int = Field(default=200, env="VAD_MIN_SILENCE_MS")
    VAD_ENERGY_THRESHOLD: float = Field(default=-45.0, env="VAD_ENERGY_THRESHOLD")
    
    # 音频配置
    AUDIO_SAMPLE_RATE: int = Field(default=16000, env="AUDIO_SAMPLE_RATE")
    AUDIO_CHANNELS: int = Field(default=1, env="AUDIO_CHANNELS")
    AUDIO_CHUNK_SIZE_MS: int = Field(default=100, env="AUDIO_CHUNK_SIZE_MS")
    
    # 缓存配置
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")
    SEMANTIC_CACHE_THRESHOLD: float = Field(default=0.85, env="SEMANTIC_CACHE_THRESHOLD")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# 全局配置实例
settings = Settings()
