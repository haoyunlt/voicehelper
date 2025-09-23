"""
请求模型定义
使用Pydantic进行数据验证和序列化
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum
import re


class AudioFormat(str, Enum):
    """音频格式枚举"""
    WAV = "wav"
    MP3 = "mp3"
    WEBM = "webm"
    M4A = "m4a"
    FLAC = "flac"
    OGG = "ogg"


class LanguageCode(str, Enum):
    """语言代码枚举"""
    ZH_CN = "zh-CN"
    ZH_TW = "zh-TW"
    EN_US = "en-US"
    EN_GB = "en-GB"
    JA_JP = "ja-JP"
    KO_KR = "ko-KR"
    FR_FR = "fr-FR"
    DE_DE = "de-DE"
    ES_ES = "es-ES"
    IT_IT = "it-IT"
    PT_BR = "pt-BR"
    RU_RU = "ru-RU"


class EmotionType(str, Enum):
    """情感类型枚举"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEAR = "fear"
    DISGUST = "disgust"
    CALM = "calm"
    EXCITED = "excited"
    FRUSTRATED = "frustrated"
    CONFIDENT = "confident"


class SearchType(str, Enum):
    """搜索类型枚举"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    FUZZY = "fuzzy"


class FusionType(str, Enum):
    """融合类型枚举"""
    EARLY = "early"
    LATE = "late"
    ATTENTION = "attention"
    GATED = "gated"
    HIERARCHICAL = "hierarchical"


# RAG相关请求模型

class DocumentInput(BaseModel):
    """文档输入模型"""
    id: str = Field(..., min_length=1, max_length=100, description="文档ID")
    title: Optional[str] = Field("", max_length=200, description="文档标题")
    content: str = Field(..., min_length=1, max_length=100000, description="文档内容")
    source: Optional[str] = Field("", max_length=500, description="文档来源")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")
    language: Optional[LanguageCode] = Field(LanguageCode.ZH_CN, description="语言代码")
    
    @validator('content')
    def validate_content(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Content cannot be empty")
        return v.strip()


class IngestRequest(BaseModel):
    """入库请求模型"""
    documents: List[DocumentInput] = Field(..., min_items=1, max_items=100, description="文档列表")
    batch_size: Optional[int] = Field(10, ge=1, le=100, description="批处理大小")
    overwrite: Optional[bool] = Field(False, description="是否覆盖现有文档")
    
    @validator('documents')
    def validate_documents(cls, v):
        if not v:
            raise ValueError("Documents list cannot be empty")
        
        # 检查文档ID唯一性
        doc_ids = [doc.id for doc in v]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("Document IDs must be unique")
        
        return v


class QueryRequest(BaseModel):
    """查询请求模型"""
    query: str = Field(..., min_length=1, max_length=1000, description="查询内容")
    top_k: Optional[int] = Field(5, ge=1, le=100, description="返回结果数量")
    threshold: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="相似度阈值")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="过滤条件")
    include_metadata: Optional[bool] = Field(True, description="是否包含元数据")
    search_type: Optional[SearchType] = Field(SearchType.SEMANTIC, description="搜索类型")
    language: Optional[LanguageCode] = Field(LanguageCode.ZH_CN, description="查询语言")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Query cannot be empty")
        return v.strip()


class DeleteRequest(BaseModel):
    """删除请求模型"""
    document_ids: List[str] = Field(..., min_items=1, max_items=1000, description="文档ID列表")
    
    @validator('document_ids')
    def validate_document_ids(cls, v):
        if not v:
            raise ValueError("Document IDs list cannot be empty")
        
        # 检查ID格式
        for doc_id in v:
            if not doc_id or doc_id.strip() == "":
                raise ValueError("Document ID cannot be empty")
        
        return [doc_id.strip() for doc_id in v]


# 语音相关请求模型

class TranscribeRequest(BaseModel):
    """语音转文字请求"""
    audio_data: Optional[bytes] = Field(None, description="音频数据")
    audio_url: Optional[str] = Field(None, description="音频URL")
    language: LanguageCode = Field(LanguageCode.ZH_CN, description="语言代码")
    audio_format: AudioFormat = Field(AudioFormat.WAV, description="音频格式")
    sample_rate: Optional[int] = Field(16000, ge=8000, le=48000, description="采样率")
    enable_emotion: Optional[bool] = Field(False, description="启用情感识别")
    enable_speaker: Optional[bool] = Field(False, description="启用说话人识别")
    max_duration: Optional[int] = Field(300, ge=1, le=3600, description="最大时长（秒）")
    
    @validator('audio_data', 'audio_url')
    def validate_audio_input(cls, v, values):
        audio_data = values.get('audio_data')
        audio_url = values.get('audio_url')
        
        if not audio_data and not audio_url:
            raise ValueError("Either audio_data or audio_url must be provided")
        
        if audio_data and audio_url:
            raise ValueError("Only one of audio_data or audio_url should be provided")
        
        return v
    
    @validator('audio_url')
    def validate_audio_url(cls, v):
        if v:
            url_pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
            if not re.match(url_pattern, v):
                raise ValueError("Invalid URL format")
        return v


class SynthesizeRequest(BaseModel):
    """文字转语音请求"""
    text: str = Field(..., min_length=1, max_length=5000, description="要合成的文本")
    voice_id: str = Field(..., min_length=1, max_length=100, description="语音ID")
    language: LanguageCode = Field(LanguageCode.ZH_CN, description="语言代码")
    speed: Optional[float] = Field(1.0, ge=0.5, le=2.0, description="语速")
    pitch: Optional[float] = Field(0.0, ge=-20.0, le=20.0, description="音调")
    volume: Optional[float] = Field(1.0, ge=0.1, le=2.0, description="音量")
    emotion: Optional[EmotionType] = Field(EmotionType.NEUTRAL, description="情感")
    output_format: Optional[AudioFormat] = Field(AudioFormat.WAV, description="输出格式")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Text cannot be empty")
        return v.strip()


class VoiceStreamRequest(BaseModel):
    """语音流请求"""
    session_id: str = Field(..., min_length=1, max_length=100, description="会话ID")
    language: LanguageCode = Field(LanguageCode.ZH_CN, description="语言代码")
    audio_format: AudioFormat = Field(AudioFormat.WAV, description="音频格式")
    sample_rate: Optional[int] = Field(16000, ge=8000, le=48000, description="采样率")
    chunk_size: Optional[int] = Field(1024, ge=512, le=8192, description="数据块大小")
    enable_vad: Optional[bool] = Field(True, description="启用语音活动检测")
    enable_nr: Optional[bool] = Field(True, description="启用噪音抑制")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="超时时间（秒）")


# 对话相关请求模型

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., min_length=1, max_length=10000, description="消息内容")
    conversation_id: Optional[str] = Field(None, min_length=1, max_length=100, description="会话ID")
    message_type: Optional[str] = Field("text", description="消息类型")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="上下文")
    stream: Optional[bool] = Field(False, description="是否流式响应")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: Optional[int] = Field(2048, ge=1, le=8192, description="最大令牌数")
    model: Optional[str] = Field("default", min_length=1, description="模型名称")
    
    @validator('message')
    def validate_message(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Message cannot be empty")
        return v.strip()


class CancelChatRequest(BaseModel):
    """取消聊天请求"""
    conversation_id: str = Field(..., min_length=1, max_length=100, description="会话ID")
    reason: Optional[str] = Field("", max_length=500, description="取消原因")


# 多模态相关请求模型

class MultimodalRequest(BaseModel):
    """多模态请求"""
    query: str = Field(..., min_length=1, max_length=1000, description="查询内容")
    text: Optional[str] = Field(None, max_length=5000, description="文本内容")
    image_data: Optional[bytes] = Field(None, description="图像数据")
    audio_data: Optional[bytes] = Field(None, description="音频数据")
    video_data: Optional[bytes] = Field(None, description="视频数据")
    fusion_type: Optional[FusionType] = Field(FusionType.ATTENTION, description="融合类型")
    language: Optional[LanguageCode] = Field(LanguageCode.ZH_CN, description="语言代码")
    output_type: Optional[str] = Field("text", description="输出类型")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Query cannot be empty")
        return v.strip()
    
    @validator('text', 'image_data', 'audio_data', 'video_data')
    def validate_multimodal_input(cls, v, values):
        # 至少需要一种模态的输入
        inputs = [
            values.get('text'),
            values.get('image_data'),
            values.get('audio_data'),
            values.get('video_data')
        ]
        
        if not any(inputs):
            raise ValueError("At least one modality input is required")
        
        return v


# Agent相关请求模型

class AgentRequest(BaseModel):
    """Agent请求"""
    query: str = Field(..., min_length=1, max_length=5000, description="查询内容")
    session_id: Optional[str] = Field(None, min_length=1, max_length=100, description="会话ID")
    tools: Optional[List[str]] = Field(default_factory=list, description="可用工具列表")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="上下文")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="温度参数")
    max_steps: Optional[int] = Field(10, ge=1, le=20, description="最大步骤数")
    timeout: Optional[int] = Field(60, ge=1, le=300, description="超时时间（秒）")
    
    @validator('query')
    def validate_query(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Query cannot be empty")
        return v.strip()


class ToolExecuteRequest(BaseModel):
    """工具执行请求"""
    tool_name: str = Field(..., min_length=1, max_length=100, description="工具名称")
    parameters: Dict[str, Any] = Field(..., description="工具参数")
    session_id: Optional[str] = Field(None, min_length=1, max_length=100, description="会话ID")
    timeout: Optional[int] = Field(30, ge=1, le=300, description="超时时间（秒）")
    
    @validator('parameters')
    def validate_parameters(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Parameters must be a dictionary")
        return v


# 批量操作请求模型

class BatchOperation(BaseModel):
    """批量操作"""
    id: str = Field(..., min_length=1, max_length=100, description="操作ID")
    type: str = Field(..., min_length=1, description="操作类型")
    parameters: Dict[str, Any] = Field(..., description="操作参数")
    
    @validator('type')
    def validate_type(cls, v):
        valid_types = ["transcribe", "synthesize", "search", "ingest", "delete"]
        if v not in valid_types:
            raise ValueError(f"Type must be one of: {', '.join(valid_types)}")
        return v


class BatchRequest(BaseModel):
    """批量请求"""
    operations: List[BatchOperation] = Field(..., min_items=1, max_items=100, description="操作列表")
    parallel: Optional[bool] = Field(True, description="是否并行执行")
    timeout: Optional[int] = Field(300, ge=1, le=600, description="超时时间（秒）")
    
    @validator('operations')
    def validate_operations(cls, v):
        if not v:
            raise ValueError("Operations list cannot be empty")
        
        # 检查操作ID唯一性
        op_ids = [op.id for op in v]
        if len(op_ids) != len(set(op_ids)):
            raise ValueError("Operation IDs must be unique")
        
        return v


# 性能监控请求模型

class MetricsRequest(BaseModel):
    """指标请求"""
    start_time: Optional[int] = Field(None, ge=0, description="开始时间戳")
    end_time: Optional[int] = Field(None, ge=0, description="结束时间戳")
    metrics: Optional[List[str]] = Field(default_factory=list, description="指标列表")
    interval: Optional[str] = Field("1m", description="时间间隔")
    
    @validator('interval')
    def validate_interval(cls, v):
        valid_intervals = ["1m", "5m", "15m", "1h", "1d"]
        if v not in valid_intervals:
            raise ValueError(f"Interval must be one of: {', '.join(valid_intervals)}")
        return v
    
    @validator('end_time')
    def validate_time_range(cls, v, values):
        start_time = values.get('start_time')
        if start_time and v and v <= start_time:
            raise ValueError("End time must be greater than start time")
        return v


# 健康检查请求模型

class HealthCheckRequest(BaseModel):
    """健康检查请求"""
    deep: Optional[bool] = Field(False, description="是否进行深度检查")
    components: Optional[List[str]] = Field(default_factory=list, description="检查组件列表")


# 配置相关请求模型

class ConfigUpdateRequest(BaseModel):
    """配置更新请求"""
    component: str = Field(..., min_length=1, description="组件名称")
    config: Dict[str, Any] = Field(..., description="配置内容")
    force: Optional[bool] = Field(False, description="是否强制更新")
    
    @validator('component')
    def validate_component(cls, v):
        valid_components = ["rag", "voice", "chat", "agent", "multimodal"]
        if v not in valid_components:
            raise ValueError(f"Component must be one of: {', '.join(valid_components)}")
        return v
    
    @validator('config')
    def validate_config(cls, v):
        if not isinstance(v, dict):
            raise ValueError("Config must be a dictionary")
        return v
