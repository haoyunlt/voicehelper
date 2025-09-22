import asyncio
import base64
import json
import io
import wave
import time
import os
from typing import AsyncGenerator, Optional, Dict, Any
from datetime import datetime

from fastapi import HTTPException

from core.config import config
from core.models import VoiceQueryRequest, VoiceQueryResponse, Reference
from core.metrics import voice_metrics_collector
from core.enhanced_voice_services import (
    EnhancedVoiceService, VoiceConfig, VoiceProvider
)

class ASRService:
    """语音识别服务（兼容性包装）"""
    
    def __init__(self):
        # 创建语音配置
        self.voice_config = self._create_voice_config()
        # 初始化增强语音服务（仅ASR部分）
        from core.enhanced_voice_services import EnhancedASRService
        self.enhanced_asr = EnhancedASRService(self.voice_config)
        
    def _create_voice_config(self) -> VoiceConfig:
        """创建语音配置"""
        # 从环境变量获取配置
        provider_configs = {}
        
        # OpenAI配置
        if os.getenv('OPENAI_API_KEY'):
            provider_configs[VoiceProvider.OPENAI] = {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            }
        
        # Azure配置
        if os.getenv('AZURE_SPEECH_KEY'):
            provider_configs[VoiceProvider.AZURE] = {
                'api_key': os.getenv('AZURE_SPEECH_KEY'),
                'region': os.getenv('AZURE_SPEECH_REGION', 'eastus')
            }
        
        # 确定主要提供商
        primary_provider = VoiceProvider.EDGE_TTS  # 默认使用免费的Edge TTS
        if os.getenv('OPENAI_API_KEY'):
            primary_provider = VoiceProvider.OPENAI
        elif os.getenv('AZURE_SPEECH_KEY'):
            primary_provider = VoiceProvider.AZURE
        
        return VoiceConfig(
            primary_asr_provider=primary_provider,
            fallback_asr_providers=[VoiceProvider.LOCAL],
            provider_configs=provider_configs,
            enable_vad=True,
            enable_cache=True
        )
        
    async def transcribe_audio(self, audio_data: bytes, is_final: bool = False, session_id: str = "") -> Optional[str]:
        """转写音频数据"""
        start_time = time.time()
        
        try:
            # 使用增强的ASR服务
            result = await self.enhanced_asr.transcribe(
                audio_data, 
                is_final=is_final, 
                session_id=session_id
            )
            
            # 记录指标
            if session_id and result:
                latency = time.time() - start_time
                voice_metrics_collector.record_asr_metrics(session_id, latency, accuracy=0.95)
            
            return result
                
        except Exception as e:
            print(f"ASR transcription error: {e}")
            return None

class TTSService:
    """文本转语音服务（兼容性包装）"""
    
    def __init__(self):
        self.voice = "zh-CN-XiaoxiaoNeural"  # 使用 Edge TTS 中文语音
        # 创建语音配置
        self.voice_config = self._create_voice_config()
        # 初始化增强语音服务（仅TTS部分）
        from core.enhanced_voice_services import EnhancedTTSService
        self.enhanced_tts = EnhancedTTSService(self.voice_config)
        
    def _create_voice_config(self) -> VoiceConfig:
        """创建语音配置"""
        # 从环境变量获取配置
        provider_configs = {}
        
        # OpenAI配置
        if os.getenv('OPENAI_API_KEY'):
            provider_configs[VoiceProvider.OPENAI] = {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            }
        
        # Azure配置
        if os.getenv('AZURE_SPEECH_KEY'):
            provider_configs[VoiceProvider.AZURE] = {
                'api_key': os.getenv('AZURE_SPEECH_KEY'),
                'region': os.getenv('AZURE_SPEECH_REGION', 'eastus')
            }
        
        # 确定主要提供商（TTS优先使用免费的Edge TTS）
        primary_provider = VoiceProvider.EDGE_TTS
        fallback_providers = []
        
        if os.getenv('AZURE_SPEECH_KEY'):
            fallback_providers.append(VoiceProvider.AZURE)
        if os.getenv('OPENAI_API_KEY'):
            fallback_providers.append(VoiceProvider.OPENAI)
        
        return VoiceConfig(
            primary_tts_provider=primary_provider,
            fallback_tts_providers=fallback_providers,
            tts_voice=self.voice,
            provider_configs=provider_configs,
            enable_cache=True
        )
        
    async def synthesize_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """流式合成语音"""
        try:
            # 使用增强的TTS服务
            async for chunk in self.enhanced_tts.synthesize_streaming(text, voice=self.voice):
                if chunk:
                    yield chunk
                    
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return
    
    async def synthesize_sentence(self, sentence: str) -> bytes:
        """合成单个句子"""
        try:
            # 使用增强的TTS服务
            result = await self.enhanced_tts.synthesize(sentence, voice=self.voice)
            return result
            
        except Exception as e:
            print(f"TTS sentence synthesis error: {e}")
            return b""

class VoiceService:
    """语音处理服务（兼容性包装）"""
    
    def __init__(self, retrieve_service):
        # 创建语音配置
        self.voice_config = self._create_voice_config()
        
        # 使用增强的语音服务
        self.enhanced_voice_service = EnhancedVoiceService(
            self.voice_config, 
            retrieve_service
        )
        
        # 保持兼容性
        self.asr_service = ASRService()
        self.tts_service = TTSService()
        self.retrieve_service = retrieve_service
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    def _create_voice_config(self) -> VoiceConfig:
        """创建语音配置"""
        # 从环境变量获取配置
        provider_configs = {}
        
        # OpenAI配置
        if os.getenv('OPENAI_API_KEY'):
            provider_configs[VoiceProvider.OPENAI] = {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
            }
        
        # Azure配置
        if os.getenv('AZURE_SPEECH_KEY'):
            provider_configs[VoiceProvider.AZURE] = {
                'api_key': os.getenv('AZURE_SPEECH_KEY'),
                'region': os.getenv('AZURE_SPEECH_REGION', 'eastus')
            }
        
        # 确定主要提供商
        primary_asr = VoiceProvider.EDGE_TTS
        primary_tts = VoiceProvider.EDGE_TTS
        
        if os.getenv('OPENAI_API_KEY'):
            primary_asr = VoiceProvider.OPENAI
        elif os.getenv('AZURE_SPEECH_KEY'):
            primary_asr = VoiceProvider.AZURE
        
        return VoiceConfig(
            primary_asr_provider=primary_asr,
            primary_tts_provider=primary_tts,
            fallback_asr_providers=[VoiceProvider.LOCAL],
            fallback_tts_providers=[VoiceProvider.AZURE, VoiceProvider.OPENAI],
            provider_configs=provider_configs,
            enable_vad=True,
            enable_cache=True
        )
        
    async def process_voice_query(self, request: VoiceQueryRequest) -> AsyncGenerator[VoiceQueryResponse, None]:
        """处理语音查询（使用增强服务）"""
        try:
            # 使用增强的语音服务处理请求
            async for response in self.enhanced_voice_service.process_voice_query(request):
                yield response
                
        except Exception as e:
            yield VoiceQueryResponse(
                type="error",
                error=f"Voice processing error: {str(e)}"
            )
    
    async def _process_rag_query(self, query: str, session_id: str) -> AsyncGenerator[VoiceQueryResponse, None]:
        """处理 RAG 查询并生成语音"""
        try:
            # 构建查询请求（复用现有的检索服务）
            from core.models import QueryRequest, Message
            
            query_request = QueryRequest(
                messages=[Message(role="user", content=query)],
                top_k=5,
                temperature=0.3
            )
            
            # 获取检索结果和 LLM 响应
            full_response = ""
            references = []
            
            async for response in self.retrieve_service.stream_query(query_request):
                response_data = json.loads(response)
                
                if response_data["type"] == "refs" and response_data.get("refs"):
                    references = response_data["refs"]
                    yield VoiceQueryResponse(
                        type="refs",
                        refs=[Reference(**ref) for ref in references]
                    )
                
                elif response_data["type"] == "delta" and response_data.get("content"):
                    content = response_data["content"]
                    full_response += content
                    
                    # 发送文本增量
                    yield VoiceQueryResponse(
                        type="llm_delta",
                        text=content
                    )
                    
                    # 检查是否为完整句子，进行 TTS
                    if self._is_complete_sentence(full_response):
                        sentence = self._extract_last_sentence(full_response)
                        if sentence:
                            async for tts_response in self._synthesize_and_stream(sentence, session_id):
                                yield tts_response
            
            # 处理剩余文本
            if full_response and not self._is_complete_sentence(full_response):
                async for tts_response in self._synthesize_and_stream(full_response, session_id):
                    yield tts_response
            
            yield VoiceQueryResponse(type="done")
            
        except Exception as e:
            yield VoiceQueryResponse(
                type="error",
                error=f"RAG processing error: {str(e)}"
            )
    
    async def _synthesize_and_stream(self, text: str, session_id: str = "") -> AsyncGenerator[VoiceQueryResponse, None]:
        """合成语音并流式返回"""
        start_time = time.time()
        first_audio_time = None
        
        try:
            # 语音友好化处理
            voice_text = self._make_voice_friendly(text)
            
            # 流式合成
            seq = 0
            async for audio_chunk in self.tts_service.synthesize_streaming(voice_text):
                if audio_chunk:
                    # 记录首音时间
                    if first_audio_time is None:
                        first_audio_time = time.time() - start_time
                    
                    # 转换为 PCM 格式并编码
                    pcm_data = self._convert_to_pcm(audio_chunk)
                    if pcm_data:
                        yield VoiceQueryResponse(
                            type="tts_chunk",
                            seq=seq,
                            pcm=base64.b64encode(pcm_data).decode('utf-8')
                        )
                        seq += 1
            
            # 记录 TTS 指标
            if session_id:
                total_latency = time.time() - start_time
                voice_metrics_collector.record_tts_metrics(
                    session_id, total_latency, first_audio_time or 0
                )
                        
        except Exception as e:
            print(f"TTS streaming error: {e}")
    
    def _make_voice_friendly(self, text: str) -> str:
        """将文本转换为语音友好格式"""
        # 移除引用编号（在屏幕上显示，语音中不读出）
        import re
        text = re.sub(r'\[\d+\]', '', text)
        
        # 简化复杂句子
        text = text.replace('根据检索到的信息', '根据资料')
        text = text.replace('基于以上内容', '综合来看')
        
        # 添加适当停顿
        text = text.replace('。', '。 ')
        text = text.replace('！', '！ ')
        text = text.replace('？', '？ ')
        
        return text.strip()
    
    def _is_complete_sentence(self, text: str) -> bool:
        """判断是否为完整句子"""
        return text.endswith(('。', '！', '？', '.', '!', '?'))
    
    def _extract_last_sentence(self, text: str) -> str:
        """提取最后一个完整句子"""
        import re
        sentences = re.split(r'[。！？.!?]', text)
        if len(sentences) >= 2:
            return sentences[-2] + text[-1]  # 包含标点符号
        return ""
    
    def _convert_to_pcm(self, audio_data: bytes) -> Optional[bytes]:
        """转换音频为 PCM 格式"""
        try:
            # 这里需要根据实际的音频格式进行转换
            # Edge TTS 返回的是 MP3 格式，需要转换为 PCM
            # 简化实现，实际需要使用 ffmpeg 或其他音频处理库
            return audio_data  # 临时返回原始数据
        except Exception as e:
            print(f"Audio conversion error: {e}")
            return None
    
    async def cancel_request(self, request_id: str):
        """取消语音请求"""
        # 实现请求取消逻辑
        if request_id in self.active_sessions:
            del self.active_sessions[request_id]
    
    def cleanup_inactive_sessions(self):
        """清理非活跃会话"""
        # 使用增强服务的清理方法
        self.enhanced_voice_service.cleanup_inactive_sessions()
        
        # 保持兼容性，也清理本地会话
        current_time = datetime.now()
        inactive_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if (current_time - session["last_activity"]).seconds > 300:  # 5分钟超时
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            del self.active_sessions[session_id]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取语音服务统计信息"""
        return self.enhanced_voice_service.get_stats()
