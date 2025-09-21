import asyncio
import base64
import json
import io
import wave
from typing import AsyncGenerator, Optional, Dict, Any
from datetime import datetime

import numpy as np
import soundfile as sf
import speech_recognition as sr
import edge_tts
from fastapi import HTTPException

from core.config import config
from core.models import VoiceQueryRequest, VoiceQueryResponse, Reference
from core.metrics import voice_metrics_collector

class ASRService:
    """语音识别服务"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # 配置识别器参数
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3
        
    async def transcribe_audio(self, audio_data: bytes, is_final: bool = False, session_id: str = "") -> Optional[str]:
        """转写音频数据"""
        start_time = time.time()
        
        try:
            # 解码 Opus 音频数据（简化处理，实际需要 opus 解码器）
            # 这里假设接收到的是 PCM 数据
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 转换为 AudioData 对象
            audio_data_obj = sr.AudioData(
                audio_array.tobytes(),
                sample_rate=16000,
                sample_width=2
            )
            
            # 使用 Google Speech Recognition（免费版本，生产环境建议使用更稳定的服务）
            try:
                if is_final:
                    # 最终识别
                    text = self.recognizer.recognize_google(audio_data_obj, language='zh-CN')
                else:
                    # 部分识别（简化实现）
                    text = self.recognizer.recognize_google(audio_data_obj, language='zh-CN')
                
                # 记录 ASR 指标
                if session_id:
                    latency = time.time() - start_time
                    voice_metrics_collector.record_asr_metrics(session_id, latency, accuracy=0.95)
                
                return text
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"ASR service error: {e}")
                return None
                
        except Exception as e:
            print(f"ASR transcription error: {e}")
            return None

class TTSService:
    """文本转语音服务"""
    
    def __init__(self):
        self.voice = "zh-CN-XiaoxiaoNeural"  # 使用 Edge TTS 中文语音
        
    async def synthesize_streaming(self, text: str) -> AsyncGenerator[bytes, None]:
        """流式合成语音"""
        try:
            # 使用 Edge TTS 进行语音合成
            communicate = edge_tts.Communicate(text, self.voice)
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                    
        except Exception as e:
            print(f"TTS synthesis error: {e}")
            return
    
    async def synthesize_sentence(self, sentence: str) -> bytes:
        """合成单个句子"""
        try:
            communicate = edge_tts.Communicate(sentence, self.voice)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
        except Exception as e:
            print(f"TTS sentence synthesis error: {e}")
            return b""

class VoiceService:
    """语音处理服务"""
    
    def __init__(self, retrieve_service):
        self.asr_service = ASRService()
        self.tts_service = TTSService()
        self.retrieve_service = retrieve_service
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def process_voice_query(self, request: VoiceQueryRequest) -> AsyncGenerator[VoiceQueryResponse, None]:
        """处理语音查询"""
        try:
            session_id = request.conversation_id
            
            # 初始化会话
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    "audio_buffer": b"",
                    "transcript_buffer": "",
                    "last_activity": datetime.now()
                }
            
            session = self.active_sessions[session_id]
            
            # 解码音频数据
            audio_chunk = base64.b64decode(request.audio_chunk)
            session["audio_buffer"] += audio_chunk
            session["last_activity"] = datetime.now()
            
            # ASR 处理
            if len(session["audio_buffer"]) > 8000:  # 约0.5秒的音频
                # 部分识别
                partial_text = await self.asr_service.transcribe_audio(
                    session["audio_buffer"], is_final=False, session_id=session_id
                )
                
                if partial_text:
                    yield VoiceQueryResponse(
                        type="asr_partial",
                        seq=request.seq,
                        text=partial_text
                    )
                    
                # 检查是否为完整句子（简化判断）
                if partial_text and (partial_text.endswith('。') or 
                                   partial_text.endswith('？') or 
                                   partial_text.endswith('！')):
                    
                    # 最终识别
                    final_text = await self.asr_service.transcribe_audio(
                        session["audio_buffer"], is_final=True, session_id=session_id
                    )
                    
                    if final_text:
                        yield VoiceQueryResponse(
                            type="asr_final",
                            seq=request.seq,
                            text=final_text
                        )
                        
                        # 处理 RAG 查询
                        async for response in self._process_rag_query(final_text, session_id):
                            yield response
                    
                    # 清空缓冲区
                    session["audio_buffer"] = b""
                    session["transcript_buffer"] = ""
                    
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
        current_time = datetime.now()
        inactive_sessions = []
        
        for session_id, session in self.active_sessions.items():
            if (current_time - session["last_activity"]).seconds > 300:  # 5分钟超时
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            del self.active_sessions[session_id]
