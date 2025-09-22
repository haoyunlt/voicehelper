"""
语音服务提供商实现
支持多个ASR和TTS提供商，包括OpenAI、Azure Speech、Edge TTS等
"""

import asyncio
import io
import time
import wave
import json
import tempfile
import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Dict, Any, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class VoiceProvider(Enum):
    """语音服务提供商"""
    OPENAI = "openai"
    AZURE = "azure"
    EDGE_TTS = "edge_tts"
    LOCAL = "local"

class AudioFormat(Enum):
    """音频格式"""
    WAV = "wav"
    MP3 = "mp3"
    PCM = "pcm"
    WEBM = "webm"

class BaseASRProvider(ABC):
    """ASR提供商基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__
    
    @abstractmethod
    async def transcribe(self, audio_data: bytes, language: str = "zh-CN", 
                        is_final: bool = False) -> Optional[str]:
        """转写音频"""
        pass
    
    @abstractmethod
    async def transcribe_streaming(self, audio_stream: AsyncGenerator[bytes, None], 
                                 language: str = "zh-CN") -> AsyncGenerator[str, None]:
        """流式转写音频"""
        pass

class BaseTTSProvider(ABC):
    """TTS提供商基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__
    
    @abstractmethod
    async def synthesize(self, text: str, voice: str = None, 
                        language: str = "zh-CN") -> bytes:
        """合成语音"""
        pass
    
    @abstractmethod
    async def synthesize_streaming(self, text: str, voice: str = None, 
                                 language: str = "zh-CN") -> AsyncGenerator[bytes, None]:
        """流式合成语音"""
        pass

class OpenAIASRProvider(BaseASRProvider):
    """OpenAI ASR提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=config.get("api_key"),
                base_url=config.get("base_url", "https://api.openai.com/v1")
            )
        except ImportError:
            logger.error("OpenAI package not installed")
            raise
    
    async def transcribe(self, audio_data: bytes, language: str = "zh-CN", 
                        is_final: bool = False) -> Optional[str]:
        """使用OpenAI Whisper转写音频"""
        try:
            # 将音频数据保存为临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                # 调用OpenAI Whisper API
                with open(temp_file_path, "rb") as audio_file:
                    transcript = await self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=language.split("-")[0] if language else "zh",
                        response_format="text"
                    )
                
                return transcript.strip() if transcript else None
                
            finally:
                # 清理临时文件
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"OpenAI ASR error: {e}")
            return None
    
    async def transcribe_streaming(self, audio_stream: AsyncGenerator[bytes, None], 
                                 language: str = "zh-CN") -> AsyncGenerator[str, None]:
        """OpenAI暂不支持流式转写，使用批量处理"""
        buffer = b""
        async for chunk in audio_stream:
            buffer += chunk
            # 每收集足够数据就处理一次
            if len(buffer) > 32000:  # 约2秒音频
                result = await self.transcribe(buffer, language, is_final=False)
                if result:
                    yield result
                buffer = b""
        
        # 处理剩余数据
        if buffer:
            result = await self.transcribe(buffer, language, is_final=True)
            if result:
                yield result

class AzureASRProvider(BaseASRProvider):
    """Azure Speech ASR提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import azure.cognitiveservices.speech as speechsdk
            self.speechsdk = speechsdk
            
            speech_config = speechsdk.SpeechConfig(
                subscription=config.get("api_key"),
                region=config.get("region", "eastus")
            )
            speech_config.speech_recognition_language = "zh-CN"
            self.speech_config = speech_config
            
        except ImportError:
            logger.error("Azure Speech SDK not installed")
            raise
    
    async def transcribe(self, audio_data: bytes, language: str = "zh-CN", 
                        is_final: bool = False) -> Optional[str]:
        """使用Azure Speech转写音频"""
        try:
            # 创建音频配置
            audio_stream = self.speechsdk.audio.PushAudioInputStream()
            audio_config = self.speechsdk.audio.AudioConfig(stream=audio_stream)
            
            # 创建识别器
            self.speech_config.speech_recognition_language = language
            speech_recognizer = self.speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            # 推送音频数据
            audio_stream.write(audio_data)
            audio_stream.close()
            
            # 执行识别
            result = speech_recognizer.recognize_once()
            
            if result.reason == self.speechsdk.ResultReason.RecognizedSpeech:
                return result.text
            elif result.reason == self.speechsdk.ResultReason.NoMatch:
                logger.debug("Azure ASR: No speech could be recognized")
                return None
            else:
                logger.error(f"Azure ASR error: {result.reason}")
                return None
                
        except Exception as e:
            logger.error(f"Azure ASR error: {e}")
            return None
    
    async def transcribe_streaming(self, audio_stream: AsyncGenerator[bytes, None], 
                                 language: str = "zh-CN") -> AsyncGenerator[str, None]:
        """Azure流式转写"""
        try:
            # 创建流式音频配置
            stream = self.speechsdk.audio.PushAudioInputStream()
            audio_config = self.speechsdk.audio.AudioConfig(stream=stream)
            
            self.speech_config.speech_recognition_language = language
            speech_recognizer = self.speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # 设置事件处理器
            results = []
            
            def recognized_handler(evt):
                if evt.result.text:
                    results.append(evt.result.text)
            
            speech_recognizer.recognized.connect(recognized_handler)
            
            # 开始连续识别
            speech_recognizer.start_continuous_recognition()
            
            # 推送音频数据
            async for chunk in audio_stream:
                stream.write(chunk)
                await asyncio.sleep(0.01)  # 小延迟避免过快推送
                
                # 返回识别结果
                while results:
                    yield results.pop(0)
            
            # 停止识别
            stream.close()
            speech_recognizer.stop_continuous_recognition()
            
            # 返回剩余结果
            while results:
                yield results.pop(0)
                
        except Exception as e:
            logger.error(f"Azure streaming ASR error: {e}")

class LocalASRProvider(BaseASRProvider):
    """本地ASR提供商（使用speech_recognition库）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            # 配置识别器参数
            self.recognizer.energy_threshold = config.get("energy_threshold", 300)
            self.recognizer.dynamic_energy_threshold = config.get("dynamic_energy_threshold", True)
            self.recognizer.pause_threshold = config.get("pause_threshold", 0.8)
            self.recognizer.phrase_threshold = config.get("phrase_threshold", 0.3)
        except ImportError:
            logger.error("speech_recognition package not installed")
            raise
    
    async def transcribe(self, audio_data: bytes, language: str = "zh-CN", 
                        is_final: bool = False) -> Optional[str]:
        """使用本地speech_recognition转写音频"""
        try:
            import speech_recognition as sr
            
            # 将音频数据转换为AudioData对象
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                with sr.AudioFile(temp_file_path) as source:
                    audio = self.recognizer.record(source)
                
                # 使用Google Web Speech API（免费但有限制）
                text = self.recognizer.recognize_google(audio, language=language)
                return text
                
            finally:
                os.unlink(temp_file_path)
                
        except sr.UnknownValueError:
            logger.debug("Local ASR: Could not understand audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Local ASR service error: {e}")
            return None
        except Exception as e:
            logger.error(f"Local ASR error: {e}")
            return None
    
    async def transcribe_streaming(self, audio_stream: AsyncGenerator[bytes, None], 
                                 language: str = "zh-CN") -> AsyncGenerator[str, None]:
        """本地流式转写（批量处理）"""
        buffer = b""
        async for chunk in audio_stream:
            buffer += chunk
            if len(buffer) > 16000:  # 约1秒音频
                result = await self.transcribe(buffer, language)
                if result:
                    yield result
                buffer = b""
        
        if buffer:
            result = await self.transcribe(buffer, language, is_final=True)
            if result:
                yield result

class OpenAITTSProvider(BaseTTSProvider):
    """OpenAI TTS提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import openai
            self.client = openai.AsyncOpenAI(
                api_key=config.get("api_key"),
                base_url=config.get("base_url", "https://api.openai.com/v1")
            )
        except ImportError:
            logger.error("OpenAI package not installed")
            raise
    
    async def synthesize(self, text: str, voice: str = None, 
                        language: str = "zh-CN") -> bytes:
        """使用OpenAI TTS合成语音"""
        try:
            response = await self.client.audio.speech.create(
                model="tts-1",
                voice=voice or "alloy",
                input=text,
                response_format="mp3"
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"OpenAI TTS error: {e}")
            return b""
    
    async def synthesize_streaming(self, text: str, voice: str = None, 
                                 language: str = "zh-CN") -> AsyncGenerator[bytes, None]:
        """OpenAI TTS流式合成（分句处理）"""
        try:
            # 分句处理
            sentences = self._split_sentences(text)
            
            for sentence in sentences:
                if sentence.strip():
                    audio_data = await self.synthesize(sentence, voice, language)
                    if audio_data:
                        yield audio_data
                        await asyncio.sleep(0.01)  # 小延迟
                        
        except Exception as e:
            logger.error(f"OpenAI TTS streaming error: {e}")
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        import re
        sentences = re.split(r'[。！？.!?]', text)
        return [s.strip() for s in sentences if s.strip()]

class AzureTTSProvider(BaseTTSProvider):
    """Azure Speech TTS提供商"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import azure.cognitiveservices.speech as speechsdk
            self.speechsdk = speechsdk
            
            speech_config = speechsdk.SpeechConfig(
                subscription=config.get("api_key"),
                region=config.get("region", "eastus")
            )
            self.speech_config = speech_config
            
        except ImportError:
            logger.error("Azure Speech SDK not installed")
            raise
    
    async def synthesize(self, text: str, voice: str = None, 
                        language: str = "zh-CN") -> bytes:
        """使用Azure Speech合成语音"""
        try:
            # 设置语音
            voice_name = voice or "zh-CN-XiaoxiaoNeural"
            self.speech_config.speech_synthesis_voice_name = voice_name
            
            # 创建合成器
            synthesizer = self.speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=None
            )
            
            # 合成语音
            result = synthesizer.speak_text_async(text).get()
            
            if result.reason == self.speechsdk.ResultReason.SynthesizingAudioCompleted:
                return result.audio_data
            else:
                logger.error(f"Azure TTS error: {result.reason}")
                return b""
                
        except Exception as e:
            logger.error(f"Azure TTS error: {e}")
            return b""
    
    async def synthesize_streaming(self, text: str, voice: str = None, 
                                 language: str = "zh-CN") -> AsyncGenerator[bytes, None]:
        """Azure TTS流式合成"""
        try:
            # Azure支持流式合成
            voice_name = voice or "zh-CN-XiaoxiaoNeural"
            self.speech_config.speech_synthesis_voice_name = voice_name
            
            # 创建流式配置
            stream = self.speechsdk.audio.AudioOutputStream.create_push_stream()
            audio_config = self.speechsdk.audio.AudioOutputConfig(stream=stream)
            
            synthesizer = self.speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # 设置事件处理器
            audio_chunks = []
            
            def audio_data_handler(evt):
                if evt.audio_data:
                    audio_chunks.append(evt.audio_data)
            
            synthesizer.synthesizing.connect(audio_data_handler)
            
            # 开始合成
            result = synthesizer.speak_text_async(text).get()
            
            # 返回音频块
            for chunk in audio_chunks:
                yield chunk
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Azure TTS streaming error: {e}")

class EdgeTTSProvider(BaseTTSProvider):
    """Edge TTS提供商（免费）"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import edge_tts
            self.edge_tts = edge_tts
        except ImportError:
            logger.error("edge-tts package not installed")
            raise
    
    async def synthesize(self, text: str, voice: str = None, 
                        language: str = "zh-CN") -> bytes:
        """使用Edge TTS合成语音"""
        try:
            voice_name = voice or "zh-CN-XiaoxiaoNeural"
            
            communicate = self.edge_tts.Communicate(text, voice_name)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Edge TTS error: {e}")
            return b""
    
    async def synthesize_streaming(self, text: str, voice: str = None, 
                                 language: str = "zh-CN") -> AsyncGenerator[bytes, None]:
        """Edge TTS流式合成"""
        try:
            voice_name = voice or "zh-CN-XiaoxiaoNeural"
            
            communicate = self.edge_tts.Communicate(text, voice_name)
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
                    await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Edge TTS streaming error: {e}")

class VoiceProviderFactory:
    """语音服务提供商工厂"""
    
    ASR_PROVIDERS = {
        VoiceProvider.OPENAI: OpenAIASRProvider,
        VoiceProvider.AZURE: AzureASRProvider,
        VoiceProvider.LOCAL: LocalASRProvider,
    }
    
    TTS_PROVIDERS = {
        VoiceProvider.OPENAI: OpenAITTSProvider,
        VoiceProvider.AZURE: AzureTTSProvider,
        VoiceProvider.EDGE_TTS: EdgeTTSProvider,
    }
    
    @classmethod
    def create_asr_provider(cls, provider: VoiceProvider, config: Dict[str, Any]) -> BaseASRProvider:
        """创建ASR提供商"""
        if provider not in cls.ASR_PROVIDERS:
            raise ValueError(f"Unsupported ASR provider: {provider}")
        
        provider_class = cls.ASR_PROVIDERS[provider]
        return provider_class(config)
    
    @classmethod
    def create_tts_provider(cls, provider: VoiceProvider, config: Dict[str, Any]) -> BaseTTSProvider:
        """创建TTS提供商"""
        if provider not in cls.TTS_PROVIDERS:
            raise ValueError(f"Unsupported TTS provider: {provider}")
        
        provider_class = cls.TTS_PROVIDERS[provider]
        return provider_class(config)
