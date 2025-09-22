"""
Azure ASR/TTS适配器
基于V2架构的Azure Cognitive Services实现
"""

import io
import json
import time
import asyncio
from typing import Iterator, Optional, Dict, Any
import azure.cognitiveservices.speech as speechsdk
from loguru import logger

from .base import BaseAsrAdapter, BaseTtsAdapter


class AzureAsrAdapter(BaseAsrAdapter):
    """Azure Speech ASR适配器"""
    
    provider: str = "azure"
    
    def __init__(self, subscription_key: str, region: str, **kwargs):
        super().__init__(**kwargs)
        self.subscription_key = subscription_key
        self.region = region
        self.speech_config = None
        self.audio_config = None
        self.speech_recognizer = None
        self._setup_speech_config()
    
    def _setup_speech_config(self):
        """设置Azure Speech配置"""
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.subscription_key,
            region=self.region
        )
        self.speech_config.speech_recognition_language = self.language
    
    def start(self, sr: int, codec: str, lang: str) -> str:
        """
        开始ASR会话
        
        Args:
            sr: 采样率
            codec: 音频编码
            lang: 语言代码
            
        Returns:
            会话ID
        """
        self.sample_rate = sr
        self.language = lang
        self._session_id = f"azure_asr_{int(time.time())}"
        self._is_active = True
        
        # 更新语言设置
        self.speech_config.speech_recognition_language = lang
        
        # 设置音频配置为流式输入
        self.audio_config = speechsdk.audio.AudioConfig(use_default_microphone=False)
        
        # 创建识别器
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=self.speech_config,
            audio_config=self.audio_config
        )
        
        # 设置事件处理器
        self._setup_event_handlers()
        
        # 开始连续识别
        self.speech_recognizer.start_continuous_recognition()
        
        logger.info(f"Azure ASR会话开始: {self._session_id}, sr={sr}, lang={lang}")
        return self._session_id
    
    def _setup_event_handlers(self):
        """设置事件处理器"""
        def recognizing_handler(evt):
            """部分识别结果"""
            if evt.result.reason == speechsdk.ResultReason.RecognizingSpeech:
                text = evt.result.text
                if text:
                    self._emit_partial(0, text)
        
        def recognized_handler(evt):
            """最终识别结果"""
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                text = evt.result.text
                confidence = getattr(evt.result, 'confidence', 0.9)
                if text:
                    self._emit_final(0, text, confidence)
            elif evt.result.reason == speechsdk.ResultReason.NoMatch:
                logger.debug("Azure ASR: 无匹配结果")
        
        def canceled_handler(evt):
            """取消事件"""
            logger.warning(f"Azure ASR取消: {evt}")
        
        # 绑定事件处理器
        self.speech_recognizer.recognizing.connect(recognizing_handler)
        self.speech_recognizer.recognized.connect(recognized_handler)
        self.speech_recognizer.canceled.connect(canceled_handler)
    
    def feed(self, seq: int, chunk: bytes) -> None:
        """
        输入音频数据
        
        Args:
            seq: 序列号
            chunk: 音频数据块
        """
        if not self._is_active or not self.speech_recognizer:
            return
        
        try:
            # Azure SDK需要特定的音频流处理
            # 这里简化处理，实际使用时需要更复杂的流处理
            pass
        except Exception as e:
            logger.error(f"Azure ASR音频输入失败: {e}")
    
    def stop(self) -> None:
        """停止ASR会话"""
        if not self._is_active:
            return
        
        self._is_active = False
        
        if self.speech_recognizer:
            try:
                self.speech_recognizer.stop_continuous_recognition()
            except Exception as e:
                logger.error(f"Azure ASR停止失败: {e}")
        
        logger.info(f"Azure ASR会话结束: {self._session_id}")


class AzureTtsAdapter(BaseTtsAdapter):
    """Azure Speech TTS适配器"""
    
    provider: str = "azure"
    voice: str = "zh-CN-XiaoxiaoNeural"
    
    def __init__(self, subscription_key: str, region: str, **kwargs):
        super().__init__(**kwargs)
        self.subscription_key = subscription_key
        self.region = region
        self.speech_config = None
        self._setup_speech_config()
    
    def _setup_speech_config(self):
        """设置Azure Speech配置"""
        self.speech_config = speechsdk.SpeechConfig(
            subscription=self.subscription_key,
            region=self.region
        )
        self.speech_config.speech_synthesis_voice_name = self.voice
        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
        )
    
    def synthesize(self, text_iter: Iterator[str]) -> Iterator[bytes]:
        """
        流式文本转语音
        
        Args:
            text_iter: 文本迭代器
            
        Yields:
            音频数据块
        """
        request_id = self._generate_request_id()
        
        try:
            for text in text_iter:
                if not text.strip():
                    continue
                
                logger.debug(f"Azure TTS合成: '{text[:50]}...'")
                
                def _synthesize():
                    # 创建合成器
                    synthesizer = speechsdk.SpeechSynthesizer(
                        speech_config=self.speech_config,
                        audio_config=None  # 输出到内存
                    )
                    
                    # 构建SSML
                    ssml = self._build_ssml(text)
                    
                    # 执行合成
                    result = synthesizer.speak_ssml(ssml)
                    
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        return result.audio_data
                    else:
                        raise Exception(f"Azure TTS合成失败: {result.reason}")
                
                # 使用重试机制
                audio_data = self._retry(_synthesize)
                
                if audio_data:
                    # 分块返回音频数据
                    chunk_size = 1024 * 4  # 4KB chunks
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        yield chunk
                
        except Exception as e:
            logger.error(f"Azure TTS合成失败: {e}")
            raise
        finally:
            self._untrack_request(request_id)
    
    def _build_ssml(self, text: str) -> str:
        """
        构建SSML
        
        Args:
            text: 文本内容
            
        Returns:
            SSML字符串
        """
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">
            <voice name="{self.voice}">
                <prosody rate="{self.speed}" pitch="{self.pitch:+.0%}">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        return ssml.strip()
    
    def cancel(self, request_id: str) -> None:
        """
        取消TTS请求
        
        Args:
            request_id: 请求ID
        """
        # Azure TTS不支持直接取消，只能从跟踪中移除
        self._untrack_request(request_id)
        logger.info(f"Azure TTS请求已取消: {request_id}")
