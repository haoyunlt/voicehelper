"""
语音功能专项测试用例
测试覆盖：ASR语音识别、TTS语音合成、实时语音处理、语音情感分析、语音质量评估
"""

import pytest
import asyncio
import numpy as np
import io
import wave
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from concurrent.futures import ThreadPoolExecutor
import time


class TestASRProcessing:
    """ASR语音识别测试"""
    
    @pytest.fixture
    def mock_audio_data(self):
        """模拟音频数据"""
        # 生成16kHz, 16bit, 单声道的测试音频
        sample_rate = 16000
        duration = 2.0  # 2秒
        frequency = 440  # A4音符
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.5
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        
        return {
            "audio_bytes": audio_bytes,
            "sample_rate": sample_rate,
            "duration": duration,
            "channels": 1,
            "bit_depth": 16
        }
    
    @pytest.fixture
    def mock_asr_model(self):
        """模拟ASR模型"""
        model = Mock()
        model.transcribe.return_value = Mock(
            text="这是一段测试语音转文字的内容",
            confidence=0.95,
            language="zh-CN",
            segments=[
                Mock(start=0.0, end=1.5, text="这是一段测试", confidence=0.96),
                Mock(start=1.5, end=2.0, text="语音转文字的内容", confidence=0.94)
            ]
        )
        return model
    
    def test_audio_format_validation(self, mock_audio_data):
        """测试音频格式验证"""
        def validate_audio_format(audio_data, required_sample_rate=16000):
            """验证音频格式"""
            errors = []
            
            # 检查采样率
            if audio_data["sample_rate"] != required_sample_rate:
                errors.append(f"采样率不匹配: {audio_data['sample_rate']} != {required_sample_rate}")
            
            # 检查声道数
            if audio_data["channels"] != 1:
                errors.append(f"仅支持单声道音频: {audio_data['channels']} channels")
            
            # 检查位深度
            if audio_data["bit_depth"] not in [16, 24, 32]:
                errors.append(f"不支持的位深度: {audio_data['bit_depth']}")
            
            # 检查音频长度
            if audio_data["duration"] > 30:
                errors.append(f"音频过长: {audio_data['duration']}s > 30s")
            
            # 检查音频数据大小
            expected_size = int(audio_data["sample_rate"] * audio_data["duration"] * 
                              audio_data["channels"] * audio_data["bit_depth"] / 8)
            actual_size = len(audio_data["audio_bytes"])
            
            if abs(actual_size - expected_size) > expected_size * 0.1:  # 10%误差容忍
                errors.append(f"音频数据大小异常: {actual_size} != {expected_size}")
            
            return len(errors) == 0, errors
        
        # 测试有效音频
        is_valid, errors = validate_audio_format(mock_audio_data)
        assert is_valid, f"有效音频验证失败: {errors}"
        
        # 测试无效采样率
        invalid_audio = mock_audio_data.copy()
        invalid_audio["sample_rate"] = 8000
        is_valid, errors = validate_audio_format(invalid_audio)
        assert not is_valid
        assert any("采样率不匹配" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_asr_transcription_accuracy(self, mock_asr_model, mock_audio_data):
        """测试ASR转录准确性"""
        class ASRService:
            def __init__(self, model):
                self.model = model
            
            async def transcribe(self, audio_data, language="zh-CN"):
                """转录音频"""
                try:
                    # 模拟异步转录
                    await asyncio.sleep(0.1)
                    result = self.model.transcribe(audio_data)
                    
                    return {
                        "success": True,
                        "transcript": result.text,
                        "confidence": result.confidence,
                        "language": result.language,
                        "segments": [
                            {
                                "start": seg.start,
                                "end": seg.end,
                                "text": seg.text,
                                "confidence": seg.confidence
                            }
                            for seg in result.segments
                        ],
                        "processing_time": 0.1
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "error_code": "ASR_PROCESSING_ERROR"
                    }
        
        asr_service = ASRService(mock_asr_model)
        result = await asr_service.transcribe(mock_audio_data["audio_bytes"])
        
        # 验证转录结果
        assert result["success"]
        assert result["confidence"] > 0.9
        assert len(result["transcript"]) > 0
        assert result["language"] == "zh-CN"
        assert len(result["segments"]) > 0
        
        # 验证分段信息
        for segment in result["segments"]:
            assert segment["start"] >= 0
            assert segment["end"] > segment["start"]
            assert segment["confidence"] > 0
            assert len(segment["text"]) > 0
    
    def test_asr_noise_handling(self, mock_asr_model):
        """测试ASR噪声处理"""
        def add_noise_to_audio(clean_audio, noise_level=0.1):
            """向音频添加噪声"""
            audio_array = np.frombuffer(clean_audio, dtype=np.int16)
            noise = np.random.normal(0, noise_level * 32767, len(audio_array))
            noisy_audio = audio_array + noise.astype(np.int16)
            return noisy_audio.tobytes()
        
        def assess_audio_quality(audio_data):
            """评估音频质量"""
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 计算信噪比 (简化版)
            signal_power = np.mean(audio_array ** 2)
            noise_estimate = np.var(np.diff(audio_array))  # 简单噪声估计
            
            if noise_estimate == 0:
                snr = float('inf')
            else:
                snr = 10 * np.log10(signal_power / noise_estimate)
            
            # 质量评级
            if snr > 20:
                quality = "excellent"
            elif snr > 15:
                quality = "good"
            elif snr > 10:
                quality = "fair"
            else:
                quality = "poor"
            
            return {
                "snr": snr,
                "quality": quality,
                "recommended_processing": quality in ["fair", "poor"]
            }
        
        # 生成测试音频
        clean_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)) * 0.5
        clean_bytes = (clean_audio * 32767).astype(np.int16).tobytes()
        
        # 测试不同噪声水平
        noise_levels = [0.0, 0.1, 0.3, 0.5]
        
        for noise_level in noise_levels:
            noisy_audio = add_noise_to_audio(clean_bytes, noise_level)
            quality_assessment = assess_audio_quality(noisy_audio)
            
            if noise_level == 0.0:
                assert quality_assessment["quality"] in ["excellent", "good"]
            elif noise_level >= 0.3:
                assert quality_assessment["recommended_processing"]
    
    @pytest.mark.asyncio
    async def test_real_time_asr_streaming(self):
        """测试实时ASR流式处理"""
        class StreamingASR:
            def __init__(self):
                self.buffer = b""
                self.chunk_size = 1024  # 每次处理1024字节
                self.partial_results = []
            
            async def process_audio_chunk(self, audio_chunk):
                """处理音频块"""
                self.buffer += audio_chunk
                
                # 当缓冲区足够大时处理
                if len(self.buffer) >= self.chunk_size:
                    # 模拟ASR处理
                    await asyncio.sleep(0.01)  # 10ms处理时间
                    
                    # 生成部分结果
                    partial_text = f"部分识别结果_{len(self.partial_results)}"
                    confidence = 0.7 + len(self.partial_results) * 0.05
                    
                    result = {
                        "type": "partial",
                        "text": partial_text,
                        "confidence": min(confidence, 0.95),
                        "is_final": False,
                        "timestamp": time.time()
                    }
                    
                    self.partial_results.append(result)
                    self.buffer = self.buffer[self.chunk_size:]  # 清理已处理数据
                    
                    return result
                
                return None
            
            async def finalize(self):
                """完成处理并返回最终结果"""
                if self.buffer:
                    # 处理剩余数据
                    await self.process_audio_chunk(b"")
                
                # 合并所有部分结果
                final_text = " ".join([r["text"] for r in self.partial_results])
                avg_confidence = np.mean([r["confidence"] for r in self.partial_results])
                
                return {
                    "type": "final",
                    "text": final_text,
                    "confidence": avg_confidence,
                    "is_final": True,
                    "segments": len(self.partial_results),
                    "total_processing_time": sum(0.01 for _ in self.partial_results)
                }
        
        # 测试流式处理
        streaming_asr = StreamingASR()
        
        # 模拟音频流
        audio_stream = [b"chunk_" + str(i).encode() * 200 for i in range(10)]
        
        partial_results = []
        for chunk in audio_stream:
            result = await streaming_asr.process_audio_chunk(chunk)
            if result:
                partial_results.append(result)
        
        # 获取最终结果
        final_result = await streaming_asr.finalize()
        
        # 验证结果
        assert len(partial_results) > 0
        assert final_result["type"] == "final"
        assert final_result["is_final"]
        assert final_result["segments"] == len(partial_results)
        assert final_result["confidence"] > 0.7


class TestTTSProcessing:
    """TTS语音合成测试"""
    
    @pytest.fixture
    def mock_tts_model(self):
        """模拟TTS模型"""
        model = Mock()
        
        def mock_synthesize(text, voice_config=None):
            # 生成模拟音频数据
            duration = len(text) * 0.1  # 假设每字符0.1秒
            sample_rate = 22050
            samples = int(duration * sample_rate)
            
            # 生成简单的正弦波作为音频
            t = np.linspace(0, duration, samples)
            audio = np.sin(2 * np.pi * 440 * t) * 0.3
            audio_bytes = (audio * 32767).astype(np.int16).tobytes()
            
            return Mock(
                audio_data=audio_bytes,
                sample_rate=sample_rate,
                duration=duration,
                voice_id=voice_config.get("voice_id", "default") if voice_config else "default"
            )
        
        model.synthesize = mock_synthesize
        return model
    
    def test_text_preprocessing_for_tts(self):
        """测试TTS文本预处理"""
        def preprocess_text_for_tts(text):
            """TTS文本预处理"""
            if not text or not isinstance(text, str):
                return {"success": False, "error": "无效的输入文本"}
            
            # 清理文本
            processed_text = text.strip()
            
            # 处理数字
            import re
            
            # 简单的数字转换（实际应用中需要更复杂的处理）
            number_map = {
                '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
                '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
            }
            
            for digit, chinese in number_map.items():
                processed_text = processed_text.replace(digit, chinese)
            
            # 处理特殊符号
            symbol_map = {
                '&': '和',
                '@': '艾特',
                '#': '井号',
                '%': '百分号',
                '$': '美元'
            }
            
            for symbol, pronunciation in symbol_map.items():
                processed_text = processed_text.replace(symbol, pronunciation)
            
            # 移除不支持的字符
            processed_text = re.sub(r'[^\u4e00-\u9fff\u3400-\u4dbf\w\s，。！？；：""''（）【】《》、]', '', processed_text)
            
            # 检查长度限制
            if len(processed_text) > 1000:
                return {"success": False, "error": "文本过长，超过1000字符限制"}
            
            if len(processed_text) == 0:
                return {"success": False, "error": "处理后文本为空"}
            
            return {
                "success": True,
                "original_text": text,
                "processed_text": processed_text,
                "character_count": len(processed_text)
            }
        
        # 测试正常文本
        result = preprocess_text_for_tts("你好，这是测试文本123！")
        assert result["success"]
        assert "一二三" in result["processed_text"]
        
        # 测试包含特殊符号的文本
        result = preprocess_text_for_tts("发送邮件到user@example.com")
        assert result["success"]
        assert "艾特" in result["processed_text"]
        
        # 测试空文本
        result = preprocess_text_for_tts("")
        assert not result["success"]
        assert "为空" in result["error"]
        
        # 测试过长文本
        long_text = "测试" * 501  # 超过1000字符
        result = preprocess_text_for_tts(long_text)
        assert not result["success"]
        assert "过长" in result["error"]
    
    @pytest.mark.asyncio
    async def test_tts_voice_configuration(self, mock_tts_model):
        """测试TTS语音配置"""
        class TTSService:
            def __init__(self, model):
                self.model = model
                self.supported_voices = {
                    "zh-CN-XiaoxiaoNeural": {"gender": "female", "age": "adult", "style": "friendly"},
                    "zh-CN-YunxiNeural": {"gender": "male", "age": "adult", "style": "calm"},
                    "zh-CN-XiaoyiNeural": {"gender": "female", "age": "child", "style": "cheerful"}
                }
            
            def validate_voice_config(self, voice_config):
                """验证语音配置"""
                errors = []
                
                # 检查语音ID
                voice_id = voice_config.get("voice_id")
                if not voice_id:
                    errors.append("缺少voice_id")
                elif voice_id not in self.supported_voices:
                    errors.append(f"不支持的语音ID: {voice_id}")
                
                # 检查语速
                speed = voice_config.get("speed", 1.0)
                if not isinstance(speed, (int, float)) or speed < 0.5 or speed > 2.0:
                    errors.append("语速必须在0.5-2.0之间")
                
                # 检查音调
                pitch = voice_config.get("pitch", 0.0)
                if not isinstance(pitch, (int, float)) or pitch < -50 or pitch > 50:
                    errors.append("音调必须在-50到50之间")
                
                # 检查音量
                volume = voice_config.get("volume", 1.0)
                if not isinstance(volume, (int, float)) or volume < 0.0 or volume > 2.0:
                    errors.append("音量必须在0.0-2.0之间")
                
                return len(errors) == 0, errors
            
            async def synthesize_with_config(self, text, voice_config):
                """使用配置合成语音"""
                # 验证配置
                is_valid, errors = self.validate_voice_config(voice_config)
                if not is_valid:
                    return {"success": False, "errors": errors}
                
                try:
                    # 合成语音
                    result = self.model.synthesize(text, voice_config)
                    
                    return {
                        "success": True,
                        "audio_data": result.audio_data,
                        "duration": result.duration,
                        "voice_id": result.voice_id,
                        "config_applied": voice_config
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}
        
        tts_service = TTSService(mock_tts_model)
        
        # 测试有效配置
        valid_config = {
            "voice_id": "zh-CN-XiaoxiaoNeural",
            "speed": 1.2,
            "pitch": 5.0,
            "volume": 0.8
        }
        
        result = await tts_service.synthesize_with_config("测试文本", valid_config)
        assert result["success"]
        assert result["voice_id"] == "zh-CN-XiaoxiaoNeural"
        
        # 测试无效配置
        invalid_configs = [
            {"voice_id": "invalid-voice", "speed": 1.0},  # 无效语音ID
            {"voice_id": "zh-CN-XiaoxiaoNeural", "speed": 3.0},  # 语速超范围
            {"voice_id": "zh-CN-XiaoxiaoNeural", "pitch": 100},  # 音调超范围
        ]
        
        for invalid_config in invalid_configs:
            result = await tts_service.synthesize_with_config("测试", invalid_config)
            assert not result["success"]
            assert "errors" in result
    
    def test_tts_audio_quality_assessment(self):
        """测试TTS音频质量评估"""
        def assess_tts_audio_quality(audio_data, sample_rate=22050):
            """评估TTS音频质量"""
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 计算音频特征
            # 1. 动态范围
            max_amplitude = np.max(np.abs(audio_array))
            dynamic_range = 20 * np.log10(max_amplitude / (np.mean(np.abs(audio_array)) + 1e-10))
            
            # 2. 频谱分析
            fft = np.fft.fft(audio_array)
            freqs = np.fft.fftfreq(len(fft), 1/sample_rate)
            magnitude = np.abs(fft)
            
            # 找到主要频率成分
            dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
            dominant_freq = abs(freqs[dominant_freq_idx])
            
            # 3. 静音检测
            silence_threshold = max_amplitude * 0.01
            silence_ratio = np.sum(np.abs(audio_array) < silence_threshold) / len(audio_array)
            
            # 4. 削波检测
            clipping_threshold = 32767 * 0.95
            clipping_ratio = np.sum(np.abs(audio_array) > clipping_threshold) / len(audio_array)
            
            # 质量评分
            quality_score = 100
            
            # 扣分项
            if dynamic_range < 10:
                quality_score -= 20  # 动态范围过小
            
            if silence_ratio > 0.3:
                quality_score -= 15  # 静音过多
            
            if clipping_ratio > 0.01:
                quality_score -= 25  # 削波过多
            
            if dominant_freq < 100 or dominant_freq > 8000:
                quality_score -= 10  # 主频异常
            
            quality_score = max(0, quality_score)
            
            # 质量等级
            if quality_score >= 90:
                quality_level = "excellent"
            elif quality_score >= 75:
                quality_level = "good"
            elif quality_score >= 60:
                quality_level = "fair"
            else:
                quality_level = "poor"
            
            return {
                "quality_score": quality_score,
                "quality_level": quality_level,
                "dynamic_range": dynamic_range,
                "dominant_frequency": dominant_freq,
                "silence_ratio": silence_ratio,
                "clipping_ratio": clipping_ratio,
                "recommendations": []
            }
        
        # 生成测试音频
        sample_rate = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # 高质量音频
        high_quality_audio = np.sin(2 * np.pi * 440 * t) * 0.7
        high_quality_bytes = (high_quality_audio * 32767).astype(np.int16).tobytes()
        
        quality = assess_tts_audio_quality(high_quality_bytes, sample_rate)
        assert quality["quality_level"] in ["excellent", "good"]
        assert quality["quality_score"] > 75
        
        # 低质量音频（削波）
        clipped_audio = np.sin(2 * np.pi * 440 * t) * 1.2  # 超过范围
        clipped_audio = np.clip(clipped_audio, -1, 1)
        clipped_bytes = (clipped_audio * 32767).astype(np.int16).tobytes()
        
        quality = assess_tts_audio_quality(clipped_bytes, sample_rate)
        assert quality["clipping_ratio"] > 0
        assert quality["quality_score"] < 90


class TestVoiceEmotionAnalysis:
    """语音情感分析测试"""
    
    @pytest.fixture
    def mock_emotion_analyzer(self):
        """模拟情感分析器"""
        analyzer = Mock()
        
        def mock_analyze(audio_data, text=None):
            # 基于文本内容模拟情感分析
            if text:
                if any(word in text for word in ["高兴", "开心", "快乐"]):
                    primary_emotion = "happy"
                    confidence = 0.9
                elif any(word in text for word in ["生气", "愤怒", "不满"]):
                    primary_emotion = "angry"
                    confidence = 0.85
                elif any(word in text for word in ["伤心", "难过", "悲伤"]):
                    primary_emotion = "sad"
                    confidence = 0.8
                else:
                    primary_emotion = "neutral"
                    confidence = 0.7
            else:
                # 仅基于音频的分析
                primary_emotion = "neutral"
                confidence = 0.6
            
            return Mock(
                primary_emotion=primary_emotion,
                confidence=confidence,
                all_emotions={
                    "happy": 0.8 if primary_emotion == "happy" else 0.1,
                    "sad": 0.8 if primary_emotion == "sad" else 0.1,
                    "angry": 0.85 if primary_emotion == "angry" else 0.05,
                    "neutral": 0.7 if primary_emotion == "neutral" else 0.2,
                    "surprised": 0.1,
                    "fearful": 0.05
                },
                arousal=0.6,  # 激活度
                valence=0.7 if primary_emotion == "happy" else 0.3  # 效价
            )
        
        analyzer.analyze = mock_analyze
        return analyzer
    
    @pytest.mark.asyncio
    async def test_multimodal_emotion_analysis(self, mock_emotion_analyzer):
        """测试多模态情感分析"""
        class MultimodalEmotionAnalyzer:
            def __init__(self, audio_analyzer):
                self.audio_analyzer = audio_analyzer
            
            async def analyze_emotion(self, audio_data, transcript=None, context=None):
                """多模态情感分析"""
                try:
                    # 音频情感分析
                    audio_emotion = self.audio_analyzer.analyze(audio_data)
                    
                    # 文本情感分析（如果有转录文本）
                    text_emotion = None
                    if transcript:
                        text_emotion = self.audio_analyzer.analyze(audio_data, transcript)
                    
                    # 融合分析结果
                    if text_emotion and audio_emotion:
                        # 加权融合
                        audio_weight = 0.4
                        text_weight = 0.6
                        
                        fused_confidence = (
                            audio_emotion.confidence * audio_weight +
                            text_emotion.confidence * text_weight
                        )
                        
                        # 选择置信度更高的情感
                        if text_emotion.confidence > audio_emotion.confidence:
                            primary_emotion = text_emotion.primary_emotion
                        else:
                            primary_emotion = audio_emotion.primary_emotion
                    else:
                        primary_emotion = audio_emotion.primary_emotion
                        fused_confidence = audio_emotion.confidence
                    
                    return {
                        "success": True,
                        "primary_emotion": primary_emotion,
                        "confidence": fused_confidence,
                        "audio_emotion": {
                            "emotion": audio_emotion.primary_emotion,
                            "confidence": audio_emotion.confidence,
                            "arousal": audio_emotion.arousal,
                            "valence": audio_emotion.valence
                        },
                        "text_emotion": {
                            "emotion": text_emotion.primary_emotion,
                            "confidence": text_emotion.confidence
                        } if text_emotion else None,
                        "all_emotions": text_emotion.all_emotions if text_emotion else audio_emotion.all_emotions
                    }
                
                except Exception as e:
                    return {
                        "success": False,
                        "error": str(e),
                        "error_code": "EMOTION_ANALYSIS_ERROR"
                    }
        
        analyzer = MultimodalEmotionAnalyzer(mock_emotion_analyzer)
        
        # 测试仅音频分析
        result = await analyzer.analyze_emotion(b"fake_audio_data")
        assert result["success"]
        assert "primary_emotion" in result
        assert result["audio_emotion"]["confidence"] > 0
        
        # 测试音频+文本分析
        result = await analyzer.analyze_emotion(
            b"fake_audio_data",
            transcript="我今天很高兴"
        )
        assert result["success"]
        assert result["primary_emotion"] == "happy"
        assert result["confidence"] > 0.7
        assert result["text_emotion"] is not None
    
    def test_emotion_trend_analysis(self):
        """测试情感趋势分析"""
        class EmotionTrendAnalyzer:
            def __init__(self):
                self.emotion_history = []
            
            def add_emotion_data(self, emotion_data):
                """添加情感数据"""
                timestamp = time.time()
                self.emotion_history.append({
                    "timestamp": timestamp,
                    "emotion": emotion_data["primary_emotion"],
                    "confidence": emotion_data["confidence"],
                    "valence": emotion_data.get("valence", 0.5),
                    "arousal": emotion_data.get("arousal", 0.5)
                })
                
                # 保持最近100条记录
                if len(self.emotion_history) > 100:
                    self.emotion_history = self.emotion_history[-100:]
            
            def analyze_trend(self, time_window=300):  # 5分钟窗口
                """分析情感趋势"""
                if len(self.emotion_history) < 2:
                    return {"success": False, "error": "数据不足"}
                
                current_time = time.time()
                recent_emotions = [
                    e for e in self.emotion_history
                    if current_time - e["timestamp"] <= time_window
                ]
                
                if len(recent_emotions) < 2:
                    return {"success": False, "error": "时间窗口内数据不足"}
                
                # 计算趋势
                valences = [e["valence"] for e in recent_emotions]
                arousals = [e["arousal"] for e in recent_emotions]
                
                # 简单线性趋势
                valence_trend = (valences[-1] - valences[0]) / len(valences)
                arousal_trend = (arousals[-1] - arousals[0]) / len(arousals)
                
                # 情感稳定性（方差）
                valence_stability = 1 / (1 + np.var(valences))
                arousal_stability = 1 / (1 + np.var(arousals))
                
                # 主导情感
                emotion_counts = {}
                for e in recent_emotions:
                    emotion_counts[e["emotion"]] = emotion_counts.get(e["emotion"], 0) + 1
                
                dominant_emotion = max(emotion_counts, key=emotion_counts.get)
                
                return {
                    "success": True,
                    "time_window": time_window,
                    "sample_count": len(recent_emotions),
                    "valence_trend": valence_trend,
                    "arousal_trend": arousal_trend,
                    "valence_stability": valence_stability,
                    "arousal_stability": arousal_stability,
                    "dominant_emotion": dominant_emotion,
                    "emotion_distribution": emotion_counts,
                    "trend_description": self._describe_trend(valence_trend, arousal_trend)
                }
            
            def _describe_trend(self, valence_trend, arousal_trend):
                """描述情感趋势"""
                if valence_trend > 0.1:
                    valence_desc = "情绪变得更积极"
                elif valence_trend < -0.1:
                    valence_desc = "情绪变得更消极"
                else:
                    valence_desc = "情绪相对稳定"
                
                if arousal_trend > 0.1:
                    arousal_desc = "激活度上升"
                elif arousal_trend < -0.1:
                    arousal_desc = "激活度下降"
                else:
                    arousal_desc = "激活度稳定"
                
                return f"{valence_desc}，{arousal_desc}"
        
        # 测试情感趋势分析
        trend_analyzer = EmotionTrendAnalyzer()
        
        # 添加模拟情感数据
        emotions = [
            {"primary_emotion": "neutral", "confidence": 0.7, "valence": 0.5, "arousal": 0.4},
            {"primary_emotion": "happy", "confidence": 0.8, "valence": 0.8, "arousal": 0.6},
            {"primary_emotion": "happy", "confidence": 0.9, "valence": 0.9, "arousal": 0.7},
            {"primary_emotion": "excited", "confidence": 0.85, "valence": 0.9, "arousal": 0.9},
        ]
        
        for emotion in emotions:
            trend_analyzer.add_emotion_data(emotion)
            time.sleep(0.01)  # 小间隔
        
        # 分析趋势
        trend = trend_analyzer.analyze_trend(time_window=60)
        
        assert trend["success"]
        assert trend["sample_count"] == 4
        assert trend["valence_trend"] > 0  # 情绪变积极
        assert trend["dominant_emotion"] == "happy"
        assert "积极" in trend["trend_description"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
