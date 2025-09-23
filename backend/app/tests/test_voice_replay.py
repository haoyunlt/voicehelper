"""
语音回放测试 - 用于评测STT/TTS/VAD性能
"""
import asyncio
import json
import time
import wave
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import pytest
import structlog

from app.services.stt.deepgram import DeepgramSTT
from app.services.stt.riva import MockRivaSTT
from app.services.tts.aura import AuraTTS
from app.services.tts.base import MockTTSService
from app.services.vad.semantic_vad import SemanticVAD
from app.models.schemas import AudioChunk, TTSRequest, Transcription

logger = structlog.get_logger()


class AudioTestData:
    """测试音频数据"""
    
    def __init__(self, file_path: str, expected_text: str, metadata: Dict[str, Any] = None):
        self.file_path = file_path
        self.expected_text = expected_text
        self.metadata = metadata or {}
        self.audio_data: Optional[bytes] = None
        self.sample_rate = 16000
        self.channels = 1
    
    def load_audio(self) -> bytes:
        """加载音频文件"""
        if self.audio_data is not None:
            return self.audio_data
        
        file_path = Path(self.file_path)
        
        if file_path.suffix.lower() == '.wav':
            with wave.open(str(file_path), 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                self.sample_rate = wav_file.getframerate()
                self.channels = wav_file.getnchannels()
                self.audio_data = frames
        else:
            # 生成合成音频用于测试
            duration = 2.0  # 2秒
            samples = int(self.sample_rate * duration)
            
            # 生成正弦波 + 噪声
            t = np.linspace(0, duration, samples)
            frequency = 440  # A4音符
            signal = 0.3 * np.sin(2 * np.pi * frequency * t)
            noise = 0.1 * np.random.randn(samples)
            audio_signal = signal + noise
            
            # 转换为16位PCM
            audio_int16 = (audio_signal * 32767).astype(np.int16)
            self.audio_data = audio_int16.tobytes()
        
        return self.audio_data
    
    def get_chunks(self, chunk_size_ms: int = 100) -> List[AudioChunk]:
        """将音频分割为块"""
        audio_data = self.load_audio()
        
        chunk_size_bytes = int(self.sample_rate * chunk_size_ms / 1000) * 2  # 16-bit = 2 bytes
        chunks = []
        
        for i in range(0, len(audio_data), chunk_size_bytes):
            chunk_data = audio_data[i:i + chunk_size_bytes]
            timestamp_ms = i * 1000 // (self.sample_rate * 2)
            
            chunk = AudioChunk(
                data=chunk_data,
                timestamp_ms=timestamp_ms,
                sample_rate=self.sample_rate,
                channels=self.channels,
                format="pcm"
            )
            chunks.append(chunk)
        
        return chunks


class VoiceReplayTester:
    """语音回放测试器"""
    
    def __init__(self):
        self.test_data: List[AudioTestData] = []
        self.results: List[Dict[str, Any]] = []
        
        # 初始化测试数据
        self._init_test_data()
    
    def _init_test_data(self):
        """初始化测试数据集"""
        
        # 基础测试用例
        basic_tests = [
            AudioTestData("test_hello.wav", "你好，我是语音助手", {"category": "greeting", "noise_level": "clean"}),
            AudioTestData("test_question.wav", "今天天气怎么样？", {"category": "question", "noise_level": "clean"}),
            AudioTestData("test_command.wav", "请帮我设置一个提醒", {"category": "command", "noise_level": "clean"}),
            AudioTestData("test_long.wav", "这是一段比较长的语音测试内容，用来测试系统在处理长语音时的性能表现", {"category": "long_text", "noise_level": "clean"}),
        ]
        
        # 噪声测试用例
        noise_tests = [
            AudioTestData("test_noise_light.wav", "轻微噪声环境下的语音", {"category": "greeting", "noise_level": "light"}),
            AudioTestData("test_noise_heavy.wav", "重噪声环境下的语音", {"category": "greeting", "noise_level": "heavy"}),
        ]
        
        # 口音测试用例
        accent_tests = [
            AudioTestData("test_accent_northern.wav", "北方口音测试", {"category": "accent", "accent_type": "northern"}),
            AudioTestData("test_accent_southern.wav", "南方口音测试", {"category": "accent", "accent_type": "southern"}),
        ]
        
        # 语速测试用例
        speed_tests = [
            AudioTestData("test_speed_slow.wav", "慢速语音测试", {"category": "speed", "speed_type": "slow"}),
            AudioTestData("test_speed_fast.wav", "快速语音测试", {"category": "speed", "speed_type": "fast"}),
        ]
        
        self.test_data.extend(basic_tests)
        self.test_data.extend(noise_tests)
        self.test_data.extend(accent_tests)
        self.test_data.extend(speed_tests)
    
    async def run_stt_tests(self, stt_service, test_cases: Optional[List[AudioTestData]] = None) -> List[Dict[str, Any]]:
        """运行STT测试"""
        test_cases = test_cases or self.test_data
        results = []
        
        for test_case in test_cases:
            logger.info(f"Testing STT with: {test_case.file_path}")
            
            session_id = f"test_stt_{int(time.time())}"
            start_time = time.time()
            
            try:
                # 开始STT流
                await stt_service.start_stream(session_id, language="zh-CN")
                
                # 发送音频块
                chunks = test_case.get_chunks()
                transcriptions = []
                
                for chunk in chunks:
                    result = await stt_service.ingest_audio(session_id, chunk)
                    if result:
                        transcriptions.append(result)
                
                # 结束流并获取最终结果
                final_result = await stt_service.finish_stream(session_id)
                if final_result:
                    transcriptions.append(final_result)
                
                end_time = time.time()
                
                # 计算指标
                total_duration = end_time - start_time
                audio_duration = len(test_case.load_audio()) / (test_case.sample_rate * 2)  # 16-bit
                rtf = total_duration / audio_duration  # Real-time Factor
                
                # 获取最佳转录结果
                best_transcription = max(transcriptions, key=lambda x: x.confidence) if transcriptions else None
                recognized_text = best_transcription.text if best_transcription else ""
                confidence = best_transcription.confidence if best_transcription else 0.0
                
                # 计算准确性（简单的字符匹配）
                accuracy = self._calculate_text_accuracy(test_case.expected_text, recognized_text)
                
                result = {
                    "test_case": test_case.file_path,
                    "expected_text": test_case.expected_text,
                    "recognized_text": recognized_text,
                    "confidence": confidence,
                    "accuracy": accuracy,
                    "rtf": rtf,
                    "total_duration_s": total_duration,
                    "audio_duration_s": audio_duration,
                    "metadata": test_case.metadata,
                    "transcriptions_count": len(transcriptions),
                    "success": len(transcriptions) > 0
                }
                
                results.append(result)
                
                logger.info(f"STT test completed: accuracy={accuracy:.2f}, rtf={rtf:.2f}")
                
            except Exception as e:
                logger.error(f"STT test failed: {e}")
                results.append({
                    "test_case": test_case.file_path,
                    "expected_text": test_case.expected_text,
                    "error": str(e),
                    "success": False,
                    "metadata": test_case.metadata
                })
        
        return results
    
    async def run_tts_tests(self, tts_service, test_texts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """运行TTS测试"""
        test_texts = test_texts or [
            "你好，我是语音助手",
            "今天天气很好",
            "请问有什么可以帮助您的吗？",
            "这是一段较长的测试文本，用来评估TTS系统在处理长文本时的性能表现和音质效果"
        ]
        
        results = []
        
        for i, text in enumerate(test_texts):
            logger.info(f"Testing TTS with: {text[:20]}...")
            
            session_id = f"test_tts_{int(time.time())}_{i}"
            start_time = time.time()
            
            try:
                request = TTSRequest(
                    session_id=session_id,
                    text=text,
                    language="zh-CN"
                )
                
                # 收集音频块
                audio_chunks = []
                first_chunk_time = None
                
                async for chunk in tts_service.stream_synthesis(request):
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                    
                    audio_chunks.append(chunk)
                
                end_time = time.time()
                
                # 计算指标
                total_duration = end_time - start_time
                first_chunk_latency = first_chunk_time - start_time if first_chunk_time else 0
                total_audio_size = sum(len(chunk.data) for chunk in audio_chunks)
                
                result = {
                    "text": text,
                    "text_length": len(text),
                    "total_duration_s": total_duration,
                    "first_chunk_latency_s": first_chunk_latency,
                    "chunks_count": len(audio_chunks),
                    "total_audio_bytes": total_audio_size,
                    "success": len(audio_chunks) > 0
                }
                
                results.append(result)
                
                logger.info(f"TTS test completed: {len(audio_chunks)} chunks, {first_chunk_latency:.3f}s first chunk")
                
            except Exception as e:
                logger.error(f"TTS test failed: {e}")
                results.append({
                    "text": text,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    async def run_vad_tests(self, vad_service: SemanticVAD) -> List[Dict[str, Any]]:
        """运行VAD测试"""
        results = []
        
        # 测试不同的音频场景
        test_scenarios = [
            {"name": "silence", "audio_type": "silence", "expected_events": []},
            {"name": "speech", "audio_type": "speech", "expected_events": ["SPEECH_START", "SPEECH_END"]},
            {"name": "speech_with_pause", "audio_type": "speech_pause", "expected_events": ["SPEECH_START", "SPEECH_END", "SPEECH_START", "SPEECH_END"]},
        ]
        
        for scenario in test_scenarios:
            logger.info(f"Testing VAD scenario: {scenario['name']}")
            
            session_id = f"test_vad_{scenario['name']}"
            
            try:
                # 生成测试音频
                audio_chunks = self._generate_test_audio(scenario["audio_type"])
                
                # 处理音频并收集事件
                events = []
                for chunk in audio_chunks:
                    chunk_events = await vad_service.feed(session_id, chunk.data, chunk.timestamp_ms)
                    events.extend(chunk_events)
                
                # 分析结果
                detected_events = [event[0] for event in events]
                expected_events = scenario["expected_events"]
                
                # 计算准确性
                accuracy = self._calculate_event_accuracy(expected_events, detected_events)
                
                result = {
                    "scenario": scenario["name"],
                    "expected_events": expected_events,
                    "detected_events": detected_events,
                    "accuracy": accuracy,
                    "total_events": len(events),
                    "success": True
                }
                
                results.append(result)
                
                logger.info(f"VAD test completed: {scenario['name']}, accuracy={accuracy:.2f}")
                
            except Exception as e:
                logger.error(f"VAD test failed: {e}")
                results.append({
                    "scenario": scenario["name"],
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    async def run_e2e_latency_test(self) -> Dict[str, Any]:
        """端到端延迟测试"""
        logger.info("Running E2E latency test")
        
        # 使用Mock服务进行快速测试
        stt_service = MockRivaSTT()
        tts_service = MockTTSService()
        
        session_id = f"test_e2e_{int(time.time())}"
        
        try:
            # 测试流程：音频输入 -> STT -> LLM -> TTS -> 音频输出
            start_time = time.time()
            
            # 1. STT阶段
            stt_start = time.time()
            await stt_service.start_stream(session_id)
            
            # 发送测试音频
            test_audio = AudioTestData("test_e2e.wav", "你好")
            chunks = test_audio.get_chunks()
            
            for chunk in chunks:
                await stt_service.ingest_audio(session_id, chunk)
            
            stt_result = await stt_service.finish_stream(session_id)
            stt_end = time.time()
            stt_latency = stt_end - stt_start
            
            # 2. LLM阶段（模拟）
            llm_start = time.time()
            await asyncio.sleep(0.2)  # 模拟LLM处理时间
            llm_end = time.time()
            llm_latency = llm_end - llm_start
            
            # 3. TTS阶段
            tts_start = time.time()
            tts_request = TTSRequest(
                session_id=session_id,
                text="你好，很高兴见到你！",
                language="zh-CN"
            )
            
            first_audio_chunk = None
            async for chunk in tts_service.stream_synthesis(tts_request):
                if first_audio_chunk is None:
                    first_audio_chunk = time.time()
                    break
            
            tts_latency = first_audio_chunk - tts_start if first_audio_chunk else 0
            
            end_time = time.time()
            total_latency = end_time - start_time
            
            result = {
                "total_latency_ms": total_latency * 1000,
                "stt_latency_ms": stt_latency * 1000,
                "llm_latency_ms": llm_latency * 1000,
                "tts_latency_ms": tts_latency * 1000,
                "success": True,
                "timestamp": time.time()
            }
            
            logger.info(f"E2E latency test completed: {total_latency * 1000:.1f}ms total")
            
            return result
            
        except Exception as e:
            logger.error(f"E2E latency test failed: {e}")
            return {
                "error": str(e),
                "success": False,
                "timestamp": time.time()
            }
    
    def _calculate_text_accuracy(self, expected: str, actual: str) -> float:
        """计算文本准确性（简化版）"""
        if not expected or not actual:
            return 0.0
        
        # 简单的字符级别匹配
        expected_chars = set(expected.replace(" ", ""))
        actual_chars = set(actual.replace(" ", ""))
        
        if not expected_chars:
            return 1.0 if not actual_chars else 0.0
        
        intersection = expected_chars.intersection(actual_chars)
        return len(intersection) / len(expected_chars)
    
    def _calculate_event_accuracy(self, expected: List[str], actual: List[str]) -> float:
        """计算事件检测准确性"""
        if not expected and not actual:
            return 1.0
        
        if not expected:
            return 0.0 if actual else 1.0
        
        # 简单的序列匹配
        matches = 0
        for i, event in enumerate(expected):
            if i < len(actual) and actual[i] == event:
                matches += 1
        
        return matches / len(expected)
    
    def _generate_test_audio(self, audio_type: str) -> List[AudioChunk]:
        """生成测试音频"""
        sample_rate = 16000
        chunk_duration_ms = 100
        chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        if audio_type == "silence":
            # 2秒静音
            duration_chunks = 20
            chunks = []
            for i in range(duration_chunks):
                silence_data = np.zeros(chunk_size, dtype=np.int16)
                chunk = AudioChunk(
                    data=silence_data.tobytes(),
                    timestamp_ms=i * chunk_duration_ms,
                    sample_rate=sample_rate,
                    channels=1,
                    format="pcm"
                )
                chunks.append(chunk)
            return chunks
        
        elif audio_type == "speech":
            # 1秒语音信号
            duration_chunks = 10
            chunks = []
            for i in range(duration_chunks):
                # 生成正弦波模拟语音
                t = np.linspace(0, chunk_duration_ms / 1000, chunk_size)
                signal = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440Hz
                audio_data = (signal * 32767).astype(np.int16)
                
                chunk = AudioChunk(
                    data=audio_data.tobytes(),
                    timestamp_ms=i * chunk_duration_ms,
                    sample_rate=sample_rate,
                    channels=1,
                    format="pcm"
                )
                chunks.append(chunk)
            return chunks
        
        elif audio_type == "speech_pause":
            # 语音-停顿-语音模式
            chunks = []
            
            # 0.5秒语音
            for i in range(5):
                t = np.linspace(0, chunk_duration_ms / 1000, chunk_size)
                signal = 0.3 * np.sin(2 * np.pi * 440 * t)
                audio_data = (signal * 32767).astype(np.int16)
                
                chunk = AudioChunk(
                    data=audio_data.tobytes(),
                    timestamp_ms=i * chunk_duration_ms,
                    sample_rate=sample_rate,
                    channels=1,
                    format="pcm"
                )
                chunks.append(chunk)
            
            # 0.5秒静音
            for i in range(5, 10):
                silence_data = np.zeros(chunk_size, dtype=np.int16)
                chunk = AudioChunk(
                    data=silence_data.tobytes(),
                    timestamp_ms=i * chunk_duration_ms,
                    sample_rate=sample_rate,
                    channels=1,
                    format="pcm"
                )
                chunks.append(chunk)
            
            # 0.5秒语音
            for i in range(10, 15):
                t = np.linspace(0, chunk_duration_ms / 1000, chunk_size)
                signal = 0.3 * np.sin(2 * np.pi * 880 * t)  # 不同频率
                audio_data = (signal * 32767).astype(np.int16)
                
                chunk = AudioChunk(
                    data=audio_data.tobytes(),
                    timestamp_ms=i * chunk_duration_ms,
                    sample_rate=sample_rate,
                    channels=1,
                    format="pcm"
                )
                chunks.append(chunk)
            
            return chunks
        
        return []
    
    def save_results(self, results: Dict[str, Any], output_file: str = "voice_test_results.json"):
        """保存测试结果"""
        output_path = Path(output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Test results saved to: {output_path}")


# pytest测试用例

@pytest.mark.asyncio
async def test_stt_accuracy():
    """测试STT准确性"""
    tester = VoiceReplayTester()
    stt_service = MockRivaSTT()  # 使用Mock服务进行测试
    
    results = await tester.run_stt_tests(stt_service)
    
    # 验证测试结果
    assert len(results) > 0
    
    successful_tests = [r for r in results if r.get('success', False)]
    assert len(successful_tests) > 0
    
    # 检查平均准确性
    avg_accuracy = sum(r.get('accuracy', 0) for r in successful_tests) / len(successful_tests)
    assert avg_accuracy > 0.5  # 至少50%准确性


@pytest.mark.asyncio
async def test_tts_latency():
    """测试TTS延迟"""
    tester = VoiceReplayTester()
    tts_service = MockTTSService()
    
    results = await tester.run_tts_tests(tts_service)
    
    # 验证测试结果
    assert len(results) > 0
    
    successful_tests = [r for r in results if r.get('success', False)]
    assert len(successful_tests) > 0
    
    # 检查首包延迟
    avg_first_chunk_latency = sum(r.get('first_chunk_latency_s', 0) for r in successful_tests) / len(successful_tests)
    assert avg_first_chunk_latency < 1.0  # 首包延迟小于1秒


@pytest.mark.asyncio
async def test_vad_accuracy():
    """测试VAD准确性"""
    tester = VoiceReplayTester()
    vad_service = SemanticVAD()
    
    results = await tester.run_vad_tests(vad_service)
    
    # 验证测试结果
    assert len(results) > 0
    
    successful_tests = [r for r in results if r.get('success', False)]
    assert len(successful_tests) > 0
    
    # 检查平均准确性
    avg_accuracy = sum(r.get('accuracy', 0) for r in successful_tests) / len(successful_tests)
    assert avg_accuracy > 0.7  # 至少70%准确性


@pytest.mark.asyncio
async def test_e2e_latency():
    """测试端到端延迟"""
    tester = VoiceReplayTester()
    
    result = await tester.run_e2e_latency_test()
    
    # 验证测试结果
    assert result.get('success', False)
    assert result.get('total_latency_ms', float('inf')) < 1000  # 总延迟小于1秒


if __name__ == "__main__":
    async def main():
        """运行完整的语音测试套件"""
        tester = VoiceReplayTester()
        
        # 运行所有测试
        logger.info("Starting voice replay tests...")
        
        # STT测试
        stt_service = MockRivaSTT()
        stt_results = await tester.run_stt_tests(stt_service)
        
        # TTS测试
        tts_service = MockTTSService()
        tts_results = await tester.run_tts_tests(tts_service)
        
        # VAD测试
        vad_service = SemanticVAD()
        vad_results = await tester.run_vad_tests(vad_service)
        
        # E2E延迟测试
        e2e_result = await tester.run_e2e_latency_test()
        
        # 汇总结果
        all_results = {
            "timestamp": time.time(),
            "stt_tests": stt_results,
            "tts_tests": tts_results,
            "vad_tests": vad_results,
            "e2e_test": e2e_result,
            "summary": {
                "total_tests": len(stt_results) + len(tts_results) + len(vad_results) + 1,
                "successful_tests": (
                    len([r for r in stt_results if r.get('success')]) +
                    len([r for r in tts_results if r.get('success')]) +
                    len([r for r in vad_results if r.get('success')]) +
                    (1 if e2e_result.get('success') else 0)
                )
            }
        }
        
        # 保存结果
        tester.save_results(all_results)
        
        logger.info("Voice replay tests completed!")
        
        # 打印摘要
        print("\n=== 测试结果摘要 ===")
        print(f"总测试数: {all_results['summary']['total_tests']}")
        print(f"成功测试数: {all_results['summary']['successful_tests']}")
        print(f"成功率: {all_results['summary']['successful_tests'] / all_results['summary']['total_tests'] * 100:.1f}%")
        
        if e2e_result.get('success'):
            print(f"端到端延迟: {e2e_result['total_latency_ms']:.1f}ms")
    
    asyncio.run(main())
