"""
NVIDIA Riva STT实现 - 本地化语音识别
"""
import asyncio
import grpc
from typing import Optional, List
import structlog

from app.config import settings
from app.models.schemas import Transcription, AudioChunk
from .base import STTService

logger = structlog.get_logger()

try:
    # NVIDIA Riva gRPC客户端
    import riva.client as riva_client
    import riva.client.proto.riva_asr_pb2 as rasr
    import riva.client.proto.riva_asr_pb2_grpc as rasr_srv
    RIVA_AVAILABLE = True
except ImportError:
    RIVA_AVAILABLE = False
    logger.warning("NVIDIA Riva client not available, install nvidia-riva-client")


class RivaSTT(STTService):
    """NVIDIA Riva STT服务实现"""
    
    def __init__(self):
        super().__init__()
        
        if not RIVA_AVAILABLE:
            raise ImportError("NVIDIA Riva client not available")
        
        self.server_url = settings.RIVA_SERVER
        if not self.server_url:
            raise ValueError("RIVA_SERVER is required")
        
        # 初始化Riva客户端
        self.auth = riva_client.Auth(uri=self.server_url)
        self.asr_service = riva_client.ASRService(self.auth)
        
        # 配置
        self.config = riva_client.StreamingRecognitionConfig(
            config=riva_client.RecognitionConfig(
                encoding=riva_client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=16000,
                language_code="zh-CN",
                max_alternatives=1,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
            ),
            interim_results=True,
        )
    
    async def start_stream(
        self, 
        session_id: str,
        language: str = "zh-CN",
        hints: Optional[List[str]] = None
    ) -> None:
        """开始Riva流式识别"""
        try:
            # 更新语言配置
            config = self.config
            config.config.language_code = language
            
            # 添加语音提示
            if hints:
                config.config.speech_contexts.extend([
                    riva_client.SpeechContext(phrases=hints)
                ])
            
            # 创建流式识别请求生成器
            def request_generator():
                # 首先发送配置
                yield rasr.StreamingRecognizeRequest(streaming_config=config)
                
                # 然后等待音频数据
                while session_id in self.active_streams:
                    stream_info = self.active_streams[session_id]
                    if stream_info["audio_queue"]:
                        audio_data = stream_info["audio_queue"].pop(0)
                        yield rasr.StreamingRecognizeRequest(audio_content=audio_data)
                    else:
                        # 短暂等待
                        asyncio.sleep(0.01)
            
            # 初始化会话信息
            self.active_streams[session_id] = {
                "language": language,
                "hints": hints or [],
                "audio_queue": [],
                "results": [],
                "request_generator": request_generator,
                "recognition_task": None
            }
            
            # 启动识别任务
            recognition_task = asyncio.create_task(
                self._run_recognition(session_id, request_generator())
            )
            self.active_streams[session_id]["recognition_task"] = recognition_task
            
            logger.info(
                "Riva STT stream started", 
                session_id=session_id,
                language=language
            )
            
        except Exception as e:
            logger.error(
                "Failed to start Riva stream",
                session_id=session_id,
                error=str(e)
            )
            raise
    
    async def ingest_audio(
        self, 
        session_id: str,
        audio_chunk: AudioChunk
    ) -> Optional[Transcription]:
        """发送音频数据到Riva"""
        if session_id not in self.active_streams:
            logger.warning("STT stream not found", session_id=session_id)
            return None
        
        stream_info = self.active_streams[session_id]
        
        try:
            # 添加音频数据到队列
            stream_info["audio_queue"].append(audio_chunk.data)
            
            # 检查是否有新的识别结果
            if stream_info["results"]:
                return stream_info["results"].pop(0)
            
        except Exception as e:
            logger.error(
                "Failed to send audio to Riva",
                session_id=session_id,
                error=str(e)
            )
        
        return None
    
    async def finish_stream(self, session_id: str) -> Optional[Transcription]:
        """结束Riva流式识别"""
        if session_id not in self.active_streams:
            return None
        
        stream_info = self.active_streams[session_id]
        
        try:
            # 取消识别任务
            if stream_info["recognition_task"]:
                stream_info["recognition_task"].cancel()
                try:
                    await stream_info["recognition_task"]
                except asyncio.CancelledError:
                    pass
            
            # 获取最终结果
            final_result = None
            if stream_info["results"]:
                final_result = stream_info["results"][-1]
                final_result.is_final = True
            
            logger.info("Riva STT stream finished", session_id=session_id)
            
        except Exception as e:
            logger.error(
                "Error finishing Riva stream",
                session_id=session_id,
                error=str(e)
            )
        finally:
            await self.cleanup_session(session_id)
        
        return final_result
    
    async def _run_recognition(self, session_id: str, request_generator):
        """运行Riva识别"""
        try:
            # 调用Riva流式识别
            responses = self.asr_service.streaming_response_generator(
                requests=request_generator
            )
            
            stream_info = self.active_streams[session_id]
            
            for response in responses:
                if session_id not in self.active_streams:
                    break
                
                for result in response.results:
                    if result.alternatives:
                        alternative = result.alternatives[0]
                        
                        transcription = Transcription(
                            text=alternative.transcript,
                            confidence=alternative.confidence,
                            start_time_ms=int(result.result_end_time * 1000),
                            end_time_ms=int(result.result_end_time * 1000),
                            is_final=result.is_final,
                            language=stream_info["language"]
                        )
                        
                        # 添加到结果队列
                        stream_info["results"].append(transcription)
                        
                        logger.debug(
                            "Riva result received",
                            session_id=session_id,
                            text=alternative.transcript,
                            is_final=result.is_final,
                            confidence=alternative.confidence
                        )
        
        except Exception as e:
            logger.error(
                "Error in Riva recognition",
                session_id=session_id,
                error=str(e)
            )


class MockRivaSTT(STTService):
    """Mock Riva STT服务，用于没有Riva服务器时的测试"""
    
    async def start_stream(
        self, 
        session_id: str,
        language: str = "zh-CN",
        hints: Optional[List[str]] = None
    ) -> None:
        self.active_streams[session_id] = {
            "language": language,
            "hints": hints or [],
            "buffer": []
        }
        logger.info("Mock Riva STT stream started", session_id=session_id)
    
    async def ingest_audio(
        self, 
        session_id: str,
        audio_chunk: AudioChunk
    ) -> Optional[Transcription]:
        if session_id not in self.active_streams:
            return None
        
        # 模拟识别延迟
        await asyncio.sleep(0.15)
        
        # 返回模拟结果
        return Transcription(
            text="这是模拟的Riva语音识别结果",
            confidence=0.92,
            start_time_ms=audio_chunk.timestamp_ms,
            end_time_ms=audio_chunk.timestamp_ms + 1000,
            is_final=False,
            language="zh-CN"
        )
    
    async def finish_stream(self, session_id: str) -> Optional[Transcription]:
        if session_id not in self.active_streams:
            return None
        
        result = Transcription(
            text="这是最终的Riva语音识别结果",
            confidence=0.95,
            start_time_ms=0,
            end_time_ms=2000,
            is_final=True,
            language="zh-CN"
        )
        
        await self.cleanup_session(session_id)
        return result
