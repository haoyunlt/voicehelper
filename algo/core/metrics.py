import time
import asyncio
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

@dataclass
class VoiceMetrics:
    """语音处理指标"""
    session_id: str
    
    # ASR 指标
    asr_latency: float = 0.0  # ASR 延迟（秒）
    asr_accuracy: float = 0.0  # ASR 准确率
    word_error_rate: float = 0.0  # 词错误率
    asr_rtf: float = 0.0  # Real-time Factor
    
    # TTS 指标
    tts_latency: float = 0.0  # TTS 延迟（秒）
    tts_first_audio: float = 0.0  # 首音延迟（秒）
    tts_rtf: float = 0.0  # Real-time Factor
    
    # VAD 指标
    vad_latency: float = 0.0  # VAD 检测延迟
    false_positive_rate: float = 0.0  # 误检率
    false_negative_rate: float = 0.0  # 漏检率
    
    # 端到端指标
    end_to_end_latency: float = 0.0  # 端到端延迟
    first_response_time: float = 0.0  # 首响时间
    
    # 质量指标
    audio_quality_score: float = 0.0  # 音频质量评分
    user_satisfaction: float = 0.0  # 用户满意度
    
    # 打断指标
    barge_in_success_rate: float = 0.0  # 打断成功率
    interrupt_response_time: float = 0.0  # 打断响应时间
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class VoiceMetricsCollector:
    """语音指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, VoiceMetrics] = {}
        self.session_timers: Dict[str, Dict[str, float]] = {}
        
    def start_timer(self, session_id: str, timer_name: str):
        """开始计时"""
        if session_id not in self.session_timers:
            self.session_timers[session_id] = {}
        self.session_timers[session_id][timer_name] = time.time()
    
    def end_timer(self, session_id: str, timer_name: str) -> float:
        """结束计时并返回耗时"""
        if session_id in self.session_timers and timer_name in self.session_timers[session_id]:
            start_time = self.session_timers[session_id][timer_name]
            duration = time.time() - start_time
            del self.session_timers[session_id][timer_name]
            return duration
        return 0.0
    
    def get_or_create_metrics(self, session_id: str) -> VoiceMetrics:
        """获取或创建指标对象"""
        if session_id not in self.metrics:
            self.metrics[session_id] = VoiceMetrics(session_id=session_id)
        return self.metrics[session_id]
    
    def record_asr_metrics(self, session_id: str, latency: float, accuracy: float = 0.0, wer: float = 0.0):
        """记录 ASR 指标"""
        metrics = self.get_or_create_metrics(session_id)
        metrics.asr_latency = latency
        metrics.asr_accuracy = accuracy
        metrics.word_error_rate = wer
        metrics.updated_at = datetime.now()
        
        print(f"ASR Metrics - Session: {session_id}, Latency: {latency:.3f}s, Accuracy: {accuracy:.2f}")
    
    def record_tts_metrics(self, session_id: str, latency: float, first_audio: float = 0.0):
        """记录 TTS 指标"""
        metrics = self.get_or_create_metrics(session_id)
        metrics.tts_latency = latency
        metrics.tts_first_audio = first_audio
        metrics.updated_at = datetime.now()
        
        print(f"TTS Metrics - Session: {session_id}, Latency: {latency:.3f}s, First Audio: {first_audio:.3f}s")
    
    def record_vad_metrics(self, session_id: str, latency: float, fp_rate: float = 0.0, fn_rate: float = 0.0):
        """记录 VAD 指标"""
        metrics = self.get_or_create_metrics(session_id)
        metrics.vad_latency = latency
        metrics.false_positive_rate = fp_rate
        metrics.false_negative_rate = fn_rate
        metrics.updated_at = datetime.now()
        
        print(f"VAD Metrics - Session: {session_id}, Latency: {latency:.3f}s")
    
    def record_end_to_end_metrics(self, session_id: str, total_latency: float, first_response: float = 0.0):
        """记录端到端指标"""
        metrics = self.get_or_create_metrics(session_id)
        metrics.end_to_end_latency = total_latency
        metrics.first_response_time = first_response
        metrics.updated_at = datetime.now()
        
        print(f"E2E Metrics - Session: {session_id}, Total: {total_latency:.3f}s, First Response: {first_response:.3f}s")
    
    def record_barge_in_metrics(self, session_id: str, success: bool, response_time: float):
        """记录打断指标"""
        metrics = self.get_or_create_metrics(session_id)
        metrics.barge_in_success_rate = 1.0 if success else 0.0
        metrics.interrupt_response_time = response_time
        metrics.updated_at = datetime.now()
        
        print(f"Barge-in Metrics - Session: {session_id}, Success: {success}, Response Time: {response_time:.3f}s")
    
    def record_quality_metrics(self, session_id: str, audio_quality: float, satisfaction: float = 0.0):
        """记录质量指标"""
        metrics = self.get_or_create_metrics(session_id)
        metrics.audio_quality_score = audio_quality
        metrics.user_satisfaction = satisfaction
        metrics.updated_at = datetime.now()
        
        print(f"Quality Metrics - Session: {session_id}, Audio Quality: {audio_quality:.2f}")
    
    def get_metrics(self, session_id: str) -> Optional[VoiceMetrics]:
        """获取会话指标"""
        return self.metrics.get(session_id)
    
    def get_all_metrics(self) -> Dict[str, VoiceMetrics]:
        """获取所有指标"""
        return self.metrics.copy()
    
    def get_aggregated_metrics(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, float]:
        """获取聚合指标"""
        current_time = datetime.now()
        recent_metrics = [
            m for m in self.metrics.values()
            if current_time - m.updated_at <= time_window
        ]
        
        if not recent_metrics:
            return {}
        
        return {
            "avg_asr_latency": sum(m.asr_latency for m in recent_metrics) / len(recent_metrics),
            "avg_tts_latency": sum(m.tts_latency for m in recent_metrics) / len(recent_metrics),
            "avg_e2e_latency": sum(m.end_to_end_latency for m in recent_metrics) / len(recent_metrics),
            "avg_first_response": sum(m.first_response_time for m in recent_metrics) / len(recent_metrics),
            "barge_in_success_rate": sum(m.barge_in_success_rate for m in recent_metrics) / len(recent_metrics),
            "avg_audio_quality": sum(m.audio_quality_score for m in recent_metrics) / len(recent_metrics),
            "total_sessions": len(recent_metrics)
        }
    
    def cleanup_old_metrics(self, max_age: timedelta = timedelta(hours=24)):
        """清理过期指标"""
        current_time = datetime.now()
        expired_sessions = [
            session_id for session_id, metrics in self.metrics.items()
            if current_time - metrics.updated_at > max_age
        ]
        
        for session_id in expired_sessions:
            del self.metrics[session_id]
            if session_id in self.session_timers:
                del self.session_timers[session_id]
        
        print(f"Cleaned up {len(expired_sessions)} expired metric sessions")
    
    def export_metrics_json(self) -> str:
        """导出 JSON 格式指标"""
        export_data = {}
        for session_id, metrics in self.metrics.items():
            export_data[session_id] = {
                "asr_latency": metrics.asr_latency,
                "asr_accuracy": metrics.asr_accuracy,
                "word_error_rate": metrics.word_error_rate,
                "tts_latency": metrics.tts_latency,
                "tts_first_audio": metrics.tts_first_audio,
                "vad_latency": metrics.vad_latency,
                "end_to_end_latency": metrics.end_to_end_latency,
                "first_response_time": metrics.first_response_time,
                "barge_in_success_rate": metrics.barge_in_success_rate,
                "interrupt_response_time": metrics.interrupt_response_time,
                "audio_quality_score": metrics.audio_quality_score,
                "user_satisfaction": metrics.user_satisfaction,
                "created_at": metrics.created_at.isoformat(),
                "updated_at": metrics.updated_at.isoformat()
            }
        
        return json.dumps(export_data, indent=2)
    
    def get_performance_summary(self) -> Dict[str, any]:
        """获取性能摘要"""
        if not self.metrics:
            return {"status": "no_data"}
        
        aggregated = self.get_aggregated_metrics()
        
        # 性能等级评估
        def get_performance_grade(latency: float, target: float) -> str:
            if latency <= target:
                return "excellent"
            elif latency <= target * 1.5:
                return "good"
            elif latency <= target * 2:
                return "fair"
            else:
                return "poor"
        
        return {
            "total_sessions": aggregated.get("total_sessions", 0),
            "asr_performance": {
                "avg_latency": aggregated.get("avg_asr_latency", 0),
                "grade": get_performance_grade(aggregated.get("avg_asr_latency", 0), 0.4)
            },
            "tts_performance": {
                "avg_latency": aggregated.get("avg_tts_latency", 0),
                "avg_first_audio": aggregated.get("avg_first_response", 0),
                "grade": get_performance_grade(aggregated.get("avg_tts_latency", 0), 0.5)
            },
            "e2e_performance": {
                "avg_latency": aggregated.get("avg_e2e_latency", 0),
                "grade": get_performance_grade(aggregated.get("avg_e2e_latency", 0), 0.7)
            },
            "barge_in_success_rate": aggregated.get("barge_in_success_rate", 0),
            "audio_quality": aggregated.get("avg_audio_quality", 0),
            "last_updated": datetime.now().isoformat()
        }

# 全局指标收集器实例
voice_metrics_collector = VoiceMetricsCollector()

# 定期清理任务
async def cleanup_metrics_task():
    """定期清理过期指标的后台任务"""
    while True:
        try:
            voice_metrics_collector.cleanup_old_metrics()
            await asyncio.sleep(3600)  # 每小时清理一次
        except Exception as e:
            print(f"Metrics cleanup error: {e}")
            await asyncio.sleep(300)  # 出错后5分钟重试
