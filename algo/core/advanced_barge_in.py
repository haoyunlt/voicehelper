"""
高级语音打断系统 - v1.8.0
智能barge-in处理，支持自然语音打断和上下文保存
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque
import threading
import queue

logger = logging.getLogger(__name__)

class InterruptionType(Enum):
    """打断类型"""
    INTENTIONAL = "intentional"      # 有意打断
    ACCIDENTAL = "accidental"        # 意外打断
    BACKGROUND_NOISE = "background"  # 背景噪音
    UNCLEAR = "unclear"              # 不明确

class InterruptionIntent(Enum):
    """打断意图"""
    STOP_RESPONSE = "stop"           # 停止回复
    CLARIFY_QUESTION = "clarify"     # 澄清问题
    NEW_QUESTION = "new_question"    # 新问题
    AGREEMENT = "agreement"          # 同意/确认
    DISAGREEMENT = "disagreement"    # 不同意/否定
    CONTINUE = "continue"            # 继续之前的话题

class VADState(Enum):
    """语音活动检测状态"""
    SILENCE = "silence"
    SPEECH = "speech"
    UNCERTAIN = "uncertain"

@dataclass
class InterruptionEvent:
    """打断事件"""
    timestamp: float
    interruption_type: InterruptionType
    intent: InterruptionIntent
    confidence: float
    audio_data: bytes
    context_before: str = ""
    context_after: str = ""
    user_speech: str = ""
    
@dataclass
class TTSPlaybackState:
    """TTS播放状态"""
    is_playing: bool = False
    current_sentence: str = ""
    sentence_position: int = 0
    total_sentences: int = 0
    start_time: float = 0
    pause_time: Optional[float] = None
    resume_time: Optional[float] = None

class RealTimeVAD:
    """实时语音活动检测器"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 frame_size_ms: int = 30,
                 sensitivity: float = 0.5):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_size_ms / 1000)
        self.sensitivity = sensitivity
        
        # VAD参数
        self.energy_threshold = 0.01
        self.zero_crossing_threshold = 0.1
        self.speech_frames_required = 3
        self.silence_frames_required = 10
        
        # 状态跟踪
        self.current_state = VADState.SILENCE
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.energy_history = deque(maxlen=50)
        
    async def detect_voice_activity(self, audio_chunk: bytes) -> Tuple[VADState, float]:
        """
        检测语音活动
        
        Args:
            audio_chunk: 音频数据块
            
        Returns:
            Tuple[VADState, float]: (VAD状态, 置信度)
        """
        try:
            # 转换音频数据
            audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
            
            if len(audio_array) == 0:
                return VADState.SILENCE, 1.0
            
            # 计算能量特征
            energy = np.mean(audio_array ** 2)
            self.energy_history.append(energy)
            
            # 计算零交叉率
            zero_crossings = np.sum(np.diff(np.sign(audio_array)) != 0)
            zcr = zero_crossings / len(audio_array)
            
            # 动态调整阈值
            if len(self.energy_history) > 10:
                avg_energy = np.mean(list(self.energy_history))
                self.energy_threshold = avg_energy * (1 + self.sensitivity)
            
            # 判断是否为语音
            is_speech = (energy > self.energy_threshold and 
                        zcr > self.zero_crossing_threshold)
            
            # 状态机逻辑
            if is_speech:
                self.speech_frame_count += 1
                self.silence_frame_count = 0
                
                if self.speech_frame_count >= self.speech_frames_required:
                    new_state = VADState.SPEECH
                    confidence = min(self.speech_frame_count / 10, 1.0)
                else:
                    new_state = VADState.UNCERTAIN
                    confidence = self.speech_frame_count / self.speech_frames_required
            else:
                self.silence_frame_count += 1
                self.speech_frame_count = 0
                
                if self.silence_frame_count >= self.silence_frames_required:
                    new_state = VADState.SILENCE
                    confidence = min(self.silence_frame_count / 20, 1.0)
                else:
                    new_state = VADState.UNCERTAIN
                    confidence = 1.0 - (self.silence_frame_count / self.silence_frames_required)
            
            # 更新状态
            if new_state != VADState.UNCERTAIN:
                self.current_state = new_state
            
            return self.current_state, confidence
            
        except Exception as e:
            logger.error(f"VAD detection error: {e}")
            return VADState.UNCERTAIN, 0.0
    
    def reset(self):
        """重置VAD状态"""
        self.current_state = VADState.SILENCE
        self.speech_frame_count = 0
        self.silence_frame_count = 0
        self.energy_history.clear()

class InterruptionIntentClassifier:
    """打断意图分类器"""
    
    def __init__(self):
        # 意图关键词映射
        self.intent_keywords = {
            InterruptionIntent.STOP_RESPONSE: [
                '停', '别说了', '够了', '不用了', '停止', '暂停', '等等'
            ],
            InterruptionIntent.CLARIFY_QUESTION: [
                '什么', '怎么', '为什么', '哪个', '哪里', '什么意思', '不明白', '解释'
            ],
            InterruptionIntent.NEW_QUESTION: [
                '我想问', '另外', '还有', '对了', '顺便问', '换个话题'
            ],
            InterruptionIntent.AGREEMENT: [
                '对', '是的', '没错', '正确', '好的', '可以', '同意', '赞成'
            ],
            InterruptionIntent.DISAGREEMENT: [
                '不对', '错了', '不是', '不同意', '反对', '不行', '不可以'
            ],
            InterruptionIntent.CONTINUE: [
                '继续', '接着说', '然后呢', '还有吗', '下一个', '往下说'
            ]
        }
        
        # 语音特征到意图的映射
        self.acoustic_patterns = {
            InterruptionIntent.STOP_RESPONSE: {
                'energy_level': 'high',
                'pitch_change': 'rising',
                'duration': 'short'
            },
            InterruptionIntent.CLARIFY_QUESTION: {
                'energy_level': 'medium',
                'pitch_change': 'rising',
                'duration': 'medium'
            },
            InterruptionIntent.AGREEMENT: {
                'energy_level': 'low',
                'pitch_change': 'falling',
                'duration': 'short'
            }
        }
    
    async def classify_interruption_intent(self, 
                                         user_speech: str,
                                         audio_features: Dict[str, Any],
                                         context: str) -> Tuple[InterruptionIntent, float]:
        """
        分类打断意图
        
        Args:
            user_speech: 用户语音转文本
            audio_features: 音频特征
            context: 上下文信息
            
        Returns:
            Tuple[InterruptionIntent, float]: (意图, 置信度)
        """
        try:
            intent_scores = {}
            
            # 1. 基于关键词的意图识别
            for intent, keywords in self.intent_keywords.items():
                score = 0
                for keyword in keywords:
                    if keyword in user_speech:
                        score += 1
                
                if score > 0:
                    intent_scores[intent] = score / len(keywords)
            
            # 2. 基于语音特征的意图识别
            acoustic_score = self._analyze_acoustic_patterns(audio_features)
            for intent, score in acoustic_score.items():
                intent_scores[intent] = intent_scores.get(intent, 0) + score * 0.3
            
            # 3. 基于上下文的意图推理
            context_score = self._analyze_context_patterns(user_speech, context)
            for intent, score in context_score.items():
                intent_scores[intent] = intent_scores.get(intent, 0) + score * 0.2
            
            # 选择得分最高的意图
            if intent_scores:
                best_intent = max(intent_scores.items(), key=lambda x: x[1])
                return best_intent[0], min(best_intent[1], 1.0)
            else:
                return InterruptionIntent.UNCLEAR, 0.5
                
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return InterruptionIntent.UNCLEAR, 0.0
    
    def _analyze_acoustic_patterns(self, audio_features: Dict[str, Any]) -> Dict[InterruptionIntent, float]:
        """分析声学模式"""
        scores = {}
        
        energy = audio_features.get('energy', 0)
        pitch_change = audio_features.get('pitch_change', 0)
        duration = audio_features.get('duration', 0)
        
        # 停止回复：高能量，上升音调，短时长
        if energy > 0.8 and pitch_change > 0.5 and duration < 1.0:
            scores[InterruptionIntent.STOP_RESPONSE] = 0.8
        
        # 澄清问题：中等能量，上升音调，中等时长
        if 0.3 < energy < 0.7 and pitch_change > 0.3 and 1.0 < duration < 3.0:
            scores[InterruptionIntent.CLARIFY_QUESTION] = 0.7
        
        # 同意：低能量，下降音调，短时长
        if energy < 0.4 and pitch_change < -0.2 and duration < 0.8:
            scores[InterruptionIntent.AGREEMENT] = 0.6
        
        return scores
    
    def _analyze_context_patterns(self, user_speech: str, context: str) -> Dict[InterruptionIntent, float]:
        """分析上下文模式"""
        scores = {}
        
        # 如果上下文是问题，用户可能在澄清
        if '?' in context or '吗' in context or '什么' in context:
            if any(word in user_speech for word in ['什么', '怎么', '为什么']):
                scores[InterruptionIntent.CLARIFY_QUESTION] = 0.6
        
        # 如果上下文是陈述，用户可能在表达同意/不同意
        if '。' in context:
            if any(word in user_speech for word in ['对', '是的', '没错']):
                scores[InterruptionIntent.AGREEMENT] = 0.7
            elif any(word in user_speech for word in ['不对', '错了', '不是']):
                scores[InterruptionIntent.DISAGREEMENT] = 0.7
        
        return scores

class ContextManager:
    """上下文管理器"""
    
    def __init__(self):
        self.conversation_context = deque(maxlen=100)
        self.current_tts_context = ""
        self.interrupted_contexts = []
        
    def save_context_before_interruption(self, 
                                       tts_text: str, 
                                       position: int,
                                       user_input: str) -> str:
        """
        保存打断前的上下文
        
        Args:
            tts_text: TTS文本
            position: 打断位置
            user_input: 用户输入
            
        Returns:
            str: 上下文ID
        """
        context_id = f"ctx_{int(time.time() * 1000)}"
        
        # 分割文本
        sentences = tts_text.split('。')
        completed_text = '。'.join(sentences[:position])
        remaining_text = '。'.join(sentences[position:])
        
        context = {
            'id': context_id,
            'timestamp': time.time(),
            'completed_text': completed_text,
            'remaining_text': remaining_text,
            'full_text': tts_text,
            'interruption_position': position,
            'user_input': user_input,
            'conversation_history': list(self.conversation_context)[-10:]  # 最近10条
        }
        
        self.interrupted_contexts.append(context)
        
        # 保持最近20个中断上下文
        if len(self.interrupted_contexts) > 20:
            self.interrupted_contexts.pop(0)
        
        logger.info(f"Context saved: {context_id}, position: {position}")
        return context_id
    
    def restore_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """恢复上下文"""
        for context in self.interrupted_contexts:
            if context['id'] == context_id:
                return context
        return None
    
    def generate_continuation_prompt(self, context_id: str, new_user_input: str) -> str:
        """生成继续对话的提示"""
        context = self.restore_context(context_id)
        if not context:
            return new_user_input
        
        prompt = f"""
        上下文信息：
        - 之前的回复：{context['completed_text']}
        - 被打断的内容：{context['remaining_text']}
        - 用户打断时说：{context['user_input']}
        - 用户现在说：{new_user_input}
        
        请根据上下文继续对话。
        """
        
        return prompt.strip()
    
    def update_conversation_context(self, role: str, content: str):
        """更新对话上下文"""
        self.conversation_context.append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })

class AdvancedBargeIn:
    """高级语音打断处理器"""
    
    def __init__(self, 
                 vad_sensitivity: float = 0.5,
                 interruption_threshold: float = 0.7):
        self.vad = RealTimeVAD(sensitivity=vad_sensitivity)
        self.intent_classifier = InterruptionIntentClassifier()
        self.context_manager = ContextManager()
        
        self.interruption_threshold = interruption_threshold
        self.tts_state = TTSPlaybackState()
        self.is_monitoring = False
        self.interruption_callbacks = []
        
        # 音频流处理
        self.audio_buffer = deque(maxlen=1000)
        self.processing_task: Optional[asyncio.Task] = None
        
    async def start_tts_playback(self, 
                               tts_text: str, 
                               interruption_callback: Callable[[InterruptionEvent], None]):
        """
        开始TTS播放并监控打断
        
        Args:
            tts_text: TTS文本内容
            interruption_callback: 打断回调函数
        """
        try:
            # 更新TTS状态
            sentences = tts_text.split('。')
            self.tts_state = TTSPlaybackState(
                is_playing=True,
                current_sentence=tts_text,
                sentence_position=0,
                total_sentences=len(sentences),
                start_time=time.time()
            )
            
            # 注册回调
            if interruption_callback not in self.interruption_callbacks:
                self.interruption_callbacks.append(interruption_callback)
            
            # 开始监控
            await self.start_monitoring()
            
            logger.info(f"Started TTS playback monitoring for: {tts_text[:50]}...")
            
        except Exception as e:
            logger.error(f"Failed to start TTS playback monitoring: {e}")
    
    async def stop_tts_playback(self):
        """停止TTS播放监控"""
        self.tts_state.is_playing = False
        await self.stop_monitoring()
        self.interruption_callbacks.clear()
        logger.info("Stopped TTS playback monitoring")
    
    async def start_monitoring(self):
        """开始监控音频流"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.vad.reset()
        
        # 启动音频处理任务
        self.processing_task = asyncio.create_task(self._audio_processing_loop())
        logger.info("Started interruption monitoring")
    
    async def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped interruption monitoring")
    
    async def process_audio_chunk(self, audio_chunk: bytes):
        """处理音频块"""
        if not self.is_monitoring or not self.tts_state.is_playing:
            return
        
        # 添加到缓冲区
        self.audio_buffer.append({
            'data': audio_chunk,
            'timestamp': time.time()
        })
    
    async def _audio_processing_loop(self):
        """音频处理循环"""
        try:
            while self.is_monitoring:
                if not self.audio_buffer:
                    await asyncio.sleep(0.01)
                    continue
                
                # 处理音频块
                audio_item = self.audio_buffer.popleft()
                await self._process_audio_for_interruption(
                    audio_item['data'], 
                    audio_item['timestamp']
                )
                
        except asyncio.CancelledError:
            logger.info("Audio processing loop cancelled")
        except Exception as e:
            logger.error(f"Audio processing loop error: {e}")
    
    async def _process_audio_for_interruption(self, audio_chunk: bytes, timestamp: float):
        """处理音频以检测打断"""
        try:
            # VAD检测
            vad_state, vad_confidence = await self.vad.detect_voice_activity(audio_chunk)
            
            # 如果检测到语音活动
            if vad_state == VADState.SPEECH and vad_confidence > self.interruption_threshold:
                
                # 模拟ASR处理（实际应调用真实ASR）
                user_speech = await self._simulate_asr(audio_chunk)
                
                if user_speech.strip():
                    # 分析打断类型和意图
                    interruption_type = await self._classify_interruption_type(
                        audio_chunk, user_speech
                    )
                    
                    # 提取音频特征
                    audio_features = await self._extract_audio_features(audio_chunk)
                    
                    # 分类打断意图
                    intent, intent_confidence = await self.intent_classifier.classify_interruption_intent(
                        user_speech, audio_features, self.tts_state.current_sentence
                    )
                    
                    # 创建打断事件
                    interruption_event = InterruptionEvent(
                        timestamp=timestamp,
                        interruption_type=interruption_type,
                        intent=intent,
                        confidence=min(vad_confidence, intent_confidence),
                        audio_data=audio_chunk,
                        context_before=self.tts_state.current_sentence,
                        user_speech=user_speech
                    )
                    
                    # 处理打断
                    await self._handle_interruption(interruption_event)
                    
        except Exception as e:
            logger.error(f"Interruption processing error: {e}")
    
    async def _classify_interruption_type(self, 
                                        audio_chunk: bytes, 
                                        user_speech: str) -> InterruptionType:
        """分类打断类型"""
        # 简单的打断类型分类逻辑
        if len(user_speech.strip()) < 2:
            return InterruptionType.BACKGROUND_NOISE
        
        # 检查是否包含明确的停止词
        stop_words = ['停', '等等', '暂停', '别说了']
        if any(word in user_speech for word in stop_words):
            return InterruptionType.INTENTIONAL
        
        # 检查是否是问题
        question_words = ['什么', '怎么', '为什么', '哪个']
        if any(word in user_speech for word in question_words):
            return InterruptionType.INTENTIONAL
        
        # 默认为意外打断
        return InterruptionType.ACCIDENTAL
    
    async def _extract_audio_features(self, audio_chunk: bytes) -> Dict[str, Any]:
        """提取音频特征"""
        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.float32)
            
            if len(audio_array) == 0:
                return {}
            
            # 计算基本特征
            energy = np.mean(audio_array ** 2)
            duration = len(audio_array) / 16000  # 假设16kHz采样率
            
            # 简单的音调变化检测
            pitch_change = np.std(audio_array) / np.mean(np.abs(audio_array)) if np.mean(np.abs(audio_array)) > 0 else 0
            
            return {
                'energy': energy,
                'duration': duration,
                'pitch_change': pitch_change
            }
            
        except Exception as e:
            logger.error(f"Audio feature extraction error: {e}")
            return {}
    
    async def _simulate_asr(self, audio_chunk: bytes) -> str:
        """模拟ASR处理"""
        # 模拟ASR延迟
        await asyncio.sleep(0.02)
        
        # 简单的模拟ASR结果
        chunk_hash = hash(audio_chunk) % 100
        
        if chunk_hash < 20:
            return "等等"
        elif chunk_hash < 40:
            return "什么意思"
        elif chunk_hash < 60:
            return "不对"
        elif chunk_hash < 80:
            return "继续说"
        else:
            return "好的"
    
    async def _handle_interruption(self, event: InterruptionEvent):
        """处理打断事件"""
        try:
            logger.info(f"Interruption detected: {event.intent.value} - {event.user_speech}")
            
            # 保存上下文
            context_id = self.context_manager.save_context_before_interruption(
                self.tts_state.current_sentence,
                self.tts_state.sentence_position,
                event.user_speech
            )
            
            # 根据打断意图执行相应动作
            if event.intent == InterruptionIntent.STOP_RESPONSE:
                # 停止TTS播放
                await self.stop_tts_playback()
                
            elif event.intent == InterruptionIntent.CLARIFY_QUESTION:
                # 暂停TTS，等待澄清
                self.tts_state.pause_time = time.time()
                
            elif event.intent == InterruptionIntent.NEW_QUESTION:
                # 停止当前回复，处理新问题
                await self.stop_tts_playback()
            
            # 调用注册的回调函数
            for callback in self.interruption_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.error(f"Interruption callback error: {e}")
                    
        except Exception as e:
            logger.error(f"Interruption handling error: {e}")
    
    async def resume_after_interruption(self, context_id: str, user_response: str) -> str:
        """打断后恢复对话"""
        try:
            # 生成继续对话的提示
            continuation_prompt = self.context_manager.generate_continuation_prompt(
                context_id, user_response
            )
            
            # 恢复TTS状态
            if self.tts_state.pause_time:
                self.tts_state.resume_time = time.time()
                self.tts_state.pause_time = None
            
            return continuation_prompt
            
        except Exception as e:
            logger.error(f"Resume after interruption error: {e}")
            return user_response
    
    def get_interruption_stats(self) -> Dict[str, Any]:
        """获取打断统计信息"""
        return {
            'is_monitoring': self.is_monitoring,
            'tts_playing': self.tts_state.is_playing,
            'current_sentence': self.tts_state.current_sentence[:50] + "..." if len(self.tts_state.current_sentence) > 50 else self.tts_state.current_sentence,
            'sentence_position': self.tts_state.sentence_position,
            'total_sentences': self.tts_state.total_sentences,
            'interrupted_contexts_count': len(self.context_manager.interrupted_contexts),
            'audio_buffer_size': len(self.audio_buffer)
        }

# 使用示例
async def main():
    """示例用法"""
    
    def interruption_callback(event: InterruptionEvent):
        print(f"🚨 打断检测: {event.intent.value}")
        print(f"   用户说: {event.user_speech}")
        print(f"   置信度: {event.confidence:.2f}")
        print(f"   类型: {event.interruption_type.value}")
    
    barge_in = AdvancedBargeIn(
        vad_sensitivity=0.6,
        interruption_threshold=0.7
    )
    
    # 模拟TTS播放
    tts_text = "今天天气很好。我们可以出去散步。你觉得怎么样？"
    await barge_in.start_tts_playback(tts_text, interruption_callback)
    
    # 模拟音频输入
    for i in range(10):
        mock_audio = b"mock_audio_data" * (50 + i * 10)
        await barge_in.process_audio_chunk(mock_audio)
        await asyncio.sleep(0.1)
    
    # 获取统计信息
    stats = barge_in.get_interruption_stats()
    print("\n=== 打断系统统计 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 停止监控
    await barge_in.stop_tts_playback()

if __name__ == "__main__":
    asyncio.run(main())
