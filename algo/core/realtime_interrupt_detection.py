"""
VoiceHelper v1.23.0 - 实时语音打断检测系统
实现智能打断检测、上下文保持、多轮对话管理
"""

import asyncio
import time
import logging
import json
# import numpy as np  # 暂时注释掉numpy依赖
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import uuid

logger = logging.getLogger(__name__)

class InterruptType(Enum):
    """打断类型"""
    VOICE_INTERRUPT = "voice_interrupt"
    GESTURE_INTERRUPT = "gesture_interrupt"
    TEXT_INTERRUPT = "text_interrupt"
    EMERGENCY_INTERRUPT = "emergency_interrupt"
    SILENCE_INTERRUPT = "silence_interrupt"

class InterruptConfidence(Enum):
    """打断置信度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class ConversationState(Enum):
    """对话状态"""
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    WAITING = "waiting"

@dataclass
class InterruptSignal:
    """打断信号"""
    interrupt_id: str
    interrupt_type: InterruptType
    confidence: float
    timestamp: float
    audio_features: Optional[Dict[str, Any]] = None
    text_content: Optional[str] = None
    gesture_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConversationContext:
    """对话上下文"""
    session_id: str
    user_id: str
    current_state: ConversationState
    conversation_history: List[Dict[str, Any]]
    pending_requests: List[str]
    interrupt_history: List[InterruptSignal]
    context_vectors: Dict[str, List[float]]
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

@dataclass
class InterruptResponse:
    """打断响应"""
    should_interrupt: bool
    interrupt_confidence: float
    response_action: str
    context_preservation: bool
    recovery_strategy: str
    estimated_recovery_time: float

class VoiceFeatureExtractor:
    """语音特征提取器"""
    
    def __init__(self):
        self.feature_dim = 128
        self.sample_rate = 16000
        self.window_size = 1024
        self.hop_length = 512
        
    def extract_features(self, audio_data: bytes) -> Dict[str, Any]:
        """提取语音特征"""
        import random
        # 模拟语音特征提取
        features = {
            "energy": random.random(),
            "zero_crossing_rate": random.random(),
            "spectral_centroid": random.random() * 4000,
            "spectral_rolloff": random.random() * 4000,
            "mfcc": [random.random() for _ in range(13)],
            "chroma": [random.random() for _ in range(12)],
            "mel_spectrogram": [random.random() for _ in range(128)],
            "pitch": random.random() * 500,
            "formants": [random.random() for _ in range(3)],
            "voice_activity": random.random() > 0.5
        }
        
        return features
    
    def detect_voice_activity(self, audio_data: bytes) -> bool:
        """检测语音活动"""
        import random
        # 模拟语音活动检测
        return random.random() > 0.3
    
    def extract_emotion_features(self, audio_data: bytes) -> Dict[str, float]:
        """提取情感特征"""
        import random
        return {
            "valence": random.random() * 2 - 1,  # -1 to 1
            "arousal": random.random() * 2 - 1,   # -1 to 1
            "dominance": random.random() * 2 - 1, # -1 to 1
            "confidence": random.random()
        }

class InterruptDetector:
    """打断检测器"""
    
    def __init__(self):
        self.feature_extractor = VoiceFeatureExtractor()
        self.interrupt_thresholds = {
            "voice_energy": 0.7,
            "voice_confidence": 0.8,
            "gesture_confidence": 0.9,
            "text_urgency": 0.8
        }
        self.interrupt_history = deque(maxlen=100)
        self.detection_stats = defaultdict(int)
        
    async def detect_interrupt(self, audio_data: bytes, 
                             text_input: Optional[str] = None,
                             gesture_data: Optional[Dict[str, Any]] = None) -> Optional[InterruptSignal]:
        """检测打断信号"""
        try:
            # 提取语音特征
            audio_features = self.feature_extractor.extract_features(audio_data)
            
            # 检测语音活动
            voice_activity = self.feature_extractor.detect_voice_activity(audio_data)
            
            # 分析打断可能性
            interrupt_confidence = await self._analyze_interrupt_confidence(
                audio_features, text_input, gesture_data, voice_activity
            )
            
            if interrupt_confidence > 0.6:  # 打断阈值
                interrupt_signal = InterruptSignal(
                    interrupt_id=str(uuid.uuid4()),
                    interrupt_type=self._determine_interrupt_type(
                        audio_features, text_input, gesture_data
                    ),
                    confidence=interrupt_confidence,
                    timestamp=time.time(),
                    audio_features=audio_features,
                    text_content=text_input,
                    gesture_data=gesture_data
                )
                
                self.interrupt_history.append(interrupt_signal)
                self.detection_stats["total_interrupts"] += 1
                
                logger.info(f"Interrupt detected: {interrupt_signal.interrupt_type.value} (confidence: {interrupt_confidence:.2f})")
                return interrupt_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Interrupt detection error: {e}")
            return None
    
    async def _analyze_interrupt_confidence(self, audio_features: Dict[str, Any],
                                         text_input: Optional[str],
                                         gesture_data: Optional[Dict[str, Any]],
                                         voice_activity: bool) -> float:
        """分析打断置信度"""
        confidence_scores = []
        
        # 语音特征分析
        if voice_activity:
            energy_score = min(audio_features["energy"] / self.interrupt_thresholds["voice_energy"], 1.0)
            confidence_scores.append(energy_score)
            
            # 情感特征分析
            emotion_features = self.feature_extractor.extract_emotion_features(b"fake_audio")
            if emotion_features["arousal"] > 0.5:  # 高唤醒度
                confidence_scores.append(0.8)
        
        # 文本分析
        if text_input:
            urgency_score = self._analyze_text_urgency(text_input)
            confidence_scores.append(urgency_score)
        
        # 手势分析
        if gesture_data:
            gesture_score = self._analyze_gesture_urgency(gesture_data)
            confidence_scores.append(gesture_score)
        
        # 综合置信度
        if confidence_scores:
            return max(confidence_scores)
        else:
            return 0.0
    
    def _determine_interrupt_type(self, audio_features: Dict[str, Any],
                                text_input: Optional[str],
                                gesture_data: Optional[Dict[str, Any]]) -> InterruptType:
        """确定打断类型"""
        if text_input and "紧急" in text_input:
            return InterruptType.EMERGENCY_INTERRUPT
        elif gesture_data and gesture_data.get("urgent", False):
            return InterruptType.GESTURE_INTERRUPT
        elif text_input:
            return InterruptType.TEXT_INTERRUPT
        elif audio_features.get("voice_activity", False):
            return InterruptType.VOICE_INTERRUPT
        else:
            return InterruptType.SILENCE_INTERRUPT
    
    def _analyze_text_urgency(self, text: str) -> float:
        """分析文本紧急程度"""
        urgent_keywords = ["停止", "暂停", "打断", "紧急", "立即", "马上"]
        urgent_count = sum(1 for keyword in urgent_keywords if keyword in text)
        return min(urgent_count * 0.3, 1.0)
    
    def _analyze_gesture_urgency(self, gesture_data: Dict[str, Any]) -> float:
        """分析手势紧急程度"""
        if gesture_data.get("urgent", False):
            return 0.9
        elif gesture_data.get("stop", False):
            return 0.7
        else:
            return 0.3

class ContextManager:
    """上下文管理器"""
    
    def __init__(self):
        self.conversation_contexts = {}
        self.context_vectors = {}
        self.max_context_length = 50
        
    def create_context(self, session_id: str, user_id: str) -> ConversationContext:
        """创建对话上下文"""
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            current_state=ConversationState.LISTENING,
            conversation_history=[],
            pending_requests=[],
            interrupt_history=[],
            context_vectors={}
        )
        
        self.conversation_contexts[session_id] = context
        logger.info(f"Created context for session: {session_id}")
        return context
    
    def update_context(self, session_id: str, update_data: Dict[str, Any]) -> bool:
        """更新对话上下文"""
        if session_id not in self.conversation_contexts:
            return False
        
        context = self.conversation_contexts[session_id]
        context.last_activity = time.time()
        
        # 更新对话历史
        if "message" in update_data:
            context.conversation_history.append({
                "timestamp": time.time(),
                "content": update_data["message"],
                "type": update_data.get("type", "user")
            })
            
            # 限制历史长度
            if len(context.conversation_history) > self.max_context_length:
                context.conversation_history = context.conversation_history[-self.max_context_length:]
        
        # 更新状态
        if "state" in update_data:
            context.current_state = ConversationState(update_data["state"])
        
        # 更新待处理请求
        if "pending_request" in update_data:
            context.pending_requests.append(update_data["pending_request"])
        
        return True
    
    def preserve_context(self, session_id: str) -> Dict[str, Any]:
        """保存上下文"""
        if session_id not in self.conversation_contexts:
            return {}
        
        context = self.conversation_contexts[session_id]
        
        return {
            "session_id": session_id,
            "user_id": context.user_id,
            "conversation_history": context.conversation_history[-10:],  # 最近10条
            "pending_requests": context.pending_requests,
            "context_vectors": context.context_vectors,
            "last_activity": context.last_activity
        }
    
    def restore_context(self, session_id: str, preserved_context: Dict[str, Any]) -> bool:
        """恢复上下文"""
        try:
            context = ConversationContext(
                session_id=session_id,
                user_id=preserved_context["user_id"],
                current_state=ConversationState.LISTENING,
                conversation_history=preserved_context["conversation_history"],
                pending_requests=preserved_context["pending_requests"],
                interrupt_history=[],
                context_vectors=preserved_context["context_vectors"]
            )
            
            self.conversation_contexts[session_id] = context
            logger.info(f"Restored context for session: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Context restoration failed: {e}")
            return False
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """获取上下文摘要"""
        if session_id not in self.conversation_contexts:
            return {}
        
        context = self.conversation_contexts[session_id]
        
        return {
            "session_id": session_id,
            "user_id": context.user_id,
            "current_state": context.current_state.value,
            "history_length": len(context.conversation_history),
            "pending_requests_count": len(context.pending_requests),
            "interrupt_count": len(context.interrupt_history),
            "last_activity": context.last_activity,
            "session_duration": time.time() - context.created_at
        }

class MultiTurnDialogManager:
    """多轮对话管理器"""
    
    def __init__(self):
        self.dialog_states = {}
        self.intent_history = defaultdict(list)
        self.entity_tracking = defaultdict(dict)
        self.dialog_flows = {}
        
    async def process_turn(self, session_id: str, user_input: str, 
                         context: ConversationContext) -> Dict[str, Any]:
        """处理对话轮次"""
        try:
            # 意图识别
            intent = await self._recognize_intent(user_input, context)
            
            # 实体提取
            entities = await self._extract_entities(user_input)
            
            # 对话状态更新
            dialog_state = await self._update_dialog_state(session_id, intent, entities, context)
            
            # 生成响应
            response = await self._generate_response(intent, entities, dialog_state, context)
            
            # 更新历史
            self.intent_history[session_id].append(intent)
            self.entity_tracking[session_id].update(entities)
            
            return {
                "intent": intent,
                "entities": entities,
                "dialog_state": dialog_state,
                "response": response,
                "confidence": dialog_state.get("confidence", 0.8)
            }
            
        except Exception as e:
            logger.error(f"Dialog processing error: {e}")
            return {
                "intent": "unknown",
                "entities": {},
                "dialog_state": {"state": "error"},
                "response": "抱歉，我遇到了一些问题，请重试。",
                "confidence": 0.0
            }
    
    async def _recognize_intent(self, user_input: str, context: ConversationContext) -> str:
        """识别用户意图"""
        # 简化的意图识别
        intent_keywords = {
            "greeting": ["你好", "hi", "hello", "早上好", "晚上好"],
            "question": ["什么", "怎么", "为什么", "如何", "？", "?"],
            "request": ["请", "帮我", "需要", "想要"],
            "interrupt": ["停止", "暂停", "打断", "等一下"],
            "confirmation": ["是的", "对", "好的", "确认"],
            "negation": ["不是", "不对", "错误", "取消"]
        }
        
        user_input_lower = user_input.lower()
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return intent
        
        return "general"
    
    async def _extract_entities(self, user_input: str) -> Dict[str, Any]:
        """提取实体"""
        entities = {}
        
        # 简化的实体提取
        if "时间" in user_input:
            entities["time"] = "时间相关"
        if "地点" in user_input:
            entities["location"] = "地点相关"
        if "人名" in user_input:
            entities["person"] = "人名相关"
        
        return entities
    
    async def _update_dialog_state(self, session_id: str, intent: str, 
                                entities: Dict[str, Any], context: ConversationContext) -> Dict[str, Any]:
        """更新对话状态"""
        if session_id not in self.dialog_states:
            self.dialog_states[session_id] = {
                "current_intent": intent,
                "entity_slots": {},
                "turn_count": 0,
                "confidence": 0.8
            }
        
        dialog_state = self.dialog_states[session_id]
        dialog_state["turn_count"] += 1
        dialog_state["current_intent"] = intent
        dialog_state["entity_slots"].update(entities)
        
        return dialog_state
    
    async def _generate_response(self, intent: str, entities: Dict[str, Any], 
                               dialog_state: Dict[str, Any], context: ConversationContext) -> str:
        """生成响应"""
        responses = {
            "greeting": "你好！有什么可以帮助您的吗？",
            "question": "这是一个很好的问题，让我为您详细解答。",
            "request": "好的，我来帮您处理这个请求。",
            "interrupt": "好的，我暂停了。请告诉我您需要什么。",
            "confirmation": "明白了，我会按照您的要求执行。",
            "negation": "好的，我取消了之前的操作。",
            "general": "我理解了，请继续。"
        }
        
        return responses.get(intent, "我明白了，请继续。")

class RealtimeInterruptSystem:
    """实时打断系统"""
    
    def __init__(self):
        self.interrupt_detector = InterruptDetector()
        self.context_manager = ContextManager()
        self.dialog_manager = MultiTurnDialogManager()
        self.active_sessions = {}
        self.interrupt_handlers = {}
        
    async def start_session(self, session_id: str, user_id: str) -> ConversationContext:
        """开始会话"""
        context = self.context_manager.create_context(session_id, user_id)
        self.active_sessions[session_id] = {
            "context": context,
            "start_time": time.time(),
            "interrupt_count": 0
        }
        
        logger.info(f"Started session: {session_id} for user: {user_id}")
        return context
    
    async def process_input(self, session_id: str, audio_data: bytes = None,
                          text_input: str = None, gesture_data: Dict[str, Any] = None) -> InterruptResponse:
        """处理输入并检测打断"""
        if session_id not in self.active_sessions:
            return InterruptResponse(
                should_interrupt=False,
                interrupt_confidence=0.0,
                response_action="no_action",
                context_preservation=False,
                recovery_strategy="none",
                estimated_recovery_time=0.0
            )
        
        session = self.active_sessions[session_id]
        context = session["context"]
        
        # 检测打断信号
        interrupt_signal = await self.interrupt_detector.detect_interrupt(
            audio_data or b"", text_input, gesture_data
        )
        
        if interrupt_signal:
            # 处理打断
            response = await self._handle_interrupt(session_id, interrupt_signal, context)
            session["interrupt_count"] += 1
            return response
        else:
            # 正常处理
            if text_input:
                dialog_result = await self.dialog_manager.process_turn(
                    session_id, text_input, context
                )
                self.context_manager.update_context(session_id, {
                    "message": text_input,
                    "type": "user",
                    "dialog_result": dialog_result
                })
            
            return InterruptResponse(
                should_interrupt=False,
                interrupt_confidence=0.0,
                response_action="continue",
                context_preservation=True,
                recovery_strategy="none",
                estimated_recovery_time=0.0
            )
    
    async def _handle_interrupt(self, session_id: str, interrupt_signal: InterruptSignal,
                              context: ConversationContext) -> InterruptResponse:
        """处理打断"""
        # 保存上下文
        preserved_context = self.context_manager.preserve_context(session_id)
        
        # 根据打断类型确定响应策略
        if interrupt_signal.interrupt_type == InterruptType.EMERGENCY_INTERRUPT:
            return InterruptResponse(
                should_interrupt=True,
                interrupt_confidence=interrupt_signal.confidence,
                response_action="immediate_stop",
                context_preservation=True,
                recovery_strategy="quick_recovery",
                estimated_recovery_time=0.5
            )
        elif interrupt_signal.interrupt_type == InterruptType.VOICE_INTERRUPT:
            return InterruptResponse(
                should_interrupt=True,
                interrupt_confidence=interrupt_signal.confidence,
                response_action="pause_and_listen",
                context_preservation=True,
                recovery_strategy="context_aware_recovery",
                estimated_recovery_time=1.0
            )
        else:
            return InterruptResponse(
                should_interrupt=True,
                interrupt_confidence=interrupt_signal.confidence,
                response_action="graceful_pause",
                context_preservation=True,
                recovery_strategy="smooth_recovery",
                estimated_recovery_time=0.8
            )
    
    async def recover_from_interrupt(self, session_id: str) -> bool:
        """从打断中恢复"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        context = session["context"]
        
        # 恢复上下文
        preserved_context = self.context_manager.preserve_context(session_id)
        success = self.context_manager.restore_context(session_id, preserved_context)
        
        if success:
            context.current_state = ConversationState.LISTENING
            logger.info(f"Recovered from interrupt for session: {session_id}")
        
        return success
    
    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """获取会话统计"""
        if session_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[session_id]
        context = session["context"]
        
        return {
            "session_id": session_id,
            "user_id": context.user_id,
            "session_duration": time.time() - session["start_time"],
            "interrupt_count": session["interrupt_count"],
            "current_state": context.current_state.value,
            "context_summary": self.context_manager.get_context_summary(session_id)
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            "active_sessions": len(self.active_sessions),
            "total_interrupts": self.interrupt_detector.detection_stats["total_interrupts"],
            "detection_stats": dict(self.interrupt_detector.detection_stats),
            "context_manager_stats": {
                "total_contexts": len(self.context_manager.conversation_contexts),
                "dialog_states": len(self.dialog_manager.dialog_states)
            }
        }

# 全局实时打断系统实例
realtime_interrupt_system = RealtimeInterruptSystem()

async def start_interrupt_session(session_id: str, user_id: str) -> ConversationContext:
    """开始打断会话"""
    return await realtime_interrupt_system.start_session(session_id, user_id)

async def process_interrupt_input(session_id: str, audio_data: bytes = None,
                                text_input: str = None, gesture_data: Dict[str, Any] = None) -> InterruptResponse:
    """处理打断输入"""
    return await realtime_interrupt_system.process_input(session_id, audio_data, text_input, gesture_data)

async def recover_from_interrupt(session_id: str) -> bool:
    """从打断中恢复"""
    return await realtime_interrupt_system.recover_from_interrupt(session_id)

def get_interrupt_system_stats() -> Dict[str, Any]:
    """获取打断系统统计"""
    return realtime_interrupt_system.get_system_stats()

if __name__ == "__main__":
    # 测试代码
    async def test_interrupt_system():
        # 开始会话
        session_id = "test_session_001"
        user_id = "test_user_001"
        context = await start_interrupt_session(session_id, user_id)
        print(f"Started session: {session_id}")
        
        # 模拟打断
        audio_data = b"fake_audio_data"
        text_input = "停止，我需要打断一下"
        
        response = await process_interrupt_input(session_id, audio_data, text_input)
        print(f"Interrupt response: {response.should_interrupt}")
        
        # 恢复
        recovered = await recover_from_interrupt(session_id)
        print(f"Recovered: {recovered}")
        
        # 获取统计
        stats = get_interrupt_system_stats()
        print("System stats:", stats)
    
    asyncio.run(test_interrupt_system())
