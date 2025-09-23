import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class DialogueState:
    """对话状态"""
    user_id: str
    session_id: str
    current_intent: Optional[str] = None
    entities: Dict[str, Any] = None
    context: Dict[str, Any] = None
    history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.entities is None:
            self.entities = {}
        if self.context is None:
            self.context = {}
        if self.history is None:
            self.history = []

@dataclass
class DialogueResponse:
    """对话响应"""
    text: str
    intent: str
    confidence: float
    entities: Dict[str, Any]
    actions: List[str]
    context_updates: Dict[str, Any]

class RasaDialogueManager:
    """基于Rasa的对话管理器"""
    
    def __init__(self, rasa_server_url: str = "http://localhost:5005"):
        self.rasa_server_url = rasa_server_url
        self.session = None
        self.dialogue_states: Dict[str, DialogueState] = {}
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def process_message(
        self, 
        user_id: str, 
        session_id: str, 
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DialogueResponse:
        """处理用户消息"""
        try:
            # 获取或创建对话状态
            state_key = f"{user_id}:{session_id}"
            if state_key not in self.dialogue_states:
                self.dialogue_states[state_key] = DialogueState(
                    user_id=user_id,
                    session_id=session_id
                )
            
            dialogue_state = self.dialogue_states[state_key]
            
            # 调用Rasa进行意图识别和实体抽取
            nlu_result = await self._call_rasa_nlu(message)
            
            # 调用Rasa Core进行对话管理
            core_result = await self._call_rasa_core(
                user_id, session_id, message, nlu_result, metadata
            )
            
            # 更新对话状态
            dialogue_state.current_intent = nlu_result.get("intent", {}).get("name")
            dialogue_state.entities.update(
                {entity["entity"]: entity["value"] for entity in nlu_result.get("entities", [])}
            )
            dialogue_state.history.append({
                "user_message": message,
                "bot_response": core_result.get("text", ""),
                "intent": dialogue_state.current_intent,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # 构建响应
            response = DialogueResponse(
                text=core_result.get("text", ""),
                intent=dialogue_state.current_intent or "unknown",
                confidence=nlu_result.get("intent", {}).get("confidence", 0.0),
                entities=dialogue_state.entities.copy(),
                actions=[action.get("action") for action in core_result.get("actions", [])],
                context_updates={}
            )
            
            return response
            
        except Exception as e:
            logger.error(f"对话处理错误: {e}")
            return DialogueResponse(
                text="抱歉，我遇到了一些问题，请稍后再试。",
                intent="error",
                confidence=0.0,
                entities={},
                actions=[],
                context_updates={}
            )
    
    async def _call_rasa_nlu(self, message: str) -> Dict[str, Any]:
        """调用Rasa NLU进行意图识别和实体抽取"""
        try:
            async with self.session.post(
                f"{self.rasa_server_url}/model/parse",
                json={"text": message}
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Rasa NLU调用失败: {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Rasa NLU调用异常: {e}")
            return {}
    
    async def _call_rasa_core(
        self, 
        user_id: str, 
        session_id: str, 
        message: str, 
        nlu_result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """调用Rasa Core进行对话管理"""
        try:
            payload = {
                "sender": f"{user_id}:{session_id}",
                "message": message,
                "metadata": metadata or {}
            }
            
            async with self.session.post(
                f"{self.rasa_server_url}/webhooks/rest/webhook",
                json=payload
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    if results:
                        return results[0]  # 返回第一个响应
                    else:
                        return {"text": "我不太明白您的意思，能再说一遍吗？"}
                else:
                    logger.error(f"Rasa Core调用失败: {response.status}")
                    return {"text": "服务暂时不可用，请稍后再试。"}
        except Exception as e:
            logger.error(f"Rasa Core调用异常: {e}")
            return {"text": "服务出现异常，请稍后再试。"}
    
    async def get_dialogue_state(self, user_id: str, session_id: str) -> Optional[DialogueState]:
        """获取对话状态"""
        state_key = f"{user_id}:{session_id}"
        return self.dialogue_states.get(state_key)
    
    async def clear_dialogue_state(self, user_id: str, session_id: str):
        """清除对话状态"""
        state_key = f"{user_id}:{session_id}"
        if state_key in self.dialogue_states:
            del self.dialogue_states[state_key]

# Rasa配置文件示例
RASA_CONFIG_YML = """
# config.yml
language: zh

pipeline:
  - name: JiebaTokenizer
  - name: RegexFeaturizer
  - name: LexicalSyntacticFeaturizer
  - name: CountVectorsFeaturizer
  - name: CountVectorsFeaturizer
    analyzer: char_wb
    min_ngram: 1
    max_ngram: 4
  - name: DIETClassifier
    epochs: 100
    constrain_similarities: true
  - name: EntitySynonymMapper
  - name: ResponseSelector
    epochs: 100
    constrain_similarities: true
  - name: FallbackClassifier
    threshold: 0.3
    ambiguity_threshold: 0.1

policies:
  - name: MemoizationPolicy
  - name: RulePolicy
  - name: UnexpecTEDIntentPolicy
    max_history: 5
    epochs: 100
  - name: TEDPolicy
    max_history: 5
    epochs: 100
    constrain_similarities: true
"""

RASA_DOMAIN_YML = """
# domain.yml
version: "3.1"

intents:
  - greet
  - goodbye
  - affirm
  - deny
  - mood_great
  - mood_unhappy
  - bot_challenge
  - ask_weather
  - ask_time
  - play_music
  - set_reminder

entities:
  - city
  - time
  - music_genre
  - reminder_content

slots:
  city:
    type: text
    mappings:
    - type: from_entity
      entity: city
  
  music_genre:
    type: text
    mappings:
    - type: from_entity
      entity: music_genre

responses:
  utter_greet:
  - text: "你好！我是VoiceHelper，很高兴为您服务！"
  
  utter_cheer_up:
  - text: "希望我能让您开心起来！"
  
  utter_did_that_help:
  - text: "这样有帮助吗？"
  
  utter_happy:
  - text: "太好了！"
  
  utter_goodbye:
  - text: "再见！祝您有美好的一天！"
  
  utter_iamabot:
  - text: "我是VoiceHelper AI助手，由人工智能驱动。"

actions:
  - action_weather_query
  - action_play_music
  - action_set_reminder

session_config:
  session_expiration_time: 60
  carry_over_slots_to_new_session: true
"""

# 使用示例
async def main():
    async with RasaDialogueManager("http://localhost:5005") as dialogue_manager:
        response = await dialogue_manager.process_message(
            user_id="user123",
            session_id="session456",
            message="你好，今天天气怎么样？"
        )
        
        print(f"Bot回复: {response.text}")
        print(f"识别意图: {response.intent}")
        print(f"置信度: {response.confidence}")
        print(f"实体: {response.entities}")

if __name__ == "__main__":
    asyncio.run(main())
