"""
上下文记忆增强系统
长期记忆和个性化学习系统
基于向量数据库和知识图谱的记忆管理
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import networkx as nx
from collections import defaultdict, deque
import sqlite3
import pickle

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    EPISODIC = "episodic"      # 情节记忆
    SEMANTIC = "semantic"      # 语义记忆
    PROCEDURAL = "procedural"  # 程序记忆
    WORKING = "working"        # 工作记忆

class MemoryImportance(Enum):
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    MINIMAL = 1

@dataclass
class Memory:
    id: str
    user_id: str
    session_id: str
    memory_type: MemoryType
    content: str
    embedding: Optional[np.ndarray]
    importance: MemoryImportance
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    decay_factor: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class PersonalityProfile:
    user_id: str
    preferences: Dict[str, float]
    communication_style: Dict[str, float]
    topics_of_interest: List[str]
    emotional_patterns: Dict[str, float]
    learning_style: str
    context_preferences: Dict[str, Any]
    updated_at: float

class ContextMemorySystem:
    """上下文记忆系统"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", 
                 vector_dim: int = 384, max_memories: int = 10000):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_dim = vector_dim
        self.max_memories = max_memories
        
        # 向量索引
        self.faiss_index = faiss.IndexFlatIP(vector_dim)  # 内积相似度
        self.memory_store: Dict[str, Memory] = {}
        self.user_memories: Dict[str, List[str]] = defaultdict(list)
        
        # 知识图谱
        self.knowledge_graph = nx.DiGraph()
        
        # 个性化配置
        self.personality_profiles: Dict[str, PersonalityProfile] = {}
        
        # 工作记忆
        self.working_memory: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        
        # 记忆衰减参数
        self.decay_rate = 0.1
        self.access_boost = 0.1
        
    async def initialize(self):
        """初始化记忆系统"""
        # 加载持久化数据
        await self._load_persistent_data()
        
        # 启动记忆维护任务
        asyncio.create_task(self._memory_maintenance_loop())
        
        logger.info("Context memory system initialized")
    
    async def store_memory(self, user_id: str, session_id: str, content: str,
                          memory_type: MemoryType = MemoryType.EPISODIC,
                          importance: MemoryImportance = MemoryImportance.MEDIUM,
                          metadata: Dict[str, Any] = None) -> str:
        """存储记忆"""
        
        # 生成嵌入向量
        embedding = self.embedding_model.encode(content)
        
        # 创建记忆对象
        memory_id = f"mem_{int(time.time() * 1000)}_{user_id}"
        memory = Memory(
            id=memory_id,
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=importance,
            timestamp=time.time(),
            metadata=metadata or {}
        )
        
        # 存储到内存
        self.memory_store[memory_id] = memory
        self.user_memories[user_id].append(memory_id)
        
        # 添加到向量索引
        self.faiss_index.add(embedding.reshape(1, -1))
        
        # 更新知识图谱
        await self._update_knowledge_graph(memory)
        
        # 更新个性化配置
        await self._update_personality_profile(user_id, content, metadata)
        
        # 记忆容量管理
        await self._manage_memory_capacity(user_id)
        
        logger.debug(f"Stored memory {memory_id} for user {user_id}")
        return memory_id
    
    async def retrieve_memories(self, user_id: str, query: str, 
                              top_k: int = 5, memory_types: List[MemoryType] = None) -> List[Memory]:
        """检索相关记忆"""
        
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode(query)
        
        # 获取用户记忆
        user_memory_ids = self.user_memories.get(user_id, [])
        if not user_memory_ids:
            return []
        
        # 过滤记忆类型
        if memory_types:
            filtered_ids = []
            for mem_id in user_memory_ids:
                memory = self.memory_store.get(mem_id)
                if memory and memory.memory_type in memory_types:
                    filtered_ids.append(mem_id)
            user_memory_ids = filtered_ids
        
        # 计算相似度
        similarities = []
        for mem_id in user_memory_ids:
            memory = self.memory_store.get(mem_id)
            if memory and memory.embedding is not None:
                similarity = np.dot(query_embedding, memory.embedding)
                # 考虑重要性和时间衰减
                importance_weight = memory.importance.value / 5.0
                time_decay = np.exp(-self.decay_rate * (time.time() - memory.timestamp) / (24 * 3600))
                access_boost = 1 + memory.access_count * self.access_boost
                
                final_score = similarity * importance_weight * time_decay * access_boost
                similarities.append((mem_id, final_score))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for mem_id, score in similarities[:top_k]:
            memory = self.memory_store[mem_id]
            # 更新访问统计
            memory.access_count += 1
            memory.last_accessed = time.time()
            results.append(memory)
        
        return results
    
    async def get_context_for_conversation(self, user_id: str, current_message: str,
                                         max_context_length: int = 2000) -> str:
        """获取对话上下文"""
        
        # 获取工作记忆
        working_mem = list(self.working_memory[user_id])
        
        # 检索相关长期记忆
        relevant_memories = await self.retrieve_memories(
            user_id=user_id,
            query=current_message,
            top_k=5,
            memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC]
        )
        
        # 获取个性化配置
        personality = self.personality_profiles.get(user_id)
        
        # 构建上下文
        context_parts = []
        
        # 添加个性化信息
        if personality:
            context_parts.append(f"用户偏好: {self._format_personality(personality)}")
        
        # 添加相关记忆
        if relevant_memories:
            context_parts.append("相关记忆:")
            for memory in relevant_memories:
                context_parts.append(f"- {memory.content}")
        
        # 添加工作记忆
        if working_mem:
            context_parts.append("近期对话:")
            for item in working_mem[-5:]:  # 最近5条
                context_parts.append(f"- {item}")
        
        # 组合上下文并限制长度
        full_context = "\n".join(context_parts)
        if len(full_context) > max_context_length:
            full_context = full_context[:max_context_length] + "..."
        
        return full_context
    
    async def update_working_memory(self, user_id: str, message: str, is_user: bool = True):
        """更新工作记忆"""
        prefix = "用户" if is_user else "助手"
        formatted_message = f"{prefix}: {message}"
        
        self.working_memory[user_id].append(formatted_message)
    
    async def learn_from_interaction(self, user_id: str, session_id: str, 
                                   user_message: str, bot_response: str,
                                   feedback: Optional[Dict[str, Any]] = None):
        """从交互中学习"""
        
        # 存储交互记忆
        interaction_content = f"用户说: {user_message}\n助手回复: {bot_response}"
        
        # 根据反馈确定重要性
        importance = MemoryImportance.MEDIUM
        if feedback:
            rating = feedback.get("rating", 3)
            if rating >= 4:
                importance = MemoryImportance.HIGH
            elif rating <= 2:
                importance = MemoryImportance.LOW
        
        await self.store_memory(
            user_id=user_id,
            session_id=session_id,
            content=interaction_content,
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            metadata={"feedback": feedback} if feedback else None
        )
        
        # 更新工作记忆
        await self.update_working_memory(user_id, user_message, is_user=True)
        await self.update_working_memory(user_id, bot_response, is_user=False)
        
        # 学习用户偏好
        await self._learn_preferences(user_id, user_message, bot_response, feedback)
    
    async def _update_knowledge_graph(self, memory: Memory):
        """更新知识图谱"""
        # 简化实现：基于关键词构建图谱
        content = memory.content.lower()
        words = content.split()
        
        # 添加节点
        for word in words:
            if len(word) > 3:  # 过滤短词
                self.knowledge_graph.add_node(word, 
                                            user_id=memory.user_id,
                                            memory_id=memory.id,
                                            timestamp=memory.timestamp)
        
        # 添加边（词汇共现关系）
        for i, word1 in enumerate(words):
            for word2 in words[i+1:i+3]:  # 窗口大小为2
                if len(word1) > 3 and len(word2) > 3:
                    if self.knowledge_graph.has_edge(word1, word2):
                        self.knowledge_graph[word1][word2]['weight'] += 1
                    else:
                        self.knowledge_graph.add_edge(word1, word2, weight=1)
    
    async def _update_personality_profile(self, user_id: str, content: str, metadata: Dict[str, Any]):
        """更新个性化配置"""
        if user_id not in self.personality_profiles:
            self.personality_profiles[user_id] = PersonalityProfile(
                user_id=user_id,
                preferences={},
                communication_style={},
                topics_of_interest=[],
                emotional_patterns={},
                learning_style="adaptive",
                context_preferences={},
                updated_at=time.time()
            )
        
        profile = self.personality_profiles[user_id]
        
        # 分析内容特征
        content_lower = content.lower()
        
        # 更新主题兴趣
        topics = self._extract_topics(content)
        for topic in topics:
            if topic not in profile.topics_of_interest:
                profile.topics_of_interest.append(topic)
        
        # 保持列表长度
        profile.topics_of_interest = profile.topics_of_interest[-20:]
        
        # 更新沟通风格
        if "?" in content:
            profile.communication_style["questioning"] = profile.communication_style.get("questioning", 0) + 0.1
        if "!" in content:
            profile.communication_style["enthusiastic"] = profile.communication_style.get("enthusiastic", 0) + 0.1
        if len(content) > 100:
            profile.communication_style["detailed"] = profile.communication_style.get("detailed", 0) + 0.1
        else:
            profile.communication_style["concise"] = profile.communication_style.get("concise", 0) + 0.1
        
        # 归一化权重
        total_weight = sum(profile.communication_style.values())
        if total_weight > 0:
            for key in profile.communication_style:
                profile.communication_style[key] /= total_weight
        
        profile.updated_at = time.time()
    
    def _extract_topics(self, content: str) -> List[str]:
        """提取主题关键词"""
        # 简化实现：基于关键词
        keywords = ["技术", "编程", "AI", "音乐", "电影", "旅行", "美食", "运动", "学习", "工作"]
        content_lower = content.lower()
        
        found_topics = []
        for keyword in keywords:
            if keyword.lower() in content_lower:
                found_topics.append(keyword)
        
        return found_topics
    
    def _format_personality(self, personality: PersonalityProfile) -> str:
        """格式化个性化信息"""
        parts = []
        
        if personality.topics_of_interest:
            parts.append(f"感兴趣的话题: {', '.join(personality.topics_of_interest[-5:])}")
        
        if personality.communication_style:
            top_style = max(personality.communication_style.items(), key=lambda x: x[1])
            parts.append(f"沟通风格: {top_style[0]}")
        
        return "; ".join(parts)
    
    async def _learn_preferences(self, user_id: str, user_message: str, 
                               bot_response: str, feedback: Optional[Dict[str, Any]]):
        """学习用户偏好"""
        if not feedback:
            return
        
        profile = self.personality_profiles.get(user_id)
        if not profile:
            return
        
        rating = feedback.get("rating", 3)
        
        # 基于反馈调整偏好
        if rating >= 4:  # 正面反馈
            # 分析成功因素
            if len(bot_response) > 100:
                profile.preferences["detailed_responses"] = profile.preferences.get("detailed_responses", 0) + 0.1
            else:
                profile.preferences["concise_responses"] = profile.preferences.get("concise_responses", 0) + 0.1
        
        elif rating <= 2:  # 负面反馈
            # 调整策略
            if len(bot_response) > 100:
                profile.preferences["detailed_responses"] = profile.preferences.get("detailed_responses", 0) - 0.1
            else:
                profile.preferences["concise_responses"] = profile.preferences.get("concise_responses", 0) - 0.1
        
        # 限制权重范围
        for key in profile.preferences:
            profile.preferences[key] = max(-1.0, min(1.0, profile.preferences[key]))
    
    async def _manage_memory_capacity(self, user_id: str):
        """管理记忆容量"""
        user_memory_ids = self.user_memories[user_id]
        
        if len(user_memory_ids) > self.max_memories:
            # 按重要性和时间衰减排序
            memories_with_scores = []
            for mem_id in user_memory_ids:
                memory = self.memory_store[mem_id]
                importance_score = memory.importance.value
                time_decay = np.exp(-self.decay_rate * (time.time() - memory.timestamp) / (24 * 3600))
                access_score = 1 + memory.access_count * self.access_boost
                
                total_score = importance_score * time_decay * access_score
                memories_with_scores.append((mem_id, total_score))
            
            # 排序并保留top记忆
            memories_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 删除低分记忆
            to_remove = memories_with_scores[self.max_memories:]
            for mem_id, _ in to_remove:
                if mem_id in self.memory_store:
                    del self.memory_store[mem_id]
                user_memory_ids.remove(mem_id)
    
    async def _memory_maintenance_loop(self):
        """记忆维护循环"""
        while True:
            try:
                # 定期清理和优化记忆
                await self._cleanup_expired_memories()
                await self._optimize_memory_structure()
                
                # 保存持久化数据
                await self._save_persistent_data()
                
                await asyncio.sleep(3600)  # 每小时维护一次
                
            except Exception as e:
                logger.error(f"Memory maintenance error: {e}")
                await asyncio.sleep(300)  # 出错时5分钟后重试
    
    async def _cleanup_expired_memories(self):
        """清理过期记忆"""
        current_time = time.time()
        expired_threshold = 30 * 24 * 3600  # 30天
        
        for user_id, memory_ids in self.user_memories.items():
            to_remove = []
            for mem_id in memory_ids:
                memory = self.memory_store.get(mem_id)
                if memory and (current_time - memory.timestamp) > expired_threshold:
                    if memory.importance.value <= 2:  # 只删除低重要性的记忆
                        to_remove.append(mem_id)
            
            for mem_id in to_remove:
                if mem_id in self.memory_store:
                    del self.memory_store[mem_id]
                memory_ids.remove(mem_id)
    
    async def _optimize_memory_structure(self):
        """优化记忆结构"""
        # 重建FAISS索引以提高效率
        if len(self.memory_store) > 1000:
            embeddings = []
            memory_ids = []
            
            for mem_id, memory in self.memory_store.items():
                if memory.embedding is not None:
                    embeddings.append(memory.embedding)
                    memory_ids.append(mem_id)
            
            if embeddings:
                embeddings_array = np.vstack(embeddings)
                new_index = faiss.IndexFlatIP(self.vector_dim)
                new_index.add(embeddings_array)
                self.faiss_index = new_index
    
    async def _save_persistent_data(self):
        """保存持久化数据"""
        try:
            # 保存记忆数据
            with open("memories.pkl", "wb") as f:
                pickle.dump({
                    "memory_store": self.memory_store,
                    "user_memories": dict(self.user_memories),
                    "personality_profiles": self.personality_profiles
                }, f)
            
            logger.debug("Persistent data saved")
            
        except Exception as e:
            logger.error(f"Failed to save persistent data: {e}")
    
    async def _load_persistent_data(self):
        """加载持久化数据"""
        try:
            with open("memories.pkl", "rb") as f:
                data = pickle.load(f)
                self.memory_store = data.get("memory_store", {})
                self.user_memories = defaultdict(list, data.get("user_memories", {}))
                self.personality_profiles = data.get("personality_profiles", {})
            
            # 重建FAISS索引
            embeddings = []
            for memory in self.memory_store.values():
                if memory.embedding is not None:
                    embeddings.append(memory.embedding)
            
            if embeddings:
                embeddings_array = np.vstack(embeddings)
                self.faiss_index = faiss.IndexFlatIP(self.vector_dim)
                self.faiss_index.add(embeddings_array)
            
            logger.info("Persistent data loaded")
            
        except FileNotFoundError:
            logger.info("No persistent data found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load persistent data: {e}")

# 使用示例
async def create_context_memory_system():
    """创建上下文记忆系统"""
    memory_system = ContextMemorySystem()
    await memory_system.initialize()
    return memory_system

if __name__ == "__main__":
    # 测试代码
    async def test_memory_system():
        memory_system = await create_context_memory_system()
        
        user_id = "test_user_123"
        session_id = "session_456"
        
        # 存储一些记忆
        await memory_system.store_memory(
            user_id=user_id,
            session_id=session_id,
            content="我喜欢编程和人工智能",
            memory_type=MemoryType.SEMANTIC,
            importance=MemoryImportance.HIGH
        )
        
        await memory_system.store_memory(
            user_id=user_id,
            session_id=session_id,
            content="昨天我学习了机器学习算法",
            memory_type=MemoryType.EPISODIC,
            importance=MemoryImportance.MEDIUM
        )
        
        # 模拟对话交互
        await memory_system.learn_from_interaction(
            user_id=user_id,
            session_id=session_id,
            user_message="能告诉我更多关于深度学习的内容吗？",
            bot_response="深度学习是机器学习的一个分支...",
            feedback={"rating": 5, "helpful": True}
        )
        
        # 获取对话上下文
        context = await memory_system.get_context_for_conversation(
            user_id=user_id,
            current_message="我想了解神经网络"
        )
        
        print("对话上下文:")
        print(context)
        
        # 检索相关记忆
        memories = await memory_system.retrieve_memories(
            user_id=user_id,
            query="机器学习",
            top_k=3
        )
        
        print("\n相关记忆:")
        for memory in memories:
            print(f"- {memory.content}")
    
    asyncio.run(test_memory_system())
