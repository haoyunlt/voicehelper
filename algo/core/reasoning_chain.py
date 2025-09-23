"""
推理链路可视化和管理模块
支持推理步骤追踪、可视化和缓存
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from loguru import logger


class ReasoningStepType(Enum):
    """推理步骤类型"""
    PLANNING = "planning"           # 规划
    RETRIEVAL = "retrieval"         # 检索
    TOOL_CALL = "tool_call"        # 工具调用
    SYNTHESIS = "synthesis"         # 综合
    VALIDATION = "validation"       # 验证
    REFLECTION = "reflection"       # 反思
    DECISION = "decision"          # 决策


class ReasoningStepStatus(Enum):
    """推理步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: str = field(default_factory=lambda: str(uuid4()))
    step_type: ReasoningStepType = ReasoningStepType.SYNTHESIS
    status: ReasoningStepStatus = ReasoningStepStatus.PENDING
    title: str = ""
    description: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    reasoning_text: str = ""
    confidence_score: float = 0.0
    execution_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_step_id: Optional[str] = None
    child_step_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def start(self):
        """开始执行步骤"""
        self.status = ReasoningStepStatus.RUNNING
        self.started_at = datetime.now()

    def complete(self, output_data: Dict[str, Any] = None, reasoning_text: str = "", confidence_score: float = 0.0):
        """完成步骤"""
        self.status = ReasoningStepStatus.COMPLETED
        self.completed_at = datetime.now()
        if output_data:
            self.output_data = output_data
        if reasoning_text:
            self.reasoning_text = reasoning_text
        if confidence_score > 0:
            self.confidence_score = confidence_score
        
        # 计算执行时间
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()

    def fail(self, error_message: str):
        """标记步骤失败"""
        self.status = ReasoningStepStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
        
        if self.started_at:
            self.execution_time = (self.completed_at - self.started_at).total_seconds()

    def skip(self, reason: str = ""):
        """跳过步骤"""
        self.status = ReasoningStepStatus.SKIPPED
        self.completed_at = datetime.now()
        self.metadata["skip_reason"] = reason

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "step_id": self.step_id,
            "step_type": self.step_type.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "reasoning_text": self.reasoning_text,
            "confidence_score": self.confidence_score,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "parent_step_id": self.parent_step_id,
            "child_step_ids": self.child_step_ids,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class ReasoningChain:
    """推理链路"""
    chain_id: str = field(default_factory=lambda: str(uuid4()))
    conversation_id: str = ""
    user_query: str = ""
    final_answer: str = ""
    steps: List[ReasoningStep] = field(default_factory=list)
    total_execution_time: float = 0.0
    overall_confidence: float = 0.0
    success: bool = False
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    def add_step(self, step: ReasoningStep) -> str:
        """添加推理步骤"""
        self.steps.append(step)
        logger.debug(f"推理链路 {self.chain_id} 添加步骤: {step.step_type.value} - {step.title}")
        return step.step_id

    def get_step(self, step_id: str) -> Optional[ReasoningStep]:
        """获取推理步骤"""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None

    def get_steps_by_type(self, step_type: ReasoningStepType) -> List[ReasoningStep]:
        """根据类型获取步骤"""
        return [step for step in self.steps if step.step_type == step_type]

    def get_active_steps(self) -> List[ReasoningStep]:
        """获取活跃步骤"""
        return [step for step in self.steps if step.status == ReasoningStepStatus.RUNNING]

    def get_completed_steps(self) -> List[ReasoningStep]:
        """获取已完成步骤"""
        return [step for step in self.steps if step.status == ReasoningStepStatus.COMPLETED]

    def get_failed_steps(self) -> List[ReasoningStep]:
        """获取失败步骤"""
        return [step for step in self.steps if step.status == ReasoningStepStatus.FAILED]

    def complete(self, final_answer: str = "", success: bool = True, error_message: str = ""):
        """完成推理链路"""
        self.completed_at = datetime.now()
        self.final_answer = final_answer
        self.success = success
        self.error_message = error_message
        
        # 计算总执行时间
        self.total_execution_time = (self.completed_at - self.created_at).total_seconds()
        
        # 计算整体置信度
        completed_steps = self.get_completed_steps()
        if completed_steps:
            self.overall_confidence = sum(step.confidence_score for step in completed_steps) / len(completed_steps)

        logger.info(f"推理链路 {self.chain_id} 完成: 成功={success}, 总时间={self.total_execution_time:.2f}s, 置信度={self.overall_confidence:.2f}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "chain_id": self.chain_id,
            "conversation_id": self.conversation_id,
            "user_query": self.user_query,
            "final_answer": self.final_answer,
            "steps": [step.to_dict() for step in self.steps],
            "total_execution_time": self.total_execution_time,
            "overall_confidence": self.overall_confidence,
            "success": self.success,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class ReasoningChainManager:
    """推理链路管理器"""
    
    def __init__(self, enable_caching: bool = True, cache_ttl: int = 3600):
        self.chains: Dict[str, ReasoningChain] = {}
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.step_cache: Dict[str, Dict[str, Any]] = {}
        
    def create_chain(self, conversation_id: str, user_query: str) -> ReasoningChain:
        """创建推理链路"""
        chain = ReasoningChain(
            conversation_id=conversation_id,
            user_query=user_query
        )
        self.chains[chain.chain_id] = chain
        
        logger.info(f"创建推理链路: {chain.chain_id} for conversation: {conversation_id}")
        return chain
    
    def get_chain(self, chain_id: str) -> Optional[ReasoningChain]:
        """获取推理链路"""
        return self.chains.get(chain_id)
    
    def create_step(
        self,
        chain_id: str,
        step_type: ReasoningStepType,
        title: str,
        description: str = "",
        input_data: Dict[str, Any] = None,
        parent_step_id: str = None
    ) -> Optional[ReasoningStep]:
        """创建推理步骤"""
        chain = self.get_chain(chain_id)
        if not chain:
            logger.error(f"推理链路不存在: {chain_id}")
            return None
        
        step = ReasoningStep(
            step_type=step_type,
            title=title,
            description=description,
            input_data=input_data or {},
            parent_step_id=parent_step_id
        )
        
        chain.add_step(step)
        
        # 更新父步骤的子步骤列表
        if parent_step_id:
            parent_step = chain.get_step(parent_step_id)
            if parent_step:
                parent_step.child_step_ids.append(step.step_id)
        
        return step
    
    def start_step(self, chain_id: str, step_id: str) -> bool:
        """开始执行步骤"""
        chain = self.get_chain(chain_id)
        if not chain:
            return False
        
        step = chain.get_step(step_id)
        if not step:
            return False
        
        step.start()
        logger.debug(f"开始执行步骤: {step_id} - {step.title}")
        return True
    
    def complete_step(
        self,
        chain_id: str,
        step_id: str,
        output_data: Dict[str, Any] = None,
        reasoning_text: str = "",
        confidence_score: float = 0.0
    ) -> bool:
        """完成步骤"""
        chain = self.get_chain(chain_id)
        if not chain:
            return False
        
        step = chain.get_step(step_id)
        if not step:
            return False
        
        step.complete(output_data, reasoning_text, confidence_score)
        
        # 缓存步骤结果
        if self.enable_caching:
            cache_key = self._generate_cache_key(step.step_type, step.input_data)
            self.step_cache[cache_key] = {
                "output_data": output_data or {},
                "reasoning_text": reasoning_text,
                "confidence_score": confidence_score,
                "cached_at": time.time()
            }
        
        logger.debug(f"完成步骤: {step_id} - {step.title}, 置信度: {confidence_score}")
        return True
    
    def fail_step(self, chain_id: str, step_id: str, error_message: str) -> bool:
        """标记步骤失败"""
        chain = self.get_chain(chain_id)
        if not chain:
            return False
        
        step = chain.get_step(step_id)
        if not step:
            return False
        
        step.fail(error_message)
        logger.error(f"步骤失败: {step_id} - {step.title}, 错误: {error_message}")
        return True
    
    def get_cached_result(self, step_type: ReasoningStepType, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取缓存结果"""
        if not self.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(step_type, input_data)
        cached_result = self.step_cache.get(cache_key)
        
        if cached_result:
            # 检查缓存是否过期
            if time.time() - cached_result["cached_at"] > self.cache_ttl:
                del self.step_cache[cache_key]
                return None
            
            logger.debug(f"使用缓存结果: {step_type.value}")
            return cached_result
        
        return None
    
    def _generate_cache_key(self, step_type: ReasoningStepType, input_data: Dict[str, Any]) -> str:
        """生成缓存键"""
        import hashlib
        
        # 创建输入数据的哈希
        input_str = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
        input_hash = hashlib.md5(input_str.encode()).hexdigest()
        
        return f"{step_type.value}:{input_hash}"
    
    def get_chain_summary(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """获取推理链路摘要"""
        chain = self.get_chain(chain_id)
        if not chain:
            return None
        
        completed_steps = chain.get_completed_steps()
        failed_steps = chain.get_failed_steps()
        active_steps = chain.get_active_steps()
        
        return {
            "chain_id": chain_id,
            "conversation_id": chain.conversation_id,
            "user_query": chain.user_query,
            "total_steps": len(chain.steps),
            "completed_steps": len(completed_steps),
            "failed_steps": len(failed_steps),
            "active_steps": len(active_steps),
            "total_execution_time": chain.total_execution_time,
            "overall_confidence": chain.overall_confidence,
            "success": chain.success,
            "created_at": chain.created_at.isoformat(),
            "completed_at": chain.completed_at.isoformat() if chain.completed_at else None,
        }
    
    def export_chain_visualization(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """导出推理链路可视化数据"""
        chain = self.get_chain(chain_id)
        if not chain:
            return None
        
        # 构建节点和边的数据结构，用于前端可视化
        nodes = []
        edges = []
        
        for step in chain.steps:
            # 节点数据
            node = {
                "id": step.step_id,
                "type": step.step_type.value,
                "status": step.status.value,
                "title": step.title,
                "description": step.description,
                "confidence": step.confidence_score,
                "execution_time": step.execution_time,
                "created_at": step.created_at.isoformat(),
            }
            nodes.append(node)
            
            # 边数据（父子关系）
            if step.parent_step_id:
                edge = {
                    "source": step.parent_step_id,
                    "target": step.step_id,
                    "type": "parent_child"
                }
                edges.append(edge)
        
        return {
            "chain_id": chain_id,
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_steps": len(nodes),
                "total_execution_time": chain.total_execution_time,
                "overall_confidence": chain.overall_confidence,
                "success": chain.success
            }
        }
    
    def cleanup_old_chains(self, max_age_hours: int = 24):
        """清理旧的推理链路"""
        current_time = datetime.now()
        chains_to_remove = []
        
        for chain_id, chain in self.chains.items():
            age_hours = (current_time - chain.created_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                chains_to_remove.append(chain_id)
        
        for chain_id in chains_to_remove:
            del self.chains[chain_id]
            logger.info(f"清理旧推理链路: {chain_id}")
        
        # 清理过期缓存
        current_timestamp = time.time()
        cache_keys_to_remove = []
        
        for cache_key, cached_data in self.step_cache.items():
            if current_timestamp - cached_data["cached_at"] > self.cache_ttl:
                cache_keys_to_remove.append(cache_key)
        
        for cache_key in cache_keys_to_remove:
            del self.step_cache[cache_key]
        
        logger.info(f"清理完成: 删除 {len(chains_to_remove)} 个推理链路, {len(cache_keys_to_remove)} 个缓存项")


# 全局推理链路管理器实例
reasoning_chain_manager = ReasoningChainManager()
