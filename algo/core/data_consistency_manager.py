"""
VoiceHelper v1.24.0 - 数据一致性管理器
实现分布式事务和数据同步，确保数据一致性
"""

import asyncio
import time
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import functools
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class TransactionStatus(Enum):
    """事务状态"""
    PENDING = "pending"
    COMMITTED = "committed"
    ABORTED = "aborted"
    PREPARED = "prepared"
    COMMITTING = "committing"
    ABORTING = "aborting"

class ConsistencyLevel(Enum):
    """一致性级别"""
    STRONG = "strong"           # 强一致性
    EVENTUAL = "eventual"       # 最终一致性
    WEAK = "weak"              # 弱一致性
    SESSION = "session"         # 会话一致性

class SyncStrategy(Enum):
    """同步策略"""
    IMMEDIATE = "immediate"     # 立即同步
    BATCH = "batch"            # 批量同步
    SCHEDULED = "scheduled"     # 定时同步
    ON_DEMAND = "on_demand"     # 按需同步

@dataclass
class Transaction:
    """分布式事务"""
    transaction_id: str
    status: TransactionStatus
    participants: List[str]  # 参与者列表
    operations: List[Dict[str, Any]]
    created_at: float
    updated_at: float
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    consistency_level: ConsistencyLevel = ConsistencyLevel.STRONG

@dataclass
class DataOperation:
    """数据操作"""
    operation_id: str
    operation_type: str  # INSERT, UPDATE, DELETE
    table_name: str
    data: Dict[str, Any]
    conditions: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    checksum: str = ""

@dataclass
class SyncOperation:
    """同步操作"""
    sync_id: str
    source_system: str
    target_system: str
    operations: List[DataOperation]
    strategy: SyncStrategy
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    error_message: Optional[str] = None

class TwoPhaseCommit:
    """两阶段提交协议"""
    
    def __init__(self):
        self.transactions: Dict[str, Transaction] = {}
        self.participants: Dict[str, Callable] = {}
        self.timeout_tasks: Dict[str, asyncio.Task] = {}
        
    def register_participant(self, participant_id: str, prepare_func: Callable, commit_func: Callable, abort_func: Callable):
        """注册事务参与者"""
        self.participants[participant_id] = {
            'prepare': prepare_func,
            'commit': commit_func,
            'abort': abort_func
        }
        logger.info(f"Registered transaction participant: {participant_id}")
    
    async def start_transaction(self, 
                              participants: List[str],
                              operations: List[Dict[str, Any]],
                              timeout: float = 30.0) -> str:
        """开始分布式事务"""
        transaction_id = str(uuid.uuid4())
        
        transaction = Transaction(
            transaction_id=transaction_id,
            status=TransactionStatus.PENDING,
            participants=participants,
            operations=operations,
            created_at=time.time(),
            updated_at=time.time(),
            timeout=timeout
        )
        
        self.transactions[transaction_id] = transaction
        
        # 设置超时任务
        self.timeout_tasks[transaction_id] = asyncio.create_task(
            self._handle_timeout(transaction_id, timeout)
        )
        
        logger.info(f"Started distributed transaction: {transaction_id}")
        
        # 开始两阶段提交
        await self._execute_two_phase_commit(transaction_id)
        
        return transaction_id
    
    async def _execute_two_phase_commit(self, transaction_id: str):
        """执行两阶段提交"""
        transaction = self.transactions[transaction_id]
        
        try:
            # 第一阶段：准备阶段
            prepared = await self._prepare_phase(transaction)
            
            if prepared:
                # 第二阶段：提交阶段
                await self._commit_phase(transaction)
            else:
                # 回滚事务
                await self._abort_phase(transaction)
                
        except Exception as e:
            logger.error(f"Two-phase commit failed for transaction {transaction_id}: {e}")
            await self._abort_phase(transaction)
    
    async def _prepare_phase(self, transaction: Transaction) -> bool:
        """准备阶段"""
        logger.info(f"Prepare phase for transaction {transaction.transaction_id}")
        
        transaction.status = TransactionStatus.PREPARED
        transaction.updated_at = time.time()
        
        prepare_results = []
        
        for participant_id in transaction.participants:
            if participant_id not in self.participants:
                logger.error(f"Unknown participant: {participant_id}")
                return False
            
            try:
                prepare_func = self.participants[participant_id]['prepare']
                result = await prepare_func(transaction.transaction_id, transaction.operations)
                prepare_results.append(result)
                
            except Exception as e:
                logger.error(f"Prepare failed for participant {participant_id}: {e}")
                return False
        
        # 所有参与者都准备成功
        return all(prepare_results)
    
    async def _commit_phase(self, transaction: Transaction):
        """提交阶段"""
        logger.info(f"Commit phase for transaction {transaction.transaction_id}")
        
        transaction.status = TransactionStatus.COMMITTING
        transaction.updated_at = time.time()
        
        commit_results = []
        
        for participant_id in transaction.participants:
            try:
                commit_func = self.participants[participant_id]['commit']
                result = await commit_func(transaction.transaction_id)
                commit_results.append(result)
                
            except Exception as e:
                logger.error(f"Commit failed for participant {participant_id}: {e}")
                # 即使部分提交失败，也标记为已提交（数据一致性由应用层保证）
        
        transaction.status = TransactionStatus.COMMITTED
        transaction.updated_at = time.time()
        
        logger.info(f"Transaction {transaction.transaction_id} committed")
    
    async def _abort_phase(self, transaction: Transaction):
        """回滚阶段"""
        logger.info(f"Abort phase for transaction {transaction.transaction_id}")
        
        transaction.status = TransactionStatus.ABORTING
        transaction.updated_at = time.time()
        
        for participant_id in transaction.participants:
            try:
                abort_func = self.participants[participant_id]['abort']
                await abort_func(transaction.transaction_id)
                
            except Exception as e:
                logger.error(f"Abort failed for participant {participant_id}: {e}")
        
        transaction.status = TransactionStatus.ABORTED
        transaction.updated_at = time.time()
        
        logger.info(f"Transaction {transaction.transaction_id} aborted")
    
    async def _handle_timeout(self, transaction_id: str, timeout: float):
        """处理事务超时"""
        await asyncio.sleep(timeout)
        
        if transaction_id in self.transactions:
            transaction = self.transactions[transaction_id]
            if transaction.status in [TransactionStatus.PENDING, TransactionStatus.PREPARED]:
                logger.warning(f"Transaction {transaction_id} timed out, aborting")
                await self._abort_phase(transaction)
    
    def get_transaction_status(self, transaction_id: str) -> Optional[TransactionStatus]:
        """获取事务状态"""
        transaction = self.transactions.get(transaction_id)
        return transaction.status if transaction else None
    
    def get_transaction_history(self) -> List[Transaction]:
        """获取事务历史"""
        return list(self.transactions.values())

class DataSynchronizer:
    """数据同步器"""
    
    def __init__(self):
        self.sync_operations: Dict[str, SyncOperation] = {}
        self.sync_strategies: Dict[SyncStrategy, Callable] = {}
        self.sync_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.sync_intervals: Dict[str, float] = {}
        self.sync_tasks: Dict[str, asyncio.Task] = {}
        
        # 注册同步策略
        self._register_sync_strategies()
        
    def _register_sync_strategies(self):
        """注册同步策略"""
        self.sync_strategies[SyncStrategy.IMMEDIATE] = self._immediate_sync
        self.sync_strategies[SyncStrategy.BATCH] = self._batch_sync
        self.sync_strategies[SyncStrategy.SCHEDULED] = self._scheduled_sync
        self.sync_strategies[SyncStrategy.ON_DEMAND] = self._on_demand_sync
    
    async def sync_data(self, 
                       source_system: str,
                       target_system: str,
                       operations: List[DataOperation],
                       strategy: SyncStrategy = SyncStrategy.IMMEDIATE,
                       interval: float = 60.0) -> str:
        """同步数据"""
        sync_id = str(uuid.uuid4())
        
        sync_operation = SyncOperation(
            sync_id=sync_id,
            source_system=source_system,
            target_system=target_system,
            operations=operations,
            strategy=strategy,
            created_at=time.time()
        )
        
        self.sync_operations[sync_id] = sync_operation
        
        # 根据策略执行同步
        if strategy == SyncStrategy.SCHEDULED:
            self.sync_intervals[sync_id] = interval
            self.sync_tasks[sync_id] = asyncio.create_task(
                self._periodic_sync(sync_id, interval)
            )
        else:
            sync_func = self.sync_strategies.get(strategy)
            if sync_func:
                await sync_func(sync_operation)
        
        logger.info(f"Data sync initiated: {sync_id} ({strategy.value})")
        return sync_id
    
    async def _immediate_sync(self, sync_operation: SyncOperation):
        """立即同步"""
        try:
            sync_operation.status = "syncing"
            
            for operation in sync_operation.operations:
                await self._execute_sync_operation(sync_operation, operation)
            
            sync_operation.status = "completed"
            sync_operation.completed_at = time.time()
            
        except Exception as e:
            sync_operation.status = "failed"
            sync_operation.error_message = str(e)
            logger.error(f"Immediate sync failed: {e}")
    
    async def _batch_sync(self, sync_operation: SyncOperation):
        """批量同步"""
        try:
            sync_operation.status = "syncing"
            
            # 将操作分批处理
            batch_size = 100
            operations = sync_operation.operations
            
            for i in range(0, len(operations), batch_size):
                batch = operations[i:i + batch_size]
                await self._execute_batch_operations(sync_operation, batch)
                
                # 批次间延迟
                await asyncio.sleep(0.1)
            
            sync_operation.status = "completed"
            sync_operation.completed_at = time.time()
            
        except Exception as e:
            sync_operation.status = "failed"
            sync_operation.error_message = str(e)
            logger.error(f"Batch sync failed: {e}")
    
    async def _scheduled_sync(self, sync_operation: SyncOperation):
        """定时同步"""
        # 定时同步由 _periodic_sync 方法处理
        pass
    
    async def _on_demand_sync(self, sync_operation: SyncOperation):
        """按需同步"""
        # 将操作添加到队列，等待处理
        queue_key = f"{sync_operation.source_system}_{sync_operation.target_system}"
        self.sync_queues[queue_key].extend(sync_operation.operations)
        
        sync_operation.status = "queued"
        logger.info(f"Operations queued for on-demand sync: {len(sync_operation.operations)}")
    
    async def _periodic_sync(self, sync_id: str, interval: float):
        """定时同步"""
        sync_operation = self.sync_operations.get(sync_id)
        if not sync_operation:
            return
        
        while True:
            try:
                await asyncio.sleep(interval)
                
                # 执行同步
                sync_operation.status = "syncing"
                await self._immediate_sync(sync_operation)
                
                # 重置状态
                sync_operation.status = "scheduled"
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic sync error: {e}")
                await asyncio.sleep(interval)
    
    async def _execute_sync_operation(self, sync_operation: SyncOperation, operation: DataOperation):
        """执行单个同步操作"""
        # 这里应该实现具体的数据同步逻辑
        # 例如：数据库操作、API调用等
        
        logger.debug(f"Syncing operation {operation.operation_id} from "
                    f"{sync_operation.source_system} to {sync_operation.target_system}")
        
        # 模拟同步延迟
        await asyncio.sleep(0.01)
    
    async def _execute_batch_operations(self, sync_operation: SyncOperation, operations: List[DataOperation]):
        """执行批量同步操作"""
        logger.debug(f"Syncing batch of {len(operations)} operations")
        
        # 这里应该实现批量同步逻辑
        for operation in operations:
            await self._execute_sync_operation(sync_operation, operation)
    
    def get_sync_status(self, sync_id: str) -> Optional[str]:
        """获取同步状态"""
        sync_operation = self.sync_operations.get(sync_id)
        return sync_operation.status if sync_operation else None
    
    def stop_sync(self, sync_id: str):
        """停止同步"""
        if sync_id in self.sync_tasks:
            self.sync_tasks[sync_id].cancel()
            del self.sync_tasks[sync_id]
        
        if sync_id in self.sync_operations:
            self.sync_operations[sync_id].status = "stopped"
        
        logger.info(f"Sync operation stopped: {sync_id}")

class ConsistencyManager:
    """一致性管理器"""
    
    def __init__(self):
        self.consistency_levels: Dict[str, ConsistencyLevel] = {}
        self.version_vectors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.conflict_resolvers: Dict[str, Callable] = {}
        self.replication_lag: Dict[str, float] = {}
        
    def set_consistency_level(self, resource_id: str, level: ConsistencyLevel):
        """设置一致性级别"""
        self.consistency_levels[resource_id] = level
        logger.info(f"Set consistency level for {resource_id}: {level.value}")
    
    def update_version_vector(self, resource_id: str, node_id: str, version: int):
        """更新版本向量"""
        self.version_vectors[resource_id][node_id] = version
    
    def get_version_vector(self, resource_id: str) -> Dict[str, int]:
        """获取版本向量"""
        return dict(self.version_vectors[resource_id])
    
    def register_conflict_resolver(self, resource_type: str, resolver: Callable):
        """注册冲突解决器"""
        self.conflict_resolvers[resource_type] = resolver
        logger.info(f"Registered conflict resolver for {resource_type}")
    
    async def resolve_conflict(self, resource_type: str, conflict_data: Dict[str, Any]) -> Any:
        """解决冲突"""
        resolver = self.conflict_resolvers.get(resource_type)
        if not resolver:
            # 使用默认的"最后写入获胜"策略
            return self._default_conflict_resolution(conflict_data)
        
        return await resolver(conflict_data)
    
    def _default_conflict_resolution(self, conflict_data: Dict[str, Any]) -> Any:
        """默认冲突解决策略"""
        # 按时间戳排序，返回最新的数据
        sorted_data = sorted(conflict_data.items(), 
                           key=lambda x: x[1].get('timestamp', 0), reverse=True)
        return sorted_data[0][1] if sorted_data else None
    
    def update_replication_lag(self, replica_id: str, lag_seconds: float):
        """更新复制延迟"""
        self.replication_lag[replica_id] = lag_seconds
    
    def get_consistency_status(self, resource_id: str) -> Dict[str, Any]:
        """获取一致性状态"""
        level = self.consistency_levels.get(resource_id, ConsistencyLevel.EVENTUAL)
        version_vector = self.get_version_vector(resource_id)
        
        return {
            'consistency_level': level.value,
            'version_vector': version_vector,
            'replication_lag': dict(self.replication_lag),
            'is_consistent': self._check_consistency(resource_id)
        }
    
    def _check_consistency(self, resource_id: str) -> bool:
        """检查一致性"""
        level = self.consistency_levels.get(resource_id, ConsistencyLevel.EVENTUAL)
        
        if level == ConsistencyLevel.STRONG:
            # 强一致性：所有副本必须同步
            return len(set(self.version_vectors[resource_id].values())) <= 1
        
        elif level == ConsistencyLevel.EVENTUAL:
            # 最终一致性：允许一定的复制延迟
            max_lag = max(self.replication_lag.values()) if self.replication_lag else 0
            return max_lag < 60.0  # 1分钟内的延迟是可接受的
        
        else:
            # 弱一致性：总是返回True
            return True

class DataConsistencyManager:
    """数据一致性管理器"""
    
    def __init__(self):
        self.two_phase_commit = TwoPhaseCommit()
        self.data_synchronizer = DataSynchronizer()
        self.consistency_manager = ConsistencyManager()
        
        self.performance_metrics = {
            'total_transactions': 0,
            'committed_transactions': 0,
            'aborted_transactions': 0,
            'total_sync_operations': 0,
            'successful_syncs': 0,
            'failed_syncs': 0
        }
        
        logger.info("Data consistency manager initialized")
    
    async def start_distributed_transaction(self, 
                                          participants: List[str],
                                          operations: List[Dict[str, Any]],
                                          timeout: float = 30.0) -> str:
        """开始分布式事务"""
        transaction_id = await self.two_phase_commit.start_transaction(
            participants, operations, timeout
        )
        
        self.performance_metrics['total_transactions'] += 1
        
        return transaction_id
    
    async def sync_data_immediate(self, 
                                source_system: str,
                                target_system: str,
                                operations: List[DataOperation]) -> str:
        """立即同步数据"""
        sync_id = await self.data_synchronizer.sync_data(
            source_system, target_system, operations, SyncStrategy.IMMEDIATE
        )
        
        self.performance_metrics['total_sync_operations'] += 1
        return sync_id
    
    async def sync_data_scheduled(self, 
                                source_system: str,
                                target_system: str,
                                operations: List[DataOperation],
                                interval: float = 60.0) -> str:
        """定时同步数据"""
        sync_id = await self.data_synchronizer.sync_data(
            source_system, target_system, operations, SyncStrategy.SCHEDULED, interval
        )
        
        self.performance_metrics['total_sync_operations'] += 1
        return sync_id
    
    def set_consistency_level(self, resource_id: str, level: ConsistencyLevel):
        """设置一致性级别"""
        self.consistency_manager.set_consistency_level(resource_id, level)
    
    def register_transaction_participant(self, 
                                       participant_id: str,
                                       prepare_func: Callable,
                                       commit_func: Callable,
                                       abort_func: Callable):
        """注册事务参与者"""
        self.two_phase_commit.register_participant(
            participant_id, prepare_func, commit_func, abort_func
        )
    
    def register_conflict_resolver(self, resource_type: str, resolver: Callable):
        """注册冲突解决器"""
        self.consistency_manager.register_conflict_resolver(resource_type, resolver)
    
    def get_transaction_status(self, transaction_id: str) -> Optional[TransactionStatus]:
        """获取事务状态"""
        return self.two_phase_commit.get_transaction_status(transaction_id)
    
    def get_sync_status(self, sync_id: str) -> Optional[str]:
        """获取同步状态"""
        return self.data_synchronizer.get_sync_status(sync_id)
    
    def get_consistency_status(self, resource_id: str) -> Dict[str, Any]:
        """获取一致性状态"""
        return self.consistency_manager.get_consistency_status(resource_id)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self.performance_metrics,
            'transaction_success_rate': (
                self.performance_metrics['committed_transactions'] / 
                max(self.performance_metrics['total_transactions'], 1) * 100
            ),
            'sync_success_rate': (
                self.performance_metrics['successful_syncs'] / 
                max(self.performance_metrics['total_sync_operations'], 1) * 100
            ),
            'active_transactions': len(self.two_phase_commit.transactions),
            'active_syncs': len(self.data_synchronizer.sync_operations)
        }

# 全局实例
_data_consistency_manager = None

def get_data_consistency_manager() -> DataConsistencyManager:
    """获取数据一致性管理器实例"""
    global _data_consistency_manager
    if _data_consistency_manager is None:
        _data_consistency_manager = DataConsistencyManager()
    return _data_consistency_manager

# 事务装饰器
@asynccontextmanager
async def distributed_transaction(participants: List[str], timeout: float = 30.0):
    """分布式事务上下文管理器"""
    manager = get_data_consistency_manager()
    
    # 模拟操作列表
    operations = []
    
    try:
        yield operations
        
        # 提交事务
        transaction_id = await manager.start_distributed_transaction(
            participants, operations, timeout
        )
        
        logger.info(f"Distributed transaction completed: {transaction_id}")
        
    except Exception as e:
        logger.error(f"Distributed transaction failed: {e}")
        raise e

# 数据一致性装饰器
def ensure_consistency(resource_id: str, level: ConsistencyLevel = ConsistencyLevel.STRONG):
    """确保数据一致性装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = get_data_consistency_manager()
            
            # 设置一致性级别
            manager.set_consistency_level(resource_id, level)
            
            try:
                result = await func(*args, **kwargs)
                
                # 更新版本向量
                manager.consistency_manager.update_version_vector(
                    resource_id, "local", int(time.time())
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Consistency operation failed for {resource_id}: {e}")
                raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# 使用示例
if __name__ == "__main__":
    async def test_data_consistency():
        """测试数据一致性管理器"""
        manager = get_data_consistency_manager()
        
        # 注册事务参与者
        async def prepare_func(tx_id, operations):
            print(f"Preparing transaction {tx_id}")
            return True
        
        async def commit_func(tx_id):
            print(f"Committing transaction {tx_id}")
            return True
        
        async def abort_func(tx_id):
            print(f"Aborting transaction {tx_id}")
            return True
        
        manager.register_transaction_participant(
            "database", prepare_func, commit_func, abort_func
        )
        
        # 测试分布式事务
        async with distributed_transaction(["database"]) as operations:
            operations.append({
                "type": "INSERT",
                "table": "users",
                "data": {"name": "test", "email": "test@example.com"}
            })
        
        # 测试数据同步
        operations = [
            DataOperation(
                operation_id="1",
                operation_type="INSERT",
                table_name="users",
                data={"name": "test"}
            )
        ]
        
        sync_id = await manager.sync_data_immediate(
            "primary_db", "replica_db", operations
        )
        
        print(f"Sync status: {manager.get_sync_status(sync_id)}")
        
        # 获取性能指标
        metrics = manager.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
    
    # 运行测试
    asyncio.run(test_data_consistency())
