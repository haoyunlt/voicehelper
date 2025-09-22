"""
VoiceHelper v1.24.0 - 监控告警系统
实现全链路监控和智能告警，提升系统可观测性
"""

import asyncio
import time
import logging
import json
import psutil
import threading
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import functools
import traceback

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class Metric:
    """指标"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    metric_name: str
    condition: str  # 例如: "value > 100"
    severity: AlertSeverity
    duration: float = 0.0  # 持续时间（秒）
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    description: str = ""

@dataclass
class Alert:
    """告警"""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    timestamp: float
    resolved_at: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

@dataclass
class NotificationChannel:
    """通知渠道"""
    channel_id: str
    channel_type: str  # email, webhook, slack, etc.
    config: Dict[str, Any]
    enabled: bool = True

class MetricCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.collectors: List[Callable] = []
        self.collection_interval = 10.0  # 10秒
        self.is_running = False
        self.collection_task: Optional[asyncio.Task] = None
        
    def register_collector(self, collector: Callable):
        """注册指标收集器"""
        self.collectors.append(collector)
        logger.info(f"Registered metric collector: {collector.__name__}")
    
    async def start_collection(self):
        """开始指标收集"""
        if self.is_running:
            return
        
        self.is_running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metric collection started")
    
    async def stop_collection(self):
        """停止指标收集"""
        self.is_running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        logger.info("Metric collection stopped")
    
    async def _collection_loop(self):
        """指标收集循环"""
        while self.is_running:
            try:
                # 收集系统指标
                await self._collect_system_metrics()
                
                # 执行自定义收集器
                for collector in self.collectors:
                    try:
                        metrics = await collector() if asyncio.iscoroutinefunction(collector) else collector()
                        if metrics:
                            for metric in metrics:
                                self.add_metric(metric)
                    except Exception as e:
                        logger.error(f"Metric collector {collector.__name__} failed: {e}")
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Metric collection loop error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.add_metric(Metric(
                name="system_cpu_percent",
                value=cpu_percent,
                metric_type=MetricType.GAUGE,
                timestamp=time.time(),
                labels={"host": "localhost"},
                description="CPU usage percentage"
            ))
            
            # 内存使用率
            memory = psutil.virtual_memory()
            self.add_metric(Metric(
                name="system_memory_percent",
                value=memory.percent,
                metric_type=MetricType.GAUGE,
                timestamp=time.time(),
                labels={"host": "localhost"},
                description="Memory usage percentage"
            ))
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.add_metric(Metric(
                name="system_disk_percent",
                value=disk_percent,
                metric_type=MetricType.GAUGE,
                timestamp=time.time(),
                labels={"host": "localhost", "mount": "/"},
                description="Disk usage percentage"
            ))
            
            # 网络IO
            net_io = psutil.net_io_counters()
            self.add_metric(Metric(
                name="system_network_bytes_sent",
                value=net_io.bytes_sent,
                metric_type=MetricType.COUNTER,
                timestamp=time.time(),
                labels={"host": "localhost"},
                description="Network bytes sent"
            ))
            
            self.add_metric(Metric(
                name="system_network_bytes_recv",
                value=net_io.bytes_recv,
                metric_type=MetricType.COUNTER,
                timestamp=time.time(),
                labels={"host": "localhost"},
                description="Network bytes received"
            ))
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    def add_metric(self, metric: Metric):
        """添加指标"""
        self.metrics[metric.name].append(metric)
    
    def get_metric(self, name: str, duration: float = 3600) -> List[Metric]:
        """获取指标"""
        if name not in self.metrics:
            return []
        
        cutoff_time = time.time() - duration
        return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
    
    def get_latest_metric(self, name: str) -> Optional[Metric]:
        """获取最新指标"""
        if name not in self.metrics or not self.metrics[name]:
            return None
        return self.metrics[name][-1]

class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.notification_channels: Dict[str, NotificationChannel] = {}
        self.evaluation_interval = 30.0  # 30秒
        self.is_running = False
        self.evaluation_task: Optional[asyncio.Task] = None
        
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def add_notification_channel(self, channel: NotificationChannel):
        """添加通知渠道"""
        self.notification_channels[channel.channel_id] = channel
        logger.info(f"Added notification channel: {channel.channel_type}")
    
    async def start_evaluation(self):
        """开始告警评估"""
        if self.is_running:
            return
        
        self.is_running = True
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Alert evaluation started")
    
    async def stop_evaluation(self):
        """停止告警评估"""
        self.is_running = False
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert evaluation stopped")
    
    async def _evaluation_loop(self):
        """告警评估循环"""
        while self.is_running:
            try:
                await self._evaluate_alerts()
                await asyncio.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Alert evaluation loop error: {e}")
                await asyncio.sleep(self.evaluation_interval)
    
    async def _evaluate_alerts(self):
        """评估告警"""
        metric_collector = get_metric_collector()
        
        for rule_id, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # 获取指标数据
                metrics = metric_collector.get_metric(rule.metric_name, rule.duration)
                if not metrics:
                    continue
                
                # 评估告警条件
                should_alert = await self._evaluate_condition(rule, metrics)
                
                if should_alert:
                    await self._trigger_alert(rule, metrics)
                else:
                    await self._resolve_alert(rule_id)
                    
            except Exception as e:
                logger.error(f"Alert evaluation error for rule {rule_id}: {e}")
    
    async def _evaluate_condition(self, rule: AlertRule, metrics: List[Metric]) -> bool:
        """评估告警条件"""
        if not metrics:
            return False
        
        # 简单条件评估（可以扩展为更复杂的表达式解析器）
        latest_metric = metrics[-1]
        value = latest_metric.value
        
        try:
            # 解析条件（这里简化处理）
            if ">" in rule.condition:
                threshold = float(rule.condition.split(">")[1].strip())
                return value > threshold
            elif "<" in rule.condition:
                threshold = float(rule.condition.split("<")[1].strip())
                return value < threshold
            elif ">=" in rule.condition:
                threshold = float(rule.condition.split(">=")[1].strip())
                return value >= threshold
            elif "<=" in rule.condition:
                threshold = float(rule.condition.split("<=")[1].strip())
                return value <= threshold
            elif "==" in rule.condition:
                threshold = float(rule.condition.split("==")[1].strip())
                return value == threshold
            
            return False
            
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False
    
    async def _trigger_alert(self, rule: AlertRule, metrics: List[Metric]):
        """触发告警"""
        alert_id = f"{rule.rule_id}_{int(time.time())}"
        
        if alert_id in self.active_alerts:
            return  # 告警已存在
        
        latest_metric = metrics[-1]
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=f"{rule.name}: {rule.metric_name} = {latest_metric.value}",
            timestamp=time.time(),
            labels=latest_metric.labels,
            annotations={
                "rule_name": rule.name,
                "metric_name": rule.metric_name,
                "condition": rule.condition,
                "description": rule.description
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # 发送通知
        await self._send_notifications(alert)
        
        logger.warning(f"Alert triggered: {alert.message}")
    
    async def _resolve_alert(self, rule_id: str):
        """解决告警"""
        alerts_to_resolve = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.rule_id == rule_id and alert.status == AlertStatus.ACTIVE
        ]
        
        for alert_id in alerts_to_resolve:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = time.time()
            
            logger.info(f"Alert resolved: {alert.message}")
    
    async def _send_notifications(self, alert: Alert):
        """发送通知"""
        rule = self.alert_rules.get(alert.rule_id)
        if not rule:
            return
        
        for channel_id in rule.notification_channels:
            channel = self.notification_channels.get(channel_id)
            if not channel or not channel.enabled:
                continue
            
            try:
                if channel.channel_type == "email":
                    await self._send_email_notification(alert, channel)
                elif channel.channel_type == "webhook":
                    await self._send_webhook_notification(alert, channel)
                elif channel.channel_type == "slack":
                    await self._send_slack_notification(alert, channel)
                    
            except Exception as e:
                logger.error(f"Notification failed for channel {channel_id}: {e}")
    
    async def _send_email_notification(self, alert: Alert, channel: NotificationChannel):
        """发送邮件通知"""
        config = channel.config
        
        msg = MIMEMultipart()
        msg['From'] = config['from_email']
        msg['To'] = config['to_email']
        msg['Subject'] = f"Alert: {alert.message}"
        
        body = f"""
        Alert Details:
        - Severity: {alert.severity.value}
        - Message: {alert.message}
        - Time: {time.ctime(alert.timestamp)}
        - Labels: {alert.labels}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # 发送邮件（这里简化处理）
        logger.info(f"Email notification sent: {alert.message}")
    
    async def _send_webhook_notification(self, alert: Alert, channel: NotificationChannel):
        """发送Webhook通知"""
        config = channel.config
        webhook_url = config['url']
        
        payload = {
            "alert_id": alert.alert_id,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "labels": alert.labels,
            "annotations": alert.annotations
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Webhook notification sent: {alert.message}")
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
    
    async def _send_slack_notification(self, alert: Alert, channel: NotificationChannel):
        """发送Slack通知"""
        config = channel.config
        webhook_url = config['webhook_url']
        
        # Slack消息格式
        color_map = {
            AlertSeverity.INFO: "good",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "danger",
            AlertSeverity.CRITICAL: "danger"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.severity, "good"),
                "title": f"Alert: {alert.message}",
                "fields": [
                    {"title": "Severity", "value": alert.severity.value, "short": True},
                    {"title": "Time", "value": time.ctime(alert.timestamp), "short": True},
                    {"title": "Labels", "value": str(alert.labels), "short": False}
                ]
            }]
        }
        
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info(f"Slack notification sent: {alert.message}")
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")

class MonitoringAlertSystem:
    """监控告警系统"""
    
    def __init__(self):
        self.metric_collector = MetricCollector()
        self.alert_manager = AlertManager()
        self.performance_metrics = {
            "total_metrics_collected": 0,
            "active_alerts": 0,
            "notifications_sent": 0,
            "system_uptime": time.time()
        }
        
        # 注册默认指标收集器
        self._register_default_collectors()
        
        # 添加默认告警规则
        self._add_default_alert_rules()
        
        logger.info("Monitoring and alert system initialized")
    
    def _register_default_collectors(self):
        """注册默认指标收集器"""
        # 应用性能指标收集器
        self.metric_collector.register_collector(self._collect_application_metrics)
    
    async def _collect_application_metrics(self) -> List[Metric]:
        """收集应用性能指标"""
        metrics = []
        
        try:
            # 当前时间戳
            timestamp = time.time()
            
            # 活跃连接数（模拟）
            active_connections = 100  # 这里应该从实际应用中获取
            metrics.append(Metric(
                name="application_active_connections",
                value=active_connections,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                description="Number of active connections"
            ))
            
            # 请求处理时间（模拟）
            avg_response_time = 150.0  # 毫秒
            metrics.append(Metric(
                name="application_avg_response_time",
                value=avg_response_time,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                description="Average response time in milliseconds"
            ))
            
            # 错误率（模拟）
            error_rate = 0.01  # 1%
            metrics.append(Metric(
                name="application_error_rate",
                value=error_rate,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                description="Error rate percentage"
            ))
            
            # 吞吐量（模拟）
            throughput = 1000  # 请求/秒
            metrics.append(Metric(
                name="application_throughput",
                value=throughput,
                metric_type=MetricType.GAUGE,
                timestamp=timestamp,
                description="Requests per second"
            ))
            
        except Exception as e:
            logger.error(f"Application metrics collection error: {e}")
        
        return metrics
    
    def _add_default_alert_rules(self):
        """添加默认告警规则"""
        # CPU使用率告警
        self.alert_manager.add_alert_rule(AlertRule(
            rule_id="high_cpu_usage",
            name="High CPU Usage",
            metric_name="system_cpu_percent",
            condition="> 80",
            severity=AlertSeverity.WARNING,
            duration=300.0,  # 5分钟
            description="CPU usage is above 80%"
        ))
        
        # 内存使用率告警
        self.alert_manager.add_alert_rule(AlertRule(
            rule_id="high_memory_usage",
            name="High Memory Usage",
            metric_name="system_memory_percent",
            condition="> 85",
            severity=AlertSeverity.WARNING,
            duration=300.0,
            description="Memory usage is above 85%"
        ))
        
        # 磁盘使用率告警
        self.alert_manager.add_alert_rule(AlertRule(
            rule_id="high_disk_usage",
            name="High Disk Usage",
            metric_name="system_disk_percent",
            condition="> 90",
            severity=AlertSeverity.CRITICAL,
            duration=300.0,
            description="Disk usage is above 90%"
        ))
        
        # 应用错误率告警
        self.alert_manager.add_alert_rule(AlertRule(
            rule_id="high_error_rate",
            name="High Error Rate",
            metric_name="application_error_rate",
            condition="> 0.05",  # 5%
            severity=AlertSeverity.ERROR,
            duration=60.0,
            description="Application error rate is above 5%"
        ))
    
    async def start(self):
        """启动监控告警系统"""
        await self.metric_collector.start_collection()
        await self.alert_manager.start_evaluation()
        logger.info("Monitoring and alert system started")
    
    async def stop(self):
        """停止监控告警系统"""
        await self.metric_collector.stop_collection()
        await self.alert_manager.stop_evaluation()
        logger.info("Monitoring and alert system stopped")
    
    def add_custom_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """添加自定义指标"""
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            labels=labels or {},
            description=f"Custom metric: {name}"
        )
        self.metric_collector.add_metric(metric)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        return {
            "total_metrics": sum(len(metrics) for metrics in self.metric_collector.metrics.values()),
            "metric_names": list(self.metric_collector.metrics.keys()),
            "latest_metrics": {
                name: metrics[-1].value if metrics else None
                for name, metrics in self.metric_collector.metrics.items()
            }
        }
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        return {
            "active_alerts": len(self.alert_manager.active_alerts),
            "total_alert_rules": len(self.alert_manager.alert_rules),
            "recent_alerts": list(self.alert_manager.alert_history)[-10:],
            "alert_rules": list(self.alert_manager.alert_rules.values())
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self.performance_metrics,
            "uptime": time.time() - self.performance_metrics["system_uptime"],
            "metrics_summary": self.get_metrics_summary(),
            "alerts_summary": self.get_alerts_summary()
        }

# 全局实例
_monitoring_system = None

def get_metric_collector() -> MetricCollector:
    """获取指标收集器实例"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringAlertSystem()
    return _monitoring_system.metric_collector

def get_monitoring_system() -> MonitoringAlertSystem:
    """获取监控告警系统实例"""
    global _monitoring_system
    if _monitoring_system is None:
        _monitoring_system = MonitoringAlertSystem()
    return _monitoring_system

# 监控装饰器
def monitor_performance(metric_name: str, labels: Dict[str, str] = None):
    """性能监控装饰器"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # 记录成功指标
                monitoring_system = get_monitoring_system()
                monitoring_system.add_custom_metric(
                    f"{metric_name}_success", 1.0, labels
                )
                monitoring_system.add_custom_metric(
                    f"{metric_name}_duration", execution_time * 1000, labels  # 毫秒
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # 记录失败指标
                monitoring_system = get_monitoring_system()
                monitoring_system.add_custom_metric(
                    f"{metric_name}_error", 1.0, labels
                )
                monitoring_system.add_custom_metric(
                    f"{metric_name}_duration", execution_time * 1000, labels
                )
                
                raise e
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# 使用示例
if __name__ == "__main__":
    async def test_monitoring_system():
        """测试监控告警系统"""
        monitoring_system = get_monitoring_system()
        
        # 添加通知渠道
        monitoring_system.alert_manager.add_notification_channel(
            NotificationChannel(
                channel_id="email_alerts",
                channel_type="email",
                config={
                    "from_email": "alerts@voicehelper.com",
                    "to_email": "admin@voicehelper.com"
                }
            )
        )
        
        # 启动系统
        await monitoring_system.start()
        
        # 模拟一些指标
        monitoring_system.add_custom_metric("test_metric", 95.0, {"service": "test"})
        
        # 等待一段时间
        await asyncio.sleep(5)
        
        # 获取摘要
        metrics_summary = monitoring_system.get_metrics_summary()
        alerts_summary = monitoring_system.get_alerts_summary()
        
        print(f"Metrics summary: {metrics_summary}")
        print(f"Alerts summary: {alerts_summary}")
        
        # 停止系统
        await monitoring_system.stop()
    
    # 运行测试
    asyncio.run(test_monitoring_system())
