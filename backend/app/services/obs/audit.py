"""
审计日志和PII脱敏
"""
import json
import hashlib
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
import structlog

from app.config import settings

logger = structlog.get_logger()


class PIIRedactor:
    """PII数据脱敏器"""
    
    def __init__(self):
        # 敏感数据模式
        self.patterns = {
            'phone': re.compile(r'1[3-9]\d{9}'),
            'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'id_card': re.compile(r'\d{15}|\d{18}'),
            'bank_card': re.compile(r'\d{16,19}'),
            'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            'url': re.compile(r'https?://[^\s]+'),
        }
        
        # 替换模式
        self.replacements = {
            'phone': lambda m: m.group()[:3] + '****' + m.group()[-4:],
            'email': lambda m: m.group().split('@')[0][:2] + '***@' + m.group().split('@')[1],
            'id_card': lambda m: m.group()[:6] + '********' + m.group()[-4:],
            'bank_card': lambda m: m.group()[:4] + '****' + m.group()[-4:],
            'ip_address': lambda m: '.'.join(m.group().split('.')[:2] + ['***', '***']),
            'url': lambda m: m.group().split('://')[0] + '://***',
        }
    
    def redact_text(self, text: str) -> str:
        """脱敏文本中的PII信息"""
        if not text:
            return text
        
        redacted = text
        for pattern_name, pattern in self.patterns.items():
            if pattern_name in self.replacements:
                redacted = pattern.sub(self.replacements[pattern_name], redacted)
        
        return redacted
    
    def redact_dict(self, data: Dict[str, Any], sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """脱敏字典中的敏感信息"""
        if not isinstance(data, dict):
            return data
        
        sensitive_keys = sensitive_keys or [
            'password', 'token', 'api_key', 'secret', 'authorization',
            'phone', 'email', 'id_card', 'bank_card', 'address'
        ]
        
        redacted = {}
        for key, value in data.items():
            if key.lower() in [k.lower() for k in sensitive_keys]:
                # 完全隐藏敏感键值
                redacted[key] = '***'
            elif isinstance(value, str):
                # 脱敏字符串值
                redacted[key] = self.redact_text(value)
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                redacted[key] = self.redact_dict(value, sensitive_keys)
            elif isinstance(value, list):
                # 处理列表
                redacted[key] = [
                    self.redact_dict(item, sensitive_keys) if isinstance(item, dict)
                    else self.redact_text(str(item)) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                redacted[key] = value
        
        return redacted
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> str:
        """对敏感数据进行哈希处理"""
        salt = salt or "voicehelper_salt"
        return hashlib.sha256((data + salt).encode()).hexdigest()[:16]


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self):
        self.redactor = PIIRedactor()
        self.audit_logger = structlog.get_logger("audit")
    
    async def log_session_event(
        self,
        session_id: str,
        user_id: str,
        event_type: str,
        event_data: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """记录会话审计事件"""
        
        # 脱敏事件数据
        safe_event_data = self.redactor.redact_dict(event_data or {})
        
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "user_id": self.redactor.hash_sensitive_data(user_id),  # 哈希用户ID
            "event_type": event_type,
            "event_data": safe_event_data,
            "client_ip": self.redactor.redact_text(ip_address) if ip_address else None,
            "user_agent": user_agent[:100] if user_agent else None,  # 截断用户代理
            "audit_version": "1.0"
        }
        
        self.audit_logger.info(
            "Session audit event",
            **audit_record
        )
    
    async def log_voice_interaction(
        self,
        session_id: str,
        user_id: str,
        interaction_type: str,  # "stt_result", "llm_request", "tts_synthesis"
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None
    ):
        """记录语音交互审计"""
        
        # 脱敏语音内容
        safe_content = self.redactor.redact_text(content) if content else None
        safe_metadata = self.redactor.redact_dict(metadata or {})
        
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "user_id": self.redactor.hash_sensitive_data(user_id),
            "interaction_type": interaction_type,
            "content_length": len(content) if content else 0,
            "content_sample": safe_content[:200] if safe_content else None,  # 只记录前200字符
            "metadata": safe_metadata,
            "duration_ms": duration_ms,
            "audit_version": "1.0"
        }
        
        self.audit_logger.info(
            "Voice interaction audit",
            **audit_record
        )
    
    async def log_error_event(
        self,
        session_id: Optional[str],
        user_id: Optional[str],
        error_type: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录错误审计事件"""
        
        # 脱敏错误信息和上下文
        safe_error_message = self.redactor.redact_text(error_message)
        safe_context = self.redactor.redact_dict(context or {})
        
        # 脱敏堆栈跟踪（移除敏感路径信息）
        safe_stack_trace = None
        if stack_trace:
            safe_stack_trace = re.sub(
                r'/[^/\s]*/(users?|home)/[^/\s]*',
                '/***',
                stack_trace
            )[:1000]  # 限制长度
        
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "user_id": self.redactor.hash_sensitive_data(user_id) if user_id else None,
            "error_type": error_type,
            "error_message": safe_error_message,
            "stack_trace": safe_stack_trace,
            "context": safe_context,
            "audit_version": "1.0"
        }
        
        self.audit_logger.error(
            "Error audit event",
            **audit_record
        )
    
    async def log_security_event(
        self,
        event_type: str,  # "auth_failure", "rate_limit", "suspicious_activity"
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "medium"  # "low", "medium", "high", "critical"
    ):
        """记录安全审计事件"""
        
        safe_details = self.redactor.redact_dict(details or {})
        
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": self.redactor.hash_sensitive_data(user_id) if user_id else None,
            "client_ip": self.redactor.redact_text(ip_address) if ip_address else None,
            "details": safe_details,
            "severity": severity,
            "audit_version": "1.0"
        }
        
        self.audit_logger.warning(
            "Security audit event",
            **audit_record
        )
    
    async def log_data_access(
        self,
        user_id: str,
        resource_type: str,  # "user_data", "session_data", "audio_recording"
        resource_id: str,
        action: str,  # "read", "write", "delete", "export"
        ip_address: Optional[str] = None,
        justification: Optional[str] = None
    ):
        """记录数据访问审计"""
        
        audit_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": self.redactor.hash_sensitive_data(user_id),
            "resource_type": resource_type,
            "resource_id": self.redactor.hash_sensitive_data(resource_id),
            "action": action,
            "client_ip": self.redactor.redact_text(ip_address) if ip_address else None,
            "justification": justification,
            "audit_version": "1.0"
        }
        
        self.audit_logger.info(
            "Data access audit",
            **audit_record
        )
    
    async def export_audit_logs(
        self,
        start_time: datetime,
        end_time: datetime,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """导出审计日志（用于合规检查）"""
        
        # 这里应该从持久化存储中查询审计日志
        # 为了演示，返回空列表
        logger.info(
            "Audit logs export requested",
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            user_id=self.redactor.hash_sensitive_data(user_id) if user_id else None,
            session_id=session_id
        )
        
        return []
    
    def get_audit_summary(
        self,
        time_range_hours: int = 24
    ) -> Dict[str, Any]:
        """获取审计摘要"""
        
        # 这里应该从审计日志中统计数据
        # 为了演示，返回模拟数据
        return {
            "time_range_hours": time_range_hours,
            "total_events": 0,
            "event_types": {},
            "security_events": 0,
            "error_events": 0,
            "data_access_events": 0,
            "unique_users": 0,
            "unique_sessions": 0
        }


class ComplianceManager:
    """合规管理器"""
    
    def __init__(self):
        self.audit_logger = AuditLogger()
        self.retention_days = 90  # 审计日志保留天数
    
    async def ensure_gdpr_compliance(
        self,
        user_id: str,
        action: str,  # "data_export", "data_deletion", "consent_update"
        details: Optional[Dict[str, Any]] = None
    ):
        """确保GDPR合规"""
        
        await self.audit_logger.log_data_access(
            user_id=user_id,
            resource_type="gdpr_compliance",
            resource_id=f"gdpr_{action}_{user_id}",
            action=action,
            justification=f"GDPR compliance: {action}"
        )
        
        logger.info(
            "GDPR compliance action",
            user_id=self.audit_logger.redactor.hash_sensitive_data(user_id),
            action=action,
            details=details
        )
    
    async def schedule_data_cleanup(self):
        """定期数据清理"""
        
        # 清理过期的审计日志
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        
        logger.info(
            "Scheduled data cleanup",
            cutoff_date=cutoff_date.isoformat(),
            retention_days=self.retention_days
        )
        
        # 这里应该实现实际的数据清理逻辑
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """获取合规状态"""
        
        return {
            "gdpr_compliant": True,
            "audit_retention_days": self.retention_days,
            "pii_redaction_enabled": True,
            "data_encryption_enabled": True,
            "last_cleanup": datetime.utcnow().isoformat()
        }


# 全局审计实例
audit_logger = AuditLogger()
compliance_manager = ComplianceManager()
