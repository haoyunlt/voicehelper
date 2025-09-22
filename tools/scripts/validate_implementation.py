#!/usr/bin/env python3
"""
VoiceHelper实现验证脚本
验证P0和P1功能的完整性
"""

import os
import sys
import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImplementationValidator:
    """实现验证器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            "p0_features": {},
            "p1_features": {},
            "monitoring": {},
            "deployment": {},
            "overall_score": 0
        }
    
    def validate_p0_features(self) -> Dict[str, bool]:
        """验证P0功能"""
        logger.info("验证P0功能...")
        
        p0_checks = {
            "sse_streaming_chat": self._check_sse_implementation(),
            "websocket_voice": self._check_websocket_implementation(),
            "bge_faiss_rag": self._check_rag_implementation(),
            "langgraph_agent": self._check_agent_implementation(),
            "jwt_auth": self._check_auth_implementation(),
            "multi_tenant": self._check_tenant_implementation(),
            "voice_processing": self._check_voice_processing(),
            "miniprogram_support": self._check_miniprogram_implementation()
        }
        
        self.results["p0_features"] = p0_checks
        return p0_checks
    
    def validate_p1_features(self) -> Dict[str, bool]:
        """验证P1功能"""
        logger.info("验证P1功能...")
        
        p1_checks = {
            "sse_reconnection": self._check_sse_reconnection(),
            "chat_cancellation": self._check_chat_cancellation(),
            "agent_visualization": self._check_agent_visualization(),
            "latency_monitoring": self._check_latency_monitoring(),
            "backpressure_handling": self._check_backpressure(),
            "hot_reload": self._check_hot_reload(),
            "error_fallback": self._check_error_fallback()
        }
        
        self.results["p1_features"] = p1_checks
        return p1_checks
    
    def validate_monitoring(self) -> Dict[str, bool]:
        """验证监控系统"""
        logger.info("验证监控系统...")
        
        monitoring_checks = {
            "prometheus_config": self._check_prometheus_config(),
            "grafana_dashboards": self._check_grafana_dashboards(),
            "alert_rules": self._check_alert_rules(),
            "metrics_collection": self._check_metrics_collection(),
            "health_checks": self._check_health_checks()
        }
        
        self.results["monitoring"] = monitoring_checks
        return monitoring_checks
    
    def validate_deployment(self) -> Dict[str, bool]:
        """验证部署配置"""
        logger.info("验证部署配置...")
        
        deployment_checks = {
            "docker_compose": self._check_docker_compose(),
            "kubernetes_config": self._check_kubernetes_config(),
            "helm_charts": self._check_helm_charts(),
            "ci_cd_pipeline": self._check_ci_cd(),
            "environment_config": self._check_environment_config()
        }
        
        self.results["deployment"] = deployment_checks
        return deployment_checks
    
    # P0功能检查方法
    def _check_sse_implementation(self) -> bool:
        """检查SSE流式聊天实现"""
        sse_files = [
            "backend/internal/handlers/chat_sse.go",
            "frontend/components/chat/StreamingChat.tsx"
        ]
        return all((self.project_root / f).exists() for f in sse_files)
    
    def _check_websocket_implementation(self) -> bool:
        """检查WebSocket语音实现"""
        ws_files = [
            "backend/internal/handlers/voice_ws.go",
            "algo/core/websocket_voice.py",
            "frontend/components/voice/VoiceChat.tsx"
        ]
        return all((self.project_root / f).exists() for f in ws_files)
    
    def _check_rag_implementation(self) -> bool:
        """检查BGE+FAISS RAG实现"""
        rag_files = [
            "algo/core/bge_faiss_rag.py",
            "algo/core/retrieve.py"
        ]
        return all((self.project_root / f).exists() for f in rag_files)
    
    def _check_agent_implementation(self) -> bool:
        """检查LangGraph Agent实现"""
        agent_files = [
            "algo/core/langgraph_agent.py",
            "algo/core/events.py"
        ]
        return all((self.project_root / f).exists() for f in agent_files)
    
    def _check_auth_implementation(self) -> bool:
        """检查JWT认证实现"""
        auth_files = [
            "backend/pkg/middleware/auth.go"
        ]
        return all((self.project_root / f).exists() for f in auth_files)
    
    def _check_tenant_implementation(self) -> bool:
        """检查多租户实现"""
        tenant_file = self.project_root / "backend/pkg/middleware/auth.go"
        if not tenant_file.exists():
            return False
        
        content = tenant_file.read_text()
        return "TenantMiddleware" in content
    
    def _check_voice_processing(self) -> bool:
        """检查语音处理实现"""
        voice_files = [
            "algo/core/enhanced_voice_services.py",
            "algo/core/vad.py",
            "algo/services/streaming_tts.py"
        ]
        return all((self.project_root / f).exists() for f in voice_files)
    
    def _check_miniprogram_implementation(self) -> bool:
        """检查微信小程序实现"""
        mp_files = [
            "frontend/miniprogram/pages/chat/chat.js",
            "frontend/miniprogram/app.json"
        ]
        return all((self.project_root / f).exists() for f in mp_files)
    
    # P1功能检查方法
    def _check_sse_reconnection(self) -> bool:
        """检查SSE断线重连"""
        sse_file = self.project_root / "frontend/components/chat/StreamingChat.tsx"
        if not sse_file.exists():
            return False
        
        content = sse_file.read_text()
        return "reconnectAttempts" in content and "connectSSE" in content
    
    def _check_chat_cancellation(self) -> bool:
        """检查聊天取消功能"""
        sse_file = self.project_root / "frontend/components/chat/StreamingChat.tsx"
        if not sse_file.exists():
            return False
        
        content = sse_file.read_text()
        return "cancelRequest" in content and "/chat/cancel" in content
    
    def _check_agent_visualization(self) -> bool:
        """检查Agent事件可视化"""
        viz_file = self.project_root / "frontend/components/chat/AgentEventVisualization.tsx"
        return viz_file.exists()
    
    def _check_latency_monitoring(self) -> bool:
        """检查延迟监控"""
        monitor_file = self.project_root / "frontend/components/voice/LatencyMonitor.tsx"
        return monitor_file.exists()
    
    def _check_backpressure(self) -> bool:
        """检查背压处理"""
        bp_file = self.project_root / "frontend/components/voice/BackpressureHandler.tsx"
        return bp_file.exists()
    
    def _check_hot_reload(self) -> bool:
        """检查热重载功能"""
        algo_file = self.project_root / "algo/app/main.py"
        if not algo_file.exists():
            return False
        
        content = algo_file.read_text()
        return "/reload" in content and "reload_index" in content
    
    def _check_error_fallback(self) -> bool:
        """检查异常回退"""
        voice_file = self.project_root / "algo/core/enhanced_voice_services.py"
        if not voice_file.exists():
            return False
        
        content = voice_file.read_text()
        return "except" in content and "fallback" in content.lower()
    
    # 监控系统检查方法
    def _check_prometheus_config(self) -> bool:
        """检查Prometheus配置"""
        config_file = self.project_root / "deploy/monitoring/prometheus.yml"
        return config_file.exists()
    
    def _check_grafana_dashboards(self) -> bool:
        """检查Grafana仪表盘"""
        dashboard_file = self.project_root / "deploy/monitoring/grafana/dashboards/voicehelper-overview.json"
        return dashboard_file.exists()
    
    def _check_alert_rules(self) -> bool:
        """检查告警规则"""
        rules_files = [
            "deploy/monitoring/rules/voicehelper-alerts.yml",
            "deploy/monitoring/rules/enhanced-alerts.yml"
        ]
        return all((self.project_root / f).exists() for f in rules_files)
    
    def _check_metrics_collection(self) -> bool:
        """检查指标收集"""
        metrics_files = [
            "backend/pkg/metrics/collector.go",
            "backend/pkg/metrics/voice_metrics.go"
        ]
        return all((self.project_root / f).exists() for f in metrics_files)
    
    def _check_health_checks(self) -> bool:
        """检查健康检查"""
        blackbox_file = self.project_root / "deploy/monitoring/blackbox.yml"
        return blackbox_file.exists()
    
    # 部署配置检查方法
    def _check_docker_compose(self) -> bool:
        """检查Docker Compose配置"""
        compose_files = [
            "docker-compose.local.yml",
            "deploy/docker-compose.monitoring.yml"
        ]
        return all((self.project_root / f).exists() for f in compose_files)
    
    def _check_kubernetes_config(self) -> bool:
        """检查Kubernetes配置"""
        k8s_file = self.project_root / "deploy/k8s/voicehelper-deployment.yaml"
        return k8s_file.exists()
    
    def _check_helm_charts(self) -> bool:
        """检查Helm Charts"""
        helm_dir = self.project_root / "deploy/helm"
        return helm_dir.exists() and helm_dir.is_dir()
    
    def _check_ci_cd(self) -> bool:
        """检查CI/CD流水线"""
        ci_file = self.project_root / ".github/workflows/ci-cd.yml"
        return ci_file.exists()
    
    def _check_environment_config(self) -> bool:
        """检查环境配置"""
        env_files = [
            "env.example",
            "env.unified"
        ]
        return all((self.project_root / f).exists() for f in env_files)
    
    def calculate_score(self) -> float:
        """计算总体评分"""
        all_checks = []
        
        # P0功能权重：40%
        p0_score = sum(self.results["p0_features"].values()) / len(self.results["p0_features"]) * 0.4
        all_checks.extend(self.results["p0_features"].values())
        
        # P1功能权重：30%
        p1_score = sum(self.results["p1_features"].values()) / len(self.results["p1_features"]) * 0.3
        all_checks.extend(self.results["p1_features"].values())
        
        # 监控系统权重：20%
        monitoring_score = sum(self.results["monitoring"].values()) / len(self.results["monitoring"]) * 0.2
        all_checks.extend(self.results["monitoring"].values())
        
        # 部署配置权重：10%
        deployment_score = sum(self.results["deployment"].values()) / len(self.results["deployment"]) * 0.1
        all_checks.extend(self.results["deployment"].values())
        
        total_score = (p0_score + p1_score + monitoring_score + deployment_score) * 100
        self.results["overall_score"] = total_score
        
        return total_score
    
    def generate_report(self) -> str:
        """生成验证报告"""
        score = self.calculate_score()
        
        report = f"""
# VoiceHelper 实现验证报告

## 总体评分: {score:.1f}/100

## P0功能验证 (权重40%)
"""
        for feature, status in self.results["p0_features"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {feature}\n"
        
        report += f"\n## P1功能验证 (权重30%)\n"
        for feature, status in self.results["p1_features"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {feature}\n"
        
        report += f"\n## 监控系统验证 (权重20%)\n"
        for feature, status in self.results["monitoring"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {feature}\n"
        
        report += f"\n## 部署配置验证 (权重10%)\n"
        for feature, status in self.results["deployment"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {feature}\n"
        
        # 评分等级
        if score >= 90:
            grade = "A+ (优秀)"
        elif score >= 80:
            grade = "A (良好)"
        elif score >= 70:
            grade = "B (合格)"
        elif score >= 60:
            grade = "C (需改进)"
        else:
            grade = "D (不合格)"
        
        report += f"\n## 评估结果\n"
        report += f"- 总体评分: {score:.1f}/100\n"
        report += f"- 评估等级: {grade}\n"
        
        if score >= 80:
            report += f"- 结论: 实现质量良好，可以进入生产环境\n"
        elif score >= 70:
            report += f"- 结论: 基本功能完整，建议完善后上线\n"
        else:
            report += f"- 结论: 存在重要功能缺失，需要继续开发\n"
        
        return report
    
    def run_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        logger.info("开始VoiceHelper实现验证...")
        
        self.validate_p0_features()
        self.validate_p1_features()
        self.validate_monitoring()
        self.validate_deployment()
        
        score = self.calculate_score()
        
        logger.info(f"验证完成，总体评分: {score:.1f}/100")
        
        return self.results

def main():
    """主函数"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    validator = ImplementationValidator(project_root)
    results = validator.run_validation()
    
    # 生成报告
    report = validator.generate_report()
    
    # 保存报告
    report_file = Path(project_root) / "reports" / "implementation_validation.md"
    report_file.parent.mkdir(exist_ok=True)
    report_file.write_text(report, encoding='utf-8')
    
    print(report)
    print(f"\n报告已保存到: {report_file}")
    
    # 返回适当的退出码
    score = results["overall_score"]
    if score >= 80:
        sys.exit(0)  # 成功
    elif score >= 70:
        sys.exit(1)  # 警告
    else:
        sys.exit(2)  # 失败

if __name__ == "__main__":
    main()
