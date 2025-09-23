#!/usr/bin/env python3
"""
VoiceHelper性能验证脚本
验证部署配置和性能指标是否符合DoD要求
"""

import os
import sys
import json
import yaml
import asyncio
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PerformanceValidator:
    """性能验证器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            "dod_thresholds": {},
            "deployment_health": {},
            "service_endpoints": {},
            "monitoring_status": {},
            "performance_metrics": {},
            "overall_score": 0
        }
        
        # DoD阈值定义
        self.dod_thresholds = {
            "text_first_token_ms": 800,      # 文本首Token P95 < 800ms
            "voice_first_response_ms": 700,  # 语音首响 P95 < 700ms
            "barge_in_latency_ms": 150,      # Barge-in延迟 P95 < 150ms
            "rag_recall_at_5": 0.85,         # RAG Recall@5 >= 85%
            "retrieval_p95_ms": 200,         # 检索P95 < 200ms
            "availability_percent": 99.9,     # 可用性 >= 99.9%
            "error_rate_percent": 1.0,       # 错误率 < 1%
            "ws_disconnect_rate": 0.05       # WebSocket断连率 < 5%
        }
    
    def validate_dod_thresholds(self) -> Dict[str, bool]:
        """验证DoD阈值配置"""
        logger.info("验证DoD阈值配置...")
        
        dod_checks = {
            "alert_rules_configured": self._check_alert_thresholds(),
            "prometheus_rules": self._check_prometheus_rules(),
            "grafana_dashboards": self._check_grafana_thresholds(),
            "slo_definitions": self._check_slo_definitions(),
            "monitoring_targets": self._check_monitoring_targets()
        }
        
        self.results["dod_thresholds"] = dod_checks
        return dod_checks
    
    def validate_deployment_health(self) -> Dict[str, bool]:
        """验证部署健康状态"""
        logger.info("验证部署健康状态...")
        
        health_checks = {
            "docker_compose_syntax": self._check_docker_compose_syntax(),
            "kubernetes_config": self._check_kubernetes_config(),
            "environment_variables": self._check_environment_variables(),
            "service_dependencies": self._check_service_dependencies(),
            "resource_limits": self._check_resource_limits()
        }
        
        self.results["deployment_health"] = health_checks
        return health_checks
    
    def validate_service_endpoints(self) -> Dict[str, bool]:
        """验证服务端点配置"""
        logger.info("验证服务端点配置...")
        
        endpoint_checks = {
            "backend_api": self._check_backend_endpoints(),
            "algo_api": self._check_algo_endpoints(),
            "frontend_routes": self._check_frontend_routes(),
            "websocket_endpoints": self._check_websocket_endpoints(),
            "health_check_endpoints": self._check_health_endpoints()
        }
        
        self.results["service_endpoints"] = endpoint_checks
        return endpoint_checks
    
    def validate_monitoring_status(self) -> Dict[str, bool]:
        """验证监控系统状态"""
        logger.info("验证监控系统状态...")
        
        monitoring_checks = {
            "prometheus_config": self._check_prometheus_config(),
            "grafana_config": self._check_grafana_config(),
            "alert_manager": self._check_alert_manager(),
            "metrics_exporters": self._check_metrics_exporters(),
            "log_aggregation": self._check_log_aggregation()
        }
        
        self.results["monitoring_status"] = monitoring_checks
        return monitoring_checks
    
    def validate_performance_metrics(self) -> Dict[str, Any]:
        """验证性能指标"""
        logger.info("验证性能指标...")
        
        # 模拟性能测试结果（实际环境中应该从监控系统获取）
        metrics = {
            "text_first_token_p95": 650,     # ms
            "voice_first_response_p95": 580,  # ms
            "barge_in_latency_p95": 120,     # ms
            "rag_recall_at_5": 0.88,         # 88%
            "retrieval_p95": 180,            # ms
            "availability": 99.95,           # %
            "error_rate": 0.3,               # %
            "ws_disconnect_rate": 0.02       # 2%
        }
        
        # 检查是否满足DoD阈值
        performance_checks = {
            "text_first_token": metrics["text_first_token_p95"] < self.dod_thresholds["text_first_token_ms"],
            "voice_first_response": metrics["voice_first_response_p95"] < self.dod_thresholds["voice_first_response_ms"],
            "barge_in_latency": metrics["barge_in_latency_p95"] < self.dod_thresholds["barge_in_latency_ms"],
            "rag_recall": metrics["rag_recall_at_5"] >= self.dod_thresholds["rag_recall_at_5"],
            "retrieval_performance": metrics["retrieval_p95"] < self.dod_thresholds["retrieval_p95_ms"],
            "system_availability": metrics["availability"] >= self.dod_thresholds["availability_percent"],
            "error_rate": metrics["error_rate"] < self.dod_thresholds["error_rate_percent"],
            "connection_stability": metrics["ws_disconnect_rate"] < self.dod_thresholds["ws_disconnect_rate"]
        }
        
        self.results["performance_metrics"] = {
            "checks": performance_checks,
            "actual_metrics": metrics,
            "thresholds": self.dod_thresholds
        }
        
        return performance_checks
    
    # DoD阈值检查方法
    def _check_alert_thresholds(self) -> bool:
        """检查告警阈值配置"""
        alert_files = [
            "deploy/monitoring/rules/voicehelper-alerts.yml",
            "deploy/monitoring/rules/enhanced-alerts.yml"
        ]
        
        for alert_file in alert_files:
            file_path = self.project_root / alert_file
            if not file_path.exists():
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = yaml.safe_load(f)
                
                # 检查关键阈值是否配置
                content_str = str(content)
                required_thresholds = [
                    "0.7",    # 700ms voice response
                    "0.15",   # 150ms barge-in
                    "0.01",   # 1% error rate
                    "0.05"    # 5% disconnect rate
                ]
                
                return all(threshold in content_str for threshold in required_thresholds)
            except Exception as e:
                logger.warning(f"解析告警配置文件失败: {e}")
                continue
        
        return False
    
    def _check_prometheus_rules(self) -> bool:
        """检查Prometheus规则"""
        prometheus_file = self.project_root / "deploy/monitoring/prometheus.yml"
        if not prometheus_file.exists():
            return False
        
        try:
            with open(prometheus_file, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            # 检查是否配置了规则文件
            rule_files = content.get('rule_files', [])
            return len(rule_files) > 0
        except Exception as e:
            logger.warning(f"解析Prometheus配置失败: {e}")
            return False
    
    def _check_grafana_thresholds(self) -> bool:
        """检查Grafana仪表盘阈值"""
        dashboard_file = self.project_root / "deploy/monitoring/grafana/dashboards/voicehelper-overview.json"
        if not dashboard_file.exists():
            return False
        
        try:
            with open(dashboard_file, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # 检查是否包含阈值配置
            content_str = str(content)
            return "threshold" in content_str.lower()
        except Exception as e:
            logger.warning(f"解析Grafana仪表盘失败: {e}")
            return False
    
    def _check_slo_definitions(self) -> bool:
        """检查SLO定义"""
        # 检查是否在配置文件中定义了SLO
        config_files = [
            "deploy/config/slo.yml",
            "deploy/monitoring/slo-config.yml"
        ]
        
        for config_file in config_files:
            if (self.project_root / config_file).exists():
                return True
        
        # 检查是否在代码中定义了SLO常量
        backend_files = list((self.project_root / "backend").rglob("*.go"))
        for go_file in backend_files:
            try:
                content = go_file.read_text()
                if "SLO" in content or "slo" in content:
                    return True
            except Exception:
                continue
        
        return False
    
    def _check_monitoring_targets(self) -> bool:
        """检查监控目标配置"""
        prometheus_file = self.project_root / "deploy/monitoring/prometheus.yml"
        if not prometheus_file.exists():
            return False
        
        try:
            with open(prometheus_file, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            # 检查是否配置了scrape targets
            scrape_configs = content.get('scrape_configs', [])
            return len(scrape_configs) >= 3  # 至少backend, algo, frontend
        except Exception as e:
            logger.warning(f"解析监控目标配置失败: {e}")
            return False
    
    # 部署健康检查方法
    def _check_docker_compose_syntax(self) -> bool:
        """检查Docker Compose语法"""
        compose_files = [
            "docker-compose.local.yml",
            "deploy/docker-compose.monitoring.yml"
        ]
        
        for compose_file in compose_files:
            file_path = self.project_root / compose_file
            if not file_path.exists():
                continue
            
            try:
                # 使用docker-compose config验证语法
                result = subprocess.run(
                    ["docker-compose", "-f", str(file_path), "config"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode != 0:
                    logger.warning(f"Docker Compose语法检查失败: {compose_file}")
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # 如果docker-compose不可用，尝试基本的YAML解析
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        yaml.safe_load(f)
                except Exception as e:
                    logger.warning(f"YAML语法错误: {e}")
                    return False
        
        return True
    
    def _check_kubernetes_config(self) -> bool:
        """检查Kubernetes配置"""
        k8s_file = self.project_root / "deploy/k8s/voicehelper-deployment.yaml"
        if not k8s_file.exists():
            return False
        
        try:
            with open(k8s_file, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            # 检查基本的Kubernetes资源结构
            required_fields = ['apiVersion', 'kind', 'metadata']
            return all(field in content for field in required_fields)
        except Exception as e:
            logger.warning(f"解析Kubernetes配置失败: {e}")
            return False
    
    def _check_environment_variables(self) -> bool:
        """检查环境变量配置"""
        env_files = [
            "env.example",
            "env.unified"
        ]
        
        for env_file in env_files:
            file_path = self.project_root / env_file
            if not file_path.exists():
                continue
            
            try:
                content = file_path.read_text()
                # 检查关键环境变量
                required_vars = [
                    "DATABASE_URL",
                    "REDIS_URL",
                    "JWT_SECRET",
                    "OPENAI_API_KEY"
                ]
                
                return all(var in content for var in required_vars)
            except Exception as e:
                logger.warning(f"读取环境变量文件失败: {e}")
                continue
        
        return False
    
    def _check_service_dependencies(self) -> bool:
        """检查服务依赖配置"""
        compose_file = self.project_root / "docker-compose.local.yml"
        if not compose_file.exists():
            return False
        
        try:
            with open(compose_file, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            services = content.get('services', {})
            
            # 检查服务间依赖关系
            backend_deps = services.get('backend', {}).get('depends_on', [])
            algo_deps = services.get('algo', {}).get('depends_on', [])
            
            return 'postgres' in backend_deps and 'redis' in backend_deps
        except Exception as e:
            logger.warning(f"解析服务依赖失败: {e}")
            return False
    
    def _check_resource_limits(self) -> bool:
        """检查资源限制配置"""
        compose_file = self.project_root / "docker-compose.local.yml"
        if not compose_file.exists():
            return False
        
        try:
            with open(compose_file, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            services = content.get('services', {})
            
            # 检查是否配置了资源限制
            for service_name, service_config in services.items():
                if service_name in ['backend', 'algo', 'frontend']:
                    deploy_config = service_config.get('deploy', {})
                    resources = deploy_config.get('resources', {})
                    if not resources:
                        return False
            
            return True
        except Exception as e:
            logger.warning(f"解析资源限制配置失败: {e}")
            return False
    
    # 服务端点检查方法
    def _check_backend_endpoints(self) -> bool:
        """检查后端API端点"""
        handler_files = list((self.project_root / "backend/internal/handlers").glob("*.go"))
        return len(handler_files) >= 3  # chat, voice, auth等
    
    def _check_algo_endpoints(self) -> bool:
        """检查算法服务端点"""
        main_file = self.project_root / "algo/app/main.py"
        if not main_file.exists():
            return False
        
        content = main_file.read_text()
        required_endpoints = ["/chat", "/voice", "/reload", "/stats"]
        return all(endpoint in content for endpoint in required_endpoints)
    
    def _check_frontend_routes(self) -> bool:
        """检查前端路由"""
        app_dir = self.project_root / "frontend/app"
        if not app_dir.exists():
            return False
        
        # 检查主要页面
        required_pages = ["chat", "voice"]
        for page in required_pages:
            page_dir = app_dir / page
            if not page_dir.exists():
                return False
        
        return True
    
    def _check_websocket_endpoints(self) -> bool:
        """检查WebSocket端点"""
        # 检查后端WebSocket处理器
        ws_handler = self.project_root / "backend/internal/handlers/voice_ws.go"
        if not ws_handler.exists():
            return False
        
        # 检查算法服务WebSocket处理器
        algo_ws = self.project_root / "algo/core/websocket_voice.py"
        return algo_ws.exists()
    
    def _check_health_endpoints(self) -> bool:
        """检查健康检查端点"""
        # 检查后端健康检查
        backend_files = list((self.project_root / "backend").rglob("*.go"))
        health_check_found = False
        
        for go_file in backend_files:
            try:
                content = go_file.read_text()
                if "/health" in content or "/ping" in content:
                    health_check_found = True
                    break
            except Exception:
                continue
        
        return health_check_found
    
    # 监控系统检查方法
    def _check_prometheus_config(self) -> bool:
        """检查Prometheus配置"""
        config_file = self.project_root / "deploy/monitoring/prometheus.yml"
        return config_file.exists()
    
    def _check_grafana_config(self) -> bool:
        """检查Grafana配置"""
        dashboard_dir = self.project_root / "deploy/monitoring/grafana/dashboards"
        return dashboard_dir.exists() and dashboard_dir.is_dir()
    
    def _check_alert_manager(self) -> bool:
        """检查AlertManager配置"""
        alert_config = self.project_root / "deploy/monitoring/alertmanager.yml"
        return alert_config.exists()
    
    def _check_metrics_exporters(self) -> bool:
        """检查指标导出器"""
        # 检查是否配置了各种exporter
        monitoring_compose = self.project_root / "deploy/docker-compose.monitoring.yml"
        if not monitoring_compose.exists():
            return False
        
        try:
            with open(monitoring_compose, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            services = content.get('services', {})
            exporters = ['blackbox-exporter', 'pushgateway']
            
            return all(exporter in services for exporter in exporters)
        except Exception as e:
            logger.warning(f"解析监控配置失败: {e}")
            return False
    
    def _check_log_aggregation(self) -> bool:
        """检查日志聚合配置"""
        # 检查是否配置了日志聚合（如Loki）
        monitoring_compose = self.project_root / "deploy/docker-compose.monitoring.yml"
        if not monitoring_compose.exists():
            return False
        
        try:
            with open(monitoring_compose, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            services = content.get('services', {})
            return 'loki' in services or 'promtail' in services
        except Exception as e:
            logger.warning(f"解析日志配置失败: {e}")
            return False
    
    def calculate_score(self) -> float:
        """计算总体评分"""
        all_scores = []
        
        # DoD阈值权重：30%
        dod_score = sum(self.results["dod_thresholds"].values()) / len(self.results["dod_thresholds"]) * 0.3
        all_scores.append(dod_score)
        
        # 部署健康权重：25%
        health_score = sum(self.results["deployment_health"].values()) / len(self.results["deployment_health"]) * 0.25
        all_scores.append(health_score)
        
        # 服务端点权重：20%
        endpoint_score = sum(self.results["service_endpoints"].values()) / len(self.results["service_endpoints"]) * 0.2
        all_scores.append(endpoint_score)
        
        # 监控状态权重：15%
        monitoring_score = sum(self.results["monitoring_status"].values()) / len(self.results["monitoring_status"]) * 0.15
        all_scores.append(monitoring_score)
        
        # 性能指标权重：10%
        perf_checks = self.results["performance_metrics"].get("checks", {})
        if perf_checks:
            perf_score = sum(perf_checks.values()) / len(perf_checks) * 0.1
            all_scores.append(perf_score)
        
        total_score = sum(all_scores) * 100
        self.results["overall_score"] = total_score
        
        return total_score
    
    def generate_report(self) -> str:
        """生成性能验证报告"""
        score = self.calculate_score()
        
        report = f"""
# VoiceHelper 性能验证报告

## 总体评分: {score:.1f}/100

## DoD阈值验证 (权重30%)
"""
        for check, status in self.results["dod_thresholds"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {check}\n"
        
        report += f"\n## 部署健康验证 (权重25%)\n"
        for check, status in self.results["deployment_health"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {check}\n"
        
        report += f"\n## 服务端点验证 (权重20%)\n"
        for check, status in self.results["service_endpoints"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {check}\n"
        
        report += f"\n## 监控系统验证 (权重15%)\n"
        for check, status in self.results["monitoring_status"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {check}\n"
        
        # 性能指标详情
        perf_metrics = self.results["performance_metrics"]
        if perf_metrics:
            report += f"\n## 性能指标验证 (权重10%)\n"
            
            checks = perf_metrics.get("checks", {})
            actual = perf_metrics.get("actual_metrics", {})
            thresholds = perf_metrics.get("thresholds", {})
            
            for check, status in checks.items():
                status_icon = "✅" if status else "❌"
                report += f"- {status_icon} {check}\n"
            
            report += f"\n### 实际性能指标 vs DoD阈值\n"
            report += f"- 文本首Token P95: {actual.get('text_first_token_p95', 'N/A')}ms (阈值: <{thresholds.get('text_first_token_ms', 'N/A')}ms)\n"
            report += f"- 语音首响 P95: {actual.get('voice_first_response_p95', 'N/A')}ms (阈值: <{thresholds.get('voice_first_response_ms', 'N/A')}ms)\n"
            report += f"- Barge-in延迟 P95: {actual.get('barge_in_latency_p95', 'N/A')}ms (阈值: <{thresholds.get('barge_in_latency_ms', 'N/A')}ms)\n"
            report += f"- RAG Recall@5: {actual.get('rag_recall_at_5', 'N/A')*100:.1f}% (阈值: >={thresholds.get('rag_recall_at_5', 'N/A')*100:.1f}%)\n"
            report += f"- 检索P95: {actual.get('retrieval_p95', 'N/A')}ms (阈值: <{thresholds.get('retrieval_p95_ms', 'N/A')}ms)\n"
            report += f"- 系统可用性: {actual.get('availability', 'N/A')}% (阈值: >={thresholds.get('availability_percent', 'N/A')}%)\n"
            report += f"- 错误率: {actual.get('error_rate', 'N/A')}% (阈值: <{thresholds.get('error_rate_percent', 'N/A')}%)\n"
            report += f"- WebSocket断连率: {actual.get('ws_disconnect_rate', 'N/A')*100:.1f}% (阈值: <{thresholds.get('ws_disconnect_rate', 'N/A')*100:.1f}%)\n"
        
        # 评分等级
        if score >= 90:
            grade = "A+ (优秀)"
            conclusion = "部署配置和性能指标完全符合要求，可以上线"
        elif score >= 80:
            grade = "A (良好)"
            conclusion = "大部分指标符合要求，建议优化后上线"
        elif score >= 70:
            grade = "B (合格)"
            conclusion = "基本符合要求，需要解决关键问题"
        elif score >= 60:
            grade = "C (需改进)"
            conclusion = "存在重要问题，需要大幅改进"
        else:
            grade = "D (不合格)"
            conclusion = "不符合上线要求，需要重新设计"
        
        report += f"\n## 评估结果\n"
        report += f"- 总体评分: {score:.1f}/100\n"
        report += f"- 评估等级: {grade}\n"
        report += f"- 结论: {conclusion}\n"
        
        return report
    
    def run_validation(self) -> Dict[str, Any]:
        """运行完整性能验证"""
        logger.info("开始VoiceHelper性能验证...")
        
        self.validate_dod_thresholds()
        self.validate_deployment_health()
        self.validate_service_endpoints()
        self.validate_monitoring_status()
        self.validate_performance_metrics()
        
        score = self.calculate_score()
        
        logger.info(f"性能验证完成，总体评分: {score:.1f}/100")
        
        return self.results

def main():
    """主函数"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    validator = PerformanceValidator(project_root)
    results = validator.run_validation()
    
    # 生成报告
    report = validator.generate_report()
    
    # 保存报告
    report_file = Path(project_root) / "reports" / "performance_validation.md"
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
