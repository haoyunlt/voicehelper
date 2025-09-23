#!/usr/bin/env python3
"""
VoiceHelper简化验证脚本
不依赖外部库，仅使用Python标准库
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleValidator:
    """简化验证器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.results = {
            "file_structure": {},
            "configuration": {},
            "deployment": {},
            "overall_score": 0
        }
    
    def validate_file_structure(self) -> Dict[str, bool]:
        """验证文件结构"""
        logger.info("验证文件结构...")
        
        # 关键文件检查
        critical_files = {
            "backend_main": "backend/cmd/server/main.go",
            "algo_main": "algo/app/main.py",
            "frontend_chat": "frontend/components/chat/StreamingChat.tsx",
            "websocket_voice": "algo/core/websocket_voice.py",
            "rag_service": "algo/core/bge_faiss_rag.py",
            "agent_service": "algo/core/langgraph_agent.py",
            "monitoring_config": "deploy/monitoring/prometheus.yml",
            "grafana_dashboard": "deploy/monitoring/grafana/dashboards/voicehelper-overview.json",
            "alert_rules": "deploy/monitoring/rules/enhanced-alerts.yml",
            "docker_compose": "docker-compose.local.yml",
            "k8s_deployment": "deploy/k8s/voicehelper-deployment.yaml"
        }
        
        file_checks = {}
        for name, file_path in critical_files.items():
            full_path = self.project_root / file_path
            file_checks[name] = full_path.exists()
            if not full_path.exists():
                logger.warning(f"缺失关键文件: {file_path}")
        
        self.results["file_structure"] = file_checks
        return file_checks
    
    def validate_configuration(self) -> Dict[str, bool]:
        """验证配置文件"""
        logger.info("验证配置文件...")
        
        config_checks = {
            "environment_config": self._check_env_config(),
            "docker_compose_syntax": self._check_compose_syntax(),
            "monitoring_setup": self._check_monitoring_setup(),
            "service_endpoints": self._check_service_endpoints(),
            "security_config": self._check_security_config()
        }
        
        self.results["configuration"] = config_checks
        return config_checks
    
    def validate_deployment(self) -> Dict[str, bool]:
        """验证部署配置"""
        logger.info("验证部署配置...")
        
        deployment_checks = {
            "docker_images": self._check_docker_images(),
            "service_dependencies": self._check_service_deps(),
            "resource_limits": self._check_resource_limits(),
            "health_checks": self._check_health_checks(),
            "scaling_config": self._check_scaling_config()
        }
        
        self.results["deployment"] = deployment_checks
        return deployment_checks
    
    def _check_env_config(self) -> bool:
        """检查环境配置"""
        env_files = ["env.example", "env.unified"]
        
        for env_file in env_files:
            file_path = self.project_root / env_file
            if file_path.exists():
                try:
                    content = file_path.read_text()
                    required_vars = [
                        "DATABASE_URL", "REDIS_URL", "JWT_SECRET", 
                        "OPENAI_API_KEY", "BACKEND_PORT", "ALGO_PORT"
                    ]
                    
                    missing_vars = [var for var in required_vars if var not in content]
                    if missing_vars:
                        logger.warning(f"缺失环境变量: {missing_vars}")
                        return False
                    return True
                except Exception as e:
                    logger.warning(f"读取环境配置失败: {e}")
                    continue
        
        return False
    
    def _check_compose_syntax(self) -> bool:
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
                content = file_path.read_text()
                # 基本语法检查
                if "version:" not in content or "services:" not in content:
                    logger.warning(f"Docker Compose语法错误: {compose_file}")
                    return False
            except Exception as e:
                logger.warning(f"读取Compose文件失败: {e}")
                return False
        
        return True
    
    def _check_monitoring_setup(self) -> bool:
        """检查监控设置"""
        monitoring_files = [
            "deploy/monitoring/prometheus.yml",
            "deploy/monitoring/grafana/dashboards/voicehelper-overview.json",
            "deploy/monitoring/rules/enhanced-alerts.yml"
        ]
        
        return all((self.project_root / f).exists() for f in monitoring_files)
    
    def _check_service_endpoints(self) -> bool:
        """检查服务端点"""
        # 检查算法服务主文件
        algo_main = self.project_root / "algo/app/main.py"
        if not algo_main.exists():
            return False
        
        try:
            content = algo_main.read_text()
            required_endpoints = ["/chat", "/voice", "/reload", "/stats"]
            return all(endpoint in content for endpoint in required_endpoints)
        except Exception as e:
            logger.warning(f"检查服务端点失败: {e}")
            return False
    
    def _check_security_config(self) -> bool:
        """检查安全配置"""
        # 检查JWT中间件
        auth_files = list((self.project_root / "backend").rglob("*auth*.go"))
        if not auth_files:
            return False
        
        # 检查是否有JWT相关代码
        for auth_file in auth_files:
            try:
                content = auth_file.read_text()
                if "JWT" in content or "jwt" in content:
                    return True
            except Exception:
                continue
        
        return False
    
    def _check_docker_images(self) -> bool:
        """检查Docker镜像配置"""
        dockerfiles = [
            "backend/Dockerfile",
            "algo/Dockerfile",
            "frontend/Dockerfile"
        ]
        
        return all((self.project_root / f).exists() for f in dockerfiles)
    
    def _check_service_deps(self) -> bool:
        """检查服务依赖"""
        compose_file = self.project_root / "docker-compose.local.yml"
        if not compose_file.exists():
            return False
        
        try:
            content = compose_file.read_text()
            # 检查基本服务
            required_services = ["backend", "algo", "frontend", "postgres", "redis"]
            return all(service in content for service in required_services)
        except Exception as e:
            logger.warning(f"检查服务依赖失败: {e}")
            return False
    
    def _check_resource_limits(self) -> bool:
        """检查资源限制"""
        k8s_file = self.project_root / "deploy/k8s/voicehelper-deployment.yaml"
        if not k8s_file.exists():
            return False
        
        try:
            content = k8s_file.read_text()
            # 检查是否配置了资源限制
            return "resources:" in content and ("limits:" in content or "requests:" in content)
        except Exception as e:
            logger.warning(f"检查资源限制失败: {e}")
            return False
    
    def _check_health_checks(self) -> bool:
        """检查健康检查"""
        # 检查后端健康检查端点
        backend_files = list((self.project_root / "backend").rglob("*.go"))
        
        for go_file in backend_files:
            try:
                content = go_file.read_text()
                if "/health" in content or "/ping" in content:
                    return True
            except Exception:
                continue
        
        return False
    
    def _check_scaling_config(self) -> bool:
        """检查扩缩容配置"""
        k8s_file = self.project_root / "deploy/k8s/voicehelper-deployment.yaml"
        if not k8s_file.exists():
            return False
        
        try:
            content = k8s_file.read_text()
            # 检查是否配置了副本数
            return "replicas:" in content
        except Exception as e:
            logger.warning(f"检查扩缩容配置失败: {e}")
            return False
    
    def calculate_score(self) -> float:
        """计算总体评分"""
        all_checks = []
        
        # 文件结构权重：40%
        file_score = sum(self.results["file_structure"].values()) / len(self.results["file_structure"]) * 0.4
        all_checks.extend(self.results["file_structure"].values())
        
        # 配置文件权重：35%
        config_score = sum(self.results["configuration"].values()) / len(self.results["configuration"]) * 0.35
        all_checks.extend(self.results["configuration"].values())
        
        # 部署配置权重：25%
        deploy_score = sum(self.results["deployment"].values()) / len(self.results["deployment"]) * 0.25
        all_checks.extend(self.results["deployment"].values())
        
        total_score = (file_score + config_score + deploy_score) * 100
        self.results["overall_score"] = total_score
        
        return total_score
    
    def generate_report(self) -> str:
        """生成验证报告"""
        score = self.calculate_score()
        
        report = f"""
# VoiceHelper 部署验证报告

## 总体评分: {score:.1f}/100

## 文件结构验证 (权重40%)
"""
        for check, status in self.results["file_structure"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {check}\n"
        
        report += f"\n## 配置文件验证 (权重35%)\n"
        for check, status in self.results["configuration"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {check}\n"
        
        report += f"\n## 部署配置验证 (权重25%)\n"
        for check, status in self.results["deployment"].items():
            status_icon = "✅" if status else "❌"
            report += f"- {status_icon} {check}\n"
        
        # 评分等级和建议
        if score >= 90:
            grade = "A+ (优秀)"
            recommendation = "所有关键组件已就绪，可以开始部署"
        elif score >= 80:
            grade = "A (良好)"
            recommendation = "大部分组件已就绪，建议完善后部署"
        elif score >= 70:
            grade = "B (合格)"
            recommendation = "基本组件已就绪，需要解决关键问题"
        elif score >= 60:
            grade = "C (需改进)"
            recommendation = "存在重要缺失，需要补充关键组件"
        else:
            grade = "D (不合格)"
            recommendation = "缺失太多关键组件，需要重新检查"
        
        report += f"\n## 评估结果\n"
        report += f"- 总体评分: {score:.1f}/100\n"
        report += f"- 评估等级: {grade}\n"
        report += f"- 建议: {recommendation}\n"
        
        # DoD阈值提醒
        report += f"\n## DoD性能阈值提醒\n"
        report += f"请确保生产环境满足以下性能要求：\n"
        report += f"- 文本首Token P95 < 800ms\n"
        report += f"- 语音首响 P95 < 700ms\n"
        report += f"- Barge-in延迟 P95 < 150ms\n"
        report += f"- RAG Recall@5 >= 85%\n"
        report += f"- 检索P95 < 200ms\n"
        report += f"- 系统可用性 >= 99.9%\n"
        report += f"- 错误率 < 1%\n"
        report += f"- WebSocket断连率 < 5%\n"
        
        return report
    
    def run_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        logger.info("开始VoiceHelper部署验证...")
        
        self.validate_file_structure()
        self.validate_configuration()
        self.validate_deployment()
        
        score = self.calculate_score()
        
        logger.info(f"部署验证完成，总体评分: {score:.1f}/100")
        
        return self.results

def main():
    """主函数"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    validator = SimpleValidator(project_root)
    results = validator.run_validation()
    
    # 生成报告
    report = validator.generate_report()
    
    # 保存报告
    report_file = Path(project_root) / "reports" / "deployment_validation.md"
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
