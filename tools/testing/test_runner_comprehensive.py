"""
综合测试运行器
统一管理和执行所有类型的测试用例，生成详细的测试报告
"""

import pytest
import asyncio
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import argparse


@dataclass
class TestSuiteResult:
    """测试套件结果"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    success_rate: float
    error_messages: List[str]
    performance_metrics: Dict[str, float]


@dataclass
class ComprehensiveTestReport:
    """综合测试报告"""
    execution_timestamp: float
    total_execution_time: float
    overall_success: bool
    suite_results: List[TestSuiteResult]
    summary_metrics: Dict[str, Any]
    recommendations: List[str]


class ComprehensiveTestRunner:
    """综合测试运行器"""
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.test_suites = {
            "security": {
                "path": "tools/testing/unit/security/test_security.py",
                "description": "安全测试套件",
                "timeout": 300,
                "critical": True
            },
            "error_handling": {
                "path": "tools/testing/unit/error_handling/test_error_handling.py", 
                "description": "异常处理测试套件",
                "timeout": 180,
                "critical": True
            },
            "voice_processing": {
                "path": "tools/testing/unit/voice/test_voice_processing.py",
                "description": "语音处理测试套件", 
                "timeout": 240,
                "critical": False
            },
            "multimodal_fusion": {
                "path": "tools/testing/unit/multimodal/test_multimodal_fusion.py",
                "description": "多模态融合测试套件",
                "timeout": 300,
                "critical": False
            },
            "service_integration": {
                "path": "tools/testing/integration/test_service_integration.py",
                "description": "服务集成测试套件",
                "timeout": 360,
                "critical": True
            },
            "performance": {
                "path": "tools/testing/performance/test_comprehensive_performance.py",
                "description": "性能测试套件",
                "timeout": 600,
                "critical": False
            },
            "business_workflows": {
                "path": "tools/testing/e2e/test_business_workflows.py",
                "description": "端到端业务流程测试套件",
                "timeout": 480,
                "critical": True
            }
        }
        
        self.results = []
        self.start_time = None
        self.end_time = None
    
    def setup_test_environment(self):
        """设置测试环境"""
        print("设置测试环境...")
        
        # 检查Python环境
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            raise RuntimeError(f"需要Python 3.8+，当前版本: {python_version.major}.{python_version.minor}")
        
        # 检查必要的包
        required_packages = [
            "pytest", "pytest-asyncio", "pytest-cov", "aiohttp", 
            "numpy", "psutil", "cryptography"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"警告: 缺少以下包: {missing_packages}")
            print("请运行: pip install -r requirements-test.txt")
        
        # 创建测试报告目录
        report_dir = self.project_root / "reports" / "testing"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        print("测试环境设置完成")
    
    def run_test_suite(self, suite_name: str, suite_config: Dict[str, Any]) -> TestSuiteResult:
        """运行单个测试套件"""
        print(f"\n{'='*60}")
        print(f"运行测试套件: {suite_config['description']}")
        print(f"路径: {suite_config['path']}")
        print(f"{'='*60}")
        
        suite_start_time = time.time()
        test_file_path = self.project_root / suite_config["path"]
        
        if not test_file_path.exists():
            print(f"警告: 测试文件不存在: {test_file_path}")
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                execution_time=0,
                success_rate=0,
                error_messages=[f"测试文件不存在: {test_file_path}"],
                performance_metrics={}
            )
        
        try:
            # 构建pytest命令
            pytest_args = [
                str(test_file_path),
                "-v",
                "--tb=short",
                "--timeout=" + str(suite_config["timeout"]),
                "--json-report",
                "--json-report-file=" + str(self.project_root / "reports" / "testing" / f"{suite_name}_report.json")
            ]
            
            # 运行pytest
            result = subprocess.run(
                [sys.executable, "-m", "pytest"] + pytest_args,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=suite_config["timeout"]
            )
            
            execution_time = time.time() - suite_start_time
            
            # 解析pytest输出
            total_tests = 0
            passed_tests = 0
            failed_tests = 0
            skipped_tests = 0
            error_messages = []
            
            # 尝试从JSON报告解析结果
            json_report_path = self.project_root / "reports" / "testing" / f"{suite_name}_report.json"
            if json_report_path.exists():
                try:
                    with open(json_report_path, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    summary = report_data.get("summary", {})
                    total_tests = summary.get("total", 0)
                    passed_tests = summary.get("passed", 0)
                    failed_tests = summary.get("failed", 0)
                    skipped_tests = summary.get("skipped", 0)
                    
                    # 提取错误信息
                    for test in report_data.get("tests", []):
                        if test.get("outcome") == "failed":
                            error_messages.append(f"{test.get('nodeid', 'Unknown')}: {test.get('call', {}).get('longrepr', 'Unknown error')}")
                
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"解析JSON报告失败: {e}")
            
            # 如果JSON解析失败，从stdout解析
            if total_tests == 0:
                stdout_lines = result.stdout.split('\n')
                for line in stdout_lines:
                    if "failed" in line and "passed" in line:
                        # 解析类似 "5 failed, 10 passed in 2.34s" 的行
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "failed" and i > 0:
                                failed_tests = int(parts[i-1])
                            elif part == "passed" and i > 0:
                                passed_tests = int(parts[i-1])
                            elif part == "skipped" and i > 0:
                                skipped_tests = int(parts[i-1])
                
                total_tests = passed_tests + failed_tests + skipped_tests
            
            # 如果仍然没有解析到结果，使用返回码判断
            if total_tests == 0:
                if result.returncode == 0:
                    total_tests = 1
                    passed_tests = 1
                else:
                    total_tests = 1
                    failed_tests = 1
                    error_messages.append(f"测试执行失败，返回码: {result.returncode}")
            
            # 添加stderr中的错误信息
            if result.stderr:
                error_messages.append(f"stderr: {result.stderr}")
            
            success_rate = passed_tests / total_tests if total_tests > 0 else 0
            
            # 提取性能指标
            performance_metrics = {
                "execution_time": execution_time,
                "tests_per_second": total_tests / execution_time if execution_time > 0 else 0
            }
            
            print(f"测试套件 {suite_name} 完成:")
            print(f"  总测试数: {total_tests}")
            print(f"  通过: {passed_tests}")
            print(f"  失败: {failed_tests}")
            print(f"  跳过: {skipped_tests}")
            print(f"  成功率: {success_rate:.2%}")
            print(f"  执行时间: {execution_time:.2f}s")
            
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=total_tests,
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=skipped_tests,
                execution_time=execution_time,
                success_rate=success_rate,
                error_messages=error_messages,
                performance_metrics=performance_metrics
            )
        
        except subprocess.TimeoutExpired:
            execution_time = time.time() - suite_start_time
            error_msg = f"测试套件 {suite_name} 超时 ({suite_config['timeout']}s)"
            print(f"错误: {error_msg}")
            
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                execution_time=execution_time,
                success_rate=0,
                error_messages=[error_msg],
                performance_metrics={}
            )
        
        except Exception as e:
            execution_time = time.time() - suite_start_time
            error_msg = f"测试套件 {suite_name} 执行异常: {str(e)}"
            print(f"错误: {error_msg}")
            
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                execution_time=execution_time,
                success_rate=0,
                error_messages=[error_msg],
                performance_metrics={}
            )
    
    def run_all_tests(self, selected_suites: List[str] = None, 
                     skip_non_critical: bool = False) -> ComprehensiveTestReport:
        """运行所有测试套件"""
        print("开始综合测试执行...")
        self.start_time = time.time()
        
        # 设置测试环境
        self.setup_test_environment()
        
        # 确定要运行的测试套件
        suites_to_run = selected_suites if selected_suites else list(self.test_suites.keys())
        
        if skip_non_critical:
            suites_to_run = [
                suite for suite in suites_to_run 
                if self.test_suites[suite].get("critical", False)
            ]
        
        print(f"将运行以下测试套件: {suites_to_run}")
        
        # 运行每个测试套件
        suite_results = []
        for suite_name in suites_to_run:
            if suite_name not in self.test_suites:
                print(f"警告: 未知的测试套件: {suite_name}")
                continue
            
            suite_config = self.test_suites[suite_name]
            result = self.run_test_suite(suite_name, suite_config)
            suite_results.append(result)
        
        self.end_time = time.time()
        total_execution_time = self.end_time - self.start_time
        
        # 计算总体指标
        total_tests = sum(r.total_tests for r in suite_results)
        total_passed = sum(r.passed_tests for r in suite_results)
        total_failed = sum(r.failed_tests for r in suite_results)
        total_skipped = sum(r.skipped_tests for r in suite_results)
        
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        overall_success = total_failed == 0
        
        # 检查关键测试套件
        critical_failures = [
            r for r in suite_results 
            if r.failed_tests > 0 and self.test_suites[r.suite_name].get("critical", False)
        ]
        
        if critical_failures:
            overall_success = False
        
        # 生成汇总指标
        summary_metrics = {
            "total_suites": len(suite_results),
            "successful_suites": len([r for r in suite_results if r.failed_tests == 0]),
            "failed_suites": len([r for r in suite_results if r.failed_tests > 0]),
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_skipped": total_skipped,
            "overall_success_rate": overall_success_rate,
            "average_execution_time": sum(r.execution_time for r in suite_results) / len(suite_results) if suite_results else 0,
            "critical_failures": len(critical_failures)
        }
        
        # 生成建议
        recommendations = self._generate_recommendations(suite_results, summary_metrics)
        
        # 创建综合报告
        report = ComprehensiveTestReport(
            execution_timestamp=self.start_time,
            total_execution_time=total_execution_time,
            overall_success=overall_success,
            suite_results=suite_results,
            summary_metrics=summary_metrics,
            recommendations=recommendations
        )
        
        return report
    
    def _generate_recommendations(self, suite_results: List[TestSuiteResult], 
                                summary_metrics: Dict[str, Any]) -> List[str]:
        """生成测试建议"""
        recommendations = []
        
        # 检查成功率
        if summary_metrics["overall_success_rate"] < 0.9:
            recommendations.append(
                f"整体测试成功率较低 ({summary_metrics['overall_success_rate']:.1%})，"
                "建议优先修复失败的测试用例"
            )
        
        # 检查关键测试失败
        if summary_metrics["critical_failures"] > 0:
            recommendations.append(
                f"有 {summary_metrics['critical_failures']} 个关键测试套件失败，"
                "这些问题可能影响系统核心功能，建议立即修复"
            )
        
        # 检查执行时间
        slow_suites = [r for r in suite_results if r.execution_time > 300]  # 5分钟
        if slow_suites:
            suite_names = [r.suite_name for r in slow_suites]
            recommendations.append(
                f"以下测试套件执行时间较长: {suite_names}，"
                "建议优化测试用例或考虑并行化执行"
            )
        
        # 检查跳过的测试
        total_skipped = summary_metrics["total_skipped"]
        if total_skipped > 0:
            recommendations.append(
                f"有 {total_skipped} 个测试被跳过，"
                "建议检查跳过原因并完善测试覆盖"
            )
        
        # 检查测试覆盖
        if summary_metrics["total_tests"] < 50:
            recommendations.append(
                "测试用例数量较少，建议增加更多测试用例以提高代码覆盖率"
            )
        
        # 性能建议
        avg_time = summary_metrics["average_execution_time"]
        if avg_time > 180:  # 3分钟
            recommendations.append(
                f"平均测试套件执行时间较长 ({avg_time:.1f}s)，"
                "建议优化测试性能或增加并行化"
            )
        
        return recommendations
    
    def generate_html_report(self, report: ComprehensiveTestReport, output_path: str = None):
        """生成HTML测试报告"""
        if not output_path:
            output_path = self.project_root / "reports" / "testing" / "comprehensive_test_report.html"
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoiceHelper 综合测试报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
        .suite-result {{ margin-bottom: 20px; padding: 20px; border-left: 4px solid #3498db; background: #f8f9fa; }}
        .suite-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
        .suite-name {{ font-size: 1.2em; font-weight: bold; }}
        .suite-status {{ padding: 5px 10px; border-radius: 15px; color: white; font-size: 0.9em; }}
        .status-success {{ background-color: #27ae60; }}
        .status-failed {{ background-color: #e74c3c; }}
        .suite-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; margin-top: 15px; }}
        .suite-metric {{ text-align: center; }}
        .recommendations {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 20px; margin-top: 30px; }}
        .recommendation {{ margin-bottom: 10px; padding: 10px; background: white; border-radius: 5px; }}
        .timestamp {{ text-align: center; color: #7f8c8d; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>VoiceHelper 综合测试报告</h1>
        
        <div class="summary">
            <div class="metric-card">
                <div class="metric-value {'success' if report.overall_success else 'error'}">
                    {'✓' if report.overall_success else '✗'}
                </div>
                <div class="metric-label">总体状态</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.summary_metrics['total_tests']}</div>
                <div class="metric-label">总测试数</div>
            </div>
            <div class="metric-card">
                <div class="metric-value success">{report.summary_metrics['total_passed']}</div>
                <div class="metric-label">通过测试</div>
            </div>
            <div class="metric-card">
                <div class="metric-value error">{report.summary_metrics['total_failed']}</div>
                <div class="metric-label">失败测试</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.summary_metrics['overall_success_rate']:.1%}</div>
                <div class="metric-label">成功率</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{report.total_execution_time:.1f}s</div>
                <div class="metric-label">总执行时间</div>
            </div>
        </div>
        
        <h2>测试套件详情</h2>
        """
        
        for suite_result in report.suite_results:
            status_class = "status-success" if suite_result.failed_tests == 0 else "status-failed"
            status_text = "通过" if suite_result.failed_tests == 0 else "失败"
            
            html_content += f"""
        <div class="suite-result">
            <div class="suite-header">
                <div class="suite-name">{suite_result.suite_name}</div>
                <div class="suite-status {status_class}">{status_text}</div>
            </div>
            <div class="suite-metrics">
                <div class="suite-metric">
                    <strong>{suite_result.total_tests}</strong><br>总测试数
                </div>
                <div class="suite-metric">
                    <strong class="success">{suite_result.passed_tests}</strong><br>通过
                </div>
                <div class="suite-metric">
                    <strong class="error">{suite_result.failed_tests}</strong><br>失败
                </div>
                <div class="suite-metric">
                    <strong>{suite_result.skipped_tests}</strong><br>跳过
                </div>
                <div class="suite-metric">
                    <strong>{suite_result.success_rate:.1%}</strong><br>成功率
                </div>
                <div class="suite-metric">
                    <strong>{suite_result.execution_time:.1f}s</strong><br>执行时间
                </div>
            </div>
            """
            
            if suite_result.error_messages:
                html_content += "<div style='margin-top: 15px;'><strong>错误信息:</strong><ul>"
                for error in suite_result.error_messages[:5]:  # 只显示前5个错误
                    html_content += f"<li style='color: #e74c3c; margin: 5px 0;'>{error}</li>"
                html_content += "</ul></div>"
            
            html_content += "</div>"
        
        if report.recommendations:
            html_content += """
        <div class="recommendations">
            <h2>改进建议</h2>
            """
            for i, recommendation in enumerate(report.recommendations, 1):
                html_content += f"""
            <div class="recommendation">
                <strong>{i}.</strong> {recommendation}
            </div>
                """
            html_content += "</div>"
        
        html_content += f"""
        <div class="timestamp">
            报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.execution_timestamp))}
        </div>
    </div>
</body>
</html>
        """
        
        # 写入HTML文件
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML报告已生成: {output_path}")
    
    def save_json_report(self, report: ComprehensiveTestReport, output_path: str = None):
        """保存JSON格式的测试报告"""
        if not output_path:
            output_path = self.project_root / "reports" / "testing" / "comprehensive_test_report.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 转换为可序列化的字典
        report_dict = asdict(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"JSON报告已保存: {output_path}")
    
    def print_summary(self, report: ComprehensiveTestReport):
        """打印测试总结"""
        print(f"\n{'='*80}")
        print("VoiceHelper 综合测试总结")
        print(f"{'='*80}")
        
        print(f"执行时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.execution_timestamp))}")
        print(f"总执行时间: {report.total_execution_time:.2f}s")
        print(f"总体状态: {'✓ 成功' if report.overall_success else '✗ 失败'}")
        
        print(f"\n测试统计:")
        print(f"  测试套件: {report.summary_metrics['total_suites']} 个")
        print(f"  成功套件: {report.summary_metrics['successful_suites']} 个")
        print(f"  失败套件: {report.summary_metrics['failed_suites']} 个")
        print(f"  总测试数: {report.summary_metrics['total_tests']}")
        print(f"  通过测试: {report.summary_metrics['total_passed']}")
        print(f"  失败测试: {report.summary_metrics['total_failed']}")
        print(f"  跳过测试: {report.summary_metrics['total_skipped']}")
        print(f"  成功率: {report.summary_metrics['overall_success_rate']:.1%}")
        
        print(f"\n各套件详情:")
        for suite_result in report.suite_results:
            status = "✓" if suite_result.failed_tests == 0 else "✗"
            print(f"  {status} {suite_result.suite_name}: "
                  f"{suite_result.passed_tests}/{suite_result.total_tests} 通过 "
                  f"({suite_result.success_rate:.1%}) - {suite_result.execution_time:.1f}s")
        
        if report.recommendations:
            print(f"\n改进建议:")
            for i, recommendation in enumerate(report.recommendations, 1):
                print(f"  {i}. {recommendation}")
        
        print(f"\n{'='*80}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VoiceHelper 综合测试运行器")
    parser.add_argument("--suites", nargs="+", 
                       help="指定要运行的测试套件",
                       choices=["security", "error_handling", "voice_processing", 
                               "multimodal_fusion", "service_integration", 
                               "performance", "business_workflows"])
    parser.add_argument("--critical-only", action="store_true",
                       help="只运行关键测试套件")
    parser.add_argument("--output-dir", default="reports/testing",
                       help="输出目录")
    parser.add_argument("--no-html", action="store_true",
                       help="不生成HTML报告")
    parser.add_argument("--no-json", action="store_true", 
                       help="不生成JSON报告")
    
    args = parser.parse_args()
    
    # 创建测试运行器
    runner = ComprehensiveTestRunner()
    
    try:
        # 运行测试
        report = runner.run_all_tests(
            selected_suites=args.suites,
            skip_non_critical=args.critical_only
        )
        
        # 打印总结
        runner.print_summary(report)
        
        # 生成报告
        output_dir = Path(args.output_dir)
        
        if not args.no_html:
            html_path = output_dir / "comprehensive_test_report.html"
            runner.generate_html_report(report, html_path)
        
        if not args.no_json:
            json_path = output_dir / "comprehensive_test_report.json"
            runner.save_json_report(report, json_path)
        
        # 设置退出码
        sys.exit(0 if report.overall_success else 1)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        sys.exit(130)
    except Exception as e:
        print(f"测试运行器异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
