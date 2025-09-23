#!/usr/bin/env python3
"""
VoiceHelper 技术债务监控工具
分析代码库中的技术债务并生成报告
"""

import os
import re
import json
import subprocess
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class TechDebtItem:
    """技术债务项"""
    type: str  # TODO, FIXME, HACK, DEPRECATED, etc.
    file_path: str
    line_number: int
    content: str
    severity: str  # low, medium, high, critical
    category: str  # code_quality, performance, security, maintainability
    age_days: int = 0
    author: str = ""
    created_date: str = ""

@dataclass
class TechDebtMetrics:
    """技术债务指标"""
    total_items: int = 0
    by_type: Dict[str, int] = None
    by_severity: Dict[str, int] = None
    by_category: Dict[str, int] = None
    by_file_extension: Dict[str, int] = None
    average_age_days: float = 0.0
    oldest_item_days: int = 0
    trend_7d: int = 0  # 7天变化
    trend_30d: int = 0  # 30天变化

    def __post_init__(self):
        if self.by_type is None:
            self.by_type = {}
        if self.by_severity is None:
            self.by_severity = {}
        if self.by_category is None:
            self.by_category = {}
        if self.by_file_extension is None:
            self.by_file_extension = {}

class TechDebtMonitor:
    """技术债务监控器"""
    
    # 技术债务关键词及其严重程度
    DEBT_KEYWORDS = {
        'TODO': ('medium', 'maintainability'),
        'FIXME': ('high', 'code_quality'),
        'HACK': ('high', 'code_quality'),
        'XXX': ('high', 'code_quality'),
        'BUG': ('critical', 'code_quality'),
        'DEPRECATED': ('medium', 'maintainability'),
        'TEMP': ('medium', 'maintainability'),
        'TEMPORARY': ('medium', 'maintainability'),
        'WORKAROUND': ('medium', 'code_quality'),
        'KLUDGE': ('high', 'code_quality'),
        'REFACTOR': ('low', 'maintainability'),
        'OPTIMIZE': ('low', 'performance'),
        'PERFORMANCE': ('medium', 'performance'),
        'SECURITY': ('critical', 'security'),
        'VULNERABILITY': ('critical', 'security'),
    }
    
    # 文件扩展名映射
    FILE_EXTENSIONS = {
        '.go': 'Go',
        '.py': 'Python',
        '.js': 'JavaScript',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript',
        '.jsx': 'JavaScript',
        '.java': 'Java',
        '.cpp': 'C++',
        '.c': 'C',
        '.h': 'C/C++',
        '.hpp': 'C++',
        '.cs': 'C#',
        '.php': 'PHP',
        '.rb': 'Ruby',
        '.rs': 'Rust',
        '.kt': 'Kotlin',
        '.swift': 'Swift',
        '.sql': 'SQL',
        '.sh': 'Shell',
        '.yml': 'YAML',
        '.yaml': 'YAML',
        '.json': 'JSON',
        '.xml': 'XML',
        '.md': 'Markdown',
    }
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.debt_items: List[TechDebtItem] = []
        self.metrics = TechDebtMetrics()
        
    def scan_codebase(self) -> List[TechDebtItem]:
        """扫描代码库中的技术债务"""
        print("🔍 扫描技术债务...")
        
        # 忽略的目录和文件
        ignore_patterns = [
            '.git', 'node_modules', 'vendor', 'dist', 'build',
            '__pycache__', '.pytest_cache', 'coverage',
            '*.min.js', '*.min.css', '*.map'
        ]
        
        debt_items = []
        
        for file_path in self._get_source_files(ignore_patterns):
            items = self._scan_file(file_path)
            debt_items.extend(items)
            
        self.debt_items = debt_items
        return debt_items
    
    def _get_source_files(self, ignore_patterns: List[str]) -> List[Path]:
        """获取源代码文件列表"""
        source_files = []
        
        for root, dirs, files in os.walk(self.project_root):
            # 过滤忽略的目录
            dirs[:] = [d for d in dirs if not any(
                d.startswith(pattern.rstrip('*')) for pattern in ignore_patterns
            )]
            
            for file in files:
                file_path = Path(root) / file
                
                # 检查文件扩展名
                if file_path.suffix in self.FILE_EXTENSIONS:
                    # 检查是否匹配忽略模式
                    if not any(file.endswith(pattern.lstrip('*')) for pattern in ignore_patterns if '*' in pattern):
                        source_files.append(file_path)
                        
        return source_files
    
    def _scan_file(self, file_path: Path) -> List[TechDebtItem]:
        """扫描单个文件中的技术债务"""
        debt_items = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            for line_num, line in enumerate(lines, 1):
                items = self._extract_debt_from_line(file_path, line_num, line)
                debt_items.extend(items)
                
        except Exception as e:
            print(f"⚠️ 无法读取文件 {file_path}: {e}")
            
        return debt_items
    
    def _extract_debt_from_line(self, file_path: Path, line_num: int, line: str) -> List[TechDebtItem]:
        """从代码行中提取技术债务"""
        debt_items = []
        line_clean = line.strip()
        
        # 检查注释中的技术债务关键词
        comment_patterns = [
            r'//\s*(.*)',  # C-style comments
            r'#\s*(.*)',   # Python/Shell comments
            r'/\*\s*(.*?)\s*\*/',  # Multi-line C-style comments
            r'<!--\s*(.*?)\s*-->',  # HTML comments
        ]
        
        for pattern in comment_patterns:
            matches = re.finditer(pattern, line_clean, re.IGNORECASE)
            for match in matches:
                comment_text = match.group(1).strip()
                
                for keyword, (severity, category) in self.DEBT_KEYWORDS.items():
                    if keyword.lower() in comment_text.lower():
                        # 获取文件创建时间和作者信息
                        age_days, author, created_date = self._get_file_git_info(file_path, line_num)
                        
                        debt_item = TechDebtItem(
                            type=keyword,
                            file_path=str(file_path.relative_to(self.project_root)),
                            line_number=line_num,
                            content=comment_text,
                            severity=severity,
                            category=category,
                            age_days=age_days,
                            author=author,
                            created_date=created_date
                        )
                        debt_items.append(debt_item)
                        break  # 每行只记录一个债务项
                        
        return debt_items
    
    def _get_file_git_info(self, file_path: Path, line_num: int) -> Tuple[int, str, str]:
        """获取文件的Git信息"""
        try:
            # 获取文件的最后修改时间
            result = subprocess.run([
                'git', 'log', '-1', '--format=%at,%an,%ad',
                '--date=short', str(file_path)
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    timestamp = int(parts[0])
                    author = parts[1]
                    date_str = parts[2]
                    
                    # 计算天数差
                    file_date = datetime.fromtimestamp(timestamp)
                    age_days = (datetime.now() - file_date).days
                    
                    return age_days, author, date_str
                    
        except Exception:
            pass
            
        return 0, "", ""
    
    def calculate_metrics(self) -> TechDebtMetrics:
        """计算技术债务指标"""
        print("📊 计算技术债务指标...")
        
        if not self.debt_items:
            return self.metrics
            
        # 基本统计
        self.metrics.total_items = len(self.debt_items)
        
        # 按类型统计
        self.metrics.by_type = defaultdict(int)
        for item in self.debt_items:
            self.metrics.by_type[item.type] += 1
            
        # 按严重程度统计
        self.metrics.by_severity = defaultdict(int)
        for item in self.debt_items:
            self.metrics.by_severity[item.severity] += 1
            
        # 按类别统计
        self.metrics.by_category = defaultdict(int)
        for item in self.debt_items:
            self.metrics.by_category[item.category] += 1
            
        # 按文件类型统计
        self.metrics.by_file_extension = defaultdict(int)
        for item in self.debt_items:
            ext = Path(item.file_path).suffix
            lang = self.FILE_EXTENSIONS.get(ext, 'Other')
            self.metrics.by_file_extension[lang] += 1
            
        # 计算平均年龄
        if self.debt_items:
            total_age = sum(item.age_days for item in self.debt_items)
            self.metrics.average_age_days = total_age / len(self.debt_items)
            self.metrics.oldest_item_days = max(item.age_days for item in self.debt_items)
            
        # 计算趋势（需要历史数据）
        self.metrics.trend_7d = self._calculate_trend(7)
        self.metrics.trend_30d = self._calculate_trend(30)
        
        return self.metrics
    
    def _calculate_trend(self, days: int) -> int:
        """计算趋势变化"""
        # 这里需要与历史数据比较，暂时返回0
        # 实际实现中可以从数据库或文件中读取历史数据
        return 0
    
    def generate_report(self, output_format: str = 'json') -> str:
        """生成技术债务报告"""
        print("📝 生成技术债务报告...")
        
        report_data = {
            'generated_at': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'metrics': asdict(self.metrics),
            'debt_items': [asdict(item) for item in self.debt_items],
            'summary': self._generate_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        if output_format == 'json':
            return json.dumps(report_data, indent=2, ensure_ascii=False)
        elif output_format == 'markdown':
            return self._generate_markdown_report(report_data)
        else:
            raise ValueError(f"不支持的输出格式: {output_format}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成摘要"""
        return {
            'total_debt_items': self.metrics.total_items,
            'critical_items': self.metrics.by_severity.get('critical', 0),
            'high_priority_items': self.metrics.by_severity.get('high', 0),
            'average_age_days': round(self.metrics.average_age_days, 1),
            'oldest_item_days': self.metrics.oldest_item_days,
            'most_common_type': max(self.metrics.by_type.items(), key=lambda x: x[1])[0] if self.metrics.by_type else 'None',
            'most_affected_language': max(self.metrics.by_file_extension.items(), key=lambda x: x[1])[0] if self.metrics.by_file_extension else 'None'
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于严重程度的建议
        critical_count = self.metrics.by_severity.get('critical', 0)
        if critical_count > 0:
            recommendations.append(f"🚨 立即处理 {critical_count} 个严重技术债务项")
            
        high_count = self.metrics.by_severity.get('high', 0)
        if high_count > 5:
            recommendations.append(f"⚠️ 优先处理 {high_count} 个高优先级技术债务项")
            
        # 基于年龄的建议
        if self.metrics.average_age_days > 90:
            recommendations.append("📅 技术债务平均年龄过长，建议定期清理")
            
        # 基于数量的建议
        if self.metrics.total_items > 100:
            recommendations.append("📈 技术债务数量较多，建议制定清理计划")
            
        # 基于类型的建议
        todo_count = self.metrics.by_type.get('TODO', 0)
        if todo_count > 50:
            recommendations.append("📝 TODO项目过多，建议转化为正式任务")
            
        fixme_count = self.metrics.by_type.get('FIXME', 0)
        if fixme_count > 10:
            recommendations.append("🔧 FIXME项目需要及时修复")
            
        if not recommendations:
            recommendations.append("✅ 技术债务控制良好，继续保持")
            
        return recommendations
    
    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """生成Markdown格式报告"""
        md_content = f"""# VoiceHelper 技术债务报告

**生成时间**: {report_data['generated_at']}  
**项目路径**: {report_data['project_root']}

## 📊 总体概况

- **技术债务总数**: {report_data['summary']['total_debt_items']}
- **严重项目**: {report_data['summary']['critical_items']}
- **高优先级项目**: {report_data['summary']['high_priority_items']}
- **平均年龄**: {report_data['summary']['average_age_days']} 天
- **最老项目**: {report_data['summary']['oldest_item_days']} 天
- **最常见类型**: {report_data['summary']['most_common_type']}
- **最受影响语言**: {report_data['summary']['most_affected_language']}

## 📈 详细指标

### 按类型分布
"""
        
        for debt_type, count in report_data['metrics']['by_type'].items():
            md_content += f"- **{debt_type}**: {count}\n"
            
        md_content += "\n### 按严重程度分布\n"
        for severity, count in report_data['metrics']['by_severity'].items():
            md_content += f"- **{severity}**: {count}\n"
            
        md_content += "\n### 按类别分布\n"
        for category, count in report_data['metrics']['by_category'].items():
            md_content += f"- **{category}**: {count}\n"
            
        md_content += "\n### 按编程语言分布\n"
        for lang, count in report_data['metrics']['by_file_extension'].items():
            md_content += f"- **{lang}**: {count}\n"
            
        md_content += "\n## 🎯 改进建议\n\n"
        for rec in report_data['recommendations']:
            md_content += f"- {rec}\n"
            
        # 添加严重和高优先级项目详情
        critical_items = [item for item in report_data['debt_items'] if item['severity'] == 'critical']
        high_items = [item for item in report_data['debt_items'] if item['severity'] == 'high']
        
        if critical_items:
            md_content += "\n## 🚨 严重技术债务项目\n\n"
            for item in critical_items[:10]:  # 只显示前10个
                md_content += f"- **{item['file_path']}:{item['line_number']}** - {item['content']}\n"
                
        if high_items:
            md_content += "\n## ⚠️ 高优先级技术债务项目\n\n"
            for item in high_items[:10]:  # 只显示前10个
                md_content += f"- **{item['file_path']}:{item['line_number']}** - {item['content']}\n"
                
        md_content += f"\n---\n*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        
        return md_content
    
    def save_report(self, report_content: str, output_path: str):
        """保存报告到文件"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
            
        print(f"📄 报告已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='VoiceHelper 技术债务监控工具')
    parser.add_argument('--project-root', default='.', help='项目根目录路径')
    parser.add_argument('--output-format', choices=['json', 'markdown'], default='markdown', help='输出格式')
    parser.add_argument('--output-file', help='输出文件路径')
    parser.add_argument('--verbose', '-v', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = TechDebtMonitor(args.project_root)
    
    # 扫描代码库
    debt_items = monitor.scan_codebase()
    print(f"✅ 发现 {len(debt_items)} 个技术债务项目")
    
    # 计算指标
    metrics = monitor.calculate_metrics()
    
    # 生成报告
    report_content = monitor.generate_report(args.output_format)
    
    # 保存报告
    if args.output_file:
        monitor.save_report(report_content, args.output_file)
    else:
        # 默认输出路径
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = 'md' if args.output_format == 'markdown' else 'json'
        default_path = f"reports/tech_debt_{timestamp}.{ext}"
        monitor.save_report(report_content, default_path)
    
    # 打印摘要
    if args.verbose:
        print("\n📊 技术债务摘要:")
        summary = monitor._generate_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
            
        print("\n🎯 改进建议:")
        recommendations = monitor._generate_recommendations()
        for rec in recommendations:
            print(f"  {rec}")

if __name__ == '__main__':
    main()
