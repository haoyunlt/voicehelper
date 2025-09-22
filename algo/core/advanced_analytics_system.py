"""
VoiceHelper v1.22.0 - 高级分析系统
实现用户行为分析、性能分析、业务智能分析
"""

import asyncio
import time
import logging
import json
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

class AnalysisType(Enum):
    """分析类型"""
    USER_BEHAVIOR = "user_behavior"
    PERFORMANCE = "performance"
    BUSINESS_INTELLIGENCE = "business_intelligence"
    PREDICTIVE = "predictive"
    REAL_TIME = "real_time"

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"

@dataclass
class Metric:
    """指标"""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AnalysisResult:
    """分析结果"""
    analysis_id: str
    analysis_type: AnalysisType
    title: str
    description: str
    data: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    confidence: float
    created_at: float = field(default_factory=time.time)

@dataclass
class UserBehavior:
    """用户行为"""
    user_id: str
    action: str
    timestamp: float
    session_id: str
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics = deque(maxlen=max_metrics)
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
        
    def record_metric(self, metric: Metric):
        """记录指标"""
        self.metrics.append(metric)
        
        # 根据指标类型更新相应存储
        if metric.metric_type == MetricType.COUNTER:
            self.counters[metric.name] += metric.value
        elif metric.metric_type == MetricType.GAUGE:
            self.gauges[metric.name] = metric.value
        elif metric.metric_type == MetricType.HISTOGRAM:
            self.histograms[metric.name].append(metric.value)
        elif metric.metric_type == MetricType.TIMER:
            self.timers[metric.name].append(metric.value)
        
        logger.debug(f"Recorded metric: {metric.name} = {metric.value}")
    
    def get_metric_summary(self, name: str) -> Dict[str, Any]:
        """获取指标摘要"""
        if name in self.counters:
            return {
                "type": "counter",
                "value": self.counters[name],
                "count": len([m for m in self.metrics if m.name == name])
            }
        elif name in self.gauges:
            return {
                "type": "gauge",
                "value": self.gauges[name],
                "count": len([m for m in self.metrics if m.name == name])
            }
        elif name in self.histograms:
            values = self.histograms[name]
            return {
                "type": "histogram",
                "count": len(values),
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "mean": statistics.mean(values) if values else 0,
                "median": statistics.median(values) if values else 0,
                "p95": self._percentile(values, 95) if values else 0,
                "p99": self._percentile(values, 99) if values else 0
            }
        elif name in self.timers:
            values = self.timers[name]
            return {
                "type": "timer",
                "count": len(values),
                "total": sum(values),
                "min": min(values) if values else 0,
                "max": max(values) if values else 0,
                "mean": statistics.mean(values) if values else 0,
                "median": statistics.median(values) if values else 0
            }
        else:
            return {"type": "unknown", "value": 0, "count": 0}
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        return {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {name: self.get_metric_summary(name) for name in self.histograms},
            "timers": {name: self.get_metric_summary(name) for name in self.timers},
            "total_metrics": len(self.metrics)
        }

class UserBehaviorAnalyzer:
    """用户行为分析器"""
    
    def __init__(self):
        self.user_behaviors = defaultdict(list)
        self.session_data = defaultdict(dict)
        self.user_sessions = defaultdict(list)
        
    def record_behavior(self, behavior: UserBehavior):
        """记录用户行为"""
        self.user_behaviors[behavior.user_id].append(behavior)
        self.session_data[behavior.session_id][behavior.timestamp] = behavior
        
        # 更新用户会话
        if behavior.session_id not in self.user_sessions[behavior.user_id]:
            self.user_sessions[behavior.user_id].append(behavior.session_id)
        
        logger.debug(f"Recorded behavior: {behavior.user_id} - {behavior.action}")
    
    def analyze_user_patterns(self, user_id: str) -> Dict[str, Any]:
        """分析用户模式"""
        if user_id not in self.user_behaviors:
            return {"error": "User not found"}
        
        behaviors = self.user_behaviors[user_id]
        
        # 分析行为频率
        action_counts = defaultdict(int)
        for behavior in behaviors:
            action_counts[behavior.action] += 1
        
        # 分析会话模式
        sessions = self.user_sessions[user_id]
        session_durations = []
        for session_id in sessions:
            session_behaviors = [b for b in behaviors if b.session_id == session_id]
            if session_behaviors:
                duration = max(b.timestamp for b in session_behaviors) - min(b.timestamp for b in session_behaviors)
                session_durations.append(duration)
        
        # 分析时间模式
        hourly_activity = defaultdict(int)
        for behavior in behaviors:
            hour = datetime.fromtimestamp(behavior.timestamp).hour
            hourly_activity[hour] += 1
        
        return {
            "user_id": user_id,
            "total_behaviors": len(behaviors),
            "unique_sessions": len(sessions),
            "action_frequency": dict(action_counts),
            "avg_session_duration": statistics.mean(session_durations) if session_durations else 0,
            "hourly_activity": dict(hourly_activity),
            "most_active_hour": max(hourly_activity.items(), key=lambda x: x[1])[0] if hourly_activity else 0
        }
    
    def get_user_insights(self, user_id: str) -> List[str]:
        """获取用户洞察"""
        patterns = self.analyze_user_patterns(user_id)
        insights = []
        
        if patterns.get("total_behaviors", 0) > 100:
            insights.append("高活跃用户")
        
        if patterns.get("avg_session_duration", 0) > 3600:  # 1小时
            insights.append("长时间会话用户")
        
        most_active_hour = patterns.get("most_active_hour", 0)
        if 9 <= most_active_hour <= 17:
            insights.append("工作时间活跃用户")
        elif 18 <= most_active_hour <= 23:
            insights.append("晚间活跃用户")
        else:
            insights.append("非典型时间活跃用户")
        
        return insights

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.performance_metrics = defaultdict(list)
        self.alert_thresholds = {
            "response_time": 1000,  # 1秒
            "error_rate": 0.05,     # 5%
            "cpu_usage": 0.8,       # 80%
            "memory_usage": 0.8     # 80%
        }
    
    def record_performance_metric(self, metric_name: str, value: float, timestamp: float = None):
        """记录性能指标"""
        if timestamp is None:
            timestamp = time.time()
        
        self.performance_metrics[metric_name].append({
            "value": value,
            "timestamp": timestamp
        })
        
        # 检查告警阈值
        self._check_alerts(metric_name, value)
    
    def _check_alerts(self, metric_name: str, value: float):
        """检查告警"""
        if metric_name in self.alert_thresholds:
            threshold = self.alert_thresholds[metric_name]
            if value > threshold:
                logger.warning(f"Performance alert: {metric_name} = {value} > {threshold}")
    
    def analyze_performance_trends(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """分析性能趋势"""
        if metric_name not in self.performance_metrics:
            return {"error": "Metric not found"}
        
        metrics = self.performance_metrics[metric_name]
        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in metrics if m["timestamp"] >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No recent data"}
        
        values = [m["value"] for m in recent_metrics]
        
        return {
            "metric_name": metric_name,
            "time_range_hours": hours,
            "data_points": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
            "trend": self._calculate_trend(values)
        }
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return "insufficient_data"
        
        # 简单线性回归
        n = len(values)
        x = list(range(n))
        y = values
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "no_trend"
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

class BusinessIntelligenceAnalyzer:
    """商业智能分析器"""
    
    def __init__(self):
        self.business_metrics = defaultdict(list)
        self.kpi_definitions = {
            "user_retention": "用户留存率",
            "conversion_rate": "转化率",
            "revenue_per_user": "每用户收入",
            "customer_satisfaction": "客户满意度"
        }
    
    def record_business_metric(self, metric_name: str, value: float, date: str = None):
        """记录商业指标"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        self.business_metrics[metric_name].append({
            "value": value,
            "date": date,
            "timestamp": time.time()
        })
    
    def analyze_kpi_trends(self, kpi_name: str, days: int = 30) -> Dict[str, Any]:
        """分析KPI趋势"""
        if kpi_name not in self.business_metrics:
            return {"error": "KPI not found"}
        
        metrics = self.business_metrics[kpi_name]
        cutoff_time = time.time() - (days * 24 * 3600)
        recent_metrics = [m for m in metrics if m["timestamp"] >= cutoff_time]
        
        if not recent_metrics:
            return {"error": "No recent data"}
        
        values = [m["value"] for m in recent_metrics]
        dates = [m["date"] for m in recent_metrics]
        
        return {
            "kpi_name": kpi_name,
            "kpi_description": self.kpi_definitions.get(kpi_name, ""),
            "time_range_days": days,
            "data_points": len(values),
            "current_value": values[-1] if values else 0,
            "avg_value": statistics.mean(values),
            "min_value": min(values),
            "max_value": max(values),
            "trend": self._calculate_trend(values),
            "growth_rate": self._calculate_growth_rate(values),
            "dates": dates,
            "values": values
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势"""
        if len(values) < 2:
            return "insufficient_data"
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg * 1.05:
            return "increasing"
        elif second_avg < first_avg * 0.95:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """计算增长率"""
        if len(values) < 2:
            return 0.0
        
        first_value = values[0]
        last_value = values[-1]
        
        if first_value == 0:
            return 0.0
        
        return ((last_value - first_value) / first_value) * 100

class AdvancedAnalyticsSystem:
    """高级分析系统"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.user_analyzer = UserBehaviorAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.bi_analyzer = BusinessIntelligenceAnalyzer()
        self.analysis_results = {}
        
    async def generate_analysis(self, analysis_type: AnalysisType, 
                              parameters: Dict[str, Any] = None) -> AnalysisResult:
        """生成分析报告"""
        analysis_id = str(uuid.uuid4())
        parameters = parameters or {}
        
        try:
            if analysis_type == AnalysisType.USER_BEHAVIOR:
                result = await self._analyze_user_behavior(parameters)
            elif analysis_type == AnalysisType.PERFORMANCE:
                result = await self._analyze_performance(parameters)
            elif analysis_type == AnalysisType.BUSINESS_INTELLIGENCE:
                result = await self._analyze_business_intelligence(parameters)
            elif analysis_type == AnalysisType.PREDICTIVE:
                result = await self._analyze_predictive(parameters)
            elif analysis_type == AnalysisType.REAL_TIME:
                result = await self._analyze_real_time(parameters)
            else:
                result = {"error": "Unknown analysis type"}
            
            analysis_result = AnalysisResult(
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                title=f"{analysis_type.value.title()} Analysis",
                description=f"Generated analysis for {analysis_type.value}",
                data=result,
                insights=self._generate_insights(result, analysis_type),
                recommendations=self._generate_recommendations(result, analysis_type),
                confidence=0.8
            )
            
            self.analysis_results[analysis_id] = analysis_result
            return analysis_result
            
        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            return AnalysisResult(
                analysis_id=analysis_id,
                analysis_type=analysis_type,
                title="Analysis Failed",
                description=f"Failed to generate {analysis_type.value} analysis",
                data={"error": str(e)},
                insights=[],
                recommendations=[],
                confidence=0.0
            )
    
    async def _analyze_user_behavior(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """分析用户行为"""
        user_id = parameters.get("user_id")
        if not user_id:
            return {"error": "User ID required"}
        
        patterns = self.user_analyzer.analyze_user_patterns(user_id)
        insights = self.user_analyzer.get_user_insights(user_id)
        
        return {
            "user_patterns": patterns,
            "insights": insights,
            "analysis_timestamp": time.time()
        }
    
    async def _analyze_performance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能"""
        metric_name = parameters.get("metric_name", "response_time")
        hours = parameters.get("hours", 24)
        
        trend_analysis = self.performance_analyzer.analyze_performance_trends(metric_name, hours)
        return {
            "performance_analysis": trend_analysis,
            "analysis_timestamp": time.time()
        }
    
    async def _analyze_business_intelligence(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """分析商业智能"""
        kpi_name = parameters.get("kpi_name", "user_retention")
        days = parameters.get("days", 30)
        
        kpi_analysis = self.bi_analyzer.analyze_kpi_trends(kpi_name, days)
        return {
            "kpi_analysis": kpi_analysis,
            "analysis_timestamp": time.time()
        }
    
    async def _analyze_predictive(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """预测分析"""
        # 简化的预测分析
        return {
            "prediction": "基于历史数据的预测结果",
            "confidence": 0.75,
            "forecast_period": "30天",
            "analysis_timestamp": time.time()
        }
    
    async def _analyze_real_time(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """实时分析"""
        return {
            "real_time_metrics": self.metrics_collector.get_all_metrics(),
            "analysis_timestamp": time.time()
        }
    
    def _generate_insights(self, data: Dict[str, Any], analysis_type: AnalysisType) -> List[str]:
        """生成洞察"""
        insights = []
        
        if analysis_type == AnalysisType.USER_BEHAVIOR:
            if data.get("user_patterns", {}).get("total_behaviors", 0) > 100:
                insights.append("用户活跃度较高")
            if data.get("insights"):
                insights.extend(data["insights"])
        
        elif analysis_type == AnalysisType.PERFORMANCE:
            trend = data.get("performance_analysis", {}).get("trend", "unknown")
            if trend == "increasing":
                insights.append("性能指标呈上升趋势，需要关注")
            elif trend == "decreasing":
                insights.append("性能指标呈下降趋势，系统优化有效")
        
        elif analysis_type == AnalysisType.BUSINESS_INTELLIGENCE:
            growth_rate = data.get("kpi_analysis", {}).get("growth_rate", 0)
            if growth_rate > 10:
                insights.append("KPI增长显著，业务表现良好")
            elif growth_rate < -10:
                insights.append("KPI下降明显，需要业务调整")
        
        return insights
    
    def _generate_recommendations(self, data: Dict[str, Any], analysis_type: AnalysisType) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if analysis_type == AnalysisType.USER_BEHAVIOR:
            recommendations.append("优化用户体验，提高用户满意度")
            recommendations.append("分析用户流失原因，制定留存策略")
        
        elif analysis_type == AnalysisType.PERFORMANCE:
            recommendations.append("监控性能指标，及时发现问题")
            recommendations.append("优化系统架构，提升性能表现")
        
        elif analysis_type == AnalysisType.BUSINESS_INTELLIGENCE:
            recommendations.append("持续监控KPI变化，调整业务策略")
            recommendations.append("分析成功因素，复制最佳实践")
        
        return recommendations
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            "metrics_collector": self.metrics_collector.get_all_metrics(),
            "analysis_results_count": len(self.analysis_results),
            "user_behavior_count": sum(len(behaviors) for behaviors in self.user_analyzer.user_behaviors.values()),
            "performance_metrics_count": sum(len(metrics) for metrics in self.performance_analyzer.performance_metrics.values()),
            "business_metrics_count": sum(len(metrics) for metrics in self.bi_analyzer.business_metrics.values())
        }

# 全局分析系统实例
analytics_system = AdvancedAnalyticsSystem()

async def generate_analysis_report(analysis_type: AnalysisType, parameters: Dict[str, Any] = None) -> AnalysisResult:
    """生成分析报告"""
    return await analytics_system.generate_analysis(analysis_type, parameters)

def get_analytics_system_stats() -> Dict[str, Any]:
    """获取分析系统统计"""
    return analytics_system.get_system_stats()

if __name__ == "__main__":
    # 测试代码
    async def test_analytics_system():
        # 生成不同类型的分析报告
        user_analysis = await generate_analysis_report(
            AnalysisType.USER_BEHAVIOR, 
            {"user_id": "test_user_001"}
        )
        print("用户行为分析:", user_analysis.title)
        
        performance_analysis = await generate_analysis_report(
            AnalysisType.PERFORMANCE,
            {"metric_name": "response_time", "hours": 24}
        )
        print("性能分析:", performance_analysis.title)
        
        # 获取系统统计
        stats = get_analytics_system_stats()
        print("分析系统统计:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    asyncio.run(test_analytics_system())
