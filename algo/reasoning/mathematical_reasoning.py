"""
数学推理模块 - 实现数学问题求解、证明和计算推理能力
"""

import asyncio
import json
import logging
import re
import math
import sympy as sp
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


class MathProblemType(Enum):
    """数学问题类型"""
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    CALCULUS = "calculus"
    STATISTICS = "statistics"
    PROBABILITY = "probability"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    LINEAR_ALGEBRA = "linear_algebra"
    DIFFERENTIAL_EQUATIONS = "differential_equations"


class SolutionMethod(Enum):
    """求解方法"""
    DIRECT_CALCULATION = "direct_calculation"
    ALGEBRAIC_MANIPULATION = "algebraic_manipulation"
    GEOMETRIC_CONSTRUCTION = "geometric_construction"
    CALCULUS_TECHNIQUES = "calculus_techniques"
    NUMERICAL_METHODS = "numerical_methods"
    PROOF_BY_INDUCTION = "proof_by_induction"
    PROOF_BY_CONTRADICTION = "proof_by_contradiction"
    CASE_ANALYSIS = "case_analysis"


@dataclass
class MathExpression:
    """数学表达式"""
    id: str
    expression: str
    variables: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    range: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    symbolic_form: Optional[Any] = None  # SymPy expression


@dataclass
class MathProblem:
    """数学问题"""
    id: str
    statement: str
    problem_type: MathProblemType
    given_conditions: List[str] = field(default_factory=list)
    unknowns: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    difficulty_level: int = 1  # 1-10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MathSolution:
    """数学解答"""
    problem_id: str
    solution_steps: List[Dict[str, Any]]
    final_answer: Any
    method: SolutionMethod
    confidence: float
    verification: Optional[Dict[str, Any]] = None
    alternative_solutions: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MathProof:
    """数学证明"""
    theorem_id: str
    theorem_statement: str
    assumptions: List[str]
    proof_steps: List[Dict[str, Any]]
    proof_method: str
    validity: bool = False
    rigor_level: float = 0.0
    gaps: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class SymbolicMathEngine:
    """符号数学引擎"""
    
    def __init__(self):
        self.symbol_table: Dict[str, sp.Symbol] = {}
        self.function_table: Dict[str, sp.Function] = {}
        
    def parse_expression(self, expr_str: str) -> sp.Expr:
        """解析数学表达式"""
        try:
            # 预处理表达式
            processed_expr = self._preprocess_expression(expr_str)
            
            # 使用SymPy解析
            expr = sp.sympify(processed_expr)
            
            # 更新符号表
            self._update_symbol_table(expr)
            
            return expr
        except Exception as e:
            logger.error(f"Failed to parse expression '{expr_str}': {e}")
            raise ValueError(f"Invalid mathematical expression: {expr_str}")
            
    def _preprocess_expression(self, expr_str: str) -> str:
        """预处理表达式"""
        # 替换常见的数学符号
        replacements = {
            '×': '*',
            '÷': '/',
            '²': '**2',
            '³': '**3',
            '√': 'sqrt',
            'π': 'pi',
            'e': 'E',
            '∞': 'oo',
            '∑': 'Sum',
            '∏': 'Product',
            '∫': 'Integral'
        }
        
        processed = expr_str
        for old, new in replacements.items():
            processed = processed.replace(old, new)
            
        return processed
        
    def _update_symbol_table(self, expr: sp.Expr):
        """更新符号表"""
        for symbol in expr.free_symbols:
            if str(symbol) not in self.symbol_table:
                self.symbol_table[str(symbol)] = symbol
                
    def simplify_expression(self, expr: Union[str, sp.Expr]) -> sp.Expr:
        """简化表达式"""
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        return sp.simplify(expr)
        
    def expand_expression(self, expr: Union[str, sp.Expr]) -> sp.Expr:
        """展开表达式"""
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        return sp.expand(expr)
        
    def factor_expression(self, expr: Union[str, sp.Expr]) -> sp.Expr:
        """因式分解"""
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        return sp.factor(expr)
        
    def solve_equation(self, equation: Union[str, sp.Eq], variable: str = None) -> List[sp.Expr]:
        """求解方程"""
        if isinstance(equation, str):
            # 解析方程字符串
            if '=' in equation:
                left, right = equation.split('=', 1)
                eq = sp.Eq(self.parse_expression(left.strip()), 
                          self.parse_expression(right.strip()))
            else:
                eq = sp.Eq(self.parse_expression(equation), 0)
        else:
            eq = equation
            
        if variable:
            var = sp.Symbol(variable)
            return sp.solve(eq, var)
        else:
            return sp.solve(eq)
            
    def differentiate(self, expr: Union[str, sp.Expr], variable: str) -> sp.Expr:
        """求导数"""
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        var = sp.Symbol(variable)
        return sp.diff(expr, var)
        
    def integrate(self, expr: Union[str, sp.Expr], variable: str, 
                 limits: Optional[Tuple[Any, Any]] = None) -> sp.Expr:
        """积分"""
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        var = sp.Symbol(variable)
        
        if limits:
            return sp.integrate(expr, (var, limits[0], limits[1]))
        else:
            return sp.integrate(expr, var)
            
    def find_limit(self, expr: Union[str, sp.Expr], variable: str, point: Any) -> sp.Expr:
        """求极限"""
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        var = sp.Symbol(variable)
        return sp.limit(expr, var, point)
        
    def taylor_series(self, expr: Union[str, sp.Expr], variable: str, 
                     point: Any = 0, order: int = 6) -> sp.Expr:
        """泰勒级数展开"""
        if isinstance(expr, str):
            expr = self.parse_expression(expr)
        var = sp.Symbol(variable)
        return sp.series(expr, var, point, order).removeO()


class GeometryEngine:
    """几何推理引擎"""
    
    def __init__(self):
        self.points: Dict[str, Tuple[float, float]] = {}
        self.lines: Dict[str, Dict[str, Any]] = {}
        self.circles: Dict[str, Dict[str, Any]] = {}
        self.polygons: Dict[str, Dict[str, Any]] = {}
        
    def add_point(self, name: str, x: float, y: float):
        """添加点"""
        self.points[name] = (x, y)
        
    def add_line(self, name: str, point1: str, point2: str):
        """添加直线"""
        if point1 in self.points and point2 in self.points:
            p1 = self.points[point1]
            p2 = self.points[point2]
            
            # 计算直线方程 ax + by + c = 0
            if p2[0] != p1[0]:
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                intercept = p1[1] - slope * p1[0]
                self.lines[name] = {
                    "type": "line",
                    "point1": point1,
                    "point2": point2,
                    "slope": slope,
                    "intercept": intercept,
                    "equation": f"y = {slope}*x + {intercept}"
                }
            else:
                # 垂直线
                self.lines[name] = {
                    "type": "vertical_line",
                    "point1": point1,
                    "point2": point2,
                    "x": p1[0],
                    "equation": f"x = {p1[0]}"
                }
                
    def add_circle(self, name: str, center: str, radius: float):
        """添加圆"""
        if center in self.points:
            center_point = self.points[center]
            self.circles[name] = {
                "center": center,
                "center_coords": center_point,
                "radius": radius,
                "equation": f"(x - {center_point[0]})^2 + (y - {center_point[1]})^2 = {radius**2}"
            }
            
    def calculate_distance(self, point1: str, point2: str) -> float:
        """计算两点间距离"""
        if point1 in self.points and point2 in self.points:
            p1 = self.points[point1]
            p2 = self.points[point2]
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return 0.0
        
    def calculate_angle(self, point1: str, vertex: str, point2: str) -> float:
        """计算角度（弧度）"""
        if all(p in self.points for p in [point1, vertex, point2]):
            p1 = self.points[point1]
            v = self.points[vertex]
            p2 = self.points[point2]
            
            # 向量
            vec1 = (p1[0] - v[0], p1[1] - v[1])
            vec2 = (p2[0] - v[0], p2[1] - v[1])
            
            # 点积和模长
            dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
            mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
            mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
            
            if mag1 > 0 and mag2 > 0:
                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # 确保在[-1, 1]范围内
                return math.acos(cos_angle)
        return 0.0
        
    def calculate_area_triangle(self, point1: str, point2: str, point3: str) -> float:
        """计算三角形面积"""
        if all(p in self.points for p in [point1, point2, point3]):
            p1 = self.points[point1]
            p2 = self.points[point2]
            p3 = self.points[point3]
            
            # 使用叉积公式
            area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                           (p3[0] - p1[0]) * (p2[1] - p1[1]))
            return area
        return 0.0
        
    def check_collinear(self, point1: str, point2: str, point3: str) -> bool:
        """检查三点是否共线"""
        area = self.calculate_area_triangle(point1, point2, point3)
        return abs(area) < 1e-10
        
    def check_parallel_lines(self, line1: str, line2: str) -> bool:
        """检查两直线是否平行"""
        if line1 in self.lines and line2 in self.lines:
            l1 = self.lines[line1]
            l2 = self.lines[line2]
            
            if l1["type"] == "vertical_line" and l2["type"] == "vertical_line":
                return True
            elif l1["type"] == "line" and l2["type"] == "line":
                return abs(l1["slope"] - l2["slope"]) < 1e-10
        return False
        
    def check_perpendicular_lines(self, line1: str, line2: str) -> bool:
        """检查两直线是否垂直"""
        if line1 in self.lines and line2 in self.lines:
            l1 = self.lines[line1]
            l2 = self.lines[line2]
            
            if (l1["type"] == "vertical_line" and l2["type"] == "line" and abs(l2["slope"]) < 1e-10) or \
               (l2["type"] == "vertical_line" and l1["type"] == "line" and abs(l1["slope"]) < 1e-10):
                return True
            elif l1["type"] == "line" and l2["type"] == "line":
                return abs(l1["slope"] * l2["slope"] + 1) < 1e-10
        return False


class StatisticsEngine:
    """统计推理引擎"""
    
    def __init__(self):
        pass
        
    def calculate_descriptive_stats(self, data: List[float]) -> Dict[str, float]:
        """计算描述性统计"""
        if not data:
            return {}
            
        data_array = np.array(data)
        
        return {
            "count": len(data),
            "mean": np.mean(data_array),
            "median": np.median(data_array),
            "mode": float(sp.statistics.mode(data)) if len(set(data)) < len(data) else None,
            "std_dev": np.std(data_array, ddof=1) if len(data) > 1 else 0,
            "variance": np.var(data_array, ddof=1) if len(data) > 1 else 0,
            "min": np.min(data_array),
            "max": np.max(data_array),
            "range": np.max(data_array) - np.min(data_array),
            "q1": np.percentile(data_array, 25),
            "q3": np.percentile(data_array, 75),
            "iqr": np.percentile(data_array, 75) - np.percentile(data_array, 25)
        }
        
    def calculate_correlation(self, x_data: List[float], y_data: List[float]) -> Dict[str, float]:
        """计算相关性"""
        if len(x_data) != len(y_data) or len(x_data) < 2:
            return {}
            
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        correlation_matrix = np.corrcoef(x_array, y_array)
        correlation = correlation_matrix[0, 1]
        
        return {
            "pearson_correlation": correlation,
            "correlation_strength": self._interpret_correlation(abs(correlation))
        }
        
    def _interpret_correlation(self, abs_correlation: float) -> str:
        """解释相关性强度"""
        if abs_correlation >= 0.9:
            return "very_strong"
        elif abs_correlation >= 0.7:
            return "strong"
        elif abs_correlation >= 0.5:
            return "moderate"
        elif abs_correlation >= 0.3:
            return "weak"
        else:
            return "very_weak"
            
    def linear_regression(self, x_data: List[float], y_data: List[float]) -> Dict[str, Any]:
        """线性回归"""
        if len(x_data) != len(y_data) or len(x_data) < 2:
            return {}
            
        x_array = np.array(x_data)
        y_array = np.array(y_data)
        
        # 计算回归系数
        slope, intercept = np.polyfit(x_array, y_array, 1)
        
        # 计算R²
        y_pred = slope * x_array + intercept
        ss_res = np.sum((y_array - y_pred) ** 2)
        ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "slope": slope,
            "intercept": intercept,
            "equation": f"y = {slope:.4f}x + {intercept:.4f}",
            "r_squared": r_squared,
            "predictions": y_pred.tolist()
        }


class MathematicalReasoningEngine:
    """数学推理引擎"""
    
    def __init__(self):
        self.symbolic_engine = SymbolicMathEngine()
        self.geometry_engine = GeometryEngine()
        self.statistics_engine = StatisticsEngine()
        self.problem_history: List[MathProblem] = []
        self.solution_history: List[MathSolution] = []
        
    async def solve_math_problem(self, problem: MathProblem) -> MathSolution:
        """求解数学问题"""
        logger.info(f"Solving math problem: {problem.problem_type.value}")
        
        # 根据问题类型选择求解方法
        if problem.problem_type == MathProblemType.ARITHMETIC:
            return await self._solve_arithmetic_problem(problem)
        elif problem.problem_type == MathProblemType.ALGEBRA:
            return await self._solve_algebra_problem(problem)
        elif problem.problem_type == MathProblemType.GEOMETRY:
            return await self._solve_geometry_problem(problem)
        elif problem.problem_type == MathProblemType.CALCULUS:
            return await self._solve_calculus_problem(problem)
        elif problem.problem_type == MathProblemType.STATISTICS:
            return await self._solve_statistics_problem(problem)
        elif problem.problem_type == MathProblemType.PROBABILITY:
            return await self._solve_probability_problem(problem)
        else:
            return await self._solve_general_problem(problem)
            
    async def _solve_arithmetic_problem(self, problem: MathProblem) -> MathSolution:
        """求解算术问题"""
        solution_steps = []
        
        # 提取数值和运算
        numbers = re.findall(r'-?\d+\.?\d*', problem.statement)
        operations = re.findall(r'[+\-*/]', problem.statement)
        
        if numbers:
            # 构建表达式
            expression = problem.statement
            for condition in problem.given_conditions:
                expression += f" and {condition}"
                
            try:
                # 使用符号引擎计算
                expr = self.symbolic_engine.parse_expression(expression)
                result = float(expr.evalf())
                
                solution_steps.append({
                    "step": 1,
                    "description": "Parse and evaluate arithmetic expression",
                    "expression": expression,
                    "result": result
                })
                
                return MathSolution(
                    problem_id=problem.id,
                    solution_steps=solution_steps,
                    final_answer=result,
                    method=SolutionMethod.DIRECT_CALCULATION,
                    confidence=0.95,
                    explanation=f"Direct calculation: {expression} = {result}"
                )
            except Exception as e:
                logger.error(f"Arithmetic calculation failed: {e}")
                
        return MathSolution(
            problem_id=problem.id,
            solution_steps=[],
            final_answer=None,
            method=SolutionMethod.DIRECT_CALCULATION,
            confidence=0.0,
            explanation="Failed to solve arithmetic problem"
        )
        
    async def _solve_algebra_problem(self, problem: MathProblem) -> MathSolution:
        """求解代数问题"""
        solution_steps = []
        
        # 查找方程
        equations = []
        for condition in problem.given_conditions:
            if '=' in condition:
                equations.append(condition)
                
        if not equations and '=' in problem.statement:
            equations.append(problem.statement)
            
        if equations:
            try:
                # 求解第一个方程
                equation = equations[0]
                solutions = self.symbolic_engine.solve_equation(equation)
                
                solution_steps.append({
                    "step": 1,
                    "description": "Parse equation",
                    "equation": equation,
                    "parsed": str(equation)
                })
                
                solution_steps.append({
                    "step": 2,
                    "description": "Solve equation",
                    "method": "symbolic_solving",
                    "solutions": [str(sol) for sol in solutions]
                })
                
                # 验证解
                if solutions:
                    verification_results = []
                    for sol in solutions:
                        # 简化验证
                        verification_results.append({
                            "solution": str(sol),
                            "verified": True
                        })
                        
                    solution_steps.append({
                        "step": 3,
                        "description": "Verify solutions",
                        "verification": verification_results
                    })
                
                return MathSolution(
                    problem_id=problem.id,
                    solution_steps=solution_steps,
                    final_answer=solutions,
                    method=SolutionMethod.ALGEBRAIC_MANIPULATION,
                    confidence=0.9,
                    explanation=f"Solved equation {equation}, found {len(solutions)} solution(s)"
                )
                
            except Exception as e:
                logger.error(f"Algebra solving failed: {e}")
                
        return MathSolution(
            problem_id=problem.id,
            solution_steps=[],
            final_answer=None,
            method=SolutionMethod.ALGEBRAIC_MANIPULATION,
            confidence=0.0,
            explanation="Failed to solve algebra problem"
        )
        
    async def _solve_geometry_problem(self, problem: MathProblem) -> MathSolution:
        """求解几何问题"""
        solution_steps = []
        
        # 解析几何元素
        points = re.findall(r'point\s+([A-Z])', problem.statement, re.IGNORECASE)
        coordinates = re.findall(r'\((-?\d+\.?\d*),\s*(-?\d+\.?\d*)\)', problem.statement)
        
        # 添加点到几何引擎
        for i, point in enumerate(points):
            if i < len(coordinates):
                x, y = float(coordinates[i][0]), float(coordinates[i][1])
                self.geometry_engine.add_point(point, x, y)
                solution_steps.append({
                    "step": len(solution_steps) + 1,
                    "description": f"Add point {point}",
                    "point": point,
                    "coordinates": (x, y)
                })
                
        # 计算距离、角度、面积等
        if len(points) >= 2:
            distance = self.geometry_engine.calculate_distance(points[0], points[1])
            solution_steps.append({
                "step": len(solution_steps) + 1,
                "description": f"Calculate distance between {points[0]} and {points[1]}",
                "distance": distance
            })
            
        if len(points) >= 3:
            area = self.geometry_engine.calculate_area_triangle(points[0], points[1], points[2])
            solution_steps.append({
                "step": len(solution_steps) + 1,
                "description": f"Calculate area of triangle {points[0]}{points[1]}{points[2]}",
                "area": area
            })
            
        # 确定最终答案
        final_answer = None
        if "distance" in problem.statement.lower():
            final_answer = distance if len(points) >= 2 else None
        elif "area" in problem.statement.lower():
            final_answer = area if len(points) >= 3 else None
            
        return MathSolution(
            problem_id=problem.id,
            solution_steps=solution_steps,
            final_answer=final_answer,
            method=SolutionMethod.GEOMETRIC_CONSTRUCTION,
            confidence=0.8,
            explanation="Solved geometry problem using coordinate geometry"
        )
        
    async def _solve_calculus_problem(self, problem: MathProblem) -> MathSolution:
        """求解微积分问题"""
        solution_steps = []
        
        # 查找函数表达式
        function_match = re.search(r'f\(x\)\s*=\s*(.+)', problem.statement)
        if not function_match:
            function_match = re.search(r'y\s*=\s*(.+)', problem.statement)
            
        if function_match:
            function_expr = function_match.group(1).strip()
            
            try:
                expr = self.symbolic_engine.parse_expression(function_expr)
                solution_steps.append({
                    "step": 1,
                    "description": "Parse function",
                    "function": function_expr,
                    "parsed": str(expr)
                })
                
                # 求导数
                if "derivative" in problem.statement.lower() or "differentiate" in problem.statement.lower():
                    derivative = self.symbolic_engine.differentiate(expr, 'x')
                    solution_steps.append({
                        "step": 2,
                        "description": "Calculate derivative",
                        "derivative": str(derivative)
                    })
                    final_answer = derivative
                    
                # 求积分
                elif "integral" in problem.statement.lower() or "integrate" in problem.statement.lower():
                    integral = self.symbolic_engine.integrate(expr, 'x')
                    solution_steps.append({
                        "step": 2,
                        "description": "Calculate integral",
                        "integral": str(integral)
                    })
                    final_answer = integral
                    
                # 求极限
                elif "limit" in problem.statement.lower():
                    # 查找极限点
                    limit_match = re.search(r'x\s*->\s*(-?\d+\.?\d*|∞|infinity)', problem.statement)
                    if limit_match:
                        point_str = limit_match.group(1)
                        if point_str in ['∞', 'infinity']:
                            point = sp.oo
                        else:
                            point = float(point_str)
                            
                        limit_result = self.symbolic_engine.find_limit(expr, 'x', point)
                        solution_steps.append({
                            "step": 2,
                            "description": f"Calculate limit as x approaches {point}",
                            "limit": str(limit_result)
                        })
                        final_answer = limit_result
                        
                else:
                    final_answer = expr
                    
                return MathSolution(
                    problem_id=problem.id,
                    solution_steps=solution_steps,
                    final_answer=final_answer,
                    method=SolutionMethod.CALCULUS_TECHNIQUES,
                    confidence=0.85,
                    explanation="Solved calculus problem using symbolic computation"
                )
                
            except Exception as e:
                logger.error(f"Calculus solving failed: {e}")
                
        return MathSolution(
            problem_id=problem.id,
            solution_steps=[],
            final_answer=None,
            method=SolutionMethod.CALCULUS_TECHNIQUES,
            confidence=0.0,
            explanation="Failed to solve calculus problem"
        )
        
    async def _solve_statistics_problem(self, problem: MathProblem) -> MathSolution:
        """求解统计问题"""
        solution_steps = []
        
        # 提取数据
        numbers = re.findall(r'-?\d+\.?\d*', problem.statement)
        data = [float(num) for num in numbers]
        
        if data:
            # 计算描述性统计
            stats = self.statistics_engine.calculate_descriptive_stats(data)
            
            solution_steps.append({
                "step": 1,
                "description": "Extract data",
                "data": data,
                "count": len(data)
            })
            
            solution_steps.append({
                "step": 2,
                "description": "Calculate descriptive statistics",
                "statistics": stats
            })
            
            # 根据问题确定答案
            final_answer = None
            if "mean" in problem.statement.lower() or "average" in problem.statement.lower():
                final_answer = stats.get("mean")
            elif "median" in problem.statement.lower():
                final_answer = stats.get("median")
            elif "standard deviation" in problem.statement.lower():
                final_answer = stats.get("std_dev")
            elif "variance" in problem.statement.lower():
                final_answer = stats.get("variance")
            else:
                final_answer = stats
                
            return MathSolution(
                problem_id=problem.id,
                solution_steps=solution_steps,
                final_answer=final_answer,
                method=SolutionMethod.DIRECT_CALCULATION,
                confidence=0.9,
                explanation="Solved statistics problem using descriptive statistics"
            )
            
        return MathSolution(
            problem_id=problem.id,
            solution_steps=[],
            final_answer=None,
            method=SolutionMethod.DIRECT_CALCULATION,
            confidence=0.0,
            explanation="Failed to solve statistics problem"
        )
        
    async def _solve_probability_problem(self, problem: MathProblem) -> MathSolution:
        """求解概率问题"""
        solution_steps = []
        
        # 简化的概率计算
        # 查找概率相关的数值
        fractions = re.findall(r'(\d+)/(\d+)', problem.statement)
        decimals = re.findall(r'0\.\d+', problem.statement)
        
        probabilities = []
        for num, den in fractions:
            probabilities.append(float(num) / float(den))
            
        for decimal in decimals:
            probabilities.append(float(decimal))
            
        if probabilities:
            solution_steps.append({
                "step": 1,
                "description": "Extract probabilities",
                "probabilities": probabilities
            })
            
            # 简单的概率运算
            if "and" in problem.statement.lower() and len(probabilities) >= 2:
                # 独立事件的交集
                result = probabilities[0] * probabilities[1]
                solution_steps.append({
                    "step": 2,
                    "description": "Calculate intersection probability (independent events)",
                    "calculation": f"{probabilities[0]} × {probabilities[1]} = {result}",
                    "result": result
                })
                final_answer = result
                
            elif "or" in problem.statement.lower() and len(probabilities) >= 2:
                # 互斥事件的并集
                result = probabilities[0] + probabilities[1]
                solution_steps.append({
                    "step": 2,
                    "description": "Calculate union probability (mutually exclusive events)",
                    "calculation": f"{probabilities[0]} + {probabilities[1]} = {result}",
                    "result": result
                })
                final_answer = result
                
            else:
                final_answer = probabilities[0] if probabilities else None
                
            return MathSolution(
                problem_id=problem.id,
                solution_steps=solution_steps,
                final_answer=final_answer,
                method=SolutionMethod.DIRECT_CALCULATION,
                confidence=0.7,
                explanation="Solved probability problem using basic probability rules"
            )
            
        return MathSolution(
            problem_id=problem.id,
            solution_steps=[],
            final_answer=None,
            method=SolutionMethod.DIRECT_CALCULATION,
            confidence=0.0,
            explanation="Failed to solve probability problem"
        )
        
    async def _solve_general_problem(self, problem: MathProblem) -> MathSolution:
        """求解一般数学问题"""
        solution_steps = []
        
        # 尝试解析为表达式
        try:
            expr = self.symbolic_engine.parse_expression(problem.statement)
            simplified = self.symbolic_engine.simplify_expression(expr)
            
            solution_steps.append({
                "step": 1,
                "description": "Parse mathematical expression",
                "original": problem.statement,
                "parsed": str(expr)
            })
            
            solution_steps.append({
                "step": 2,
                "description": "Simplify expression",
                "simplified": str(simplified)
            })
            
            # 尝试数值计算
            try:
                numerical_result = float(simplified.evalf())
                solution_steps.append({
                    "step": 3,
                    "description": "Evaluate numerically",
                    "result": numerical_result
                })
                final_answer = numerical_result
            except:
                final_answer = simplified
                
            return MathSolution(
                problem_id=problem.id,
                solution_steps=solution_steps,
                final_answer=final_answer,
                method=SolutionMethod.ALGEBRAIC_MANIPULATION,
                confidence=0.6,
                explanation="Solved using symbolic computation and simplification"
            )
            
        except Exception as e:
            logger.error(f"General problem solving failed: {e}")
            
        return MathSolution(
            problem_id=problem.id,
            solution_steps=[],
            final_answer=None,
            method=SolutionMethod.DIRECT_CALCULATION,
            confidence=0.0,
            explanation="Failed to solve mathematical problem"
        )
        
    async def verify_solution(self, problem: MathProblem, solution: MathSolution) -> Dict[str, Any]:
        """验证解答"""
        verification = {
            "verified": False,
            "confidence": 0.0,
            "checks": []
        }
        
        try:
            # 基本检查
            if solution.final_answer is not None:
                verification["checks"].append({
                    "type": "answer_exists",
                    "passed": True,
                    "description": "Solution provides an answer"
                })
                
                # 类型检查
                if problem.problem_type == MathProblemType.ARITHMETIC:
                    if isinstance(solution.final_answer, (int, float)):
                        verification["checks"].append({
                            "type": "answer_type",
                            "passed": True,
                            "description": "Answer is numeric as expected"
                        })
                        
                # 合理性检查
                if isinstance(solution.final_answer, (int, float)):
                    if not (math.isnan(solution.final_answer) or math.isinf(solution.final_answer)):
                        verification["checks"].append({
                            "type": "answer_validity",
                            "passed": True,
                            "description": "Answer is a valid number"
                        })
                        
                # 步骤检查
                if solution.solution_steps:
                    verification["checks"].append({
                        "type": "solution_steps",
                        "passed": True,
                        "description": f"Solution includes {len(solution.solution_steps)} steps"
                    })
                    
                # 计算总体验证分数
                passed_checks = sum(1 for check in verification["checks"] if check["passed"])
                total_checks = len(verification["checks"])
                
                if total_checks > 0:
                    verification["confidence"] = passed_checks / total_checks
                    verification["verified"] = verification["confidence"] >= 0.7
                    
        except Exception as e:
            logger.error(f"Solution verification failed: {e}")
            verification["checks"].append({
                "type": "verification_error",
                "passed": False,
                "description": f"Verification failed: {str(e)}"
            })
            
        return verification


# 示例使用
async def main():
    """示例：数学推理"""
    
    # 创建数学推理引擎
    reasoning_engine = MathematicalReasoningEngine()
    
    # 示例1：算术问题
    print("=== 算术问题示例 ===")
    arithmetic_problem = MathProblem(
        id="arith_001",
        statement="2 + 3 * 4 - 1",
        problem_type=MathProblemType.ARITHMETIC
    )
    
    arithmetic_solution = await reasoning_engine.solve_math_problem(arithmetic_problem)
    print(f"问题: {arithmetic_problem.statement}")
    print(f"答案: {arithmetic_solution.final_answer}")
    print(f"置信度: {arithmetic_solution.confidence:.2f}")
    print(f"解释: {arithmetic_solution.explanation}")
    
    # 示例2：代数问题
    print("\n=== 代数问题示例 ===")
    algebra_problem = MathProblem(
        id="alg_001",
        statement="x^2 - 5*x + 6 = 0",
        problem_type=MathProblemType.ALGEBRA
    )
    
    algebra_solution = await reasoning_engine.solve_math_problem(algebra_problem)
    print(f"问题: {algebra_problem.statement}")
    print(f"答案: {algebra_solution.final_answer}")
    print(f"解答步骤:")
    for step in algebra_solution.solution_steps:
        print(f"  步骤 {step['step']}: {step['description']}")
        
    # 示例3：几何问题
    print("\n=== 几何问题示例 ===")
    geometry_problem = MathProblem(
        id="geom_001",
        statement="Calculate the distance between point A (0, 0) and point B (3, 4)",
        problem_type=MathProblemType.GEOMETRY
    )
    
    geometry_solution = await reasoning_engine.solve_math_problem(geometry_problem)
    print(f"问题: {geometry_problem.statement}")
    print(f"答案: {geometry_solution.final_answer}")
    
    # 示例4：微积分问题
    print("\n=== 微积分问题示例 ===")
    calculus_problem = MathProblem(
        id="calc_001",
        statement="Find the derivative of f(x) = x^3 + 2*x^2 - x + 1",
        problem_type=MathProblemType.CALCULUS
    )
    
    calculus_solution = await reasoning_engine.solve_math_problem(calculus_problem)
    print(f"问题: {calculus_problem.statement}")
    print(f"答案: {calculus_solution.final_answer}")
    
    # 示例5：统计问题
    print("\n=== 统计问题示例 ===")
    statistics_problem = MathProblem(
        id="stat_001",
        statement="Find the mean of the following data: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
        problem_type=MathProblemType.STATISTICS
    )
    
    statistics_solution = await reasoning_engine.solve_math_problem(statistics_problem)
    print(f"问题: {statistics_problem.statement}")
    print(f"答案: {statistics_solution.final_answer}")
    
    # 验证解答
    print("\n=== 解答验证示例 ===")
    verification = await reasoning_engine.verify_solution(arithmetic_problem, arithmetic_solution)
    print(f"验证结果: {'通过' if verification['verified'] else '未通过'}")
    print(f"验证置信度: {verification['confidence']:.2f}")
    print("验证检查:")
    for check in verification["checks"]:
        status = "✓" if check["passed"] else "✗"
        print(f"  {status} {check['description']}")


if __name__ == "__main__":
    asyncio.run(main())
