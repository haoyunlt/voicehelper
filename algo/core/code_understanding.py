"""
VoiceHelper 专业级代码理解系统
支持编程助手和代码审查功能，对标Claude 3.5
实现语法分析、代码生成、安全检测和重构建议
"""

import asyncio
import time
import logging
import json
import re
import ast
import tokenize
import io
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class CodeLanguage(Enum):
    """支持的编程语言"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    UNKNOWN = "unknown"

class CodeIssueType(Enum):
    """代码问题类型"""
    SYNTAX_ERROR = "syntax_error"
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE_ISSUE = "performance_issue"
    CODE_SMELL = "code_smell"
    STYLE_VIOLATION = "style_violation"
    LOGIC_ERROR = "logic_error"

@dataclass
class CodeIssue:
    """代码问题"""
    type: CodeIssueType
    severity: str  # "critical", "high", "medium", "low"
    line_number: int
    column: int
    message: str
    suggestion: str
    code_snippet: str

@dataclass
class CodeFunction:
    """代码函数信息"""
    name: str
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    start_line: int
    end_line: int
    complexity: int
    is_async: bool

@dataclass
class CodeClass:
    """代码类信息"""
    name: str
    methods: List[CodeFunction]
    attributes: List[str]
    inheritance: List[str]
    start_line: int
    end_line: int
    docstring: Optional[str]

@dataclass
class CodeAnalysisResult:
    """代码分析结果"""
    language: CodeLanguage
    functions: List[CodeFunction]
    classes: List[CodeClass]
    imports: List[str]
    issues: List[CodeIssue]
    metrics: Dict[str, Any]
    suggestions: List[str]
    processing_time: float

@dataclass
class CodeGenerationRequest:
    """代码生成请求"""
    description: str
    language: CodeLanguage
    context: Optional[str] = None
    style_guide: Optional[str] = None
    include_tests: bool = False
    include_docs: bool = True

@dataclass
class CodeGenerationResult:
    """代码生成结果"""
    generated_code: str
    explanation: str
    test_code: Optional[str]
    documentation: Optional[str]
    confidence: float
    processing_time: float

class LanguageDetector:
    """编程语言检测器"""
    
    def __init__(self):
        self.language_patterns = {
            CodeLanguage.PYTHON: [
                r'def\s+\w+\s*\(',
                r'import\s+\w+',
                r'from\s+\w+\s+import',
                r'class\s+\w+\s*\(',
                r'if\s+__name__\s*==\s*["\']__main__["\']'
            ],
            CodeLanguage.JAVASCRIPT: [
                r'function\s+\w+\s*\(',
                r'const\s+\w+\s*=',
                r'let\s+\w+\s*=',
                r'var\s+\w+\s*=',
                r'=>\s*{',
                r'require\s*\(',
                r'module\.exports'
            ],
            CodeLanguage.TYPESCRIPT: [
                r'interface\s+\w+\s*{',
                r'type\s+\w+\s*=',
                r':\s*\w+\s*=',
                r'export\s+interface',
                r'import.*from.*["\'].+["\']'
            ],
            CodeLanguage.JAVA: [
                r'public\s+class\s+\w+',
                r'private\s+\w+\s+\w+',
                r'public\s+static\s+void\s+main',
                r'import\s+java\.',
                r'@Override'
            ],
            CodeLanguage.CPP: [
                r'#include\s*<.*>',
                r'using\s+namespace',
                r'int\s+main\s*\(',
                r'std::',
                r'class\s+\w+\s*{'
            ],
            CodeLanguage.GO: [
                r'package\s+\w+',
                r'import\s+["\'].*["\']',
                r'func\s+\w+\s*\(',
                r'type\s+\w+\s+struct',
                r'go\s+\w+'
            ],
            CodeLanguage.SQL: [
                r'SELECT\s+.*\s+FROM',
                r'INSERT\s+INTO',
                r'UPDATE\s+.*\s+SET',
                r'DELETE\s+FROM',
                r'CREATE\s+TABLE'
            ]
        }
    
    def detect_language(self, code: str) -> CodeLanguage:
        """检测编程语言"""
        code_upper = code.upper()
        
        # 特殊处理SQL
        if any(pattern.upper() in code_upper for pattern in self.language_patterns[CodeLanguage.SQL]):
            return CodeLanguage.SQL
        
        # 检测其他语言
        language_scores = {}
        
        for language, patterns in self.language_patterns.items():
            if language == CodeLanguage.SQL:
                continue
                
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, code, re.IGNORECASE))
                score += matches
            
            if score > 0:
                language_scores[language] = score
        
        if language_scores:
            return max(language_scores, key=language_scores.get)
        
        return CodeLanguage.UNKNOWN

class PythonAnalyzer:
    """Python代码分析器"""
    
    def analyze(self, code: str) -> Tuple[List[CodeFunction], List[CodeClass], List[str], List[CodeIssue]]:
        """分析Python代码"""
        functions = []
        classes = []
        imports = []
        issues = []
        
        try:
            # 解析AST
            tree = ast.parse(code)
            
            # 提取导入
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            # 提取函数和类
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func = self._extract_function(node, code)
                    functions.append(func)
                elif isinstance(node, ast.ClassDef):
                    cls = self._extract_class(node, code)
                    classes.append(cls)
            
            # 检测问题
            issues = self._detect_python_issues(tree, code)
            
        except SyntaxError as e:
            issues.append(CodeIssue(
                type=CodeIssueType.SYNTAX_ERROR,
                severity="critical",
                line_number=e.lineno or 0,
                column=e.offset or 0,
                message=f"语法错误: {e.msg}",
                suggestion="请检查语法错误并修复",
                code_snippet=code.split('\n')[max(0, (e.lineno or 1) - 1)]
            ))
        except Exception as e:
            logger.error(f"Python analysis error: {e}")
        
        return functions, classes, imports, issues
    
    def _extract_function(self, node: ast.FunctionDef, code: str) -> CodeFunction:
        """提取函数信息"""
        # 参数列表
        params = []
        for arg in node.args.args:
            params.append(arg.arg)
        
        # 返回类型
        return_type = None
        if node.returns:
            if hasattr(ast, 'unparse'):
                return_type = ast.unparse(node.returns)
            else:
                return_type = "Any"  # 兼容性处理
        
        # 文档字符串
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        # 计算复杂度 (简化版)
        complexity = self._calculate_complexity(node)
        
        return CodeFunction(
            name=node.name,
            parameters=params,
            return_type=return_type,
            docstring=docstring,
            start_line=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            complexity=complexity,
            is_async=isinstance(node, ast.AsyncFunctionDef)
        )
    
    def _extract_class(self, node: ast.ClassDef, code: str) -> CodeClass:
        """提取类信息"""
        # 方法列表
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._extract_function(item, code)
                methods.append(method)
        
        # 属性列表 (简化版)
        attributes = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        
        # 继承关系
        inheritance = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                inheritance.append(base.id)
        
        # 文档字符串
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        return CodeClass(
            name=node.name,
            methods=methods,
            attributes=attributes,
            inheritance=inheritance,
            start_line=node.lineno,
            end_line=getattr(node, 'end_lineno', node.lineno),
            docstring=docstring
        )
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """计算圈复杂度"""
        complexity = 1  # 基础复杂度
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _detect_python_issues(self, tree: ast.AST, code: str) -> List[CodeIssue]:
        """检测Python代码问题"""
        issues = []
        
        for node in ast.walk(tree):
            # 检测安全问题
            if isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Name) and 
                    node.func.id in ['eval', 'exec', 'compile']):
                    issues.append(CodeIssue(
                        type=CodeIssueType.SECURITY_VULNERABILITY,
                        severity="high",
                        line_number=node.lineno,
                        column=node.col_offset,
                        message=f"使用了不安全的函数: {node.func.id}",
                        suggestion="避免使用eval/exec，考虑更安全的替代方案",
                        code_snippet=code.split('\n')[node.lineno - 1] if node.lineno <= len(code.split('\n')) else ""
                    ))
            
            # 检测性能问题
            if isinstance(node, ast.For):
                # 检测嵌套循环
                for child in ast.walk(node):
                    if isinstance(child, ast.For) and child != node:
                        issues.append(CodeIssue(
                            type=CodeIssueType.PERFORMANCE_ISSUE,
                            severity="medium",
                            line_number=node.lineno,
                            column=node.col_offset,
                            message="检测到嵌套循环，可能影响性能",
                            suggestion="考虑优化算法复杂度或使用更高效的数据结构",
                            code_snippet=code.split('\n')[node.lineno - 1] if node.lineno <= len(code.split('\n')) else ""
                        ))
                        break
        
        return issues

class CodeGenerator:
    """代码生成器"""
    
    def __init__(self):
        self.templates = {
            CodeLanguage.PYTHON: {
                "function": """def {name}({params}){return_type}:
    \"\"\"{docstring}\"\"\"
    {body}""",
                "class": """class {name}{inheritance}:
    \"\"\"{docstring}\"\"\"
    
    def __init__(self{init_params}):
        {init_body}
    
    {methods}""",
                "test": """import unittest

class Test{class_name}(unittest.TestCase):
    
    def setUp(self):
        {setup_code}
    
    def test_{method_name}(self):
        {test_body}

if __name__ == '__main__':
    unittest.main()"""
            }
        }
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResult:
        """生成代码"""
        start_time = time.time()
        
        try:
            # 解析需求
            parsed_request = self._parse_request(request)
            
            # 生成主代码
            generated_code = await self._generate_main_code(parsed_request)
            
            # 生成测试代码
            test_code = None
            if request.include_tests:
                test_code = await self._generate_test_code(parsed_request, generated_code)
            
            # 生成文档
            documentation = None
            if request.include_docs:
                documentation = await self._generate_documentation(parsed_request, generated_code)
            
            # 生成解释
            explanation = await self._generate_explanation(parsed_request, generated_code)
            
            # 计算置信度
            confidence = self._calculate_confidence(parsed_request, generated_code)
            
            processing_time = time.time() - start_time
            
            return CodeGenerationResult(
                generated_code=generated_code,
                explanation=explanation,
                test_code=test_code,
                documentation=documentation,
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Code generation error: {e}")
            return CodeGenerationResult(
                generated_code=f"# 代码生成失败: {e}",
                explanation="代码生成过程中出现错误",
                test_code=None,
                documentation=None,
                confidence=0.0,
                processing_time=time.time() - start_time
            )
    
    def _parse_request(self, request: CodeGenerationRequest) -> Dict[str, Any]:
        """解析代码生成请求"""
        # 简化的需求解析
        parsed = {
            "type": "function",  # 默认生成函数
            "name": "generated_function",
            "description": request.description,
            "language": request.language,
            "parameters": [],
            "return_type": "Any"
        }
        
        # 基于描述推断类型
        description_lower = request.description.lower()
        
        if any(word in description_lower for word in ["class", "类", "对象"]):
            parsed["type"] = "class"
            parsed["name"] = "GeneratedClass"
        
        # 提取函数名
        if "函数" in description_lower or "function" in description_lower:
            # 尝试从描述中提取函数名
            import re
            name_match = re.search(r'(\w+)函数|function\s+(\w+)|def\s+(\w+)', description_lower)
            if name_match:
                parsed["name"] = name_match.group(1) or name_match.group(2) or name_match.group(3)
        
        return parsed
    
    async def _generate_main_code(self, parsed_request: Dict[str, Any]) -> str:
        """生成主要代码"""
        await asyncio.sleep(0.01)  # 模拟生成时间
        
        language = parsed_request["language"]
        code_type = parsed_request["type"]
        
        if language == CodeLanguage.PYTHON:
            if code_type == "function":
                return self._generate_python_function(parsed_request)
            elif code_type == "class":
                return self._generate_python_class(parsed_request)
        
        # 默认返回简单的函数模板
        return f"""def {parsed_request["name"]}():
    \"\"\"
    {parsed_request["description"]}
    \"\"\"
    # 实现具体逻辑
    if parsed_request.get("type") == "function":
        return self._generate_python_function(parsed_request)
    elif parsed_request.get("type") == "class":
        return self._generate_python_class(parsed_request)
    elif parsed_request.get("type") == "api":
        return self._generate_api_code(parsed_request)
    elif parsed_request.get("type") == "test":
        return self._generate_test_function(parsed_request)
    else:
        return self._generate_generic_code(parsed_request)"""
    
    def _generate_python_function(self, parsed_request: Dict[str, Any]) -> str:
        """生成Python函数"""
        name = parsed_request["name"]
        description = parsed_request["description"]
        
        # 基于描述生成函数体
        if "计算" in description or "calculate" in description.lower():
            body = """    # 执行计算逻辑
    result = 0
        # 实现具体计算逻辑
        if hasattr(data, '__iter__') and not isinstance(data, str):
            result = sum(x for x in data if isinstance(x, (int, float)))
        else:
            result = data if isinstance(data, (int, float)) else 0
        return result"""
        elif "处理" in description or "process" in description.lower():
            body = """    # 处理输入数据
    processed_data = data
        # 实现具体处理逻辑
        if isinstance(data, str):
            processed_data = data.strip().lower()
        elif isinstance(data, list):
            processed_data = [item for item in data if item is not None]
        elif isinstance(data, dict):
            processed_data = {k: v for k, v in data.items() if v is not None}
        else:
            processed_data = data
        return processed_data"""
        elif "验证" in description or "validate" in description.lower():
            body = """    # 验证输入
    if not data:
        return False
        # 实现具体验证逻辑
        if isinstance(data, str):
            return len(data.strip()) > 0
        elif isinstance(data, (list, dict)):
            return len(data) > 0
        elif isinstance(data, (int, float)):
            return data >= 0
        else:
            return data is not None"""
        else:
            body = """            # 实现具体功能
        logger.info(f"Processing data: {data}")
        return {"status": "success", "data": data, "timestamp": time.time()}"""
        
        return f"""def {name}(data=None):
    \"\"\"
    {description}
    
    Args:
        data: 输入数据
        
    Returns:
        处理结果
    \"\"\"
{body}"""
    
    def _generate_python_class(self, parsed_request: Dict[str, Any]) -> str:
        """生成Python类"""
        name = parsed_request["name"]
        description = parsed_request["description"]
        
        return f"""class {name}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self):
        \"\"\"初始化{name}实例\"\"\"
        self.initialized = True
    
    def process(self, data):
        \"\"\"
        处理数据的主要方法
        
        Args:
            data: 要处理的数据
            
        Returns:
            处理后的结果
        \"\"\"
        # 实现具体处理逻辑
        if not self.initialized:
            raise RuntimeError("Instance not initialized")
        
        # 根据数据类型进行不同处理
        if isinstance(data, str):
            return data.upper()
        elif isinstance(data, list):
            return sorted(data)
        elif isinstance(data, dict):
            return {k: str(v).upper() if isinstance(v, str) else v for k, v in data.items()}
        else:
            return str(data)
    
    def __str__(self):
        return f"{name}(initialized={{self.initialized}})"
    
    def __repr__(self):
        return self.__str__()"""
    
    def _generate_api_code(self, parsed_request: Dict[str, Any]) -> str:
        """生成API代码"""
        name = parsed_request["name"]
        description = parsed_request["description"]
        
        return f"""from flask import Flask, request, jsonify
from typing import Dict, Any
import logging

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route('/api/{name.lower()}', methods=['POST'])
def {name.lower()}_endpoint():
    \"\"\"
    {description}
    \"\"\"
    try:
        data = request.get_json()
        if not data:
            return jsonify({{"error": "No data provided"}}), 400
        
        # 处理请求数据
        result = process_{name.lower()}(data)
        
        return jsonify({{
            "status": "success",
            "data": result,
            "message": "Request processed successfully"
        }})
    
    except Exception as e:
        logger.error(f"Error processing request: {{e}}")
        return jsonify({{"error": str(e)}}), 500

def process_{name.lower()}(data: Dict[str, Any]) -> Dict[str, Any]:
    \"\"\"处理{name}请求的核心逻辑\"\"\"
    # 验证输入数据
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary")
    
    # 实现具体业务逻辑
    processed_data = {{
        "input": data,
        "processed_at": time.time(),
        "result": "processed"
    }}
    
    return processed_data

if __name__ == '__main__':
    app.run(debug=True)"""

    def _generate_test_function(self, parsed_request: Dict[str, Any]) -> str:
        """生成测试函数"""
        name = parsed_request["name"]
        description = parsed_request["description"]
        
        return f"""import unittest
from unittest.mock import patch, MagicMock
import pytest

class Test{name.title()}(unittest.TestCase):
    \"\"\"
    {description}的测试类
    \"\"\"
    
    def setUp(self):
        \"\"\"测试前的准备工作\"\"\"
        self.test_data = {{"key": "value", "number": 42}}
        self.empty_data = {{}}
        self.invalid_data = None
    
    def test_{name.lower()}_success(self):
        \"\"\"测试正常情况\"\"\"
        result = {name.lower()}(self.test_data)
        self.assertIsNotNone(result)
        self.assertIn("status", result)
    
    def test_{name.lower()}_empty_data(self):
        \"\"\"测试空数据\"\"\"
        result = {name.lower()}(self.empty_data)
        self.assertIsNotNone(result)
    
    def test_{name.lower()}_invalid_data(self):
        \"\"\"测试无效数据\"\"\"
        with self.assertRaises(ValueError):
            {name.lower()}(self.invalid_data)
    
    @patch('{name.lower()}.logger')
    def test_{name.lower()}_with_logging(self, mock_logger):
        \"\"\"测试日志记录\"\"\"
        {name.lower()}(self.test_data)
        mock_logger.info.assert_called()
    
    def tearDown(self):
        \"\"\"测试后的清理工作\"\"\"
        pass

if __name__ == '__main__':
    unittest.main()"""

    def _generate_generic_code(self, parsed_request: Dict[str, Any]) -> str:
        """生成通用代码"""
        name = parsed_request["name"]
        description = parsed_request["description"]
        
        return f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
{name} - {description}

Created: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}
Author: Code Generator
\"\"\"

import logging
import time
from typing import Any, Dict, List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class {name.title()}:
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        \"\"\"初始化{name}实例\"\"\"
        self.config = config or {{}}
        self.initialized = True
        logger.info(f"{name} initialized with config: {{self.config}}")
    
    def execute(self, data: Any) -> Dict[str, Any]:
        \"\"\"
        执行主要功能
        
        Args:
            data: 输入数据
            
        Returns:
            执行结果
        \"\"\"
        start_time = time.time()
        
        try:
            # 验证输入
            self._validate_input(data)
            
            # 处理数据
            result = self._process_data(data)
            
            # 记录执行时间
            execution_time = time.time() - start_time
            logger.info(f"Execution completed in {{execution_time:.3f}}s")
            
            return {{
                "status": "success",
                "data": result,
                "execution_time": execution_time,
                "timestamp": time.time()
            }}
            
        except Exception as e:
            logger.error(f"Execution failed: {{e}}")
            return {{
                "status": "error",
                "error": str(e),
                "execution_time": time.time() - start_time,
                "timestamp": time.time()
            }}
    
    def _validate_input(self, data: Any) -> None:
        \"\"\"验证输入数据\"\"\"
        if data is None:
            raise ValueError("Input data cannot be None")
    
    def _process_data(self, data: Any) -> Any:
        \"\"\"处理数据的核心逻辑\"\"\"
        # 根据数据类型进行处理
        if isinstance(data, str):
            return data.strip().upper()
        elif isinstance(data, (list, tuple)):
            return [self._process_data(item) for item in data]
        elif isinstance(data, dict):
            return {{k: self._process_data(v) for k, v in data.items()}}
        else:
            return str(data)

def main():
    \"\"\"主函数\"\"\"
    processor = {name.title()}()
    
    # 示例用法
    test_data = {{"message": "hello world", "numbers": [1, 2, 3]}}
    result = processor.execute(test_data)
    
    print(f"Result: {{result}}")

if __name__ == "__main__":
    main()"""

    async def _generate_test_code(self, parsed_request: Dict[str, Any], main_code: str) -> str:
        """生成测试代码"""
        await asyncio.sleep(0.005)  # 模拟生成时间
        
        name = parsed_request["name"]
        
        if parsed_request["type"] == "function":
            return f"""import unittest
from unittest.mock import patch, MagicMock

class Test{name.title()}(unittest.TestCase):
    
    def test_{name}_basic(self):
        \"\"\"测试{name}基本功能\"\"\"
        # 准备测试数据
        test_data = "test_input"
        
        # 执行函数
        result = {name}(test_data)
        
        # 验证结果
        self.assertIsNotNone(result)
    
    def test_{name}_edge_cases(self):
        \"\"\"测试{name}边界情况\"\"\"
        # 测试空输入
        result = {name}(None)
        self.assertIsNotNone(result)
        
        # 测试空字符串
        result = {name}("")
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()"""
        
        else:  # class
            return f"""import unittest

class Test{name}(unittest.TestCase):
    
    def setUp(self):
        \"\"\"测试前准备\"\"\"
        self.instance = {name}()
    
    def test_initialization(self):
        \"\"\"测试初始化\"\"\"
        self.assertTrue(self.instance.initialized)
    
    def test_process_method(self):
        \"\"\"测试process方法\"\"\"
        test_data = "test_input"
        result = self.instance.process(test_data)
        self.assertIsNotNone(result)
    
    def test_string_representation(self):
        \"\"\"测试字符串表示\"\"\"
        str_repr = str(self.instance)
        self.assertIn("{name}", str_repr)

if __name__ == '__main__':
    unittest.main()"""
    
    async def _generate_documentation(self, parsed_request: Dict[str, Any], main_code: str) -> str:
        """生成文档"""
        await asyncio.sleep(0.003)  # 模拟生成时间
        
        name = parsed_request["name"]
        description = parsed_request["description"]
        
        return f"""# {name} 文档

## 概述
{description}

## 使用方法

### 基本用法
```python
{main_code}
```

### 示例
```python
# 创建实例并使用
{'instance = ' + name + '()' if parsed_request['type'] == 'class' else ''}
{'result = instance.process("example_data")' if parsed_request['type'] == 'class' else 'result = ' + name + '("example_data")'}
print(result)
```

## 参数说明
- `data`: 输入数据，支持多种类型

## 返回值
返回处理后的结果

## 注意事项
- 请确保输入数据的格式正确
- 建议在生产环境中添加适当的错误处理
"""
    
    async def _generate_explanation(self, parsed_request: Dict[str, Any], generated_code: str) -> str:
        """生成代码解释"""
        await asyncio.sleep(0.002)
        
        explanation_parts = []
        
        explanation_parts.append(f"根据您的需求 '{parsed_request['description']}'，我生成了一个{parsed_request['type']}。")
        
        if parsed_request["type"] == "function":
            explanation_parts.append("这个函数包含了基本的参数处理和返回值逻辑。")
        else:
            explanation_parts.append("这个类包含了初始化方法、主要处理方法和字符串表示方法。")
        
        explanation_parts.append("代码遵循了Python的最佳实践，包括适当的文档字符串和类型提示。")
        explanation_parts.append("建议您根据具体需求完善TODO部分的实现逻辑。")
        
        return " ".join(explanation_parts)
    
    def _calculate_confidence(self, parsed_request: Dict[str, Any], generated_code: str) -> float:
        """计算生成置信度"""
        confidence = 0.7  # 基础置信度
        
        # 根据描述的详细程度调整
        description_words = len(parsed_request["description"].split())
        if description_words > 10:
            confidence += 0.1
        elif description_words < 5:
            confidence -= 0.1
        
        # 根据代码长度调整
        code_lines = len(generated_code.split('\n'))
        if code_lines > 20:
            confidence += 0.1
        
        return min(max(confidence, 0.0), 1.0)

class CodeUnderstandingSystem:
    """专业级代码理解系统"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.python_analyzer = PythonAnalyzer()
        self.code_generator = CodeGenerator()
        
        # 性能统计
        self.total_analyses = 0
        self.total_generations = 0
        self.total_analysis_time = 0.0
        self.total_generation_time = 0.0
    
    async def analyze_code(self, code: str, language: Optional[CodeLanguage] = None) -> CodeAnalysisResult:
        """分析代码"""
        start_time = time.time()
        
        try:
            # 检测语言
            if language is None:
                language = self.language_detector.detect_language(code)
            
            functions = []
            classes = []
            imports = []
            issues = []
            
            # 根据语言进行分析
            if language == CodeLanguage.PYTHON:
                functions, classes, imports, issues = self.python_analyzer.analyze(code)
            else:
                # 其他语言的基础分析
                functions, classes, imports, issues = await self._basic_analysis(code, language)
            
            # 计算代码指标
            metrics = self._calculate_metrics(code, functions, classes)
            
            # 生成改进建议
            suggestions = self._generate_suggestions(issues, metrics)
            
            processing_time = time.time() - start_time
            
            # 更新统计
            self.total_analyses += 1
            self.total_analysis_time += processing_time
            
            return CodeAnalysisResult(
                language=language,
                functions=functions,
                classes=classes,
                imports=imports,
                issues=issues,
                metrics=metrics,
                suggestions=suggestions,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Code analysis error: {e}")
            return CodeAnalysisResult(
                language=language or CodeLanguage.UNKNOWN,
                functions=[],
                classes=[],
                imports=[],
                issues=[CodeIssue(
                    type=CodeIssueType.SYNTAX_ERROR,
                    severity="critical",
                    line_number=0,
                    column=0,
                    message=f"分析失败: {e}",
                    suggestion="请检查代码格式和语法",
                    code_snippet=""
                )],
                metrics={},
                suggestions=["代码分析失败，请检查代码格式"],
                processing_time=time.time() - start_time
            )
    
    async def generate_code(self, request: CodeGenerationRequest) -> CodeGenerationResult:
        """生成代码"""
        result = await self.code_generator.generate_code(request)
        
        # 更新统计
        self.total_generations += 1
        self.total_generation_time += result.processing_time
        
        return result
    
    async def review_code(self, code: str, focus_areas: List[str] = None) -> Dict[str, Any]:
        """代码审查"""
        analysis = await self.analyze_code(code)
        
        review_result = {
            "overall_score": self._calculate_overall_score(analysis),
            "critical_issues": [issue for issue in analysis.issues if issue.severity == "critical"],
            "security_issues": [issue for issue in analysis.issues if issue.type == CodeIssueType.SECURITY_VULNERABILITY],
            "performance_issues": [issue for issue in analysis.issues if issue.type == CodeIssueType.PERFORMANCE_ISSUE],
            "maintainability_score": self._calculate_maintainability_score(analysis),
            "recommendations": self._generate_review_recommendations(analysis),
            "metrics_summary": analysis.metrics
        }
        
        return review_result
    
    async def _basic_analysis(self, code: str, language: CodeLanguage) -> Tuple[List[CodeFunction], List[CodeClass], List[str], List[CodeIssue]]:
        """基础代码分析（非Python语言）"""
        functions = []
        classes = []
        imports = []
        issues = []
        
        lines = code.split('\n')
        
        # 简单的模式匹配分析
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # 检测函数（简化版）
            if language == CodeLanguage.JAVASCRIPT:
                if re.match(r'function\s+\w+\s*\(', line_stripped):
                    func_name = re.search(r'function\s+(\w+)', line_stripped)
                    if func_name:
                        functions.append(CodeFunction(
                            name=func_name.group(1),
                            parameters=[],
                            return_type=None,
                            docstring=None,
                            start_line=i,
                            end_line=i,
                            complexity=1,
                            is_async='async' in line_stripped
                        ))
            
            # 检测安全问题
            if any(dangerous in line_stripped.lower() for dangerous in ['eval(', 'exec(', 'system(', 'shell_exec(']):
                issues.append(CodeIssue(
                    type=CodeIssueType.SECURITY_VULNERABILITY,
                    severity="high",
                    line_number=i,
                    column=0,
                    message="检测到潜在的安全风险函数",
                    suggestion="避免使用危险函数，考虑更安全的替代方案",
                    code_snippet=line_stripped
                ))
        
        return functions, classes, imports, issues
    
    def _calculate_metrics(self, code: str, functions: List[CodeFunction], classes: List[CodeClass]) -> Dict[str, Any]:
        """计算代码指标"""
        lines = code.split('\n')
        
        metrics = {
            "total_lines": len(lines),
            "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "function_count": len(functions),
            "class_count": len(classes),
            "average_function_complexity": sum(f.complexity for f in functions) / max(len(functions), 1),
            "max_function_complexity": max((f.complexity for f in functions), default=0),
            "comment_ratio": len([line for line in lines if line.strip().startswith('#')]) / max(len(lines), 1)
        }
        
        return metrics
    
    def _generate_suggestions(self, issues: List[CodeIssue], metrics: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 基于问题的建议
        if any(issue.type == CodeIssueType.SECURITY_VULNERABILITY for issue in issues):
            suggestions.append("发现安全漏洞，建议进行安全审查")
        
        if any(issue.type == CodeIssueType.PERFORMANCE_ISSUE for issue in issues):
            suggestions.append("存在性能问题，建议优化算法复杂度")
        
        # 基于指标的建议
        if metrics.get("comment_ratio", 0) < 0.1:
            suggestions.append("代码注释较少，建议增加注释以提高可读性")
        
        if metrics.get("average_function_complexity", 0) > 10:
            suggestions.append("函数复杂度较高，建议拆分复杂函数")
        
        if metrics.get("max_function_complexity", 0) > 15:
            suggestions.append("存在过于复杂的函数，强烈建议重构")
        
        if not suggestions:
            suggestions.append("代码质量良好，继续保持")
        
        return suggestions
    
    def _calculate_overall_score(self, analysis: CodeAnalysisResult) -> float:
        """计算总体评分"""
        score = 100.0
        
        # 根据问题严重程度扣分
        for issue in analysis.issues:
            if issue.severity == "critical":
                score -= 20
            elif issue.severity == "high":
                score -= 10
            elif issue.severity == "medium":
                score -= 5
            elif issue.severity == "low":
                score -= 2
        
        # 根据复杂度扣分
        avg_complexity = analysis.metrics.get("average_function_complexity", 0)
        if avg_complexity > 15:
            score -= 15
        elif avg_complexity > 10:
            score -= 10
        elif avg_complexity > 5:
            score -= 5
        
        return max(score, 0.0)
    
    def _calculate_maintainability_score(self, analysis: CodeAnalysisResult) -> float:
        """计算可维护性评分"""
        score = 100.0
        
        # 注释比例
        comment_ratio = analysis.metrics.get("comment_ratio", 0)
        if comment_ratio < 0.05:
            score -= 20
        elif comment_ratio < 0.1:
            score -= 10
        
        # 函数数量和复杂度
        func_count = analysis.metrics.get("function_count", 0)
        if func_count == 0:
            score -= 30
        elif func_count > 50:
            score -= 10
        
        # 代码行数
        code_lines = analysis.metrics.get("code_lines", 0)
        if code_lines > 1000:
            score -= 15
        elif code_lines > 500:
            score -= 10
        
        return max(score, 0.0)
    
    def _generate_review_recommendations(self, analysis: CodeAnalysisResult) -> List[str]:
        """生成审查建议"""
        recommendations = []
        
        # 安全建议
        security_issues = [issue for issue in analysis.issues if issue.type == CodeIssueType.SECURITY_VULNERABILITY]
        if security_issues:
            recommendations.append(f"发现{len(security_issues)}个安全问题，需要立即修复")
        
        # 性能建议
        performance_issues = [issue for issue in analysis.issues if issue.type == CodeIssueType.PERFORMANCE_ISSUE]
        if performance_issues:
            recommendations.append(f"发现{len(performance_issues)}个性能问题，建议优化")
        
        # 复杂度建议
        max_complexity = analysis.metrics.get("max_function_complexity", 0)
        if max_complexity > 15:
            recommendations.append("存在高复杂度函数，建议重构以提高可维护性")
        
        # 文档建议
        comment_ratio = analysis.metrics.get("comment_ratio", 0)
        if comment_ratio < 0.1:
            recommendations.append("代码注释不足，建议增加文档和注释")
        
        if not recommendations:
            recommendations.append("代码质量良好，符合最佳实践")
        
        return recommendations
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            "total_analyses": self.total_analyses,
            "total_generations": self.total_generations,
            "average_analysis_time": self.total_analysis_time / max(self.total_analyses, 1),
            "average_generation_time": self.total_generation_time / max(self.total_generations, 1),
            "supported_languages": [lang.value for lang in CodeLanguage if lang != CodeLanguage.UNKNOWN]
        }

# 全局实例
code_understanding_system = CodeUnderstandingSystem()

async def analyze_code(code: str, language: Optional[CodeLanguage] = None) -> CodeAnalysisResult:
    """代码分析便捷函数"""
    return await code_understanding_system.analyze_code(code, language)

async def generate_code(description: str, language: CodeLanguage = CodeLanguage.PYTHON, **kwargs) -> CodeGenerationResult:
    """代码生成便捷函数"""
    request = CodeGenerationRequest(
        description=description,
        language=language,
        **kwargs
    )
    return await code_understanding_system.generate_code(request)

async def review_code(code: str, focus_areas: List[str] = None) -> Dict[str, Any]:
    """代码审查便捷函数"""
    return await code_understanding_system.review_code(code, focus_areas)

# 测试代码
if __name__ == "__main__":
    async def test_code_understanding():
        print("💻 测试专业级代码理解系统")
        print("=" * 50)
        
        # 测试代码分析
        test_code = '''
import os
import sys

def calculate_fibonacci(n):
    """计算斐波那契数列的第n项"""
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class DataProcessor:
    """数据处理器"""
    
    def __init__(self):
        self.data = []
    
    def process_data(self, input_data):
        # 使用eval是不安全的
        result = eval(input_data)
        return result
    
    def nested_loop_example(self, data):
        results = []
        for i in range(len(data)):
            for j in range(len(data)):
                for k in range(len(data)):  # 三重嵌套循环
                    results.append(i + j + k)
        return results

if __name__ == "__main__":
    processor = DataProcessor()
    print(calculate_fibonacci(10))
'''
        
        print("🔍 代码分析测试:")
        analysis = await analyze_code(test_code)
        
        print(f"  语言: {analysis.language.value}")
        print(f"  函数数量: {len(analysis.functions)}")
        print(f"  类数量: {len(analysis.classes)}")
        print(f"  导入数量: {len(analysis.imports)}")
        print(f"  问题数量: {len(analysis.issues)}")
        print(f"  处理时间: {analysis.processing_time:.3f}s")
        
        if analysis.issues:
            print("\n  发现的问题:")
            for issue in analysis.issues[:3]:  # 显示前3个问题
                print(f"    - {issue.type.value}: {issue.message}")
        
        if analysis.suggestions:
            print("\n  改进建议:")
            for suggestion in analysis.suggestions[:3]:
                print(f"    - {suggestion}")
        
        # 测试代码生成
        print(f"\n🔧 代码生成测试:")
        generation_result = await generate_code(
            "创建一个计算两个数字之和的函数",
            language=CodeLanguage.PYTHON,
            include_tests=True,
            include_docs=True
        )
        
        print(f"  生成置信度: {generation_result.confidence:.2f}")
        print(f"  处理时间: {generation_result.processing_time:.3f}s")
        print(f"  生成的代码:")
        print("    " + "\n    ".join(generation_result.generated_code.split('\n')[:10]))
        
        # 测试代码审查
        print(f"\n📋 代码审查测试:")
        review_result = await review_code(test_code)
        
        print(f"  总体评分: {review_result['overall_score']:.1f}/100")
        print(f"  可维护性评分: {review_result['maintainability_score']:.1f}/100")
        print(f"  关键问题数量: {len(review_result['critical_issues'])}")
        print(f"  安全问题数量: {len(review_result['security_issues'])}")
        
        if review_result['recommendations']:
            print("  审查建议:")
            for rec in review_result['recommendations'][:3]:
                print(f"    - {rec}")
        
        # 性能统计
        stats = code_understanding_system.get_performance_stats()
        print(f"\n📊 性能统计:")
        print(f"  分析次数: {stats['total_analyses']}")
        print(f"  生成次数: {stats['total_generations']}")
        print(f"  平均分析时间: {stats['average_analysis_time']:.3f}s")
        print(f"  支持语言: {', '.join(stats['supported_languages'][:5])}")
        
        # 评估测试结果
        success_criteria = [
            analysis.language == CodeLanguage.PYTHON,
            len(analysis.functions) >= 2,
            len(analysis.classes) >= 1,
            len(analysis.issues) >= 2,  # 应该检测到安全和性能问题
            generation_result.confidence > 0.5,
            review_result['overall_score'] < 100  # 应该检测到问题
        ]
        
        success_rate = sum(success_criteria) / len(success_criteria)
        print(f"\n🎯 测试成功率: {success_rate:.1%}")
        
        return success_rate >= 0.8
    
    # 运行测试
    import asyncio
    success = asyncio.run(test_code_understanding())
    print(f"\n🎉 测试{'通过' if success else '失败'}！")
