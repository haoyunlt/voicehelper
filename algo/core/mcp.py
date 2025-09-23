"""
MCP (Model Context Protocol) 客户端实现
支持文件系统、HTTP、数据库等工具的标准化调用
"""

import json
import asyncio
import aiohttp
import aiofiles
import asyncpg
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import hashlib

from core.config import default_rag_config

class MCPClient:
    """MCP 客户端 - 统一的工具调用接口"""
    
    def __init__(self):
        self.tools = self._register_tools()
        self.audit_log = []
        self.rate_limiter = RateLimiter()
        
    def _register_tools(self) -> Dict[str, Any]:
        """注册可用工具"""
        return {
            "filesystem": FileSystemTool(),
            "http": HTTPTool(),
            "database": DatabaseTool(),
            "github": GitHubTool(),
        }
    
    async def call_tool(
        self, 
        tool_name: str, 
        operation: str, 
        params: Dict[str, Any],
        tenant_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """调用工具"""
        
        # 检查工具是否存在
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"Tool {tool_name} not found"
            }
        
        # 速率限制检查
        if not await self.rate_limiter.check(tenant_id, tool_name):
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        # 审计日志
        audit_entry = {
            "timestamp": datetime.now().isoformat(),
            "tenant_id": tenant_id,
            "user_id": user_id,
            "tool": tool_name,
            "operation": operation,
            "params": self._sanitize_params(params)
        }
        
        try:
            # 调用工具
            tool = self.tools[tool_name]
            result = await tool.execute(operation, params)
            
            # 记录成功
            audit_entry["success"] = True
            audit_entry["result_size"] = len(str(result))
            
            return {
                "success": True,
                "result": result,
                "metadata": {
                    "tool": tool_name,
                    "operation": operation,
                    "timestamp": audit_entry["timestamp"]
                }
            }
            
        except Exception as e:
            # 记录失败
            audit_entry["success"] = False
            audit_entry["error"] = str(e)
            
            return {
                "success": False,
                "error": str(e)
            }
        
        finally:
            # 保存审计日志
            self.audit_log.append(audit_entry)
            await self._persist_audit_log(audit_entry)
    
    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """清理敏感参数"""
        sanitized = params.copy()
        sensitive_keys = ["password", "token", "key", "secret"]
        
        for key in sanitized:
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
        
        return sanitized
    
    async def _persist_audit_log(self, entry: Dict[str, Any]):
        """持久化审计日志"""
        # 这里简化处理，实际应该写入数据库
        print(f"MCP Audit: {json.dumps(entry, ensure_ascii=False)}")
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        tools_info = []
        for name, tool in self.tools.items():
            tools_info.append({
                "name": name,
                "operations": tool.get_operations(),
                "description": tool.description
            })
        return tools_info


class FileSystemTool:
    """文件系统工具（只读）"""
    
    def __init__(self):
        self.description = "Read-only file system access"
        self.allowed_paths = ["/data", "/config"]  # 白名单路径
    
    def get_operations(self) -> List[str]:
        return ["read", "list", "stat"]
    
    async def execute(self, operation: str, params: Dict[str, Any]) -> Any:
        """执行文件系统操作"""
        
        if operation == "read":
            return await self._read_file(params.get("path"))
        elif operation == "list":
            return await self._list_directory(params.get("path"))
        elif operation == "stat":
            return await self._get_file_info(params.get("path"))
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _read_file(self, filepath: str) -> str:
        """读取文件"""
        # 安全检查
        if not self._is_safe_path(filepath):
            raise PermissionError(f"Access denied: {filepath}")
        
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {filepath}")
        
        # 文件大小限制（10MB）
        if path.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("File too large (>10MB)")
        
        async with aiofiles.open(filepath, 'r') as f:
            content = await f.read()
        
        return content
    
    async def _list_directory(self, dirpath: str) -> List[Dict[str, Any]]:
        """列出目录内容"""
        if not self._is_safe_path(dirpath):
            raise PermissionError(f"Access denied: {dirpath}")
        
        path = Path(dirpath)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dirpath}")
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {dirpath}")
        
        items = []
        for item in path.iterdir():
            items.append({
                "name": item.name,
                "type": "file" if item.is_file() else "directory",
                "size": item.stat().st_size if item.is_file() else None,
                "modified": datetime.fromtimestamp(item.stat().st_mtime).isoformat()
            })
        
        return items
    
    async def _get_file_info(self, filepath: str) -> Dict[str, Any]:
        """获取文件信息"""
        if not self._is_safe_path(filepath):
            raise PermissionError(f"Access denied: {filepath}")
        
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        stat = path.stat()
        return {
            "path": str(path),
            "type": "file" if path.is_file() else "directory",
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:]
        }
    
    def _is_safe_path(self, filepath: str) -> bool:
        """检查路径是否安全"""
        path = Path(filepath).resolve()
        
        # 检查路径遍历攻击
        if ".." in str(path):
            return False
        
        # 检查白名单
        for allowed in self.allowed_paths:
            if str(path).startswith(allowed):
                return True
        
        return False


class HTTPTool:
    """HTTP工具（只读）"""
    
    def __init__(self):
        self.description = "HTTP client for fetching resources"
        self.allowed_domains = ["api.example.com", "docs.example.com"]
        self.timeout = 10
    
    def get_operations(self) -> List[str]:
        return ["get", "head"]
    
    async def execute(self, operation: str, params: Dict[str, Any]) -> Any:
        """执行HTTP请求"""
        
        if operation == "get":
            return await self._http_get(params.get("url"), params.get("headers", {}))
        elif operation == "head":
            return await self._http_head(params.get("url"))
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _http_get(self, url: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """HTTP GET请求"""
        if not self._is_allowed_url(url):
            raise PermissionError(f"URL not allowed: {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=self.timeout) as response:
                content = await response.text()
                
                return {
                    "status": response.status,
                    "headers": dict(response.headers),
                    "content": content[:10000],  # 限制返回内容大小
                    "content_type": response.content_type
                }
    
    async def _http_head(self, url: str) -> Dict[str, Any]:
        """HTTP HEAD请求"""
        if not self._is_allowed_url(url):
            raise PermissionError(f"URL not allowed: {url}")
        
        async with aiohttp.ClientSession() as session:
            async with session.head(url, timeout=self.timeout) as response:
                return {
                    "status": response.status,
                    "headers": dict(response.headers)
                }
    
    def _is_allowed_url(self, url: str) -> bool:
        """检查URL是否允许"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        # 只允许HTTPS
        if parsed.scheme != "https":
            return False
        
        # 检查域名白名单
        for domain in self.allowed_domains:
            if parsed.netloc == domain or parsed.netloc.endswith(f".{domain}"):
                return True
        
        return False


class DatabaseTool:
    """数据库工具（只读）"""
    
    def __init__(self):
        self.description = "Read-only database access"
        self.connection_pool = None
    
    def get_operations(self) -> List[str]:
        return ["query", "describe"]
    
    async def execute(self, operation: str, params: Dict[str, Any]) -> Any:
        """执行数据库操作"""
        
        if operation == "query":
            return await self._query(params.get("sql"), params.get("params", []))
        elif operation == "describe":
            return await self._describe_table(params.get("table"))
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _query(self, sql: str, params: List[Any]) -> List[Dict[str, Any]]:
        """执行查询"""
        # SQL注入防护
        if not self._is_safe_sql(sql):
            raise PermissionError("Unsafe SQL query")
        
        # 连接池初始化
        if not self.connection_pool:
            self.connection_pool = await asyncpg.create_pool(
                host=config.DB_HOST,
                port=config.DB_PORT,
                user=config.DB_USER,
                password=config.DB_PASSWORD,
                database=config.DB_NAME,
                min_size=1,
                max_size=5
            )
        
        async with self.connection_pool.acquire() as conn:
            # 设置只读事务
            async with conn.transaction(readonly=True):
                # 限制返回行数
                if "LIMIT" not in sql.upper():
                    sql += " LIMIT 100"
                
                rows = await conn.fetch(sql, *params)
                
                # 转换为字典列表
                result = []
                for row in rows:
                    result.append(dict(row))
                
                return result
    
    async def _describe_table(self, table_name: str) -> Dict[str, Any]:
        """描述表结构"""
        if not self._is_safe_table_name(table_name):
            raise PermissionError(f"Invalid table name: {table_name}")
        
        sql = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = $1
        ORDER BY ordinal_position
        """
        
        columns = await self._query(sql, [table_name])
        
        return {
            "table": table_name,
            "columns": columns
        }
    
    def _is_safe_sql(self, sql: str) -> bool:
        """检查SQL是否安全"""
        sql_upper = sql.upper()
        
        # 只允许SELECT查询
        if not sql_upper.strip().startswith("SELECT"):
            return False
        
        # 禁止危险操作
        dangerous_keywords = [
            "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
            "TRUNCATE", "EXEC", "EXECUTE", "GRANT", "REVOKE"
        ]
        
        for keyword in dangerous_keywords:
            if keyword in sql_upper:
                return False
        
        return True
    
    def _is_safe_table_name(self, table_name: str) -> bool:
        """检查表名是否安全"""
        import re
        # 只允许字母、数字、下划线
        return bool(re.match(r'^[a-zA-Z0-9_]+$', table_name))


class GitHubTool:
    """GitHub工具（只读）"""
    
    def __init__(self):
        self.description = "GitHub repository access (read-only)"
        self.api_base = "https://api.github.com"
        self.allowed_repos = ["example/repo1", "example/repo2"]
    
    def get_operations(self) -> List[str]:
        return ["get_file", "list_files", "get_commits"]
    
    async def execute(self, operation: str, params: Dict[str, Any]) -> Any:
        """执行GitHub操作"""
        
        if operation == "get_file":
            return await self._get_file(
                params.get("repo"),
                params.get("path"),
                params.get("branch", "main")
            )
        elif operation == "list_files":
            return await self._list_files(
                params.get("repo"),
                params.get("path", ""),
                params.get("branch", "main")
            )
        elif operation == "get_commits":
            return await self._get_commits(
                params.get("repo"),
                params.get("limit", 10)
            )
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    async def _get_file(self, repo: str, filepath: str, branch: str) -> Dict[str, Any]:
        """获取文件内容"""
        if not self._is_allowed_repo(repo):
            raise PermissionError(f"Repository not allowed: {repo}")
        
        url = f"{self.api_base}/repos/{repo}/contents/{filepath}?ref={branch}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to get file: {response.status}")
                
                data = await response.json()
                
                # 解码内容（GitHub返回base64编码）
                import base64
                content = base64.b64decode(data.get("content", "")).decode('utf-8')
                
                return {
                    "path": filepath,
                    "content": content,
                    "size": data.get("size"),
                    "sha": data.get("sha")
                }
    
    async def _list_files(self, repo: str, path: str, branch: str) -> List[Dict[str, Any]]:
        """列出目录内容"""
        if not self._is_allowed_repo(repo):
            raise PermissionError(f"Repository not allowed: {repo}")
        
        url = f"{self.api_base}/repos/{repo}/contents/{path}?ref={branch}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to list files: {response.status}")
                
                data = await response.json()
                
                files = []
                for item in data:
                    files.append({
                        "name": item.get("name"),
                        "type": item.get("type"),
                        "path": item.get("path"),
                        "size": item.get("size")
                    })
                
                return files
    
    async def _get_commits(self, repo: str, limit: int) -> List[Dict[str, Any]]:
        """获取提交历史"""
        if not self._is_allowed_repo(repo):
            raise PermissionError(f"Repository not allowed: {repo}")
        
        url = f"{self.api_base}/repos/{repo}/commits?per_page={limit}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to get commits: {response.status}")
                
                data = await response.json()
                
                commits = []
                for commit in data:
                    commits.append({
                        "sha": commit.get("sha"),
                        "message": commit.get("commit", {}).get("message"),
                        "author": commit.get("commit", {}).get("author", {}).get("name"),
                        "date": commit.get("commit", {}).get("author", {}).get("date")
                    })
                
                return commits
    
    def _is_allowed_repo(self, repo: str) -> bool:
        """检查仓库是否允许"""
        return repo in self.allowed_repos


class RateLimiter:
    """速率限制器"""
    
    def __init__(self):
        self.limits = {
            "filesystem": {"calls": 100, "window": 60},  # 每分钟100次
            "http": {"calls": 50, "window": 60},         # 每分钟50次
            "database": {"calls": 30, "window": 60},     # 每分钟30次
            "github": {"calls": 60, "window": 3600}      # 每小时60次
        }
        self.counters = {}
    
    async def check(self, tenant_id: str, tool_name: str) -> bool:
        """检查是否超过速率限制"""
        key = f"{tenant_id}:{tool_name}"
        current_time = datetime.now().timestamp()
        
        if key not in self.counters:
            self.counters[key] = {
                "count": 0,
                "window_start": current_time
            }
        
        counter = self.counters[key]
        limit = self.limits.get(tool_name, {"calls": 10, "window": 60})
        
        # 检查时间窗口
        if current_time - counter["window_start"] > limit["window"]:
            # 重置计数器
            counter["count"] = 0
            counter["window_start"] = current_time
        
        # 检查是否超限
        if counter["count"] >= limit["calls"]:
            return False
        
        # 增加计数
        counter["count"] += 1
        return True
