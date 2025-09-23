"""
GitHub读取工具
基于V2架构的BaseTool实现GitHub仓库内容读取功能
"""

import base64
from typing import Dict, Any
import httpx
from loguru import logger

from ..base.runnable import BaseTool


class GithubReadTool(BaseTool):
    """GitHub读取工具"""
    
    name: str = "read_github"
    description: str = "读取GitHub仓库中的文件内容"
    args_schema: dict = {
        "type": "object",
        "properties": {
            "repo": {"type": "string", "description": "仓库名称，格式: owner/repo"},
            "path": {"type": "string", "description": "文件路径"},
            "branch": {"type": "string", "default": "main", "description": "分支名称"},
            "max_length": {"type": "number", "default": 8000}
        },
        "required": ["repo", "path"]
    }
    
    # 重试配置
    max_retries: int = 2
    retry_delay: float = 1.0
    
    def __init__(self, github_token: str = None, **kwargs):
        super().__init__(**kwargs)
        self.github_token = github_token
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        执行GitHub文件读取
        
        Args:
            **kwargs: 包含repo、path、branch、max_length等参数
            
        Returns:
            包含content、repo、path、status等字段的结果
        """
        validated_args = self.validate_args(**kwargs)
        repo = validated_args["repo"]
        path = validated_args["path"]
        branch = validated_args.get("branch", "main")
        max_length = validated_args.get("max_length", 8000)
        
        def _fetch_github():
            logger.info(f"读取GitHub文件: {repo}/{path} (branch: {branch})")
            
            # 构建GitHub API URL
            api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
            params = {"ref": branch}
            
            # 设置请求头
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "VoiceHelper/2.0"
            }
            
            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"
            
            with httpx.Client() as client:
                resp = client.get(api_url, params=params, headers=headers, timeout=10)
                resp.raise_for_status()
                
                data = resp.json()
                
                # 检查是否是文件
                if data.get("type") != "file":
                    raise ValueError(f"路径不是文件: {path}")
                
                # 解码Base64内容
                content_b64 = data.get("content", "")
                content = base64.b64decode(content_b64).decode("utf-8")
                
                return content, data
        
        try:
            content, file_info = self._retry(_fetch_github)
            
            # 限制内容长度
            if len(content) > max_length:
                content = content[:max_length] + "..."
                logger.warning(f"内容被截断到 {max_length} 字符")
            
            result = {
                "content": content,
                "repo": repo,
                "path": path,
                "branch": branch,
                "status": "success",
                "size": len(content),
                "sha": file_info.get("sha", ""),
                "url": file_info.get("html_url", "")
            }
            
            logger.info(f"GitHub文件读取成功: {repo}/{path}, 大小: {len(content)}")
            return result
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                error_msg = "文件或仓库不存在"
            elif e.response.status_code == 403:
                error_msg = "访问被拒绝，可能需要认证"
            elif e.response.status_code == 401:
                error_msg = "认证失败，请检查GitHub token"
            else:
                error_msg = f"HTTP错误 {e.response.status_code}"
            
            logger.error(f"GitHub文件读取失败: {repo}/{path}, {error_msg}")
            return {
                "content": "",
                "repo": repo,
                "path": path,
                "branch": branch,
                "status": "error",
                "error": error_msg
            }
            
        except UnicodeDecodeError:
            error_msg = "文件编码错误，可能是二进制文件"
            logger.error(f"GitHub文件读取失败: {repo}/{path}, {error_msg}")
            return {
                "content": "",
                "repo": repo,
                "path": path,
                "branch": branch,
                "status": "error",
                "error": error_msg
            }
            
        except ValueError as e:
            error_msg = str(e)
            logger.error(f"GitHub文件读取失败: {repo}/{path}, {error_msg}")
            return {
                "content": "",
                "repo": repo,
                "path": path,
                "branch": branch,
                "status": "error",
                "error": error_msg
            }
            
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"GitHub文件读取失败: {repo}/{path}, {error_msg}")
            return {
                "content": "",
                "repo": repo,
                "path": path,
                "branch": branch,
                "status": "error",
                "error": error_msg
            }
