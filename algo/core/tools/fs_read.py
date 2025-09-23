"""
文件系统读取工具
基于V2架构的BaseTool实现本地文件读取功能
"""

import os
from typing import Dict, Any
from pathlib import Path
from loguru import logger

from ..base.runnable import BaseTool


class FsReadTool(BaseTool):
    """文件系统读取工具"""
    
    name: str = "read_file"
    description: str = "读取本地文件内容"
    args_schema: dict = {
        "type": "object",
        "properties": {
            "file_path": {"type": "string"},
            "encoding": {"type": "string", "default": "utf-8"},
            "max_length": {"type": "number", "default": 8000}
        },
        "required": ["file_path"]
    }
    
    # 安全配置
    allowed_extensions = {".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".csv"}
    base_path = Path("data/documents")  # 限制访问范围
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        执行文件读取
        
        Args:
            **kwargs: 包含file_path、encoding、max_length等参数
            
        Returns:
            包含content、file_path、status等字段的结果
        """
        validated_args = self.validate_args(**kwargs)
        file_path = validated_args["file_path"]
        encoding = validated_args.get("encoding", "utf-8")
        max_length = validated_args.get("max_length", 8000)
        
        try:
            # 安全检查
            full_path = self._validate_path(file_path)
            
            logger.info(f"读取文件: {full_path}")
            
            # 检查文件是否存在
            if not full_path.exists():
                return {
                    "content": "",
                    "file_path": file_path,
                    "status": "error",
                    "error": "文件不存在"
                }
            
            # 检查是否是文件
            if not full_path.is_file():
                return {
                    "content": "",
                    "file_path": file_path,
                    "status": "error",
                    "error": "路径不是文件"
                }
            
            # 读取文件内容
            with open(full_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            # 限制内容长度
            if len(content) > max_length:
                content = content[:max_length] + "..."
                logger.warning(f"文件内容被截断到 {max_length} 字符")
            
            result = {
                "content": content,
                "file_path": file_path,
                "status": "success",
                "size": len(content),
                "encoding": encoding
            }
            
            logger.info(f"文件读取成功: {file_path}, 大小: {len(content)}")
            return result
            
        except UnicodeDecodeError:
            error_msg = f"编码错误，无法使用 {encoding} 解码"
            logger.error(f"文件读取失败: {file_path}, {error_msg}")
            return {
                "content": "",
                "file_path": file_path,
                "status": "error",
                "error": error_msg
            }
            
        except PermissionError:
            error_msg = "权限不足，无法读取文件"
            logger.error(f"文件读取失败: {file_path}, {error_msg}")
            return {
                "content": "",
                "file_path": file_path,
                "status": "error",
                "error": error_msg
            }
            
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"文件读取失败: {file_path}, {error_msg}")
            return {
                "content": "",
                "file_path": file_path,
                "status": "error",
                "error": error_msg
            }
    
    def _validate_path(self, file_path: str) -> Path:
        """
        验证文件路径安全性
        
        Args:
            file_path: 文件路径
            
        Returns:
            验证后的完整路径
            
        Raises:
            ValueError: 路径不安全
        """
        path = Path(file_path)
        
        # 如果是相对路径，基于base_path解析
        if not path.is_absolute():
            full_path = self.base_path / path
        else:
            full_path = path
        
        # 解析为绝对路径
        full_path = full_path.resolve()
        
        # 检查是否在允许的基础路径内
        try:
            full_path.relative_to(self.base_path.resolve())
        except ValueError:
            raise ValueError(f"路径超出允许范围: {file_path}")
        
        # 检查文件扩展名
        if full_path.suffix.lower() not in self.allowed_extensions:
            raise ValueError(f"不支持的文件类型: {full_path.suffix}")
        
        return full_path
