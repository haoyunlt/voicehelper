"""
URL获取工具
基于V2架构的BaseTool实现HTTP请求功能
"""

from typing import Dict, Any
import httpx
from loguru import logger

from ..base.runnable import BaseTool


class FetchTool(BaseTool):
    """URL 获取工具"""
    
    name: str = "fetch_url"
    description: str = "GET 一个URL并返回文本内容"
    args_schema: dict = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "format": "uri"},
            "timeout": {"type": "number", "default": 5},
            "max_length": {"type": "number", "default": 4000}
        },
        "required": ["url"]
    }
    
    # 重试配置
    max_retries: int = 2
    retry_delay: float = 0.5
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        执行URL获取
        
        Args:
            **kwargs: 包含url、timeout、max_length等参数
            
        Returns:
            包含text、url、status等字段的结果
        """
        validated_args = self.validate_args(**kwargs)
        url = validated_args["url"]
        timeout = validated_args.get("timeout", 5)
        max_length = validated_args.get("max_length", 4000)
        
        def _fetch():
            logger.info(f"获取URL: {url}")
            
            with httpx.Client() as client:
                resp = client.get(
                    url, 
                    timeout=timeout,
                    headers={
                        "User-Agent": "VoiceHelper/2.0 (AI Assistant)"
                    }
                )
                resp.raise_for_status()
                
                # 获取文本内容并限制长度
                text = resp.text
                if len(text) > max_length:
                    text = text[:max_length] + "..."
                    logger.warning(f"内容被截断到 {max_length} 字符")
                
                return text
        
        try:
            text = self._retry(_fetch)
            
            result = {
                "text": text,
                "url": url,
                "status": "success",
                "length": len(text)
            }
            
            logger.info(f"URL获取成功: {url}, 长度: {len(text)}")
            return result
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP错误 {e.response.status_code}: {e.response.reason_phrase}"
            logger.error(f"URL获取失败: {url}, {error_msg}")
            return {
                "text": "",
                "url": url,
                "status": "error",
                "error": error_msg
            }
            
        except httpx.TimeoutException:
            error_msg = f"请求超时 ({timeout}s)"
            logger.error(f"URL获取失败: {url}, {error_msg}")
            return {
                "text": "",
                "url": url,
                "status": "error", 
                "error": error_msg
            }
            
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            logger.error(f"URL获取失败: {url}, {error_msg}")
            return {
                "text": "",
                "url": url,
                "status": "error",
                "error": error_msg
            }
