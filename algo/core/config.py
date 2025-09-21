import os
from typing import Optional

class Config:
    # Milvus 配置
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_USER: Optional[str] = os.getenv("MILVUS_USER")
    MILVUS_PASSWORD: Optional[str] = os.getenv("MILVUS_PASSWORD")
    
    # 豆包 API 配置
    ARK_API_KEY: str = os.getenv("ARK_API_KEY", "")
    ARK_BASE_URL: str = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
    ARK_MODEL: str = os.getenv("ARK_MODEL", "ep-20241201140014-vbzjz")  # 豆包模型ID
    
    # Embedding 配置
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    
    # 默认参数
    DEFAULT_CHUNK_SIZE: int = 600
    DEFAULT_CHUNK_OVERLAP: int = 80
    DEFAULT_TOP_K: int = 5
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.3
    
    # 集合名称
    DEFAULT_COLLECTION_NAME: str = "chatbot_knowledge"
    
    @classmethod
    def get_milvus_uri(cls) -> str:
        """获取 Milvus 连接 URI"""
        if cls.MILVUS_USER and cls.MILVUS_PASSWORD:
            return f"http://{cls.MILVUS_USER}:{cls.MILVUS_PASSWORD}@{cls.MILVUS_HOST}:{cls.MILVUS_PORT}"
        return f"http://{cls.MILVUS_HOST}:{cls.MILVUS_PORT}"

config = Config()
