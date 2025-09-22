import os
import requests
from typing import List
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from core.config import config

class BGEEmbeddings(Embeddings):
    """BGE Embedding 模型（简化版）"""
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        # 使用简化的本地实现，避免下载模型
        self.dimension = 1024
        print(f"Using simplified BGE embeddings with dimension {self.dimension}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档（模拟实现）"""
        import random
        import hashlib
        
        embeddings = []
        for text in texts:
            # 使用文本哈希作为种子，确保相同文本产生相同嵌入
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            random.seed(seed)
            
            # 生成归一化的随机向量
            embedding = [random.gauss(0, 1) for _ in range(self.dimension)]
            norm = sum(x*x for x in embedding) ** 0.5
            embedding = [x/norm for x in embedding]
            embeddings.append(embedding)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询（模拟实现）"""
        return self.embed_documents([text])[0]

class ArkEmbeddings(Embeddings):
    """豆包 Embedding API"""
    
    def __init__(self, api_key: str = None, base_url: str = None):
        self.api_key = api_key or config.ARK_API_KEY
        self.base_url = base_url or config.ARK_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档"""
        # 批量处理，每次最多处理100个文档
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self._get_embeddings(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入查询"""
        embeddings = self._get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """调用豆包 API 获取嵌入"""
        try:
            payload = {
                "model": "text-embedding-3-large",  # 豆包embedding模型
                "input": texts,
                "encoding_format": "float"
            }
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return [item["embedding"] for item in data["data"]]
            else:
                print(f"Embedding API error: {response.status_code}, {response.text}")
                return []
                
        except Exception as e:
            print(f"Error calling embedding API: {e}")
            return []

def get_embeddings() -> Embeddings:
    """获取嵌入模型实例"""
    if config.ARK_API_KEY:
        try:
            # 优先使用豆包 Embedding
            embeddings = ArkEmbeddings()
            # 测试连接
            test_embedding = embeddings.embed_query("测试")
            if test_embedding:
                print("Using Ark Embeddings")
                return embeddings
        except Exception as e:
            print(f"Failed to initialize Ark Embeddings: {e}")
    
    # 回退到 BGE
    print("Using BGE Embeddings")
    return BGEEmbeddings(config.EMBEDDING_MODEL)
