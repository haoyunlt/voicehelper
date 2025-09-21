import uuid
import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader, WebBaseLoader
)
from langchain_milvus import Milvus
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType

from core.config import config
from core.embeddings import get_embeddings
from core.models import IngestRequest, TaskStatus

class IngestService:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.DEFAULT_CHUNK_SIZE,
            chunk_overlap=config.DEFAULT_CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        self.tasks: Dict[str, TaskStatus] = {}
        self._init_milvus()
    
    def _init_milvus(self):
        """初始化 Milvus 连接"""
        try:
            connections.connect(
                alias="default",
                host=config.MILVUS_HOST,
                port=config.MILVUS_PORT,
                user=config.MILVUS_USER,
                password=config.MILVUS_PASSWORD
            )
            
            # 创建集合（如果不存在）
            self._create_collection_if_not_exists()
            
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            raise
    
    def _create_collection_if_not_exists(self):
        """创建集合"""
        collection_name = config.DEFAULT_COLLECTION_NAME
        
        if utility.has_collection(collection_name):
            print(f"Collection {collection_name} already exists")
            return
        
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=config.EMBEDDING_DIMENSION),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="dataset_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=50),
        ]
        
        # 创建集合
        schema = CollectionSchema(fields, f"Chatbot knowledge base collection")
        collection = Collection(collection_name, schema)
        
        # 创建索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        }
        collection.create_index("vector", index_params)
        
        print(f"Created collection {collection_name}")
    
    def generate_task_id(self) -> str:
        """生成任务ID"""
        return str(uuid.uuid4())
    
    def get_task_status(self, task_id: str) -> TaskStatus:
        """获取任务状态"""
        if task_id not in self.tasks:
            return TaskStatus(
                task_id=task_id,
                status="not_found",
                message="Task not found"
            )
        return self.tasks[task_id]
    
    async def process_ingest_task(self, task_id: str, request: IngestRequest):
        """处理入库任务"""
        # 初始化任务状态
        self.tasks[task_id] = TaskStatus(
            task_id=task_id,
            status="processing",
            progress=0,
            created_at=datetime.now().isoformat()
        )
        
        try:
            documents = []
            total_items = len(request.files or []) + len(request.urls or [])
            processed_items = 0
            
            # 处理文件
            if request.files:
                for file_path in request.files:
                    try:
                        docs = await self._load_file(file_path)
                        documents.extend(docs)
                        processed_items += 1
                        
                        # 更新进度
                        progress = int((processed_items / total_items) * 50)  # 文件加载占50%
                        self.tasks[task_id].progress = progress
                        
                    except Exception as e:
                        print(f"Error loading file {file_path}: {e}")
            
            # 处理URL
            if request.urls:
                for url in request.urls:
                    try:
                        docs = await self._load_url(url)
                        documents.extend(docs)
                        processed_items += 1
                        
                        # 更新进度
                        progress = int((processed_items / total_items) * 50)
                        self.tasks[task_id].progress = progress
                        
                    except Exception as e:
                        print(f"Error loading URL {url}: {e}")
            
            if not documents:
                self.tasks[task_id].status = "failed"
                self.tasks[task_id].message = "No documents loaded"
                return
            
            # 文本切分
            self.tasks[task_id].progress = 60
            chunks = self.text_splitter.split_documents(documents)
            
            # 向量化并存储
            self.tasks[task_id].progress = 70
            await self._store_chunks(chunks, request.dataset_id)
            
            # 完成
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].progress = 100
            self.tasks[task_id].message = f"Successfully processed {len(chunks)} chunks"
            self.tasks[task_id].updated_at = datetime.now().isoformat()
            
        except Exception as e:
            self.tasks[task_id].status = "failed"
            self.tasks[task_id].message = str(e)
            self.tasks[task_id].updated_at = datetime.now().isoformat()
            print(f"Task {task_id} failed: {e}")
    
    async def _load_file(self, file_path: str) -> List:
        """加载文件"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif file_ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_ext == '.html':
            loader = UnstructuredHTMLLoader(file_path)
        elif file_ext == '.md':
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            # 默认作为文本文件处理
            loader = TextLoader(file_path, encoding='utf-8')
        
        return loader.load()
    
    async def _load_url(self, url: str) -> List:
        """加载URL"""
        loader = WebBaseLoader([url])
        return loader.load()
    
    async def _store_chunks(self, chunks: List, dataset_id: str):
        """存储文档块到 Milvus"""
        if not chunks:
            return
        
        # 准备数据
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        
        # 构建插入数据
        entities = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = chunk.metadata.get('source', f'doc_{i}')
            chunk_id = f"{doc_id}_chunk_{i}"
            
            entities.append({
                "id": chunk_id,
                "text": chunk.page_content,
                "vector": embedding,
                "source": chunk.metadata.get('source', ''),
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "dataset_id": dataset_id,
                "created_at": datetime.now().isoformat(),
            })
        
        # 插入到 Milvus
        collection = Collection(config.DEFAULT_COLLECTION_NAME)
        collection.insert(entities)
        collection.flush()
        
        # 加载集合（用于搜索）
        collection.load()
        
        print(f"Stored {len(entities)} chunks to Milvus")
