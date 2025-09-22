import uuid
import asyncio
import json
import pickle
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader, WebBaseLoader
)

from core.config import default_rag_config
from core.embeddings import get_embeddings
from core.models import IngestRequest, TaskStatus

class IngestService:
    def __init__(self):
        self.embeddings = get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=default_rag_config.document.chunk_size,
            chunk_overlap=default_rag_config.document.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        self.tasks: Dict[str, TaskStatus] = {}
        # 使用本地存储替代 Milvus
        self.vector_store = {}  # 存储向量数据
        self.documents_store = {}  # 存储文档数据
        self.storage_path = Path("data")
        self.storage_path.mkdir(exist_ok=True)
        self._load_local_storage()
    
    def _load_local_storage(self):
        """加载本地存储的向量数据"""
        try:
            vector_file = self.storage_path / "vectors.pkl"
            docs_file = self.storage_path / "documents.pkl"
            
            if vector_file.exists():
                with open(vector_file, 'rb') as f:
                    self.vector_store = pickle.load(f)
                print(f"Loaded {len(self.vector_store)} vectors from local storage")
            
            if docs_file.exists():
                with open(docs_file, 'rb') as f:
                    self.documents_store = pickle.load(f)
                print(f"Loaded {len(self.documents_store)} documents from local storage")
                    
        except Exception as e:
            print(f"Failed to load local storage: {e}")
            self.vector_store = {}
            self.documents_store = {}
    
    def _save_local_storage(self):
        """保存向量数据到本地存储"""
        try:
            vector_file = self.storage_path / "vectors.pkl"
            docs_file = self.storage_path / "documents.pkl"
            
            with open(vector_file, 'wb') as f:
                pickle.dump(self.vector_store, f)
            
            with open(docs_file, 'wb') as f:
                pickle.dump(self.documents_store, f)
                
            print(f"Saved {len(self.vector_store)} vectors and {len(self.documents_store)} documents to local storage")
            
        except Exception as e:
            print(f"Failed to save local storage: {e}")
    
    def generate_task_id(self) -> str:
        """生成任务ID"""
        return str(uuid.uuid4())
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        if task_id not in self.tasks:
            return {
                "task_id": task_id,
                "status": "not_found",
                "message": "Task not found",
                "progress": 0
            }
        task = self.tasks[task_id]
        return {
            "task_id": task.task_id,
            "status": task.status,
            "progress": task.progress,
            "message": task.message,
            "created_at": task.created_at,
            "updated_at": task.updated_at
        }
    
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
        """存储文档块到本地存储"""
        if not chunks:
            return
        
        # 准备数据
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        
        # 构建存储数据
        stored_count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc_id = chunk.metadata.get('source', f'doc_{i}')
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # 存储向量数据
            self.vector_store[chunk_id] = {
                "vector": embedding,
                "text": chunk.page_content,
                "metadata": {
                    "source": chunk.metadata.get('source', ''),
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "dataset_id": dataset_id,
                    "created_at": datetime.now().isoformat(),
                }
            }
            
            # 存储文档数据
            self.documents_store[chunk_id] = {
                "id": chunk_id,
                "text": chunk.page_content,
                "source": chunk.metadata.get('source', ''),
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "dataset_id": dataset_id,
                "created_at": datetime.now().isoformat(),
            }
            
            stored_count += 1
        
        # 保存到本地文件
        self._save_local_storage()
        
        print(f"Stored {stored_count} chunks to local storage")
