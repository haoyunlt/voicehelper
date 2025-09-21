from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
import os
from dotenv import load_dotenv

from core.ingest import IngestService
from core.retrieve import RetrieveService
from core.voice import VoiceService
from core.models import QueryRequest, IngestRequest, IngestResponse, VoiceQueryRequest, VoiceQueryResponse

# 加载环境变量
load_dotenv()

app = FastAPI(
    title="Chatbot Algorithm Service",
    description="基于 LangChain 的 RAG 算法服务",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化服务
ingest_service = IngestService()
retrieve_service = RetrieveService()
voice_service = VoiceService(retrieve_service)

@app.get("/")
async def root():
    return {"message": "Chatbot Algorithm Service", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    request: IngestRequest,
    background_tasks: BackgroundTasks
):
    """文档入库接口"""
    try:
        # 生成任务ID
        task_id = ingest_service.generate_task_id()
        
        # 后台处理入库任务
        background_tasks.add_task(
            ingest_service.process_ingest_task,
            task_id,
            request
        )
        
        return IngestResponse(task_id=task_id)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(request: QueryRequest):
    """查询接口，返回流式响应"""
    try:
        # 生成流式响应
        return StreamingResponse(
            retrieve_service.stream_query(request),
            media_type="application/x-ndjson"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    try:
        status = ingest_service.get_task_status(task_id)
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice/query")
async def voice_query(request: VoiceQueryRequest):
    """语音查询接口"""
    try:
        return StreamingResponse(
            voice_service.process_voice_query(request),
            media_type="application/x-ndjson"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cancel")
async def cancel_request(request: Request):
    """取消请求"""
    try:
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            raise HTTPException(status_code=400, detail="Request ID required")
        
        await voice_service.cancel_request(request_id)
        return {"status": "cancelled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=True if os.getenv("ENV") == "development" else False
    )
