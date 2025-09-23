"""
带参数验证的API端点
在原有API端点基础上添加严格的参数验证
"""

from typing import List, Dict, Any, Optional
from fastapi import HTTPException, UploadFile, File, Form, Depends
from pydantic import BaseModel
import logging

from .validation.parameter_validator import ParameterValidator, ValidationException, get_validator
from .validation.request_models import (
    DocumentInput, IngestRequest, QueryRequest, DeleteRequest,
    TranscribeRequest, SynthesizeRequest, VoiceStreamRequest,
    ChatRequest, CancelChatRequest, MultimodalRequest,
    AgentRequest, ToolExecuteRequest, BatchRequest,
    MetricsRequest, HealthCheckRequest, ConfigUpdateRequest
)
from .validation.decorators import (
    validate_parameters, validate_pydantic_model, validate_required_fields,
    validate_file_upload, validate_rate_limit, validate_authentication
)

logger = logging.getLogger(__name__)


class ValidatedRAGAPI:
    """带验证的RAG API"""
    
    def __init__(self, rag_service):
        self.rag_service = rag_service
        self.validator = get_validator()
    
    @validate_pydantic_model(IngestRequest)
    @validate_rate_limit(max_requests=100, window_seconds=3600)
    @validate_authentication(required_permissions=["document:write"])
    async def ingest_documents(self, request: IngestRequest) -> Dict[str, Any]:
        """
        文档入库接口（带验证）
        
        Args:
            request: 入库请求，包含文档列表和配置
            
        Returns:
            入库结果
            
        Raises:
            ValidationException: 参数验证失败
            HTTPException: 业务逻辑错误
        """
        try:
            # 额外的业务逻辑验证
            total_content_length = sum(len(doc.content) for doc in request.documents)
            if total_content_length > 10 * 1024 * 1024:  # 10MB限制
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "CONTENT_TOO_LARGE",
                        "message": "Total content size exceeds 10MB limit",
                        "total_size": total_content_length,
                        "max_size": 10 * 1024 * 1024
                    }
                )
            
            # 检查文档ID唯一性（与现有文档）
            existing_ids = await self.rag_service.get_existing_document_ids(
                [doc.id for doc in request.documents]
            )
            
            if existing_ids and not request.overwrite:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "DOCUMENT_EXISTS",
                        "message": "Documents with these IDs already exist",
                        "existing_ids": existing_ids,
                        "suggestion": "Set overwrite=true to replace existing documents"
                    }
                )
            
            # 调用原始服务
            result = await self.rag_service.ingest_documents(
                documents=request.documents,
                batch_size=request.batch_size,
                overwrite=request.overwrite
            )
            
            logger.info(f"Successfully ingested {len(request.documents)} documents")
            return result
            
        except ValidationException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document ingestion failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "INGESTION_FAILED",
                    "message": "Failed to ingest documents",
                    "details": str(e)
                }
            )
    
    @validate_pydantic_model(QueryRequest)
    @validate_rate_limit(max_requests=1000, window_seconds=3600)
    @validate_authentication(required_permissions=["document:read"])
    async def query_documents(self, request: QueryRequest) -> Dict[str, Any]:
        """
        文档查询接口（带验证）
        
        Args:
            request: 查询请求
            
        Returns:
            查询结果
        """
        try:
            # 额外验证：检查查询复杂度
            query_words = len(request.query.split())
            if query_words > 100:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "QUERY_TOO_COMPLEX",
                        "message": "Query is too complex (max 100 words)",
                        "word_count": query_words,
                        "max_words": 100
                    }
                )
            
            # 验证过滤条件
            if request.filters:
                allowed_filter_keys = ["category", "language", "source", "created_after", "created_before"]
                invalid_keys = set(request.filters.keys()) - set(allowed_filter_keys)
                if invalid_keys:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "INVALID_FILTERS",
                            "message": "Invalid filter keys",
                            "invalid_keys": list(invalid_keys),
                            "allowed_keys": allowed_filter_keys
                        }
                    )
            
            # 调用原始服务
            result = await self.rag_service.query_documents(
                query=request.query,
                top_k=request.top_k,
                threshold=request.threshold,
                filters=request.filters,
                include_metadata=request.include_metadata,
                search_type=request.search_type.value,
                language=request.language.value
            )
            
            logger.info(f"Query executed successfully: '{request.query[:50]}...'")
            return result
            
        except ValidationException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document query failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "QUERY_FAILED",
                    "message": "Failed to query documents",
                    "details": str(e)
                }
            )
    
    @validate_pydantic_model(DeleteRequest)
    @validate_authentication(required_permissions=["document:delete"])
    async def delete_documents(self, request: DeleteRequest) -> Dict[str, Any]:
        """
        文档删除接口（带验证）
        
        Args:
            request: 删除请求
            
        Returns:
            删除结果
        """
        try:
            # 验证文档是否存在
            existing_ids = await self.rag_service.get_existing_document_ids(request.document_ids)
            non_existing_ids = set(request.document_ids) - set(existing_ids)
            
            if non_existing_ids:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "DOCUMENTS_NOT_FOUND",
                        "message": "Some documents do not exist",
                        "non_existing_ids": list(non_existing_ids),
                        "existing_ids": existing_ids
                    }
                )
            
            # 调用原始服务
            result = await self.rag_service.delete_documents(request.document_ids)
            
            logger.info(f"Successfully deleted {len(request.document_ids)} documents")
            return result
            
        except ValidationException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document deletion failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "DELETION_FAILED",
                    "message": "Failed to delete documents",
                    "details": str(e)
                }
            )


class ValidatedVoiceAPI:
    """带验证的语音API"""
    
    def __init__(self, voice_service):
        self.voice_service = voice_service
        self.validator = get_validator()
    
    @validate_file_upload(
        max_size=50 * 1024 * 1024,  # 50MB
        allowed_extensions=['.wav', '.mp3', '.webm', '.m4a', '.flac'],
        allowed_mime_types=['audio/wav', 'audio/mpeg', 'audio/webm', 'audio/x-m4a', 'audio/flac']
    )
    @validate_rate_limit(max_requests=200, window_seconds=3600)
    @validate_authentication(required_permissions=["voice:transcribe"])
    async def transcribe_audio(
        self,
        audio_file: UploadFile = File(...),
        language: str = Form(...),
        audio_format: str = Form(...),
        sample_rate: int = Form(16000),
        enable_emotion: bool = Form(False),
        enable_speaker: bool = Form(False),
        max_duration: int = Form(300)
    ) -> Dict[str, Any]:
        """
        语音转文字接口（带验证）
        
        Args:
            audio_file: 音频文件
            language: 语言代码
            audio_format: 音频格式
            sample_rate: 采样率
            enable_emotion: 是否启用情感识别
            enable_speaker: 是否启用说话人识别
            max_duration: 最大时长
            
        Returns:
            转录结果
        """
        try:
            # 创建请求对象进行验证
            request_data = {
                "language": language,
                "audio_format": audio_format,
                "sample_rate": sample_rate,
                "enable_emotion": enable_emotion,
                "enable_speaker": enable_speaker,
                "max_duration": max_duration
            }
            
            # 使用Pydantic验证（不包括文件）
            try:
                from .validation.request_models import LanguageCode, AudioFormat
                LanguageCode(language)  # 验证语言代码
                AudioFormat(audio_format)  # 验证音频格式
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "INVALID_PARAMETER",
                        "message": str(e)
                    }
                )
            
            # 验证音频文件内容
            audio_data = await audio_file.read()
            if len(audio_data) == 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "EMPTY_AUDIO_FILE",
                        "message": "Audio file is empty"
                    }
                )
            
            # 重置文件指针
            await audio_file.seek(0)
            
            # 调用原始服务
            result = await self.voice_service.transcribe_audio(
                audio_data=audio_data,
                language=language,
                audio_format=audio_format,
                sample_rate=sample_rate,
                enable_emotion=enable_emotion,
                enable_speaker=enable_speaker,
                max_duration=max_duration
            )
            
            logger.info(f"Audio transcription completed: {audio_file.filename}")
            return result
            
        except ValidationException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Audio transcription failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "TRANSCRIPTION_FAILED",
                    "message": "Failed to transcribe audio",
                    "details": str(e)
                }
            )
    
    @validate_pydantic_model(SynthesizeRequest)
    @validate_rate_limit(max_requests=500, window_seconds=3600)
    @validate_authentication(required_permissions=["voice:synthesize"])
    async def synthesize_text(self, request: SynthesizeRequest) -> Dict[str, Any]:
        """
        文字转语音接口（带验证）
        
        Args:
            request: 合成请求
            
        Returns:
            合成结果
        """
        try:
            # 额外验证：检查文本内容
            if len(request.text.split()) > 500:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "TEXT_TOO_LONG",
                        "message": "Text is too long for synthesis (max 500 words)",
                        "word_count": len(request.text.split()),
                        "max_words": 500
                    }
                )
            
            # 检查敏感内容
            sensitive_patterns = ["敏感词1", "敏感词2"]  # 实际应该从配置加载
            for pattern in sensitive_patterns:
                if pattern in request.text:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "SENSITIVE_CONTENT",
                            "message": "Text contains sensitive content"
                        }
                    )
            
            # 验证语音ID是否存在
            if not await self.voice_service.is_voice_id_valid(request.voice_id):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "INVALID_VOICE_ID",
                        "message": f"Voice ID '{request.voice_id}' is not valid",
                        "available_voices": await self.voice_service.get_available_voices()
                    }
                )
            
            # 调用原始服务
            result = await self.voice_service.synthesize_text(
                text=request.text,
                voice_id=request.voice_id,
                language=request.language.value,
                speed=request.speed,
                pitch=request.pitch,
                volume=request.volume,
                emotion=request.emotion.value if request.emotion else None,
                output_format=request.output_format.value
            )
            
            logger.info(f"Text synthesis completed: {len(request.text)} characters")
            return result
            
        except ValidationException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Text synthesis failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "SYNTHESIS_FAILED",
                    "message": "Failed to synthesize text",
                    "details": str(e)
                }
            )


class ValidatedChatAPI:
    """带验证的对话API"""
    
    def __init__(self, chat_service):
        self.chat_service = chat_service
        self.validator = get_validator()
    
    @validate_pydantic_model(ChatRequest)
    @validate_rate_limit(max_requests=1000, window_seconds=3600)
    @validate_authentication(required_permissions=["chat:send"])
    async def send_message(self, request: ChatRequest) -> Dict[str, Any]:
        """
        发送消息接口（带验证）
        
        Args:
            request: 聊天请求
            
        Returns:
            聊天响应
        """
        try:
            # 验证会话ID（如果提供）
            if request.conversation_id:
                if not await self.chat_service.is_conversation_exists(request.conversation_id):
                    raise HTTPException(
                        status_code=404,
                        detail={
                            "error": "CONVERSATION_NOT_FOUND",
                            "message": f"Conversation '{request.conversation_id}' does not exist"
                        }
                    )
            
            # 验证消息内容
            if len(request.message.strip()) == 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "EMPTY_MESSAGE",
                        "message": "Message content cannot be empty"
                    }
                )
            
            # 检查消息频率（防止刷屏）
            # 实现消息频率检查
            import time
            
            # 生成用户标识键
            user_id = getattr(request, 'user_id', 'anonymous')
            rate_limit_key = f"message_rate:{user_id}"
            
            # 简化实现：使用内存存储
            if not hasattr(self, '_message_rate_cache'):
                self._message_rate_cache = {}
            
            current_time = time.time()
            
            # 清理过期记录（超过60秒的记录）
            expired_keys = [k for k, timestamps in self._message_rate_cache.items() 
                          if all(current_time - ts > 60 for ts in timestamps)]
            for key in expired_keys:
                self._message_rate_cache.pop(key, None)
            
            # 获取用户最近的消息时间戳
            user_timestamps = self._message_rate_cache.get(rate_limit_key, [])
            
            # 移除60秒前的时间戳
            recent_timestamps = [ts for ts in user_timestamps if current_time - ts <= 60]
            
            # 检查频率限制（每分钟最多30条消息）
            max_messages_per_minute = 30
            if len(recent_timestamps) >= max_messages_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "MESSAGE_RATE_LIMIT_EXCEEDED",
                        "message": f"Too many messages. Maximum {max_messages_per_minute} messages per minute allowed",
                        "retry_after": 60
                    }
                )
            
            # 记录当前消息时间戳
            recent_timestamps.append(current_time)
            self._message_rate_cache[rate_limit_key] = recent_timestamps
            
            # 调用原始服务
            result = await self.chat_service.send_message(
                message=request.message,
                conversation_id=request.conversation_id,
                message_type=request.message_type,
                context=request.context,
                stream=request.stream,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                model=request.model
            )
            
            logger.info(f"Message sent successfully: conversation_id={request.conversation_id}")
            return result
            
        except ValidationException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Send message failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "SEND_MESSAGE_FAILED",
                    "message": "Failed to send message",
                    "details": str(e)
                }
            )
    
    @validate_pydantic_model(CancelChatRequest)
    @validate_authentication(required_permissions=["chat:cancel"])
    async def cancel_chat(self, request: CancelChatRequest) -> Dict[str, Any]:
        """
        取消聊天接口（带验证）
        
        Args:
            request: 取消请求
            
        Returns:
            取消结果
        """
        try:
            # 验证会话是否存在且可以取消
            if not await self.chat_service.is_conversation_exists(request.conversation_id):
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "CONVERSATION_NOT_FOUND",
                        "message": f"Conversation '{request.conversation_id}' does not exist"
                    }
                )
            
            if not await self.chat_service.is_conversation_active(request.conversation_id):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "CONVERSATION_NOT_ACTIVE",
                        "message": f"Conversation '{request.conversation_id}' is not active"
                    }
                )
            
            # 调用原始服务
            result = await self.chat_service.cancel_chat(
                conversation_id=request.conversation_id,
                reason=request.reason
            )
            
            logger.info(f"Chat cancelled successfully: conversation_id={request.conversation_id}")
            return result
            
        except ValidationException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Cancel chat failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "CANCEL_CHAT_FAILED",
                    "message": "Failed to cancel chat",
                    "details": str(e)
                }
            )


class ValidatedMultimodalAPI:
    """带验证的多模态API"""
    
    def __init__(self, multimodal_service):
        self.multimodal_service = multimodal_service
        self.validator = get_validator()
    
    @validate_pydantic_model(MultimodalRequest)
    @validate_rate_limit(max_requests=100, window_seconds=3600)
    @validate_authentication(required_permissions=["multimodal:process"])
    async def process_multimodal(self, request: MultimodalRequest) -> Dict[str, Any]:
        """
        多模态处理接口（带验证）
        
        Args:
            request: 多模态请求
            
        Returns:
            处理结果
        """
        try:
            # 验证至少有一种模态输入
            modality_count = sum([
                bool(request.text),
                bool(request.image_data),
                bool(request.audio_data),
                bool(request.video_data)
            ])
            
            if modality_count == 0:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "NO_MODALITY_INPUT",
                        "message": "At least one modality input is required"
                    }
                )
            
            # 验证数据大小
            total_size = 0
            if request.image_data:
                total_size += len(request.image_data)
            if request.audio_data:
                total_size += len(request.audio_data)
            if request.video_data:
                total_size += len(request.video_data)
            
            max_size = 100 * 1024 * 1024  # 100MB
            if total_size > max_size:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "DATA_TOO_LARGE",
                        "message": f"Total data size ({total_size} bytes) exceeds maximum ({max_size} bytes)",
                        "total_size": total_size,
                        "max_size": max_size
                    }
                )
            
            # 调用原始服务
            result = await self.multimodal_service.process_multimodal(
                query=request.query,
                text=request.text,
                image_data=request.image_data,
                audio_data=request.audio_data,
                video_data=request.video_data,
                fusion_type=request.fusion_type.value,
                language=request.language.value if request.language else None,
                output_type=request.output_type
            )
            
            logger.info(f"Multimodal processing completed: {modality_count} modalities")
            return result
            
        except ValidationException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Multimodal processing failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "MULTIMODAL_PROCESSING_FAILED",
                    "message": "Failed to process multimodal request",
                    "details": str(e)
                }
            )


class ValidatedAgentAPI:
    """带验证的Agent API"""
    
    def __init__(self, agent_service):
        self.agent_service = agent_service
        self.validator = get_validator()
    
    @validate_pydantic_model(AgentRequest)
    @validate_rate_limit(max_requests=200, window_seconds=3600)
    @validate_authentication(required_permissions=["agent:query"])
    async def agent_query(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Agent查询接口（带验证）
        
        Args:
            request: Agent请求
            
        Returns:
            Agent响应
        """
        try:
            # 验证工具列表
            if request.tools:
                available_tools = await self.agent_service.get_available_tools()
                invalid_tools = set(request.tools) - set(available_tools)
                if invalid_tools:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "INVALID_TOOLS",
                            "message": "Some tools are not available",
                            "invalid_tools": list(invalid_tools),
                            "available_tools": available_tools
                        }
                    )
            
            # 验证查询复杂度
            query_complexity = len(request.query.split()) + len(request.tools or []) * 10
            if query_complexity > 1000:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "QUERY_TOO_COMPLEX",
                        "message": "Query is too complex",
                        "complexity_score": query_complexity,
                        "max_complexity": 1000
                    }
                )
            
            # 调用原始服务
            result = await self.agent_service.process_query(
                query=request.query,
                session_id=request.session_id,
                tools=request.tools,
                context=request.context,
                temperature=request.temperature,
                max_steps=request.max_steps,
                timeout=request.timeout
            )
            
            logger.info(f"Agent query processed successfully: session_id={request.session_id}")
            return result
            
        except ValidationException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Agent query failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "AGENT_QUERY_FAILED",
                    "message": "Failed to process agent query",
                    "details": str(e)
                }
            )
    
    @validate_pydantic_model(ToolExecuteRequest)
    @validate_authentication(required_permissions=["agent:execute_tool"])
    async def execute_tool(self, request: ToolExecuteRequest) -> Dict[str, Any]:
        """
        执行工具接口（带验证）
        
        Args:
            request: 工具执行请求
            
        Returns:
            执行结果
        """
        try:
            # 验证工具是否可用
            available_tools = await self.agent_service.get_available_tools()
            if request.tool_name not in available_tools:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "TOOL_NOT_AVAILABLE",
                        "message": f"Tool '{request.tool_name}' is not available",
                        "available_tools": available_tools
                    }
                )
            
            # 验证工具参数
            required_params = await self.agent_service.get_tool_required_parameters(request.tool_name)
            missing_params = set(required_params) - set(request.parameters.keys())
            if missing_params:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "MISSING_TOOL_PARAMETERS",
                        "message": f"Missing required parameters for tool '{request.tool_name}'",
                        "missing_parameters": list(missing_params),
                        "required_parameters": required_params
                    }
                )
            
            # 调用原始服务
            result = await self.agent_service.execute_tool(
                tool_name=request.tool_name,
                parameters=request.parameters,
                session_id=request.session_id,
                timeout=request.timeout
            )
            
            logger.info(f"Tool executed successfully: {request.tool_name}")
            return result
            
        except ValidationException:
            raise
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "TOOL_EXECUTION_FAILED",
                    "message": f"Failed to execute tool '{request.tool_name}'",
                    "details": str(e)
                }
            )
