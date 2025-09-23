"""
端到端业务流程测试用例
测试覆盖：完整用户旅程、跨服务业务流程、数据一致性、用户体验
"""

import pytest
import asyncio
import time
import json
import uuid
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import aiohttp
import websockets
from concurrent.futures import ThreadPoolExecutor


@dataclass
class UserSession:
    """用户会话数据"""
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    conversation_history: List[Dict[str, Any]]
    voice_settings: Dict[str, Any]
    preferences: Dict[str, Any]
    
    def to_dict(self):
        return asdict(self)


@dataclass
class BusinessWorkflowResult:
    """业务流程测试结果"""
    workflow_name: str
    success: bool
    total_time: float
    steps_completed: int
    steps_failed: int
    error_messages: List[str]
    performance_metrics: Dict[str, float]
    user_experience_score: float


class TestUserOnboardingWorkflow:
    """用户入门流程测试"""
    
    @pytest.fixture
    def mock_user_service(self):
        """模拟用户服务"""
        class MockUserService:
            def __init__(self):
                self.users = {}
                self.sessions = {}
            
            async def register_user(self, user_data):
                """用户注册"""
                await asyncio.sleep(0.1)  # 模拟数据库操作
                
                user_id = str(uuid.uuid4())
                user_record = {
                    "user_id": user_id,
                    "username": user_data["username"],
                    "email": user_data["email"],
                    "created_at": time.time(),
                    "status": "active",
                    "profile": {
                        "display_name": user_data.get("display_name", user_data["username"]),
                        "language": user_data.get("language", "zh-CN"),
                        "timezone": user_data.get("timezone", "Asia/Shanghai")
                    }
                }
                
                self.users[user_id] = user_record
                return {"success": True, "user_id": user_id, "user": user_record}
            
            async def create_session(self, user_id):
                """创建用户会话"""
                await asyncio.sleep(0.05)
                
                if user_id not in self.users:
                    return {"success": False, "error": "User not found"}
                
                session_id = str(uuid.uuid4())
                session = UserSession(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=time.time(),
                    last_activity=time.time(),
                    conversation_history=[],
                    voice_settings={
                        "voice_id": "zh-CN-XiaoxiaoNeural",
                        "speed": 1.0,
                        "pitch": 0.0
                    },
                    preferences={
                        "theme": "light",
                        "notifications": True,
                        "auto_save": True
                    }
                )
                
                self.sessions[session_id] = session
                return {"success": True, "session": session.to_dict()}
            
            async def setup_user_preferences(self, user_id, preferences):
                """设置用户偏好"""
                await asyncio.sleep(0.08)
                
                if user_id not in self.users:
                    return {"success": False, "error": "User not found"}
                
                user = self.users[user_id]
                user["preferences"] = {**user.get("preferences", {}), **preferences}
                
                return {"success": True, "preferences": user["preferences"]}
        
        return MockUserService()
    
    @pytest.fixture
    def mock_onboarding_service(self):
        """模拟入门引导服务"""
        class MockOnboardingService:
            def __init__(self):
                self.onboarding_steps = [
                    {"step": "welcome", "title": "欢迎使用VoiceHelper", "required": True},
                    {"step": "voice_setup", "title": "语音设置", "required": True},
                    {"step": "first_chat", "title": "首次对话", "required": True},
                    {"step": "document_upload", "title": "文档上传", "required": False},
                    {"step": "tutorial_complete", "title": "教程完成", "required": True}
                ]
                self.user_progress = {}
            
            async def start_onboarding(self, user_id):
                """开始入门流程"""
                await asyncio.sleep(0.05)
                
                self.user_progress[user_id] = {
                    "current_step": 0,
                    "completed_steps": [],
                    "started_at": time.time(),
                    "status": "in_progress"
                }
                
                return {
                    "success": True,
                    "steps": self.onboarding_steps,
                    "current_step": self.onboarding_steps[0]
                }
            
            async def complete_step(self, user_id, step_name):
                """完成入门步骤"""
                await asyncio.sleep(0.03)
                
                if user_id not in self.user_progress:
                    return {"success": False, "error": "Onboarding not started"}
                
                progress = self.user_progress[user_id]
                
                # 查找步骤
                step_index = None
                for i, step in enumerate(self.onboarding_steps):
                    if step["step"] == step_name:
                        step_index = i
                        break
                
                if step_index is None:
                    return {"success": False, "error": "Invalid step"}
                
                # 记录完成的步骤
                if step_name not in progress["completed_steps"]:
                    progress["completed_steps"].append(step_name)
                
                # 更新当前步骤
                progress["current_step"] = min(step_index + 1, len(self.onboarding_steps))
                
                # 检查是否完成所有必需步骤
                required_steps = [s["step"] for s in self.onboarding_steps if s["required"]]
                completed_required = [s for s in progress["completed_steps"] if s in required_steps]
                
                if len(completed_required) == len(required_steps):
                    progress["status"] = "completed"
                    progress["completed_at"] = time.time()
                
                next_step = None
                if progress["current_step"] < len(self.onboarding_steps):
                    next_step = self.onboarding_steps[progress["current_step"]]
                
                return {
                    "success": True,
                    "progress": progress,
                    "next_step": next_step
                }
            
            async def get_progress(self, user_id):
                """获取入门进度"""
                await asyncio.sleep(0.02)
                
                if user_id not in self.user_progress:
                    return {"success": False, "error": "Onboarding not started"}
                
                progress = self.user_progress[user_id]
                completion_rate = len(progress["completed_steps"]) / len(self.onboarding_steps)
                
                return {
                    "success": True,
                    "progress": progress,
                    "completion_rate": completion_rate,
                    "total_steps": len(self.onboarding_steps)
                }
        
        return MockOnboardingService()
    
    @pytest.mark.asyncio
    async def test_complete_user_onboarding_workflow(self, mock_user_service, mock_onboarding_service):
        """测试完整用户入门流程"""
        workflow_start = time.time()
        steps_completed = 0
        steps_failed = 0
        error_messages = []
        performance_metrics = {}
        
        try:
            # 步骤1: 用户注册
            step_start = time.time()
            user_data = {
                "username": "test_user_001",
                "email": "test@example.com",
                "display_name": "测试用户",
                "language": "zh-CN"
            }
            
            register_result = await mock_user_service.register_user(user_data)
            performance_metrics["registration_time"] = time.time() - step_start
            
            assert register_result["success"], "用户注册失败"
            user_id = register_result["user_id"]
            steps_completed += 1
            
            # 步骤2: 创建会话
            step_start = time.time()
            session_result = await mock_user_service.create_session(user_id)
            performance_metrics["session_creation_time"] = time.time() - step_start
            
            assert session_result["success"], "会话创建失败"
            session_id = session_result["session"]["session_id"]
            steps_completed += 1
            
            # 步骤3: 开始入门流程
            step_start = time.time()
            onboarding_result = await mock_onboarding_service.start_onboarding(user_id)
            performance_metrics["onboarding_start_time"] = time.time() - step_start
            
            assert onboarding_result["success"], "入门流程启动失败"
            steps_completed += 1
            
            # 步骤4: 完成入门步骤
            onboarding_steps = ["welcome", "voice_setup", "first_chat", "tutorial_complete"]
            
            for step_name in onboarding_steps:
                step_start = time.time()
                complete_result = await mock_onboarding_service.complete_step(user_id, step_name)
                performance_metrics[f"{step_name}_time"] = time.time() - step_start
                
                if complete_result["success"]:
                    steps_completed += 1
                else:
                    steps_failed += 1
                    error_messages.append(f"步骤 {step_name} 失败: {complete_result.get('error')}")
            
            # 步骤5: 设置用户偏好
            step_start = time.time()
            preferences = {
                "theme": "dark",
                "language": "zh-CN",
                "voice_speed": 1.2,
                "auto_save": True
            }
            
            prefs_result = await mock_user_service.setup_user_preferences(user_id, preferences)
            performance_metrics["preferences_setup_time"] = time.time() - step_start
            
            if prefs_result["success"]:
                steps_completed += 1
            else:
                steps_failed += 1
                error_messages.append(f"偏好设置失败: {prefs_result.get('error')}")
            
            # 步骤6: 验证入门完成状态
            step_start = time.time()
            progress_result = await mock_onboarding_service.get_progress(user_id)
            performance_metrics["progress_check_time"] = time.time() - step_start
            
            assert progress_result["success"], "进度检查失败"
            assert progress_result["progress"]["status"] == "completed", "入门流程未完成"
            steps_completed += 1
            
        except Exception as e:
            steps_failed += 1
            error_messages.append(f"流程异常: {str(e)}")
        
        # 计算整体指标
        total_time = time.time() - workflow_start
        success_rate = steps_completed / (steps_completed + steps_failed)
        
        # 用户体验评分
        ux_score = 0.0
        if total_time < 2.0:  # 快速完成
            ux_score += 30
        elif total_time < 5.0:
            ux_score += 20
        else:
            ux_score += 10
        
        if success_rate > 0.9:  # 高成功率
            ux_score += 40
        elif success_rate > 0.7:
            ux_score += 30
        else:
            ux_score += 10
        
        if len(error_messages) == 0:  # 无错误
            ux_score += 30
        elif len(error_messages) <= 2:
            ux_score += 20
        else:
            ux_score += 10
        
        # 创建测试结果
        result = BusinessWorkflowResult(
            workflow_name="用户入门流程",
            success=steps_failed == 0,
            total_time=total_time,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            error_messages=error_messages,
            performance_metrics=performance_metrics,
            user_experience_score=ux_score
        )
        
        # 输出测试结果
        print(f"\n用户入门流程测试结果:")
        print(f"  总体成功: {result.success}")
        print(f"  总耗时: {result.total_time:.3f}s")
        print(f"  完成步骤: {result.steps_completed}")
        print(f"  失败步骤: {result.steps_failed}")
        print(f"  用户体验评分: {result.user_experience_score}/100")
        
        if result.error_messages:
            print(f"  错误信息: {result.error_messages}")
        
        print(f"  性能指标:")
        for metric, value in result.performance_metrics.items():
            print(f"    {metric}: {value:.3f}s")
        
        # 验证关键指标
        assert result.success, f"入门流程失败: {result.error_messages}"
        assert result.total_time < 10.0, f"入门流程耗时过长: {result.total_time:.3f}s"
        assert result.user_experience_score >= 70, f"用户体验评分过低: {result.user_experience_score}"
        
        return result


class TestConversationWorkflow:
    """对话流程测试"""
    
    @pytest.fixture
    def mock_conversation_service(self):
        """模拟对话服务"""
        class MockConversationService:
            def __init__(self):
                self.conversations = {}
                self.message_counter = 0
            
            async def start_conversation(self, user_id, session_id):
                """开始对话"""
                await asyncio.sleep(0.05)
                
                conversation_id = str(uuid.uuid4())
                conversation = {
                    "conversation_id": conversation_id,
                    "user_id": user_id,
                    "session_id": session_id,
                    "created_at": time.time(),
                    "messages": [],
                    "status": "active",
                    "context": {}
                }
                
                self.conversations[conversation_id] = conversation
                return {"success": True, "conversation": conversation}
            
            async def send_message(self, conversation_id, message_data):
                """发送消息"""
                await asyncio.sleep(0.1)  # 模拟处理时间
                
                if conversation_id not in self.conversations:
                    return {"success": False, "error": "Conversation not found"}
                
                conversation = self.conversations[conversation_id]
                
                # 添加用户消息
                self.message_counter += 1
                user_message = {
                    "message_id": f"msg_{self.message_counter}",
                    "role": "user",
                    "content": message_data["content"],
                    "timestamp": time.time(),
                    "message_type": message_data.get("type", "text")
                }
                
                conversation["messages"].append(user_message)
                
                # 生成AI回复
                await asyncio.sleep(0.2)  # 模拟AI处理时间
                
                self.message_counter += 1
                ai_message = {
                    "message_id": f"msg_{self.message_counter}",
                    "role": "assistant",
                    "content": f"这是对 '{message_data['content']}' 的回复",
                    "timestamp": time.time(),
                    "message_type": "text",
                    "references": []
                }
                
                conversation["messages"].append(ai_message)
                conversation["last_activity"] = time.time()
                
                return {
                    "success": True,
                    "user_message": user_message,
                    "ai_message": ai_message,
                    "conversation": conversation
                }
            
            async def get_conversation_history(self, conversation_id, limit=50):
                """获取对话历史"""
                await asyncio.sleep(0.03)
                
                if conversation_id not in self.conversations:
                    return {"success": False, "error": "Conversation not found"}
                
                conversation = self.conversations[conversation_id]
                messages = conversation["messages"][-limit:] if limit else conversation["messages"]
                
                return {
                    "success": True,
                    "messages": messages,
                    "total_messages": len(conversation["messages"]),
                    "conversation_info": {
                        "conversation_id": conversation_id,
                        "created_at": conversation["created_at"],
                        "last_activity": conversation["last_activity"]
                    }
                }
        
        return MockConversationService()
    
    @pytest.fixture
    def mock_voice_service(self):
        """模拟语音服务"""
        class MockVoiceService:
            def __init__(self):
                self.processing_times = []
            
            async def speech_to_text(self, audio_data):
                """语音转文字"""
                processing_time = 0.15 + len(audio_data) * 0.001  # 基于音频长度
                await asyncio.sleep(processing_time)
                
                self.processing_times.append(processing_time)
                
                return {
                    "success": True,
                    "transcript": "这是语音转换的文字内容",
                    "confidence": 0.95,
                    "processing_time": processing_time,
                    "language": "zh-CN"
                }
            
            async def text_to_speech(self, text, voice_config=None):
                """文字转语音"""
                processing_time = 0.1 + len(text) * 0.01
                await asyncio.sleep(processing_time)
                
                self.processing_times.append(processing_time)
                
                # 模拟音频数据
                audio_data = b"fake_audio_data" * (len(text) // 10 + 1)
                
                return {
                    "success": True,
                    "audio_data": audio_data,
                    "duration": len(text) * 0.1,
                    "processing_time": processing_time,
                    "voice_config": voice_config or {"voice_id": "default"}
                }
        
        return MockVoiceService()
    
    @pytest.mark.asyncio
    async def test_multimodal_conversation_workflow(self, mock_conversation_service, mock_voice_service):
        """测试多模态对话流程"""
        workflow_start = time.time()
        steps_completed = 0
        steps_failed = 0
        error_messages = []
        performance_metrics = {}
        
        try:
            # 步骤1: 开始对话
            step_start = time.time()
            user_id = "test_user_001"
            session_id = "test_session_001"
            
            conv_result = await mock_conversation_service.start_conversation(user_id, session_id)
            performance_metrics["conversation_start_time"] = time.time() - step_start
            
            assert conv_result["success"], "对话启动失败"
            conversation_id = conv_result["conversation"]["conversation_id"]
            steps_completed += 1
            
            # 步骤2: 文本对话
            step_start = time.time()
            text_messages = [
                "你好，我想了解VoiceHelper的功能",
                "可以帮我处理文档吗？",
                "语音功能怎么使用？"
            ]
            
            for i, message in enumerate(text_messages):
                msg_start = time.time()
                msg_result = await mock_conversation_service.send_message(
                    conversation_id, 
                    {"content": message, "type": "text"}
                )
                msg_time = time.time() - msg_start
                performance_metrics[f"text_message_{i+1}_time"] = msg_time
                
                if msg_result["success"]:
                    steps_completed += 1
                else:
                    steps_failed += 1
                    error_messages.append(f"文本消息 {i+1} 失败: {msg_result.get('error')}")
            
            performance_metrics["text_conversation_time"] = time.time() - step_start
            
            # 步骤3: 语音输入对话
            step_start = time.time()
            
            # 模拟语音输入
            audio_data = b"fake_audio_input" * 100
            
            # 语音转文字
            stt_start = time.time()
            stt_result = await mock_voice_service.speech_to_text(audio_data)
            performance_metrics["speech_to_text_time"] = time.time() - stt_start
            
            if stt_result["success"]:
                steps_completed += 1
                
                # 发送转换后的文字消息
                msg_result = await mock_conversation_service.send_message(
                    conversation_id,
                    {"content": stt_result["transcript"], "type": "voice_input"}
                )
                
                if msg_result["success"]:
                    steps_completed += 1
                    
                    # 将AI回复转换为语音
                    tts_start = time.time()
                    ai_response = msg_result["ai_message"]["content"]
                    tts_result = await mock_voice_service.text_to_speech(
                        ai_response, 
                        {"voice_id": "zh-CN-XiaoxiaoNeural", "speed": 1.0}
                    )
                    performance_metrics["text_to_speech_time"] = time.time() - tts_start
                    
                    if tts_result["success"]:
                        steps_completed += 1
                    else:
                        steps_failed += 1
                        error_messages.append(f"语音合成失败: {tts_result.get('error')}")
                else:
                    steps_failed += 1
                    error_messages.append(f"语音消息发送失败: {msg_result.get('error')}")
            else:
                steps_failed += 1
                error_messages.append(f"语音识别失败: {stt_result.get('error')}")
            
            performance_metrics["voice_conversation_time"] = time.time() - step_start
            
            # 步骤4: 获取对话历史
            step_start = time.time()
            history_result = await mock_conversation_service.get_conversation_history(conversation_id)
            performance_metrics["history_retrieval_time"] = time.time() - step_start
            
            if history_result["success"]:
                steps_completed += 1
                
                # 验证对话历史完整性
                messages = history_result["messages"]
                expected_message_count = len(text_messages) * 2 + 2  # 每个用户消息+AI回复，加上语音对话
                
                if len(messages) >= expected_message_count:
                    steps_completed += 1
                else:
                    steps_failed += 1
                    error_messages.append(f"对话历史不完整: 期望 {expected_message_count}，实际 {len(messages)}")
            else:
                steps_failed += 1
                error_messages.append(f"历史获取失败: {history_result.get('error')}")
            
        except Exception as e:
            steps_failed += 1
            error_messages.append(f"流程异常: {str(e)}")
        
        # 计算整体指标
        total_time = time.time() - workflow_start
        success_rate = steps_completed / (steps_completed + steps_failed)
        
        # 计算平均响应时间
        response_times = [
            performance_metrics.get(f"text_message_{i+1}_time", 0) 
            for i in range(len(text_messages))
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # 用户体验评分
        ux_score = 0.0
        
        # 响应速度评分 (40分)
        if avg_response_time < 0.5:
            ux_score += 40
        elif avg_response_time < 1.0:
            ux_score += 30
        elif avg_response_time < 2.0:
            ux_score += 20
        else:
            ux_score += 10
        
        # 功能完整性评分 (40分)
        if success_rate > 0.95:
            ux_score += 40
        elif success_rate > 0.8:
            ux_score += 30
        elif success_rate > 0.6:
            ux_score += 20
        else:
            ux_score += 10
        
        # 语音质量评分 (20分)
        voice_success = (
            performance_metrics.get("speech_to_text_time", 0) > 0 and
            performance_metrics.get("text_to_speech_time", 0) > 0
        )
        if voice_success:
            ux_score += 20
        else:
            ux_score += 5
        
        # 创建测试结果
        result = BusinessWorkflowResult(
            workflow_name="多模态对话流程",
            success=steps_failed == 0,
            total_time=total_time,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            error_messages=error_messages,
            performance_metrics=performance_metrics,
            user_experience_score=ux_score
        )
        
        # 输出测试结果
        print(f"\n多模态对话流程测试结果:")
        print(f"  总体成功: {result.success}")
        print(f"  总耗时: {result.total_time:.3f}s")
        print(f"  完成步骤: {result.steps_completed}")
        print(f"  失败步骤: {result.steps_failed}")
        print(f"  平均响应时间: {avg_response_time:.3f}s")
        print(f"  用户体验评分: {result.user_experience_score}/100")
        
        if result.error_messages:
            print(f"  错误信息: {result.error_messages}")
        
        # 验证关键指标
        assert result.success, f"对话流程失败: {result.error_messages}"
        assert avg_response_time < 3.0, f"平均响应时间过长: {avg_response_time:.3f}s"
        assert result.user_experience_score >= 60, f"用户体验评分过低: {result.user_experience_score}"
        
        return result


class TestDocumentManagementWorkflow:
    """文档管理流程测试"""
    
    @pytest.fixture
    def mock_document_service(self):
        """模拟文档服务"""
        class MockDocumentService:
            def __init__(self):
                self.documents = {}
                self.processing_queue = []
            
            async def upload_document(self, user_id, file_data, metadata=None):
                """上传文档"""
                await asyncio.sleep(0.2)  # 模拟上传时间
                
                doc_id = str(uuid.uuid4())
                document = {
                    "document_id": doc_id,
                    "user_id": user_id,
                    "filename": metadata.get("filename", "unknown.txt"),
                    "file_size": len(file_data),
                    "content_type": metadata.get("content_type", "text/plain"),
                    "uploaded_at": time.time(),
                    "status": "uploaded",
                    "processing_status": "pending",
                    "metadata": metadata or {}
                }
                
                self.documents[doc_id] = document
                self.processing_queue.append(doc_id)
                
                return {"success": True, "document": document}
            
            async def process_document(self, doc_id):
                """处理文档"""
                if doc_id not in self.documents:
                    return {"success": False, "error": "Document not found"}
                
                document = self.documents[doc_id]
                
                # 模拟文档处理步骤
                processing_steps = [
                    ("parsing", 0.3),      # 解析文档
                    ("chunking", 0.2),     # 分块
                    ("embedding", 0.5),    # 向量化
                    ("indexing", 0.1)      # 索引
                ]
                
                document["processing_status"] = "processing"
                document["processing_steps"] = []
                
                for step_name, duration in processing_steps:
                    step_start = time.time()
                    await asyncio.sleep(duration)
                    step_time = time.time() - step_start
                    
                    step_info = {
                        "step": step_name,
                        "duration": step_time,
                        "completed_at": time.time()
                    }
                    document["processing_steps"].append(step_info)
                
                # 模拟处理结果
                document["processing_status"] = "completed"
                document["processed_at"] = time.time()
                document["chunks_created"] = 15
                document["vectors_indexed"] = 15
                
                return {"success": True, "document": document}
            
            async def search_documents(self, user_id, query, limit=10):
                """搜索文档"""
                await asyncio.sleep(0.1)  # 模拟搜索时间
                
                # 过滤用户的已处理文档
                user_docs = [
                    doc for doc in self.documents.values()
                    if doc["user_id"] == user_id and doc["processing_status"] == "completed"
                ]
                
                # 模拟搜索结果
                results = []
                for doc in user_docs[:limit]:
                    # 模拟相关性评分
                    relevance_score = 0.8 + (hash(query + doc["document_id"]) % 20) / 100
                    
                    result = {
                        "document_id": doc["document_id"],
                        "filename": doc["filename"],
                        "relevance_score": relevance_score,
                        "snippet": f"这是来自 {doc['filename']} 的相关内容片段...",
                        "chunk_info": {
                            "chunk_id": f"chunk_{hash(query) % 100}",
                            "page": 1,
                            "position": "top"
                        }
                    }
                    results.append(result)
                
                # 按相关性排序
                results.sort(key=lambda x: x["relevance_score"], reverse=True)
                
                return {
                    "success": True,
                    "results": results,
                    "total_found": len(results),
                    "query": query,
                    "search_time": 0.1
                }
            
            async def delete_document(self, user_id, doc_id):
                """删除文档"""
                await asyncio.sleep(0.05)
                
                if doc_id not in self.documents:
                    return {"success": False, "error": "Document not found"}
                
                document = self.documents[doc_id]
                if document["user_id"] != user_id:
                    return {"success": False, "error": "Permission denied"}
                
                del self.documents[doc_id]
                
                return {"success": True, "deleted_document_id": doc_id}
        
        return MockDocumentService()
    
    @pytest.mark.asyncio
    async def test_complete_document_lifecycle_workflow(self, mock_document_service):
        """测试完整文档生命周期流程"""
        workflow_start = time.time()
        steps_completed = 0
        steps_failed = 0
        error_messages = []
        performance_metrics = {}
        
        user_id = "test_user_001"
        uploaded_docs = []
        
        try:
            # 步骤1: 批量上传文档
            step_start = time.time()
            test_documents = [
                {
                    "filename": "技术文档.pdf",
                    "content": b"这是技术文档的内容" * 100,
                    "content_type": "application/pdf"
                },
                {
                    "filename": "用户手册.docx", 
                    "content": b"这是用户手册的内容" * 80,
                    "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                },
                {
                    "filename": "FAQ.txt",
                    "content": b"这是常见问题的内容" * 60,
                    "content_type": "text/plain"
                }
            ]
            
            for i, doc_data in enumerate(test_documents):
                upload_start = time.time()
                upload_result = await mock_document_service.upload_document(
                    user_id,
                    doc_data["content"],
                    {
                        "filename": doc_data["filename"],
                        "content_type": doc_data["content_type"]
                    }
                )
                upload_time = time.time() - upload_start
                performance_metrics[f"upload_{i+1}_time"] = upload_time
                
                if upload_result["success"]:
                    uploaded_docs.append(upload_result["document"])
                    steps_completed += 1
                else:
                    steps_failed += 1
                    error_messages.append(f"文档 {i+1} 上传失败: {upload_result.get('error')}")
            
            performance_metrics["total_upload_time"] = time.time() - step_start
            
            # 步骤2: 处理所有上传的文档
            step_start = time.time()
            processed_docs = []
            
            for i, document in enumerate(uploaded_docs):
                process_start = time.time()
                process_result = await mock_document_service.process_document(document["document_id"])
                process_time = time.time() - process_start
                performance_metrics[f"process_{i+1}_time"] = process_time
                
                if process_result["success"]:
                    processed_docs.append(process_result["document"])
                    steps_completed += 1
                else:
                    steps_failed += 1
                    error_messages.append(f"文档 {i+1} 处理失败: {process_result.get('error')}")
            
            performance_metrics["total_processing_time"] = time.time() - step_start
            
            # 步骤3: 文档搜索测试
            step_start = time.time()
            search_queries = [
                "技术规范",
                "用户指南",
                "常见问题",
                "安装步骤"
            ]
            
            search_results_all = []
            for i, query in enumerate(search_queries):
                search_start = time.time()
                search_result = await mock_document_service.search_documents(user_id, query, limit=5)
                search_time = time.time() - search_start
                performance_metrics[f"search_{i+1}_time"] = search_time
                
                if search_result["success"]:
                    search_results_all.append(search_result)
                    steps_completed += 1
                else:
                    steps_failed += 1
                    error_messages.append(f"搜索 '{query}' 失败: {search_result.get('error')}")
            
            performance_metrics["total_search_time"] = time.time() - step_start
            
            # 步骤4: 验证搜索结果质量
            step_start = time.time()
            
            total_results = sum(len(sr["results"]) for sr in search_results_all)
            avg_relevance = 0
            
            if total_results > 0:
                all_scores = []
                for search_result in search_results_all:
                    for result in search_result["results"]:
                        all_scores.append(result["relevance_score"])
                
                avg_relevance = sum(all_scores) / len(all_scores)
                
                if avg_relevance > 0.7:
                    steps_completed += 1
                else:
                    steps_failed += 1
                    error_messages.append(f"搜索质量不佳，平均相关性: {avg_relevance:.3f}")
            else:
                steps_failed += 1
                error_messages.append("没有搜索结果")
            
            performance_metrics["search_quality_check_time"] = time.time() - step_start
            performance_metrics["average_relevance_score"] = avg_relevance
            
            # 步骤5: 文档管理操作（删除一个文档）
            step_start = time.time()
            
            if processed_docs:
                delete_doc = processed_docs[0]
                delete_result = await mock_document_service.delete_document(
                    user_id, 
                    delete_doc["document_id"]
                )
                
                if delete_result["success"]:
                    steps_completed += 1
                else:
                    steps_failed += 1
                    error_messages.append(f"文档删除失败: {delete_result.get('error')}")
            
            performance_metrics["document_deletion_time"] = time.time() - step_start
            
        except Exception as e:
            steps_failed += 1
            error_messages.append(f"流程异常: {str(e)}")
        
        # 计算整体指标
        total_time = time.time() - workflow_start
        success_rate = steps_completed / (steps_completed + steps_failed)
        
        # 计算处理效率
        total_file_size = sum(len(doc["content"]) for doc in test_documents)
        processing_throughput = total_file_size / performance_metrics.get("total_processing_time", 1)
        
        # 用户体验评分
        ux_score = 0.0
        
        # 上传速度评分 (25分)
        avg_upload_time = performance_metrics.get("total_upload_time", 0) / len(test_documents)
        if avg_upload_time < 0.5:
            ux_score += 25
        elif avg_upload_time < 1.0:
            ux_score += 20
        elif avg_upload_time < 2.0:
            ux_score += 15
        else:
            ux_score += 10
        
        # 处理速度评分 (25分)
        avg_process_time = performance_metrics.get("total_processing_time", 0) / len(uploaded_docs)
        if avg_process_time < 1.0:
            ux_score += 25
        elif avg_process_time < 2.0:
            ux_score += 20
        elif avg_process_time < 5.0:
            ux_score += 15
        else:
            ux_score += 10
        
        # 搜索质量评分 (30分)
        if avg_relevance > 0.8:
            ux_score += 30
        elif avg_relevance > 0.7:
            ux_score += 25
        elif avg_relevance > 0.6:
            ux_score += 20
        else:
            ux_score += 10
        
        # 整体成功率评分 (20分)
        if success_rate > 0.95:
            ux_score += 20
        elif success_rate > 0.8:
            ux_score += 15
        elif success_rate > 0.6:
            ux_score += 10
        else:
            ux_score += 5
        
        # 创建测试结果
        result = BusinessWorkflowResult(
            workflow_name="文档管理生命周期",
            success=steps_failed == 0,
            total_time=total_time,
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            error_messages=error_messages,
            performance_metrics=performance_metrics,
            user_experience_score=ux_score
        )
        
        # 输出测试结果
        print(f"\n文档管理生命周期测试结果:")
        print(f"  总体成功: {result.success}")
        print(f"  总耗时: {result.total_time:.3f}s")
        print(f"  完成步骤: {result.steps_completed}")
        print(f"  失败步骤: {result.steps_failed}")
        print(f"  处理吞吐量: {processing_throughput:.0f} bytes/s")
        print(f"  平均搜索相关性: {avg_relevance:.3f}")
        print(f"  用户体验评分: {result.user_experience_score}/100")
        
        if result.error_messages:
            print(f"  错误信息: {result.error_messages}")
        
        # 验证关键指标
        assert result.success, f"文档管理流程失败: {result.error_messages}"
        assert result.total_time < 30.0, f"文档管理流程耗时过长: {result.total_time:.3f}s"
        assert avg_relevance > 0.6, f"搜索质量过低: {avg_relevance:.3f}"
        assert result.user_experience_score >= 50, f"用户体验评分过低: {result.user_experience_score}"
        
        return result


class TestCrossServiceIntegrationWorkflow:
    """跨服务集成流程测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_user_journey(self):
        """测试端到端用户旅程"""
        workflow_start = time.time()
        
        # 集成所有服务的模拟实例
        from .test_business_workflows import TestUserOnboardingWorkflow, TestConversationWorkflow, TestDocumentManagementWorkflow
        
        onboarding_test = TestUserOnboardingWorkflow()
        conversation_test = TestConversationWorkflow()
        document_test = TestDocumentManagementWorkflow()
        
        # 创建服务实例
        user_service = onboarding_test.mock_user_service()
        onboarding_service = onboarding_test.mock_onboarding_service()
        conversation_service = conversation_test.mock_conversation_service()
        voice_service = conversation_test.mock_voice_service()
        document_service = document_test.mock_document_service()
        
        journey_results = []
        overall_success = True
        
        try:
            # 阶段1: 用户入门
            print("执行用户入门流程...")
            onboarding_result = await onboarding_test.test_complete_user_onboarding_workflow(
                user_service, onboarding_service
            )
            journey_results.append(onboarding_result)
            
            if not onboarding_result.success:
                overall_success = False
            
            # 阶段2: 多模态对话
            print("执行多模态对话流程...")
            conversation_result = await conversation_test.test_multimodal_conversation_workflow(
                conversation_service, voice_service
            )
            journey_results.append(conversation_result)
            
            if not conversation_result.success:
                overall_success = False
            
            # 阶段3: 文档管理
            print("执行文档管理流程...")
            document_result = await document_test.test_complete_document_lifecycle_workflow(
                document_service
            )
            journey_results.append(document_result)
            
            if not document_result.success:
                overall_success = False
            
        except Exception as e:
            overall_success = False
            print(f"端到端测试异常: {str(e)}")
        
        # 计算整体指标
        total_journey_time = time.time() - workflow_start
        
        # 汇总所有阶段的指标
        total_steps_completed = sum(r.steps_completed for r in journey_results)
        total_steps_failed = sum(r.steps_failed for r in journey_results)
        all_error_messages = []
        for r in journey_results:
            all_error_messages.extend(r.error_messages)
        
        # 计算综合用户体验评分
        if journey_results:
            avg_ux_score = sum(r.user_experience_score for r in journey_results) / len(journey_results)
        else:
            avg_ux_score = 0
        
        # 输出端到端测试总结
        print(f"\n=== 端到端用户旅程测试总结 ===")
        print(f"总体成功: {overall_success}")
        print(f"总耗时: {total_journey_time:.3f}s")
        print(f"总完成步骤: {total_steps_completed}")
        print(f"总失败步骤: {total_steps_failed}")
        print(f"综合用户体验评分: {avg_ux_score:.1f}/100")
        
        print(f"\n各阶段详细结果:")
        for i, result in enumerate(journey_results, 1):
            print(f"  阶段 {i} - {result.workflow_name}:")
            print(f"    成功: {result.success}")
            print(f"    耗时: {result.total_time:.3f}s")
            print(f"    UX评分: {result.user_experience_score}/100")
        
        if all_error_messages:
            print(f"\n所有错误信息:")
            for error in all_error_messages:
                print(f"  - {error}")
        
        # 验证端到端测试结果
        assert overall_success, f"端到端测试失败，错误: {all_error_messages}"
        assert total_journey_time < 60.0, f"端到端流程耗时过长: {total_journey_time:.3f}s"
        assert avg_ux_score >= 60, f"综合用户体验评分过低: {avg_ux_score:.1f}"
        
        return {
            "overall_success": overall_success,
            "total_time": total_journey_time,
            "stage_results": journey_results,
            "comprehensive_ux_score": avg_ux_score
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
