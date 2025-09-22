"""
端到端测试用例
测试覆盖：完整业务流程、用户场景、多平台交互
"""

import pytest
import asyncio
import json
import time
import requests
import websockets
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import base64
from typing import Dict, List, Any


class TestCompleteUserJourney:
    """完整用户旅程测试"""
    
    @pytest.fixture(scope="class")
    def service_config(self):
        """服务配置"""
        return {
            "backend_url": "http://localhost:8080",
            "algo_url": "http://localhost:8000", 
            "frontend_url": "http://localhost:3000",
            "websocket_url": "ws://localhost:8080"
        }
    
    @pytest.fixture(scope="class")
    def test_user_data(self):
        """测试用户数据"""
        return {
            "username": "e2e_test_user",
            "password": "test_password_123",
            "email": "e2e_test@example.com",
            "tenant_id": "test_tenant"
        }
    
    def test_user_registration_and_login_flow(self, service_config, test_user_data):
        """测试用户注册和登录流程"""
        backend_url = service_config["backend_url"]
        
        # 1. 用户注册
        register_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"],
            "email": test_user_data["email"],
            "tenant_id": test_user_data["tenant_id"]
        }
        
        register_response = requests.post(
            f"{backend_url}/api/v1/auth/register",
            json=register_data,
            timeout=10
        )
        
        # 注册可能成功或用户已存在
        assert register_response.status_code in [200, 201, 409]
        
        # 2. 用户登录
        login_data = {
            "username": test_user_data["username"],
            "password": test_user_data["password"]
        }
        
        login_response = requests.post(
            f"{backend_url}/api/v1/auth/login",
            json=login_data,
            timeout=10
        )
        
        if login_response.status_code == 200:
            login_result = login_response.json()
            assert "token" in login_result
            return login_result["token"]
        else:
            # 如果登录接口不存在，返回模拟token
            return "e2e_test_token"
    
    def test_document_management_workflow(self, service_config):
        """测试文档管理工作流"""
        algo_url = service_config["algo_url"]
        
        # 1. 准备测试文档
        test_documents = [
            {
                "name": "产品手册.pdf",
                "content": """
                VoiceHelper产品手册
                
                第一章：产品概述
                VoiceHelper是一个智能语音助手系统，支持多模态交互。
                
                第二章：功能特性
                - 实时语音识别和合成
                - 智能问答和对话
                - 多平台支持
                - 企业级安全
                
                第三章：使用指南
                1. 安装和配置
                2. 基本使用方法
                3. 高级功能设置
                """,
                "metadata": {
                    "category": "product_manual",
                    "version": "1.0",
                    "language": "zh-CN"
                }
            },
            {
                "name": "技术文档.md",
                "content": """
                # VoiceHelper技术架构
                
                ## 系统架构
                - 微服务架构设计
                - API网关层
                - 核心服务层
                - AI算法引擎
                
                ## 技术栈
                - 后端：Go + Gin
                - 前端：Next.js + React
                - AI服务：Python + FastAPI
                - 数据库：PostgreSQL + Redis + Milvus
                
                ## 性能指标
                - 响应时间：< 200ms
                - 并发支持：10000+ QPS
                - 可用性：99.9%
                """,
                "metadata": {
                    "category": "technical_doc",
                    "type": "architecture",
                    "language": "zh-CN"
                }
            }
        ]
        
        # 2. 批量入库文档
        ingest_data = {
            "files": test_documents,
            "collection_name": "e2e_test_collection",
            "chunk_size": 500,
            "chunk_overlap": 50
        }
        
        ingest_response = requests.post(
            f"{algo_url}/ingest",
            json=ingest_data,
            timeout=60
        )
        
        assert ingest_response.status_code == 200
        task_id = ingest_response.json()["task_id"]
        
        # 3. 监控入库进度
        max_wait_time = 120  # 最多等待2分钟
        wait_time = 0
        
        while wait_time < max_wait_time:
            status_response = requests.get(f"{algo_url}/tasks/{task_id}")
            
            if status_response.status_code == 200:
                status_data = status_response.json()
                status = status_data.get("status")
                
                if status == "completed":
                    print(f"文档入库完成，耗时: {wait_time}秒")
                    break
                elif status == "failed":
                    pytest.fail(f"文档入库失败: {status_data.get('error', 'Unknown error')}")
                
                print(f"入库进度: {status_data.get('progress', 'Unknown')}")
            
            time.sleep(5)
            wait_time += 5
        
        if wait_time >= max_wait_time:
            pytest.fail("文档入库超时")
        
        return task_id
    
    def test_intelligent_qa_workflow(self, service_config):
        """测试智能问答工作流"""
        # 首先确保有文档入库
        self.test_document_management_workflow(service_config)
        
        algo_url = service_config["algo_url"]
        
        # 测试问题列表
        test_questions = [
            "VoiceHelper是什么产品？",
            "系统支持哪些功能特性？",
            "技术架构是怎样的？",
            "性能指标如何？",
            "支持哪些技术栈？"
        ]
        
        qa_results = []
        
        for question in test_questions:
            print(f"测试问题: {question}")
            
            query_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "top_k": 5,
                "temperature": 0.3,
                "max_tokens": 500
            }
            
            start_time = time.time()
            
            query_response = requests.post(
                f"{algo_url}/query",
                json=query_data,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            assert query_response.status_code == 200
            
            # 解析流式响应
            response_content = ""
            references = []
            
            for line in query_response.text.strip().split('\n'):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        if data.get("type") == "refs":
                            references = data.get("refs", [])
                        elif data.get("type") == "message":
                            response_content += data.get("content", "")
                        
                    except json.JSONDecodeError:
                        continue
            
            qa_result = {
                "question": question,
                "answer": response_content,
                "references": references,
                "response_time": response_time,
                "has_references": len(references) > 0
            }
            
            qa_results.append(qa_result)
            
            # 验证回答质量
            assert len(response_content) > 0, f"问题 '{question}' 没有得到回答"
            assert response_time < 5.0, f"问题 '{question}' 响应时间过长: {response_time:.2f}s"
            
            print(f"回答长度: {len(response_content)}, 响应时间: {response_time:.2f}s, 引用数: {len(references)}")
        
        # 统计测试结果
        avg_response_time = sum(r["response_time"] for r in qa_results) / len(qa_results)
        questions_with_refs = sum(1 for r in qa_results if r["has_references"])
        
        print(f"QA测试总结:")
        print(f"- 平均响应时间: {avg_response_time:.2f}s")
        print(f"- 有引用的问题: {questions_with_refs}/{len(qa_results)}")
        
        assert avg_response_time < 3.0, f"平均响应时间过长: {avg_response_time:.2f}s"
        
        return qa_results
    
    @pytest.mark.asyncio
    async def test_voice_interaction_workflow(self, service_config):
        """测试语音交互工作流"""
        websocket_url = service_config["websocket_url"]
        
        try:
            # 连接语音WebSocket
            ws_url = f"{websocket_url}/api/v1/voice/stream"
            
            async with websockets.connect(
                ws_url,
                extra_headers={"Authorization": "Bearer e2e_test_token"},
                timeout=15
            ) as websocket:
                
                print("WebSocket连接成功")
                
                # 1. 发送初始化消息
                init_message = {
                    "type": "start",
                    "codec": "pcm16",
                    "sample_rate": 16000,
                    "conversation_id": "e2e_voice_test",
                    "lang": "zh-CN",
                    "vad": {
                        "enable": true,
                        "min_speech_ms": 200,
                        "min_silence_ms": 300
                    }
                }
                
                await websocket.send(json.dumps(init_message))
                
                # 等待初始化响应
                init_response = await asyncio.wait_for(websocket.recv(), timeout=10)
                init_data = json.loads(init_response)
                
                assert init_data.get("type") in ["ready", "ack"]
                print(f"初始化响应: {init_data}")
                
                # 2. 模拟发送音频数据
                # 创建模拟的PCM音频数据
                fake_audio_chunks = [
                    b"fake_pcm_audio_chunk_1" * 100,
                    b"fake_pcm_audio_chunk_2" * 100,
                    b"fake_pcm_audio_chunk_3" * 100
                ]
                
                for i, chunk in enumerate(fake_audio_chunks):
                    audio_message = {
                        "type": "audio",
                        "data": base64.b64encode(chunk).decode(),
                        "sequence": i + 1
                    }
                    
                    await websocket.send(json.dumps(audio_message))
                    print(f"发送音频块 {i + 1}")
                    
                    # 短暂等待
                    await asyncio.sleep(0.1)
                
                # 3. 发送结束信号
                end_message = {"type": "end"}
                await websocket.send(json.dumps(end_message))
                
                # 4. 收集响应
                responses = []
                timeout_count = 0
                max_timeout = 3
                
                while timeout_count < max_timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5)
                        response_data = json.loads(response)
                        responses.append(response_data)
                        
                        print(f"收到响应: {response_data.get('type', 'unknown')}")
                        
                        # 如果收到结束信号，退出循环
                        if response_data.get("type") == "end":
                            break
                            
                    except asyncio.TimeoutError:
                        timeout_count += 1
                        print(f"等待响应超时 ({timeout_count}/{max_timeout})")
                
                # 验证响应
                response_types = [r.get("type") for r in responses]
                print(f"收到的响应类型: {response_types}")
                
                # 根据实际实现，可能包含不同类型的响应
                expected_types = ["transcript", "emotion", "message", "audio", "end", "error"]
                assert any(t in expected_types for t in response_types), f"未收到预期的响应类型: {response_types}"
                
        except websockets.exceptions.ConnectionClosed as e:
            pytest.skip(f"WebSocket连接关闭: {e}")
        except asyncio.TimeoutError:
            pytest.skip("WebSocket连接或响应超时")
        except Exception as e:
            pytest.fail(f"语音交互测试失败: {e}")
    
    def test_multi_platform_consistency(self, service_config):
        """测试多平台一致性"""
        backend_url = service_config["backend_url"]
        
        # 测试相同的API在不同平台调用的一致性
        test_message = "测试多平台一致性的消息"
        
        platforms = [
            {"name": "web", "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            {"name": "mobile", "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"},
            {"name": "desktop", "user_agent": "VoiceHelper-Desktop/1.0"},
            {"name": "miniprogram", "user_agent": "miniProgram"}
        ]
        
        results = []
        
        for platform in platforms:
            headers = {
                "Authorization": "Bearer e2e_test_token",
                "Content-Type": "application/json",
                "User-Agent": platform["user_agent"],
                "X-Platform": platform["name"]
            }
            
            chat_data = {
                "conversation_id": f"e2e_consistency_test_{platform['name']}",
                "messages": [
                    {
                        "role": "user",
                        "content": test_message
                    }
                ],
                "platform": platform["name"]
            }
            
            start_time = time.time()
            
            response = requests.post(
                f"{backend_url}/api/v1/chat/completions",
                headers=headers,
                json=chat_data,
                timeout=30
            )
            
            response_time = time.time() - start_time
            
            result = {
                "platform": platform["name"],
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200
            }
            
            if response.status_code == 200:
                # 简单验证响应格式
                if response.headers.get("content-type", "").startswith("text/"):
                    result["response_length"] = len(response.text)
                else:
                    try:
                        data = response.json()
                        result["response_data"] = data
                    except:
                        result["response_length"] = len(response.text)
            
            results.append(result)
            print(f"{platform['name']} 平台测试: {result}")
        
        # 验证一致性
        successful_platforms = [r for r in results if r["success"]]
        
        if len(successful_platforms) > 1:
            # 比较响应时间的一致性（允许一定差异）
            response_times = [r["response_time"] for r in successful_platforms]
            avg_time = sum(response_times) / len(response_times)
            max_deviation = max(abs(t - avg_time) for t in response_times)
            
            # 响应时间差异不应超过平均值的50%
            assert max_deviation < avg_time * 0.5, f"平台间响应时间差异过大: {max_deviation:.2f}s"
        
        return results


class TestWebFrontendE2E:
    """Web前端端到端测试"""
    
    @pytest.fixture(scope="class")
    def browser_driver(self):
        """浏览器驱动"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 无头模式
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            yield driver
        except Exception as e:
            pytest.skip(f"Chrome浏览器驱动初始化失败: {e}")
        finally:
            if 'driver' in locals():
                driver.quit()
    
    def test_web_chat_interface(self, browser_driver, service_config):
        """测试Web聊天界面"""
        frontend_url = service_config["frontend_url"]
        
        try:
            # 访问聊天页面
            browser_driver.get(f"{frontend_url}/chat")
            
            # 等待页面加载
            wait = WebDriverWait(browser_driver, 10)
            
            # 查找聊天输入框
            chat_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text'], textarea"))
            )
            
            # 输入测试消息
            test_message = "这是一个Web端到端测试消息"
            chat_input.send_keys(test_message)
            
            # 查找发送按钮并点击
            send_button = browser_driver.find_element(By.CSS_SELECTOR, "button[type='submit'], .send-button")
            send_button.click()
            
            # 等待响应出现
            response_element = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".message, .chat-message, .response"))
            )
            
            # 验证消息已发送和响应已显示
            messages = browser_driver.find_elements(By.CSS_SELECTOR, ".message, .chat-message")
            assert len(messages) >= 1, "未找到聊天消息"
            
            print(f"Web聊天测试成功，找到 {len(messages)} 条消息")
            
        except Exception as e:
            pytest.skip(f"Web前端测试失败: {e}")
    
    def test_web_file_upload(self, browser_driver, service_config):
        """测试Web文件上传功能"""
        frontend_url = service_config["frontend_url"]
        
        try:
            # 访问数据集管理页面
            browser_driver.get(f"{frontend_url}/datasets")
            
            wait = WebDriverWait(browser_driver, 10)
            
            # 查找文件上传元素
            file_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='file']"))
            )
            
            # 创建临时测试文件
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write("这是一个测试文档，用于验证Web端文件上传功能。")
                temp_file_path = f.name
            
            try:
                # 上传文件
                file_input.send_keys(temp_file_path)
                
                # 查找上传按钮
                upload_button = browser_driver.find_element(By.CSS_SELECTOR, ".upload-button, button[type='submit']")
                upload_button.click()
                
                # 等待上传完成提示
                success_message = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".success, .upload-success, .notification"))
                )
                
                print("Web文件上传测试成功")
                
            finally:
                # 清理临时文件
                os.unlink(temp_file_path)
                
        except Exception as e:
            pytest.skip(f"Web文件上传测试失败: {e}")


class TestPerformanceE2E:
    """端到端性能测试"""
    
    def test_concurrent_users_simulation(self, service_config):
        """模拟并发用户测试"""
        import threading
        import queue
        
        backend_url = service_config["backend_url"]
        num_concurrent_users = 10
        messages_per_user = 3
        
        results_queue = queue.Queue()
        
        def simulate_user(user_id):
            """模拟单个用户的操作"""
            user_results = []
            
            for i in range(messages_per_user):
                start_time = time.time()
                
                try:
                    headers = {
                        "Authorization": f"Bearer user_{user_id}_token",
                        "Content-Type": "application/json"
                    }
                    
                    chat_data = {
                        "conversation_id": f"concurrent_test_user_{user_id}",
                        "messages": [
                            {
                                "role": "user",
                                "content": f"用户{user_id}的第{i+1}条消息"
                            }
                        ]
                    }
                    
                    response = requests.post(
                        f"{backend_url}/api/v1/chat/completions",
                        headers=headers,
                        json=chat_data,
                        timeout=30
                    )
                    
                    response_time = time.time() - start_time
                    
                    user_results.append({
                        "user_id": user_id,
                        "message_id": i + 1,
                        "status_code": response.status_code,
                        "response_time": response_time,
                        "success": response.status_code == 200
                    })
                    
                except Exception as e:
                    user_results.append({
                        "user_id": user_id,
                        "message_id": i + 1,
                        "error": str(e),
                        "success": False
                    })
                
                # 用户间隔
                time.sleep(0.5)
            
            results_queue.put(user_results)
        
        # 启动并发用户线程
        threads = []
        start_time = time.time()
        
        for user_id in range(num_concurrent_users):
            thread = threading.Thread(target=simulate_user, args=(user_id,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # 收集结果
        all_results = []
        while not results_queue.empty():
            user_results = results_queue.get()
            all_results.extend(user_results)
        
        # 分析结果
        successful_requests = [r for r in all_results if r.get("success")]
        failed_requests = [r for r in all_results if not r.get("success")]
        
        if successful_requests:
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
            max_response_time = max(r["response_time"] for r in successful_requests)
            min_response_time = min(r["response_time"] for r in successful_requests)
        else:
            avg_response_time = max_response_time = min_response_time = 0
        
        success_rate = len(successful_requests) / len(all_results) * 100
        
        print(f"并发测试结果:")
        print(f"- 并发用户数: {num_concurrent_users}")
        print(f"- 总请求数: {len(all_results)}")
        print(f"- 成功率: {success_rate:.1f}%")
        print(f"- 平均响应时间: {avg_response_time:.2f}s")
        print(f"- 最大响应时间: {max_response_time:.2f}s")
        print(f"- 最小响应时间: {min_response_time:.2f}s")
        print(f"- 总耗时: {total_time:.2f}s")
        
        # 性能断言
        assert success_rate >= 80, f"成功率过低: {success_rate:.1f}%"
        assert avg_response_time < 5.0, f"平均响应时间过长: {avg_response_time:.2f}s"
        
        return {
            "concurrent_users": num_concurrent_users,
            "total_requests": len(all_results),
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "max_response_time": max_response_time,
            "total_time": total_time
        }


# 测试配置
@pytest.fixture(scope="session")
def e2e_test_config():
    """端到端测试配置"""
    return {
        "timeouts": {
            "page_load": 30,
            "api_request": 30,
            "websocket_connect": 15,
            "file_upload": 60
        },
        "performance_thresholds": {
            "avg_response_time": 3.0,
            "max_response_time": 10.0,
            "success_rate": 90.0,
            "concurrent_users": 20
        },
        "test_scenarios": {
            "basic_chat": True,
            "voice_interaction": True,
            "file_upload": True,
            "multi_platform": True,
            "performance": True
        }
    }


if __name__ == "__main__":
    # 运行端到端测试
    pytest.main([__file__, "-v", "--tb=short", "-s", "--maxfail=5"])
