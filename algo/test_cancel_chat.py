#!/usr/bin/env python3
"""
测试取消聊天功能的脚本
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any


class ChatCancelTester:
    """聊天取消功能测试器"""
    
    def __init__(self, base_url: str = "http://localhost:8070"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def start_chat_stream(self, session_id: str, query: str) -> Dict[str, Any]:
        """启动流式聊天"""
        url = f"{self.base_url}/api/v1/chat/stream"
        data = {
            "query": query,
            "session_id": session_id,
            "context": {}
        }
        
        print(f"🚀 启动聊天流: session_id={session_id}, query={query}")
        
        async with self.session.post(url, json=data) as response:
            if response.status == 200:
                print("✅ 聊天流启动成功")
                return {"status": "started", "response": response}
            else:
                error_text = await response.text()
                print(f"❌ 聊天流启动失败: {response.status} - {error_text}")
                return {"status": "error", "error": error_text}
    
    async def cancel_chat(self, session_id: str) -> Dict[str, Any]:
        """取消聊天"""
        url = f"{self.base_url}/api/v1/chat/cancel"
        data = {"session_id": session_id}
        
        print(f"🛑 取消聊天: session_id={session_id}")
        
        async with self.session.post(url, json=data) as response:
            result = await response.json()
            if response.status == 200:
                print(f"✅ 聊天取消成功: {result}")
                return {"status": "cancelled", "result": result}
            else:
                print(f"❌ 聊天取消失败: {response.status} - {result}")
                return {"status": "error", "error": result}
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """获取会话状态"""
        url = f"{self.base_url}/api/v1/chat/session/{session_id}"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                result = await response.json()
                print(f"📊 会话状态: {result}")
                return {"status": "success", "result": result}
            else:
                error_text = await response.text()
                print(f"❌ 获取会话状态失败: {response.status} - {error_text}")
                return {"status": "error", "error": error_text}
    
    async def list_sessions(self) -> Dict[str, Any]:
        """列出所有会话"""
        url = f"{self.base_url}/api/v1/chat/sessions"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                result = await response.json()
                print(f"📋 会话列表: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return {"status": "success", "result": result}
            else:
                error_text = await response.text()
                print(f"❌ 获取会话列表失败: {response.status} - {error_text}")
                return {"status": "error", "error": error_text}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        url = f"{self.base_url}/api/v1/health"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                result = await response.json()
                print(f"💚 健康检查: {json.dumps(result, indent=2, ensure_ascii=False)}")
                return {"status": "success", "result": result}
            else:
                error_text = await response.text()
                print(f"❌ 健康检查失败: {response.status} - {error_text}")
                return {"status": "error", "error": error_text}


async def test_cancel_functionality():
    """测试取消功能"""
    print("🧪 开始测试聊天取消功能")
    print("=" * 50)
    
    async with ChatCancelTester() as tester:
        # 1. 健康检查
        print("\n1️⃣ 健康检查")
        await tester.health_check()
        
        # 2. 列出初始会话
        print("\n2️⃣ 列出初始会话")
        await tester.list_sessions()
        
        # 3. 启动一个聊天会话
        session_id = f"test_session_{int(time.time())}"
        print(f"\n3️⃣ 启动聊天会话: {session_id}")
        
        # 启动聊天流（在后台运行）
        chat_task = asyncio.create_task(
            tester.start_chat_stream(session_id, "请详细介绍人工智能的发展历史")
        )
        
        # 等待一小段时间让聊天开始
        await asyncio.sleep(1)
        
        # 4. 检查会话状态
        print(f"\n4️⃣ 检查会话状态")
        await tester.get_session_status(session_id)
        
        # 5. 列出当前会话
        print(f"\n5️⃣ 列出当前会话")
        await tester.list_sessions()
        
        # 6. 取消聊天
        print(f"\n6️⃣ 取消聊天")
        await tester.cancel_chat(session_id)
        
        # 7. 再次检查会话状态
        print(f"\n7️⃣ 检查取消后的会话状态")
        await tester.get_session_status(session_id)
        
        # 8. 等待聊天任务完成
        print(f"\n8️⃣ 等待聊天任务完成")
        try:
            await asyncio.wait_for(chat_task, timeout=5)
        except asyncio.TimeoutError:
            print("⏰ 聊天任务超时（预期行为，因为已取消）")
            chat_task.cancel()
        
        # 9. 最终会话列表
        print(f"\n9️⃣ 最终会话列表")
        await tester.list_sessions()
        
        # 10. 测试取消不存在的会话
        print(f"\n🔟 测试取消不存在的会话")
        await tester.cancel_chat("non_existent_session")
    
    print("\n" + "=" * 50)
    print("🎉 测试完成！")


async def test_concurrent_sessions():
    """测试并发会话取消"""
    print("\n🔄 测试并发会话取消")
    print("=" * 50)
    
    async with ChatCancelTester() as tester:
        # 创建多个并发会话
        session_ids = [f"concurrent_session_{i}_{int(time.time())}" for i in range(3)]
        
        # 启动多个聊天流
        chat_tasks = []
        for session_id in session_ids:
            task = asyncio.create_task(
                tester.start_chat_stream(session_id, f"会话 {session_id} 的查询")
            )
            chat_tasks.append(task)
        
        # 等待会话启动
        await asyncio.sleep(1)
        
        # 列出所有会话
        print("📋 并发会话列表:")
        await tester.list_sessions()
        
        # 取消部分会话
        for i, session_id in enumerate(session_ids):
            if i % 2 == 0:  # 取消偶数索引的会话
                print(f"🛑 取消会话: {session_id}")
                await tester.cancel_chat(session_id)
        
        # 再次列出会话状态
        print("📋 取消后的会话列表:")
        await tester.list_sessions()
        
        # 清理任务
        for task in chat_tasks:
            if not task.done():
                task.cancel()
        
        await asyncio.sleep(2)
        print("📋 最终会话列表:")
        await tester.list_sessions()


if __name__ == "__main__":
    print("🚀 启动聊天取消功能测试")
    
    try:
        # 基础功能测试
        asyncio.run(test_cancel_functionality())
        
        # 并发测试
        asyncio.run(test_concurrent_sessions())
        
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
