#!/usr/bin/env python3
"""
聊天取消功能使用示例
演示如何在实际应用中使用聊天取消功能
"""

import asyncio
import aiohttp
import json
import time
from typing import Optional


class VoiceHelperChatClient:
    """VoiceHelper 聊天客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8070"):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        self.active_chats = {}
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def start_chat(self, session_id: str, query: str, on_event=None):
        """
        启动聊天会话
        
        Args:
            session_id: 会话ID
            query: 查询内容
            on_event: 事件回调函数
        """
        url = f"{self.base_url}/api/v1/chat/stream"
        data = {
            "query": query,
            "session_id": session_id,
            "context": {}
        }
        
        print(f"🚀 启动聊天: {session_id}")
        print(f"📝 查询内容: {query}")
        
        try:
            async with self.session.post(url, json=data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ 聊天启动失败: {error_text}")
                    return
                
                self.active_chats[session_id] = {
                    "status": "active",
                    "start_time": time.time()
                }
                
                # 处理SSE流
                async for line in response.content:
                    if not line:
                        continue
                    
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith('data: '):
                        try:
                            event_data = json.loads(line_str[6:])
                            await self._handle_event(session_id, event_data, on_event)
                            
                            # 检查是否取消或完成
                            if event_data.get("event") in ["cancelled", "done", "error"]:
                                break
                                
                        except json.JSONDecodeError:
                            continue
                
        except Exception as e:
            print(f"❌ 聊天过程中出错: {e}")
        finally:
            if session_id in self.active_chats:
                self.active_chats[session_id]["status"] = "finished"
                print(f"🏁 聊天结束: {session_id}")
    
    async def _handle_event(self, session_id: str, event_data: dict, on_event=None):
        """处理聊天事件"""
        event_type = event_data.get("event", "unknown")
        data = event_data.get("data", {})
        
        # 更新会话状态
        if session_id in self.active_chats:
            self.active_chats[session_id]["last_event"] = event_type
            self.active_chats[session_id]["last_update"] = time.time()
        
        # 处理不同类型的事件
        if event_type == "intent":
            print(f"🧠 意图分析: {data}")
        elif event_type == "retrieve":
            print(f"🔍 检索结果: 找到 {data.get('count', 0)} 个相关文档")
        elif event_type == "plan":
            print(f"📋 执行计划: {data.get('strategy', 'unknown')}")
        elif event_type == "tool_result":
            print(f"🔧 工具执行: {data.get('tool', 'unknown')} - 成功")
        elif event_type == "answer":
            print(f"💬 AI回答: {data.get('text', '')[:100]}...")
        elif event_type == "cancelled":
            print(f"🛑 聊天已取消: {data.get('message', '')}")
        elif event_type == "done":
            print(f"✅ 聊天完成")
        elif event_type == "error":
            print(f"❌ 错误: {data.get('error', '')}")
        
        # 调用用户回调
        if on_event:
            await on_event(session_id, event_type, data)
    
    async def cancel_chat(self, session_id: str):
        """取消聊天"""
        url = f"{self.base_url}/api/v1/chat/cancel"
        data = {"session_id": session_id}
        
        print(f"🛑 取消聊天: {session_id}")
        
        async with self.session.post(url, json=data) as response:
            result = await response.json()
            if response.status == 200:
                print(f"✅ 取消成功: {result.get('message', '')}")
                if session_id in self.active_chats:
                    self.active_chats[session_id]["status"] = "cancelled"
                return True
            else:
                print(f"❌ 取消失败: {result}")
                return False
    
    async def get_session_status(self, session_id: str):
        """获取会话状态"""
        url = f"{self.base_url}/api/v1/chat/session/{session_id}"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                result = await response.json()
                print(f"📊 会话状态: {result['status']} (持续时间: {result['duration']:.1f}s)")
                return result
            else:
                print(f"❌ 获取状态失败: {response.status}")
                return None
    
    def get_active_chats(self):
        """获取本地活跃聊天列表"""
        return self.active_chats


async def example_basic_cancel():
    """基础取消示例"""
    print("🎯 示例1: 基础聊天取消")
    print("=" * 40)
    
    async with VoiceHelperChatClient() as client:
        session_id = f"basic_example_{int(time.time())}"
        
        # 启动聊天（在后台运行）
        chat_task = asyncio.create_task(
            client.start_chat(
                session_id, 
                "请详细介绍深度学习的发展历史，包括重要的里程碑事件"
            )
        )
        
        # 等待2秒后取消
        await asyncio.sleep(2)
        await client.cancel_chat(session_id)
        
        # 等待聊天任务完成
        try:
            await asyncio.wait_for(chat_task, timeout=5)
        except asyncio.TimeoutError:
            print("⏰ 聊天任务超时")
            chat_task.cancel()


async def example_conditional_cancel():
    """条件取消示例"""
    print("\n🎯 示例2: 条件取消")
    print("=" * 40)
    
    cancel_on_keyword = False
    
    async def event_handler(session_id, event_type, data):
        nonlocal cancel_on_keyword
        # 如果AI回答中包含特定关键词，则取消
        if event_type == "answer" and "神经网络" in data.get("text", ""):
            print("🔍 检测到关键词 '神经网络'，准备取消...")
            cancel_on_keyword = True
    
    async with VoiceHelperChatClient() as client:
        session_id = f"conditional_example_{int(time.time())}"
        
        # 启动聊天
        chat_task = asyncio.create_task(
            client.start_chat(
                session_id,
                "什么是神经网络？",
                on_event=event_handler
            )
        )
        
        # 监控取消条件
        while not chat_task.done():
            if cancel_on_keyword:
                await client.cancel_chat(session_id)
                break
            await asyncio.sleep(0.1)
        
        # 等待任务完成
        try:
            await asyncio.wait_for(chat_task, timeout=3)
        except asyncio.TimeoutError:
            chat_task.cancel()


async def example_multiple_sessions():
    """多会话管理示例"""
    print("\n🎯 示例3: 多会话管理")
    print("=" * 40)
    
    async with VoiceHelperChatClient() as client:
        # 创建多个会话
        sessions = [
            (f"session_1_{int(time.time())}", "介绍机器学习"),
            (f"session_2_{int(time.time())}", "什么是人工智能"),
            (f"session_3_{int(time.time())}", "深度学习的应用")
        ]
        
        # 启动所有会话
        tasks = []
        for session_id, query in sessions:
            task = asyncio.create_task(client.start_chat(session_id, query))
            tasks.append((session_id, task))
        
        # 等待1秒
        await asyncio.sleep(1)
        
        # 取消第一个和第三个会话
        await client.cancel_chat(sessions[0][0])
        await client.cancel_chat(sessions[2][0])
        
        # 等待所有任务完成
        for session_id, task in tasks:
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.TimeoutError:
                print(f"⏰ 会话 {session_id} 超时")
                task.cancel()
        
        # 显示最终状态
        print("\n📊 最终会话状态:")
        for session_id, _ in sessions:
            await client.get_session_status(session_id)


async def example_timeout_cancel():
    """超时取消示例"""
    print("\n🎯 示例4: 超时自动取消")
    print("=" * 40)
    
    async with VoiceHelperChatClient() as client:
        session_id = f"timeout_example_{int(time.time())}"
        timeout_seconds = 3
        
        print(f"⏰ 设置超时时间: {timeout_seconds} 秒")
        
        # 启动聊天
        chat_task = asyncio.create_task(
            client.start_chat(
                session_id,
                "请详细解释量子计算的原理和应用前景"
            )
        )
        
        try:
            # 等待指定时间
            await asyncio.wait_for(chat_task, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            print(f"⏰ 超时 {timeout_seconds} 秒，自动取消聊天")
            await client.cancel_chat(session_id)
            chat_task.cancel()


async def main():
    """主函数"""
    print("🚀 VoiceHelper 聊天取消功能示例")
    print("=" * 50)
    
    try:
        # 运行各种示例
        await example_basic_cancel()
        await example_conditional_cancel()
        await example_multiple_sessions()
        await example_timeout_cancel()
        
        print("\n🎉 所有示例运行完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 示例被用户中断")
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())
