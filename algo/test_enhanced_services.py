#!/usr/bin/env python3
"""
VoiceHelper 增强服务测试脚本
测试新实现的核心功能：Whisper ASR、Edge-TTS、增强版FAISS RAG、Rasa对话管理
"""

import asyncio
import logging
import time
import base64
import numpy as np
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_whisper_asr():
    """测试Whisper ASR服务"""
    print("\n" + "="*50)
    print("测试 OpenAI Whisper ASR 服务")
    print("="*50)
    
    try:
        from core.whisper_realtime_asr import WhisperRealtimeASR, ASRConfig
        
        # 创建配置
        config = ASRConfig(
            model_size="tiny",  # 使用tiny模型加快测试
            language="zh",
            vad_aggressiveness=2
        )
        
        # 初始化服务
        asr = WhisperRealtimeASR(config)
        print("正在初始化Whisper模型...")
        await asr.initialize()
        print("✅ Whisper ASR服务初始化成功")
        
        # 生成测试音频数据（模拟）
        sample_rate = 16000
        duration = 2  # 2秒
        samples = int(sample_rate * duration)
        
        # 生成简单的正弦波作为测试音频
        frequency = 440  # A4音符
        t = np.linspace(0, duration, samples)
        audio_data = (np.sin(2 * np.pi * frequency * t) * 16384).astype(np.int16)
        audio_bytes = audio_data.tobytes()
        
        print("正在处理测试音频...")
        results = []
        async for result in asr.process_audio_stream(audio_bytes):
            results.append(result)
            print(f"  识别结果: '{result.text}' (置信度: {result.confidence:.2f}, 最终: {result.is_final})")
            if result.is_final:
                break
        
        # 获取统计信息
        stats = asr.get_stats()
        print(f"✅ ASR统计: {stats}")
        
    except Exception as e:
        print(f"❌ Whisper ASR测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_edge_tts():
    """测试Edge-TTS服务"""
    print("\n" + "="*50)
    print("测试 Edge-TTS 语音合成服务")
    print("="*50)
    
    try:
        from core.edge_tts_service import EdgeTTSService, TTSConfig, TTSRequest
        
        # 创建配置
        config = TTSConfig(
            voice="zh-CN-XiaoxiaoNeural",
            cache_enabled=True,
            max_cache_size_mb=100
        )
        
        # 初始化服务
        tts = EdgeTTSService(config)
        print("正在初始化Edge-TTS服务...")
        await tts.initialize()
        print("✅ Edge-TTS服务初始化成功")
        
        # 测试语音合成
        request = TTSRequest(
            text="你好，我是VoiceHelper语音助手，这是一个测试。",
            voice="zh-CN-XiaoxiaoNeural"
        )
        
        print("正在合成语音...")
        start_time = time.time()
        response = await tts.synthesize(request)
        end_time = time.time()
        
        print(f"✅ 语音合成成功:")
        print(f"  音频大小: {len(response.audio_data)} bytes")
        print(f"  处理时间: {response.processing_time_ms:.2f}ms")
        print(f"  预估时长: {response.duration_ms}ms")
        print(f"  是否缓存: {response.cached}")
        print(f"  使用语音: {response.voice_used}")
        
        # 测试缓存命中
        print("测试缓存命中...")
        response2 = await tts.synthesize(request)
        print(f"  第二次合成是否缓存: {response2.cached}")
        print(f"  第二次处理时间: {response2.processing_time_ms:.2f}ms")
        
        # 获取可用语音
        voices = await tts.get_available_voices()
        print(f"✅ 可用语音数量: {len(voices)}")
        
        # 获取统计信息
        stats = tts.get_stats()
        print(f"✅ TTS统计: {stats}")
        
    except Exception as e:
        print(f"❌ Edge-TTS测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_enhanced_faiss_rag():
    """测试增强版FAISS RAG服务"""
    print("\n" + "="*50)
    print("测试 增强版FAISS RAG 检索服务")
    print("="*50)
    
    try:
        from core.enhanced_faiss_rag import EnhancedFAISSRAG
        
        # 初始化RAG服务
        rag = EnhancedFAISSRAG(
            embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 使用较小的模型
            index_type="Flat",  # 使用简单索引
            data_dir="data/test_rag"
        )
        
        print("正在初始化RAG服务...")
        await rag.initialize()
        print("✅ RAG服务初始化成功")
        
        # 添加测试文档
        documents = [
            {
                "id": "doc1",
                "content": "VoiceHelper是一个智能语音助手平台，支持实时语音识别和语音合成功能。",
                "metadata": {"category": "product", "language": "zh"}
            },
            {
                "id": "doc2",
                "content": "该平台基于OpenAI Whisper进行语音识别，使用Edge-TTS进行语音合成。",
                "metadata": {"category": "technology", "language": "zh"}
            },
            {
                "id": "doc3",
                "content": "系统支持多租户架构，可以为不同的客户提供定制化的语音服务。",
                "metadata": {"category": "architecture", "language": "zh"}
            },
            {
                "id": "doc4",
                "content": "RAG检索系统使用FAISS向量数据库，支持高性能的语义搜索。",
                "metadata": {"category": "technology", "language": "zh"}
            }
        ]
        
        print("正在添加测试文档...")
        result = await rag.add_documents(documents)
        print(f"✅ 文档添加结果: {result}")
        
        # 测试搜索
        queries = [
            "语音识别功能",
            "多租户架构",
            "FAISS向量搜索",
            "语音合成技术"
        ]
        
        for query in queries:
            print(f"\n搜索查询: '{query}'")
            start_time = time.time()
            search_results = await rag.search(query, top_k=3)
            end_time = time.time()
            
            print(f"  搜索时间: {(end_time - start_time)*1000:.2f}ms")
            for i, result in enumerate(search_results):
                print(f"  结果{i+1}: {result.document.content[:50]}... (分数: {result.score:.3f})")
        
        # 测试混合搜索
        print(f"\n混合搜索测试: '语音助手平台'")
        hybrid_results = await rag.hybrid_search("语音助手平台", top_k=3)
        for i, result in enumerate(hybrid_results):
            print(f"  混合结果{i+1}: {result.document.content[:50]}... (分数: {result.score:.3f})")
        
        # 获取统计信息
        stats = await rag.get_stats()
        print(f"✅ RAG统计: {stats}")
        
    except Exception as e:
        print(f"❌ FAISS RAG测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_rasa_dialogue():
    """测试Rasa对话管理服务"""
    print("\n" + "="*50)
    print("测试 Rasa 对话管理服务")
    print("="*50)
    
    try:
        from core.rasa_dialogue import RasaDialogueManager
        
        # 注意：这个测试需要Rasa服务运行在localhost:5005
        dialogue_manager = RasaDialogueManager("http://localhost:5005")
        
        print("正在测试Rasa对话管理...")
        
        # 测试消息
        test_messages = [
            "你好",
            "今天天气怎么样？",
            "播放音乐",
            "设置提醒",
            "再见"
        ]
        
        async with dialogue_manager:
            for message in test_messages:
                print(f"\n用户: {message}")
                try:
                    response = await dialogue_manager.process_message(
                        user_id="test_user",
                        session_id="test_session",
                        message=message
                    )
                    
                    print(f"  助手: {response.text}")
                    print(f"  意图: {response.intent} (置信度: {response.confidence:.2f})")
                    if response.entities:
                        print(f"  实体: {response.entities}")
                    if response.actions:
                        print(f"  动作: {response.actions}")
                        
                except Exception as e:
                    print(f"  ⚠️ 对话处理失败 (可能Rasa服务未启动): {e}")
        
        print("✅ Rasa对话管理测试完成")
        
    except Exception as e:
        print(f"❌ Rasa对话管理测试失败: {e}")
        import traceback
        traceback.print_exc()

async def test_integrated_workflow():
    """测试集成工作流"""
    print("\n" + "="*50)
    print("测试 集成工作流")
    print("="*50)
    
    try:
        print("模拟完整的语音助手工作流:")
        print("1. 用户说话 -> 2. 语音识别 -> 3. 对话理解 -> 4. 知识检索 -> 5. 生成回复 -> 6. 语音合成")
        
        # 模拟用户语音输入
        user_speech = "你好，请告诉我关于语音识别的信息"
        print(f"\n1. 模拟用户语音: '{user_speech}'")
        
        # 2. 语音识别 (这里直接使用文本，实际应该是音频)
        recognized_text = user_speech
        print(f"2. 语音识别结果: '{recognized_text}'")
        
        # 3. 对话理解 (需要Rasa服务)
        print("3. 对话理解: 跳过 (需要Rasa服务)")
        
        # 4. 知识检索
        print("4. 知识检索...")
        try:
            from core.enhanced_faiss_rag import EnhancedFAISSRAG
            
            rag = EnhancedFAISSRAG(
                embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                index_type="Flat",
                data_dir="data/test_rag"
            )
            
            await rag.initialize()
            
            # 如果之前的测试已经添加了文档，这里可以直接搜索
            search_results = await rag.search("语音识别", top_k=2)
            if search_results:
                knowledge = search_results[0].document.content
                print(f"   检索到知识: {knowledge[:100]}...")
            else:
                knowledge = "语音识别是将语音信号转换为文本的技术。"
                print(f"   使用默认知识: {knowledge}")
                
        except Exception as e:
            knowledge = "语音识别是将语音信号转换为文本的技术。"
            print(f"   知识检索失败，使用默认知识: {e}")
        
        # 5. 生成回复
        response_text = f"根据我的了解，{knowledge} VoiceHelper平台使用OpenAI Whisper技术提供高质量的语音识别服务。"
        print(f"5. 生成回复: '{response_text[:100]}...'")
        
        # 6. 语音合成
        print("6. 语音合成...")
        try:
            from core.edge_tts_service import EdgeTTSService, TTSConfig, TTSRequest
            
            config = TTSConfig(voice="zh-CN-XiaoxiaoNeural")
            tts = EdgeTTSService(config)
            await tts.initialize()
            
            request = TTSRequest(text=response_text[:100])  # 限制长度
            tts_response = await tts.synthesize(request)
            
            print(f"   语音合成成功: {len(tts_response.audio_data)} bytes")
            print(f"   处理时间: {tts_response.processing_time_ms:.2f}ms")
            
        except Exception as e:
            print(f"   语音合成失败: {e}")
        
        print("\n✅ 集成工作流测试完成")
        
    except Exception as e:
        print(f"❌ 集成工作流测试失败: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """主测试函数"""
    print("VoiceHelper 增强服务测试")
    print("="*60)
    
    # 创建测试数据目录
    Path("data/test_rag").mkdir(parents=True, exist_ok=True)
    Path("data/tts_cache").mkdir(parents=True, exist_ok=True)
    
    # 运行各项测试
    await test_whisper_asr()
    await test_edge_tts()
    await test_enhanced_faiss_rag()
    await test_rasa_dialogue()
    await test_integrated_workflow()
    
    print("\n" + "="*60)
    print("所有测试完成！")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
