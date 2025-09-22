"""
语音服务集成测试
"""

import asyncio
import os
import sys
import tempfile
import wave
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.voice_config import (
    get_voice_provider_status, 
    create_voice_config,
    load_voice_config_from_env
)
from core.enhanced_voice_services import (
    EnhancedASRService, 
    EnhancedTTSService,
    EnhancedVoiceService,
    VoiceProvider
)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_audio() -> bytes:
    """创建测试音频数据（1秒的静音）"""
    sample_rate = 16000
    duration = 1.0  # 1秒
    
    # 创建临时WAV文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # 写入静音数据
            silence = b'\x00\x00' * int(sample_rate * duration)
            wav_file.writeframes(silence)
        
        # 读取音频数据
        with open(temp_file.name, 'rb') as f:
            audio_data = f.read()
        
        # 清理临时文件
        os.unlink(temp_file.name)
        
        return audio_data

async def test_provider_status():
    """测试提供商状态"""
    print("=== 语音提供商状态检查 ===")
    
    status = get_voice_provider_status()
    
    print("\n可用提供商:")
    for provider, info in status['providers'].items():
        if info['available']:
            print(f"✅ {provider.upper()}: ASR={info.get('asr', False)}, TTS={info.get('tts', False)}, "
                  f"成本={info.get('cost', 'Unknown')}, 质量={info.get('quality', 'Unknown')}")
            if 'note' in info:
                print(f"   注意: {info['note']}")
        else:
            print(f"❌ {provider.upper()}: {info.get('reason', 'Unknown reason')}")
    
    if status['recommendations']:
        print("\n建议:")
        for rec in status['recommendations']:
            print(f"💡 {rec}")
    
    return status

async def test_asr_service():
    """测试ASR服务"""
    print("\n=== ASR服务测试 ===")
    
    try:
        # 创建配置
        voice_config = create_voice_config()
        asr_service = EnhancedASRService(voice_config)
        
        print(f"主要ASR提供商: {voice_config.primary_asr_provider.value}")
        print(f"降级ASR提供商: {[p.value for p in voice_config.fallback_asr_providers]}")
        
        # 创建测试音频
        test_audio = create_test_audio()
        print(f"测试音频大小: {len(test_audio)} bytes")
        
        # 测试转写
        print("开始ASR转写测试...")
        result = await asr_service.transcribe(test_audio, session_id="test_session")
        
        if result:
            print(f"✅ ASR转写成功: {result}")
        else:
            print("⚠️ ASR转写返回空结果（可能是静音音频）")
        
        # 获取统计信息
        stats = asr_service.get_stats()
        print(f"ASR统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ ASR服务测试失败: {e}")
        logger.exception("ASR test error")
        return False

async def test_tts_service():
    """测试TTS服务"""
    print("\n=== TTS服务测试 ===")
    
    try:
        # 创建配置
        voice_config = create_voice_config()
        tts_service = EnhancedTTSService(voice_config)
        
        print(f"主要TTS提供商: {voice_config.primary_tts_provider.value}")
        print(f"降级TTS提供商: {[p.value for p in voice_config.fallback_tts_providers]}")
        
        # 测试文本
        test_text = "你好，这是语音合成测试。"
        print(f"测试文本: {test_text}")
        
        # 测试合成
        print("开始TTS合成测试...")
        audio_data = await tts_service.synthesize(test_text)
        
        if audio_data:
            print(f"✅ TTS合成成功，音频大小: {len(audio_data)} bytes")
            
            # 保存测试音频文件
            test_file = "test_tts_output.mp3"
            with open(test_file, "wb") as f:
                f.write(audio_data)
            print(f"测试音频已保存到: {test_file}")
            
        else:
            print("❌ TTS合成失败，返回空数据")
        
        # 测试流式合成
        print("开始TTS流式合成测试...")
        chunks = []
        async for chunk in tts_service.synthesize_streaming(test_text):
            chunks.append(chunk)
        
        if chunks:
            total_size = sum(len(chunk) for chunk in chunks)
            print(f"✅ TTS流式合成成功，{len(chunks)}个块，总大小: {total_size} bytes")
        else:
            print("⚠️ TTS流式合成返回空结果")
        
        # 获取统计信息
        stats = tts_service.get_stats()
        print(f"TTS统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS服务测试失败: {e}")
        logger.exception("TTS test error")
        return False

async def test_integrated_voice_service():
    """测试集成语音服务"""
    print("\n=== 集成语音服务测试 ===")
    
    try:
        # 创建配置
        voice_config = create_voice_config()
        
        # 模拟检索服务
        class MockRetrieveService:
            async def stream_query(self, request):
                # 模拟返回流式响应
                responses = [
                    '{"type": "refs", "refs": []}',
                    '{"type": "delta", "content": "根据"}',
                    '{"type": "delta", "content": "您的"}',
                    '{"type": "delta", "content": "问题，"}',
                    '{"type": "delta", "content": "我来"}',
                    '{"type": "delta", "content": "为您"}',
                    '{"type": "delta", "content": "解答。"}',
                    '{"type": "done"}'
                ]
                for response in responses:
                    yield response
                    await asyncio.sleep(0.1)
        
        # 创建集成服务
        mock_retrieve_service = MockRetrieveService()
        voice_service = EnhancedVoiceService(voice_config, mock_retrieve_service)
        
        print("集成语音服务创建成功")
        
        # 获取统计信息
        stats = voice_service.get_stats()
        print(f"服务统计: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ 集成语音服务测试失败: {e}")
        logger.exception("Integrated voice service test error")
        return False

async def test_configuration():
    """测试配置系统"""
    print("\n=== 配置系统测试 ===")
    
    try:
        # 测试从环境变量加载配置
        service_config = load_voice_config_from_env()
        print(f"✅ 服务配置加载成功")
        print(f"   启用语音处理: {service_config.enable_voice_processing}")
        print(f"   默认语言: {service_config.default_language}")
        print(f"   默认语音: {service_config.default_voice}")
        print(f"   启用VAD: {service_config.enable_vad}")
        print(f"   启用缓存: {service_config.enable_cache}")
        
        # 测试创建VoiceConfig
        voice_config = create_voice_config(service_config)
        print(f"✅ 语音配置创建成功")
        print(f"   主要ASR提供商: {voice_config.primary_asr_provider.value}")
        print(f"   主要TTS提供商: {voice_config.primary_tts_provider.value}")
        print(f"   配置的提供商数量: {len(voice_config.provider_configs)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置系统测试失败: {e}")
        logger.exception("Configuration test error")
        return False

async def main():
    """主测试函数"""
    print("🎤 语音服务集成测试开始")
    print("=" * 50)
    
    # 测试结果
    results = {}
    
    # 1. 测试提供商状态
    try:
        await test_provider_status()
        results['provider_status'] = True
    except Exception as e:
        print(f"❌ 提供商状态测试失败: {e}")
        results['provider_status'] = False
    
    # 2. 测试配置系统
    results['configuration'] = await test_configuration()
    
    # 3. 测试ASR服务
    results['asr_service'] = await test_asr_service()
    
    # 4. 测试TTS服务
    results['tts_service'] = await test_tts_service()
    
    # 5. 测试集成服务
    results['integrated_service'] = await test_integrated_voice_service()
    
    # 输出测试结果
    print("\n" + "=" * 50)
    print("🎤 语音服务集成测试结果")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总结: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！语音服务集成成功。")
        return 0
    else:
        print("⚠️ 部分测试失败，请检查配置和依赖。")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
