"""
语音服务使用示例
演示如何使用新的语音服务进行ASR和TTS
"""

import asyncio
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.voice_config import create_voice_config, get_voice_provider_status
from core.enhanced_voice_services import EnhancedASRService, EnhancedTTSService, VoiceProvider

async def example_asr():
    """ASR使用示例"""
    print("=== ASR使用示例 ===")
    
    # 创建配置
    voice_config = create_voice_config()
    asr_service = EnhancedASRService(voice_config)
    
    print(f"使用ASR提供商: {voice_config.primary_asr_provider.value}")
    
    # 注意：这里需要真实的音频数据
    # 在实际使用中，audio_data应该是从麦克风或文件读取的音频数据
    print("注意：需要真实音频数据进行测试")
    print("示例代码:")
    print("""
    # 从文件读取音频
    with open('audio.wav', 'rb') as f:
        audio_data = f.read()
    
    # 转写音频
    text = await asr_service.transcribe(audio_data, language='zh-CN')
    print(f'识别结果: {text}')
    """)

async def example_tts():
    """TTS使用示例"""
    print("\n=== TTS使用示例 ===")
    
    # 创建配置
    voice_config = create_voice_config()
    tts_service = EnhancedTTSService(voice_config)
    
    print(f"使用TTS提供商: {voice_config.primary_tts_provider.value}")
    
    # 测试文本
    text = "你好，欢迎使用语音助手！今天天气不错。"
    print(f"合成文本: {text}")
    
    try:
        # 合成语音
        print("开始语音合成...")
        audio_data = await tts_service.synthesize(text)
        
        if audio_data:
            print(f"✅ 合成成功，音频大小: {len(audio_data)} bytes")
            
            # 保存音频文件
            output_file = "example_output.mp3"
            with open(output_file, "wb") as f:
                f.write(audio_data)
            print(f"音频已保存到: {output_file}")
            
        else:
            print("❌ 合成失败")
            
    except Exception as e:
        print(f"❌ TTS错误: {e}")

async def example_streaming_tts():
    """流式TTS使用示例"""
    print("\n=== 流式TTS使用示例 ===")
    
    # 创建配置
    voice_config = create_voice_config()
    tts_service = EnhancedTTSService(voice_config)
    
    text = "这是一个流式语音合成的示例。我们将文本分块处理，实现低延迟的语音输出。"
    print(f"合成文本: {text}")
    
    try:
        print("开始流式语音合成...")
        chunks = []
        chunk_count = 0
        
        async for chunk in tts_service.synthesize_streaming(text):
            if chunk:
                chunks.append(chunk)
                chunk_count += 1
                print(f"收到音频块 {chunk_count}: {len(chunk)} bytes")
        
        if chunks:
            # 合并所有音频块
            total_audio = b''.join(chunks)
            print(f"✅ 流式合成完成，总共 {chunk_count} 个块，总大小: {len(total_audio)} bytes")
            
            # 保存合并的音频
            output_file = "example_streaming_output.mp3"
            with open(output_file, "wb") as f:
                f.write(total_audio)
            print(f"合并音频已保存到: {output_file}")
            
        else:
            print("❌ 流式合成失败")
            
    except Exception as e:
        print(f"❌ 流式TTS错误: {e}")

async def example_provider_fallback():
    """提供商降级示例"""
    print("\n=== 提供商降级示例 ===")
    
    # 创建配置，故意设置一个不存在的主要提供商来测试降级
    voice_config = create_voice_config()
    
    # 显示配置的提供商
    print(f"主要TTS提供商: {voice_config.primary_tts_provider.value}")
    print(f"降级TTS提供商: {[p.value for p in voice_config.fallback_tts_providers]}")
    
    tts_service = EnhancedTTSService(voice_config)
    
    text = "测试提供商降级功能。"
    
    try:
        print("测试提供商降级...")
        audio_data = await tts_service.synthesize(text)
        
        if audio_data:
            print("✅ 合成成功（可能使用了降级提供商）")
            
            # 查看统计信息
            stats = tts_service.get_stats()
            print(f"统计信息: {stats}")
            
            if stats['fallback_usage'] > 0:
                print("🔄 使用了降级提供商")
            else:
                print("✨ 使用了主要提供商")
                
        else:
            print("❌ 所有提供商都失败了")
            
    except Exception as e:
        print(f"❌ 降级测试错误: {e}")

def show_configuration_guide():
    """显示配置指南"""
    print("\n=== 配置指南 ===")
    
    print("要使用语音服务，请配置以下环境变量：")
    print()
    
    print("1. OpenAI (推荐用于ASR):")
    print("   export OPENAI_API_KEY='your-openai-api-key'")
    print("   export OPENAI_BASE_URL='https://api.openai.com/v1'  # 可选")
    print()
    
    print("2. Azure Speech (推荐用于生产环境):")
    print("   export AZURE_SPEECH_KEY='your-azure-speech-key'")
    print("   export AZURE_SPEECH_REGION='eastus'  # 或其他区域")
    print()
    
    print("3. 免费选项:")
    print("   - Edge TTS: 无需配置，自动可用（仅TTS）")
    print("   - 本地ASR: 无需配置，使用Google Web Speech API（有限制）")
    print()
    
    print("4. 可选配置:")
    print("   export VOICE_ENABLE_VAD='true'        # 启用语音活动检测")
    print("   export VOICE_ENABLE_CACHE='true'      # 启用TTS缓存")
    print("   export VOICE_DEFAULT_LANGUAGE='zh-CN' # 默认语言")
    print()

async def main():
    """主函数"""
    print("🎤 语音服务使用示例")
    print("=" * 50)
    
    # 显示提供商状态
    status = get_voice_provider_status()
    print("当前提供商状态:")
    for provider, info in status['providers'].items():
        if info['available']:
            print(f"✅ {provider.upper()}")
        else:
            print(f"❌ {provider.upper()}: {info.get('reason', 'Unknown')}")
    
    # 如果没有可用的商业提供商，显示配置指南
    has_commercial = any(
        info['available'] and provider in ['openai', 'azure'] 
        for provider, info in status['providers'].items()
    )
    
    if not has_commercial:
        show_configuration_guide()
        print("\n注意: 当前只有免费提供商可用，功能可能受限。")
        print("建议配置至少一个商业提供商以获得更好的体验。")
    
    # 运行示例
    await example_asr()
    await example_tts()
    await example_streaming_tts()
    await example_provider_fallback()
    
    print("\n" + "=" * 50)
    print("✅ 示例运行完成")
    
    # 清理生成的文件
    for file in ["example_output.mp3", "example_streaming_output.mp3"]:
        if os.path.exists(file):
            print(f"生成的文件: {file}")

if __name__ == "__main__":
    asyncio.run(main())
