"""
语音服务配置管理
"""

import os
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .enhanced_voice_services import VoiceConfig, VoiceProvider

logger = logging.getLogger(__name__)

@dataclass
class VoiceServiceConfig:
    """语音服务配置"""
    
    # 基础配置
    enable_voice_processing: bool = True
    default_language: str = "zh-CN"
    default_voice: str = "zh-CN-XiaoxiaoNeural"
    
    # 提供商优先级配置
    asr_provider_priority: list = field(default_factory=lambda: [
        VoiceProvider.OPENAI,
        VoiceProvider.AZURE, 
        VoiceProvider.LOCAL
    ])
    
    tts_provider_priority: list = field(default_factory=lambda: [
        VoiceProvider.EDGE_TTS,  # 免费且质量好
        VoiceProvider.AZURE,
        VoiceProvider.OPENAI
    ])
    
    # 性能配置
    asr_timeout: float = 10.0
    tts_timeout: float = 15.0
    enable_vad: bool = True
    vad_aggressiveness: int = 2
    enable_cache: bool = True
    cache_ttl: int = 3600
    
    # 音频配置
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit
    
    # 提供商特定配置
    openai_model: str = "whisper-1"
    openai_tts_model: str = "tts-1"
    openai_voice: str = "alloy"
    
    azure_voice: str = "zh-CN-XiaoxiaoNeural"
    azure_speech_region: str = "eastus"
    
    edge_tts_voice: str = "zh-CN-XiaoxiaoNeural"
    
    # 本地ASR配置
    local_asr_energy_threshold: int = 300
    local_asr_pause_threshold: float = 0.8

def load_voice_config_from_env() -> VoiceServiceConfig:
    """从环境变量加载语音服务配置"""
    config = VoiceServiceConfig()
    
    # 基础配置
    config.enable_voice_processing = os.getenv('ENABLE_VOICE_PROCESSING', 'true').lower() == 'true'
    config.default_language = os.getenv('VOICE_DEFAULT_LANGUAGE', 'zh-CN')
    config.default_voice = os.getenv('VOICE_DEFAULT_VOICE', 'zh-CN-XiaoxiaoNeural')
    
    # 性能配置
    config.asr_timeout = float(os.getenv('VOICE_ASR_TIMEOUT', '10.0'))
    config.tts_timeout = float(os.getenv('VOICE_TTS_TIMEOUT', '15.0'))
    config.enable_vad = os.getenv('VOICE_ENABLE_VAD', 'true').lower() == 'true'
    config.vad_aggressiveness = int(os.getenv('VOICE_VAD_AGGRESSIVENESS', '2'))
    config.enable_cache = os.getenv('VOICE_ENABLE_CACHE', 'true').lower() == 'true'
    config.cache_ttl = int(os.getenv('VOICE_CACHE_TTL', '3600'))
    
    # 音频配置
    config.sample_rate = int(os.getenv('VOICE_SAMPLE_RATE', '16000'))
    config.channels = int(os.getenv('VOICE_CHANNELS', '1'))
    config.sample_width = int(os.getenv('VOICE_SAMPLE_WIDTH', '2'))
    
    # 提供商特定配置
    config.openai_model = os.getenv('OPENAI_ASR_MODEL', 'whisper-1')
    config.openai_tts_model = os.getenv('OPENAI_TTS_MODEL', 'tts-1')
    config.openai_voice = os.getenv('OPENAI_VOICE', 'alloy')
    
    config.azure_voice = os.getenv('AZURE_VOICE', 'zh-CN-XiaoxiaoNeural')
    config.azure_speech_region = os.getenv('AZURE_SPEECH_REGION', 'eastus')
    
    config.edge_tts_voice = os.getenv('EDGE_TTS_VOICE', 'zh-CN-XiaoxiaoNeural')
    
    config.local_asr_energy_threshold = int(os.getenv('LOCAL_ASR_ENERGY_THRESHOLD', '300'))
    config.local_asr_pause_threshold = float(os.getenv('LOCAL_ASR_PAUSE_THRESHOLD', '0.8'))
    
    return config

def create_voice_config(service_config: Optional[VoiceServiceConfig] = None) -> VoiceConfig:
    """创建VoiceConfig实例"""
    if service_config is None:
        service_config = load_voice_config_from_env()
    
    # 检查可用的提供商并构建配置
    provider_configs = {}
    available_asr_providers = []
    available_tts_providers = []
    
    # OpenAI配置
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if openai_api_key:
        provider_configs[VoiceProvider.OPENAI] = {
            'api_key': openai_api_key,
            'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
            'model': service_config.openai_model,
            'tts_model': service_config.openai_tts_model,
            'voice': service_config.openai_voice
        }
        available_asr_providers.append(VoiceProvider.OPENAI)
        available_tts_providers.append(VoiceProvider.OPENAI)
        logger.info("OpenAI voice provider configured")
    
    # Azure配置
    azure_speech_key = os.getenv('AZURE_SPEECH_KEY')
    if azure_speech_key:
        provider_configs[VoiceProvider.AZURE] = {
            'api_key': azure_speech_key,
            'region': service_config.azure_speech_region,
            'voice': service_config.azure_voice
        }
        available_asr_providers.append(VoiceProvider.AZURE)
        available_tts_providers.append(VoiceProvider.AZURE)
        logger.info("Azure Speech provider configured")
    
    # Edge TTS配置（总是可用）
    provider_configs[VoiceProvider.EDGE_TTS] = {
        'voice': service_config.edge_tts_voice
    }
    available_tts_providers.append(VoiceProvider.EDGE_TTS)
    logger.info("Edge TTS provider configured")
    
    # 本地ASR配置（总是可用）
    provider_configs[VoiceProvider.LOCAL] = {
        'energy_threshold': service_config.local_asr_energy_threshold,
        'pause_threshold': service_config.local_asr_pause_threshold
    }
    available_asr_providers.append(VoiceProvider.LOCAL)
    logger.info("Local ASR provider configured")
    
    # 确定主要提供商
    primary_asr_provider = VoiceProvider.LOCAL  # 默认
    for provider in service_config.asr_provider_priority:
        if provider in available_asr_providers:
            primary_asr_provider = provider
            break
    
    primary_tts_provider = VoiceProvider.EDGE_TTS  # 默认
    for provider in service_config.tts_provider_priority:
        if provider in available_tts_providers:
            primary_tts_provider = provider
            break
    
    # 构建降级提供商列表
    fallback_asr_providers = [p for p in available_asr_providers if p != primary_asr_provider]
    fallback_tts_providers = [p for p in available_tts_providers if p != primary_tts_provider]
    
    logger.info(f"Primary ASR provider: {primary_asr_provider.value}")
    logger.info(f"Fallback ASR providers: {[p.value for p in fallback_asr_providers]}")
    logger.info(f"Primary TTS provider: {primary_tts_provider.value}")
    logger.info(f"Fallback TTS providers: {[p.value for p in fallback_tts_providers]}")
    
    return VoiceConfig(
        primary_asr_provider=primary_asr_provider,
        fallback_asr_providers=fallback_asr_providers,
        asr_language=service_config.default_language,
        asr_timeout=service_config.asr_timeout,
        
        primary_tts_provider=primary_tts_provider,
        fallback_tts_providers=fallback_tts_providers,
        tts_voice=service_config.default_voice,
        tts_language=service_config.default_language,
        tts_timeout=service_config.tts_timeout,
        
        enable_vad=service_config.enable_vad,
        vad_aggressiveness=service_config.vad_aggressiveness,
        
        enable_cache=service_config.enable_cache,
        cache_ttl=service_config.cache_ttl,
        
        provider_configs=provider_configs
    )

def get_voice_provider_status() -> Dict[str, Any]:
    """获取语音提供商状态"""
    status = {
        'providers': {},
        'recommendations': []
    }
    
    # 检查OpenAI
    if os.getenv('OPENAI_API_KEY'):
        status['providers']['openai'] = {
            'available': True,
            'asr': True,
            'tts': True,
            'cost': 'Medium',
            'quality': 'High'
        }
    else:
        status['providers']['openai'] = {
            'available': False,
            'reason': 'OPENAI_API_KEY not configured'
        }
        status['recommendations'].append(
            'Configure OPENAI_API_KEY for high-quality ASR and TTS'
        )
    
    # 检查Azure
    if os.getenv('AZURE_SPEECH_KEY'):
        status['providers']['azure'] = {
            'available': True,
            'asr': True,
            'tts': True,
            'cost': 'Low',
            'quality': 'High'
        }
    else:
        status['providers']['azure'] = {
            'available': False,
            'reason': 'AZURE_SPEECH_KEY not configured'
        }
        status['recommendations'].append(
            'Configure AZURE_SPEECH_KEY for cost-effective voice services'
        )
    
    # Edge TTS（总是可用）
    status['providers']['edge_tts'] = {
        'available': True,
        'asr': False,
        'tts': True,
        'cost': 'Free',
        'quality': 'Good'
    }
    
    # 本地ASR（总是可用）
    status['providers']['local_asr'] = {
        'available': True,
        'asr': True,
        'tts': False,
        'cost': 'Free',
        'quality': 'Medium',
        'note': 'Uses Google Web Speech API (rate limited)'
    }
    
    # 生成推荐
    if not status['providers']['openai']['available'] and not status['providers']['azure']['available']:
        status['recommendations'].append(
            'Consider configuring at least one commercial provider for better quality'
        )
    
    return status

# 全局配置实例
_voice_config = None

def get_global_voice_config() -> VoiceConfig:
    """获取全局语音配置"""
    global _voice_config
    if _voice_config is None:
        _voice_config = create_voice_config()
    return _voice_config

def reload_voice_config():
    """重新加载语音配置"""
    global _voice_config
    _voice_config = create_voice_config()
    logger.info("Voice configuration reloaded")
