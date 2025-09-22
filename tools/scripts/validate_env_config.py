#!/usr/bin/env python3
"""
环境配置验证脚本
验证统一的 .env 配置文件是否正确设置
"""

import os
import sys
from pathlib import Path

# 尝试导入 dotenv，如果不存在则手动解析 .env 文件
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    
    def load_dotenv(env_file):
        """手动解析 .env 文件"""
        if not os.path.exists(env_file):
            return
        
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ[key] = value

def load_env_file():
    """加载环境变量文件"""
    project_root = Path(__file__).parent.parent
    env_files = [
        project_root / ".env",
        project_root / "env.unified.new",
        project_root / "env.unified",
        project_root / "env.example"
    ]
    
    for env_file in env_files:
        if env_file.exists():
            print(f"✅ 找到环境配置文件: {env_file}")
            load_dotenv(env_file)
            return str(env_file)
    
    print("❌ 未找到任何环境配置文件")
    return None

def validate_required_configs():
    """验证必需的配置项"""
    required_configs = {
        # 基础配置
        "ENV": "development",
        "NODE_ENV": "development", 
        "FLASK_ENV": "development",
        
        # 服务端口
        "GATEWAY_PORT": "8080",
        "ALGO_PORT": "8000",
        "VOICE_PORT": "8001",
        "FRONTEND_PORT": "3000",
        "ADMIN_PORT": "5001",
        
        # 数据库配置
        "POSTGRES_HOST": "postgres",
        "POSTGRES_DB": "voicehelper",
        "POSTGRES_USER": "voicehelper",
        "POSTGRES_PASSWORD": "voicehelper123",
        "REDIS_HOST": "redis",
        "REDIS_PASSWORD": "redis123",
        
        # AI 模型配置
        "PRIMARY_MODEL": "glm-4-flash",
        "GLM_API_KEY": None,  # 必需但不检查具体值
        "GLM_BASE_URL": "https://open.bigmodel.cn/api/paas/v4",
        
        # 安全配置
        "JWT_SECRET": None,  # 必需但不检查具体值
        "ADMIN_SECRET_KEY": None,  # 必需但不检查具体值
    }
    
    missing_configs = []
    invalid_configs = []
    
    for key, expected_value in required_configs.items():
        actual_value = os.getenv(key)
        
        if actual_value is None:
            missing_configs.append(key)
        elif expected_value is not None and actual_value != expected_value:
            invalid_configs.append(f"{key}: 期望 '{expected_value}', 实际 '{actual_value}'")
    
    return missing_configs, invalid_configs

def validate_service_configs():
    """验证各服务的配置"""
    services = {
        "Gateway": {
            "port": os.getenv("GATEWAY_PORT", "8080"),
            "service_name": os.getenv("GATEWAY_SERVICE_NAME", "voicehelper-gateway")
        },
        "Algorithm": {
            "port": os.getenv("ALGO_PORT", "8000"), 
            "service_name": os.getenv("ALGO_SERVICE_NAME", "voicehelper-algo")
        },
        "Voice": {
            "port": os.getenv("VOICE_PORT", "8001"),
            "service_name": os.getenv("VOICE_SERVICE_NAME", "voicehelper-voice")
        },
        "Frontend": {
            "port": os.getenv("FRONTEND_PORT", "3000"),
            "service_name": os.getenv("FRONTEND_SERVICE_NAME", "voicehelper-frontend")
        },
        "Admin": {
            "port": os.getenv("ADMIN_PORT", "5001"),
            "service_name": os.getenv("ADMIN_SERVICE_NAME", "voicehelper-admin")
        }
    }
    
    return services

def validate_ai_models():
    """验证AI模型配置"""
    models = {}
    
    # GLM-4 配置
    if os.getenv("GLM_API_KEY"):
        models["GLM-4"] = {
            "api_key": "已配置" if os.getenv("GLM_API_KEY") != "your-glm-api-key-here" else "未配置",
            "base_url": os.getenv("GLM_BASE_URL", ""),
            "status": "✅ 可用" if os.getenv("GLM_API_KEY") and os.getenv("GLM_API_KEY") != "your-glm-api-key-here" else "❌ 需要配置"
        }
    
    # 豆包配置
    if os.getenv("ARK_API_KEY"):
        models["豆包 (ARK)"] = {
            "api_key": "已配置" if os.getenv("ARK_API_KEY") != "your-ark-api-key-here" else "未配置",
            "base_url": os.getenv("ARK_BASE_URL", ""),
            "model": os.getenv("ARK_MODEL", ""),
            "status": "✅ 可用" if os.getenv("ARK_API_KEY") and os.getenv("ARK_API_KEY") != "your-ark-api-key-here" else "❌ 需要配置"
        }
    
    # OpenAI 配置
    if os.getenv("OPENAI_API_KEY"):
        models["OpenAI"] = {
            "api_key": "已配置" if os.getenv("OPENAI_API_KEY") != "your-openai-api-key-here" else "未配置",
            "base_url": os.getenv("OPENAI_BASE_URL", ""),
            "status": "✅ 可用" if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your-openai-api-key-here" else "⚠️ 可选"
        }
    
    return models

def main():
    """主函数"""
    print("🔍 VoiceHelper 环境配置验证")
    print("=" * 50)
    
    # 1. 加载环境文件
    env_file = load_env_file()
    if not env_file:
        print("\n❌ 验证失败: 未找到环境配置文件")
        print("请确保存在以下文件之一:")
        print("  - .env")
        print("  - env.unified.new") 
        print("  - env.unified")
        print("  - env.example")
        sys.exit(1)
    
    print(f"📁 使用配置文件: {env_file}")
    
    # 2. 验证必需配置
    print("\n🔧 验证必需配置...")
    missing_configs, invalid_configs = validate_required_configs()
    
    if missing_configs:
        print("❌ 缺少以下配置:")
        for config in missing_configs:
            print(f"  - {config}")
    
    if invalid_configs:
        print("⚠️ 以下配置值不正确:")
        for config in invalid_configs:
            print(f"  - {config}")
    
    if not missing_configs and not invalid_configs:
        print("✅ 所有必需配置都已正确设置")
    
    # 3. 验证服务配置
    print("\n🚀 服务配置:")
    services = validate_service_configs()
    for service_name, config in services.items():
        print(f"  {service_name}:")
        print(f"    端口: {config['port']}")
        print(f"    服务名: {config['service_name']}")
    
    # 4. 验证AI模型配置
    print("\n🤖 AI模型配置:")
    models = validate_ai_models()
    if models:
        for model_name, config in models.items():
            print(f"  {model_name}: {config['status']}")
            if config.get('base_url'):
                print(f"    API地址: {config['base_url']}")
    else:
        print("  ❌ 未配置任何AI模型")
    
    # 5. 数据库配置
    print("\n🗄️ 数据库配置:")
    print(f"  PostgreSQL: {os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}")
    print(f"  Redis: {os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}")
    print(f"  Neo4j: {os.getenv('NEO4J_URI', 'bolt://localhost:7687')}")
    
    # 6. 总结
    print("\n📊 配置总结:")
    total_issues = len(missing_configs) + len(invalid_configs)
    if total_issues == 0:
        print("✅ 配置验证通过，可以启动服务")
        print("\n🚀 启动命令:")
        print("  docker-compose -f docker-compose.local.yml up -d")
    else:
        print(f"❌ 发现 {total_issues} 个配置问题，请修复后重试")
        print("\n🔧 修复步骤:")
        print("  1. 复制配置文件: cp env.unified.new .env")
        print("  2. 编辑 .env 文件，填入正确的API密钥")
        print("  3. 重新运行验证: python scripts/validate_env_config.py")

if __name__ == "__main__":
    main()
